"""Checkpoint layout: download, split into per-layer shards, and read shards back.

A checkpoint is split once into one file per module (embedding, each decoder layer, final norm,
lm_head, plus any resident or CPU-resident modules), so streaming can read exactly one at a time.
"""
import json
import os
import shutil
import time
from collections import defaultdict
from glob import glob
from pathlib import Path

import huggingface_hub
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from tqdm import tqdm

from .compression_bnb import bitsandbytes_installed, compress_layer_state_dict, uncompress_layer_state_dict
from .memory import clean_memory
from .ngram import persist_ngram_mmap
from .persist import ModelPersister


class NotEnoughSpaceException(Exception):
    pass


def cpu_resident_module_names(layer_names, index_keys):
    """Module prefixes whose weights should stay on CPU after the split.

    ``cpu_resident`` is an explicit list. ``cpu_resident_marker`` (e.g. the Flash-Next PLE table)
    is scanned out of the checkpoint so we do not have to hard-code which decoder layer owns it.
    """
    names = list(layer_names.get('cpu_resident', []))
    marker = layer_names.get('cpu_resident_marker')
    if marker:
        found = []
        for key in index_keys:
            pos = key.find(marker)
            if pos == -1:
                continue
            prefix = key[:pos + len(marker)]
            if prefix not in names and prefix not in found:
                found.append(prefix)
        names.extend(found)
    return names


def layer_owner(key, layer_prefixes):
    """Longest matching prefix wins, so a nested cpu-resident module is not swallowed by its parent."""
    owner = None
    for prefix in layer_prefixes:
        if key.startswith(prefix) and (owner is None or len(prefix) > len(owner)):
            owner = prefix
    return owner


def layer_tensor_names(local_path, layer_name):
    """List the tensors in a layer shard without reading any tensor data."""
    with safe_open(str(Path(local_path) / (layer_name + ".safetensors")), framework="pt") as f:
        return list(f.keys())


def load_layer_subset(local_path, layer_name, keys):
    """Read only `keys` from a layer shard.

    safetensors can seek to individual tensors, so a single MoE expert costs its own few MB rather
    than the whole ~16GB layer file. That is what makes per-expert streaming worthwhile.
    """
    out = {}
    with safe_open(str(Path(local_path) / (layer_name + ".safetensors")), framework="pt") as f:
        for k in keys:
            out[k] = f.get_tensor(k)
    return out


def load_layer(local_path, layer_name, profiling=False, persister=None):
    #layer_state_dict = load_file(Path(local_path) / (layer_name + ".safetensors"), device="cpu")
    persister = persister or ModelPersister.get_model_persister()
    layer_state_dict = persister.load_model(layer_name, local_path)

    if profiling:
        t = time.process_time()

    to_return = uncompress_layer_state_dict(layer_state_dict)

    #clean_memory()

    if profiling:
        elapsed_time = time.process_time() - t
        return to_return, elapsed_time
    else:
        return to_return


def check_space(checkpoint_path, layer_shards_saving_path=None, compression=None, splitted_model_dir_name='splitted_model'):
    total_shard_files_size_bytes = 0
    for model_shard_file in glob(str(checkpoint_path / '*')):
        total_shard_files_size_bytes += os.path.getsize(model_shard_file)

    total_saved_split_files_size_bytes = 0
    if layer_shards_saving_path is not None:
        for saved_split_file in glob(str(Path(layer_shards_saving_path) / splitted_model_dir_name / '*')):
            total_saved_split_files_size_bytes += os.path.getsize(saved_split_file)

    if compression == '4bit':
        total_shard_files_size_bytes = int(total_shard_files_size_bytes / 0.2813)
    elif compression == '8bit':
        total_shard_files_size_bytes = total_shard_files_size_bytes // 2

    total, used, free = shutil.disk_usage(checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path)

    if free + total_saved_split_files_size_bytes < total_shard_files_size_bytes:
        raise NotEnoughSpaceException(f"Not enough space. Free space under {checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path}:"  \
                                      f" {free / 1024 / 1024 / 1024:.02f}GB. Model total size: {total_shard_files_size_bytes / 1024 / 1024 / 1024:.02f}GB. " \
                                      f"existing space under {checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path} assuming can reuse: {total_saved_split_files_size_bytes/ 1024 / 1024 / 1024:.02f}GB. "
                                      )


def remove_real_and_linked_file(to_delete):
    """Delete a file and, when it is a symlink (as in the Hugging Face cache), the blob behind it.

    Split shards that were hard-linked to the blob keep their data: removing one link of a
    hard-linked file leaves the others intact.
    """
    to_delete = str(to_delete)
    target = os.path.realpath(to_delete)
    os.remove(to_delete)
    if target != os.path.abspath(to_delete) and os.path.exists(target):
        os.remove(target)


def link_or_copy_file(src, dst):
    """Point dst at src's data without duplicating it, falling back to a real copy.

    A hard link is preferred over a symlink because it keeps the data alive even if the original
    checkpoint file is later deleted (``delete_original``), and because it costs no extra disk.
    Hard links need both paths on one filesystem, so we degrade to a symlink and finally to a copy.
    Hugging Face caches store files as symlinks into a blob dir, so we always link the real file.
    """
    src = Path(os.path.realpath(str(src)))
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    try:
        os.link(src, dst)
        return 'hardlink'
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return 'symlink'
    except OSError:
        pass
    shutil.copyfile(src, dst)
    return 'copy'


_WEIGHT_FILES = ('pytorch_model.bin.index.json', 'model.safetensors.index.json',
                 'model.safetensors', 'pytorch_model.bin')


def read_weight_map(checkpoint_path):
    """(tensor name -> file holding it, whether the files are safetensors).

    Multi-shard checkpoints ship an index.json; small/modern models often ship a single file with
    no index, so one is synthesized.
    """
    checkpoint_path = Path(checkpoint_path)
    for index_name, is_safetensors in (('pytorch_model.bin.index.json', False),
                                       ('model.safetensors.index.json', True)):
        if os.path.exists(checkpoint_path / index_name):
            with open(checkpoint_path / index_name, 'rb') as f:
                return json.load(f)['weight_map'], is_safetensors
    if os.path.exists(checkpoint_path / 'model.safetensors'):
        with safe_open(str(checkpoint_path / 'model.safetensors'), framework='pt') as f:
            return {k: 'model.safetensors' for k in f.keys()}, True
    if os.path.exists(checkpoint_path / 'pytorch_model.bin'):
        single_sd = torch.load(checkpoint_path / 'pytorch_model.bin', map_location='cpu')
        return {k: 'pytorch_model.bin' for k in single_sd.keys()}, False
    raise FileNotFoundError(
        f"No model weights found under {checkpoint_path}. Expected one of: "
        f"model.safetensors(.index.json) or pytorch_model.bin(.index.json).")


def _shard_number(filename):
    """Ordinal of a ``model-0000N-of-0000M`` shard file, or None for a single-file checkpoint."""
    parts = filename.split('-')
    if len(parts) > 1:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def module_prefixes(index, layer_names):
    """Module prefixes to split into their own shard (each ending in '.'), in safe split order."""
    if layer_names is None:
        n_layers = len({int(k.split('.')[2]) for k in index if 'model.layers' in k})
        layers = (['model.embed_tokens.'] + [f'model.layers.{i}.' for i in range(n_layers)]
                  + ['model.norm.', 'lm_head.'])
    else:
        prefix = layer_names['layer_prefix']
        n_layers = len({int(k[len(prefix):].split('.')[1]) for k in index if prefix in k})
        layers = ([layer_names['embed']] + [f'{prefix}.{i}' for i in range(n_layers)]
                  + [layer_names['norm'], layer_names['lm_head']])
        # Modules outside the streamed sequence that still need their weights on disk (a vision
        # tower, extra norms, host-only tables). They get their own shard.
        layers += list(layer_names.get('resident', []))
        layers += cpu_resident_module_names(layer_names, index.keys())
        layers = [layer + "." for layer in layers]

    # Drop modules with no weights of their own (tied lm_head), or we'd save an empty shard (which
    # fails) and never detect the split as complete.
    layers = [layer for layer in layers if any(k.startswith(layer) for k in index)]
    owned = {layer_owner(k, layers) for k in index}
    layers = [layer for layer in layers if layer in owned]

    # Split in ascending shard order: the splitter only walks its shard cursor forward, so a module
    # whose weights sit in an earlier shard than its predecessor's would be saved incomplete. The
    # sort is stable, so plain embed -> layers -> norm -> lm_head checkpoints keep their order. On a
    # tie the longer prefix goes first, so a nested cpu-resident table is extracted before its
    # parent decoder layer can swallow it.
    def last_shard(layer):
        nums = [n for k, v in index.items() if k.startswith(layer) and (n := _shard_number(v)) is not None]
        return max(nums) if nums else -1

    layers.sort(key=lambda layer: (last_shard(layer), -len(layer)))
    return layers


class _CheckpointSplitter:
    """One split run: reads checkpoint shards in order and writes one shard per module."""

    def __init__(self, checkpoint_path, layer_shards_saving_path, splitted_model_dir_name, compression,
                 layer_names, delete_original, repo_id, hf_token, persister):
        self.persister = persister
        self.compression = compression
        # A persister that quantizes in its own format (MLX) takes `compression` itself; otherwise
        # shards are compressed here with bitsandbytes, which only runs on CUDA.
        self.native_compression = compression is not None and \
            getattr(persister, 'supports_native_compression', False)
        if compression is not None:
            if not self.native_compression:
                assert bitsandbytes_installed, "when using compression bitsandbytes has to be installed."
            splitted_model_dir_name = splitted_model_dir_name + "." + compression
        self.dir_name = splitted_model_dir_name
        self.checkpoint_path = Path(checkpoint_path)
        self.layer_shards_saving_path = layer_shards_saving_path
        base = Path(layer_shards_saving_path) if layer_shards_saving_path is not None else self.checkpoint_path
        self.saving_path = base / splitted_model_dir_name
        self.layer_names = layer_names
        self.marker = (layer_names or {}).get('cpu_resident_marker')
        self.delete_original, self.repo_id, self.hf_token = delete_original, repo_id, hf_token

        self.index, self.safetensors_format = read_weight_map(self.checkpoint_path)
        self.layers = module_prefixes(self.index, layer_names)
        # Shard ordinal -> real filename, straight from the index: repos differ in zero-padding
        # width (DeepSeek uses model-00001-of-000004) and extension, so names can't be rebuilt.
        self.shard_files = {n: v for v in set(self.index.values()) if (n := _shard_number(v)) is not None}
        self.shard = 0               # highest checkpoint shard read so far
        self.state_dict = {}         # tensors read but not yet written to a module shard
        self.single_modelfile = None

    def run(self):
        if self._already_split():
            return str(self.saving_path)
        self.passthrough = self._one_module_shards()
        # Must exist before check_space, which stats the filesystem it lives on.
        self.saving_path.mkdir(parents=True, exist_ok=True)
        # A copy is only made for the modules we cannot link, so only those need free space.
        if not self.delete_original and len(self.passthrough) < len(self.layers):
            check_space(self.checkpoint_path, self.layer_shards_saving_path, self.compression,
                        splitted_model_dir_name=self.dir_name)
        for layer in tqdm(self.layers):
            self._split_module(layer)
        # A single-file checkpoint can only be deleted once every module has been written.
        if self.delete_original and self.single_modelfile is not None:
            self._delete_original(self.single_modelfile)
        return str(self.saving_path)

    def _already_split(self):
        if not os.path.exists(self.saving_path):
            return False
        found_layers = {layer: self.persister.model_persist_exist(layer, self.saving_path)
                        for layer in self.layers}
        print(f"found_layers:{found_layers}")
        if all(found_layers.values()):
            print(f"saved layers already found in {self.saving_path}")
            return True
        print("some layer splits found, some are not, re-save all layers in case there's some corruptions.")
        return False

    def _one_module_shards(self):
        """Modules whose checkpoint shard holds nothing else, so the shard can be linked, not copied.

        Kimi K3 ships one ~17GB shard per decoder layer; rewriting those would duplicate 1.5TB+ to
        produce byte-identical files. Linking only works when the persister reads that format.
        """
        passthrough = {}
        if not (self.compression is None and self.safetensors_format
                and type(self.persister).__name__ == 'SafetensorModelPersister'):
            return passthrough
        shard_contents = defaultdict(list)
        for k, v in self.index.items():
            shard_contents[v].append(k)
        for layer in self.layers:
            files = {v for k, v in self.index.items() if k.startswith(layer)}
            if len(files) == 1:
                only_file = next(iter(files))
                if all(k.startswith(layer) for k in shard_contents[only_file]):
                    passthrough[layer] = only_file
        if passthrough:
            print(f"{len(passthrough)}/{len(self.layers)} modules are already one-per-shard; "
                  f"linking to the original files instead of copying them.")
        return passthrough

    def _split_module(self, layer):
        if self.marker and self.marker in layer:
            persist_ngram_mmap(layer, self.index, self.checkpoint_path, self.saving_path,
                               repo_id=self.repo_id, hf_token=self.hf_token)
        elif layer in self.passthrough:
            self._link_module(layer)
        else:
            self._read_shards_for(layer)
            self._save_module(layer)

    def _link_module(self, layer):
        src = self._ensure_downloaded(self.checkpoint_path / self.passthrough[layer])
        dst = self.saving_path / (layer + 'safetensors')
        if not self.persister.model_persist_exist(layer, self.saving_path):
            link_or_copy_file(src, dst)
            (self.saving_path / (layer + 'safetensors.done')).touch()
        # A hard link or copy keeps the data once the original is gone; a symlink would not.
        if self.delete_original and not dst.is_symlink():
            self._delete_original(self.passthrough[layer])
        # Keep the shard cursor in step with what we skipped, so a later module that does need
        # loading doesn't walk back through (and read) every shard we just linked past.
        number = _shard_number(self.passthrough[layer])
        if number is not None:
            self.shard = max(self.shard, number)

    def _read_shards_for(self, layer):
        """Bring ``layer``'s tensors into self.state_dict.

        Safetensors checkpoints are read tensor by tensor, so memory holds one module rather than
        a whole multi-GB shard. Pickled (.bin) checkpoints can only be loaded whole.
        """
        if self.safetensors_format:
            self._read_module_tensors(layer)
        else:
            self._read_whole_shards_for(layer)

    def _module_files(self, layer):
        """{checkpoint file: [keys]} for the tensors ``layer`` owns (host-only tables excluded)."""
        files = defaultdict(list)
        for k, v in self.index.items():
            if layer_owner(k, self.layers) == layer and not (self.marker and self.marker in k):
                files[v].append(k)
        return files

    def _read_module_tensors(self, layer):
        files = self._module_files(layer)
        for filename, keys in files.items():
            if _shard_number(filename) is None:
                self.single_modelfile = filename
            path = self._ensure_downloaded(self.checkpoint_path / filename)
            with safe_open(path, framework='pt', device='cpu') as f:
                for k in keys:
                    self.state_dict[k] = f.get_tensor(k)
        if self.delete_original:
            self._delete_shards_done_after(layer, files)

    def _delete_shards_done_after(self, layer, files):
        """Delete each shard once the last module that reads from it has been read."""
        if not hasattr(self, '_last_reader'):
            # Any module may read a shard, host-only tables included (persist_ngram_mmap reads
            # them itself), so a shard they share is never deleted before they run.
            position = {m: i for i, m in enumerate(self.layers)}
            self._last_reader = {}
            for k, f in self.index.items():
                owner = layer_owner(k, self.layers)
                if owner in position and position[owner] >= position.get(self._last_reader.get(f), -1):
                    self._last_reader[f] = owner
        for filename in files:
            if self._last_reader.get(filename) == layer and _shard_number(filename) is not None:
                self._delete_original(filename)

    def _read_whole_shards_for(self, layer):
        """Bring every checkpoint shard holding part of ``layer`` into self.state_dict."""
        shards = [n for k, v in self.index.items() if k.startswith(layer) and (n := _shard_number(v)) is not None]
        if not shards:
            self.single_modelfile = next(v for k, v in self.index.items() if k.startswith(layer))
            self._read_file(self.checkpoint_path / self.single_modelfile)
            return
        # A module can span several shards (fp8 checkpoints keep each weight's weight_scale_inv
        # alongside it). Read every shard up to the highest one it references, or the module is
        # saved missing tensors, which silently corrupts fp8 weights.
        while max(shards) > self.shard:
            # Tensors of the shard we're leaving are already in RAM, so the file can go.
            if self.delete_original and self.shard != 0:
                self._delete_original(self.shard_files[self.shard])
            self.shard += 1
            print(f'Loading shard {self.shard}/{len(set(self.index.values()))}')
            self._read_file(self.checkpoint_path / self.shard_files[self.shard])

    def _read_file(self, path):
        path = self._ensure_downloaded(path)
        loaded = load_file(path, device='cpu') if self.safetensors_format else torch.load(path, map_location='cpu')
        if self.marker:
            # Host-only tables are written separately (persist_ngram_mmap), never via the state dict.
            loaded = {k: v for k, v in loaded.items() if self.marker not in k}
        self.state_dict.update(loaded)

    def _save_module(self, layer):
        # Longest prefix wins, so e.g. a PLE table nested under a decoder layer is not written into
        # that layer's shard (and later streamed onto the GPU).
        layer_state_dict = {k: v for k, v in self.state_dict.items() if layer_owner(k, self.layers) == layer}
        if self.native_compression:
            persist_kwargs = {'compression': self.compression}
        else:
            layer_state_dict = compress_layer_state_dict(layer_state_dict, self.compression)
            persist_kwargs = {}
        if not self.persister.model_persist_exist(layer, self.saving_path):
            self.persister.persist_model(layer_state_dict, layer, self.saving_path, **persist_kwargs)
        for k in layer_state_dict:
            self.state_dict.pop(k, None)
        del layer_state_dict
        # Splitting may hold a whole .bin shard in RAM, so free memory fully every module.
        clean_memory(full=True)

    def _ensure_downloaded(self, path):
        if not os.path.exists(path):
            assert self.repo_id is not None
            huggingface_hub.snapshot_download(self.repo_id, allow_patterns=os.path.basename(path),
                                              token=self.hf_token)
        return path

    def _delete_original(self, filename):
        to_delete = self.checkpoint_path / filename
        print(f"deleting original file: {to_delete}")
        remove_real_and_linked_file(to_delete)


def split_and_save_layers(checkpoint_path, layer_shards_saving_path=None, splitted_model_dir_name='splitted_model',
                          compression=None, layer_names=None, delete_original=False, repo_id=None, hf_token=None,
                          persister=None):
    """Split a checkpoint into one shard per module and return the folder that holds them.

    ``persister`` decides the on-disk format and should be the one the caller's runtime reads; the
    global default is kept for code that calls this directly. A finished split is reused.
    """
    persister = persister or ModelPersister.get_model_persister()
    return _CheckpointSplitter(checkpoint_path, layer_shards_saving_path, splitted_model_dir_name, compression,
                               layer_names, delete_original, repo_id, hf_token, persister).run()


def _download_checkpoint_metadata(repo_id, hf_token):
    """Download a repo without its weights, plus the weight file when there is no index to stream.

    Multi-shard checkpoints are streamed shard by shard during the split; a single-file checkpoint
    has no index, so its structure is only known from the file itself.
    """
    hf_cache_path = huggingface_hub.snapshot_download(repo_id, token=hf_token,
                                                      ignore_patterns=['*.safetensors', '*.bin'])
    has_index = os.path.exists(Path(hf_cache_path) / 'model.safetensors.index.json') or \
        os.path.exists(Path(hf_cache_path) / 'pytorch_model.bin.index.json')
    if not has_index:
        hf_cache_path = huggingface_hub.snapshot_download(
            repo_id, token=hf_token, allow_patterns=['model.safetensors', 'pytorch_model.bin'])
    return hf_cache_path


def find_or_create_local_splitted_path(model_local_path_or_repo_id, layer_shards_saving_path=None, compression=None,
                                       layer_names=None, hf_token=None, delete_original=False, persister=None):
    """Locate (or download) a checkpoint and split it into per-module shards.

    Returns ``(model_local_path, split_shards_path)``. ``model_local_path_or_repo_id`` is a local
    checkpoint folder or a Hugging Face repo id; ``compression`` ('4bit' / '8bit') stores the
    shards quantized; ``layer_shards_saving_path`` puts the split elsewhere than next to the model.
    """
    split_kwargs = dict(compression=compression, layer_names=layer_names, delete_original=delete_original,
                        persister=persister)
    if os.path.exists(model_local_path_or_repo_id):
        # Single-file checkpoints count too: the splitter handles them, and treating such a folder
        # as a repo id would fail as an invalid repo name.
        if any(os.path.exists(Path(model_local_path_or_repo_id) / f) for f in _WEIGHT_FILES):
            print("found local checkpoint...")
            return Path(model_local_path_or_repo_id), split_and_save_layers(
                model_local_path_or_repo_id, layer_shards_saving_path, **split_kwargs)
        print(f"Found local directory in {model_local_path_or_repo_id}, but didn't find downloaded model. "
              f"Try using {model_local_path_or_repo_id} as a HF repo...")

    hf_cache_path = _download_checkpoint_metadata(model_local_path_or_repo_id, hf_token)
    return Path(hf_cache_path), split_and_save_layers(
        hf_cache_path, layer_shards_saving_path, repo_id=model_local_path_or_repo_id, hf_token=hf_token,
        **split_kwargs)
