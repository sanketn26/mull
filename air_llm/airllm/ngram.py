"""Huge n-gram (PLE) embedding tables: merge shards, keep them file-mapped on the host.

Qwen3.8-Flash-Next ships a ~51B-entry table that must never be copied into RAM or onto the GPU.
"""
import json
import os
import re
from contextlib import ExitStack, contextmanager
from pathlib import Path

import huggingface_hub
import torch
import torch.nn as nn
from safetensors import safe_open


# Qwen4-Exp / Flash-Next stores the ~51B PLE n-gram table as ``*.ngram_embedding.shard_N.weight``.
# Transformers concatenates those shards into a single ``nn.Embedding.weight`` at load time.
# On machines that cannot hold ~102GB of host RAM we instead stream the shards into a file-backed
# mmap and gather rows from disk (the modeling code already looks up on ``weight.device``).
_NGRAM_SHARD_RE = re.compile(r'^(?P<prefix>.*\.ngram_embedding)\.shard_(?P<idx>\d+)\.weight$')


_DTYPE_TAGS = {
    torch.bfloat16: 'bf16',
    torch.float16: 'fp16',
    torch.float32: 'fp32',
}


_TAG_DTYPES = {tag: dtype for dtype, tag in _DTYPE_TAGS.items()}


def merge_ngram_embedding_shards(state_dict):
    """Concatenate ``ngram_embedding.shard_N.weight`` tensors into ``ngram_embedding.weight``.

    Keys that are not n-gram shards pass through unchanged. Shard indices must be a contiguous
    0..N-1 range; a gap would silently drop rows from the hash table.
    """
    groups = {}
    out = {}
    for key, value in state_dict.items():
        match = _NGRAM_SHARD_RE.match(key)
        if match:
            groups.setdefault(match.group('prefix'), []).append((int(match.group('idx')), value))
        else:
            out[key] = value
    for prefix, shards in groups.items():
        shards.sort(key=lambda item: item[0])
        got = [idx for idx, _ in shards]
        expected = list(range(len(shards)))
        if got != expected:
            raise ValueError(
                f"ngram embedding shards for {prefix} are not contiguous 0..{len(shards) - 1}: {got}"
            )
        out[f'{prefix}.weight'] = torch.cat([tensor for _, tensor in shards], dim=0)
    return out


def load_merged_ngram_embedding(local_path, layer_name):
    """Load a split n-gram file and concatenate shards without holding two full copies.

    ``load_file`` plus ``torch.cat`` would peak at ~2x the table (~200GB for Flash-Next). This
    reads one shard at a time into a preallocated host tensor.
    """
    filepath = Path(local_path) / (layer_name + ".safetensors")
    with safe_open(str(filepath), framework="pt") as handle:
        groups = {}
        rest = {}
        for key in handle.keys():
            match = _NGRAM_SHARD_RE.match(key)
            if match:
                groups.setdefault(match.group('prefix'), []).append((int(match.group('idx')), key))
            else:
                rest[key] = handle.get_tensor(key)
        for prefix, shards in groups.items():
            shards.sort(key=lambda item: item[0])
            got = [idx for idx, _ in shards]
            expected = list(range(len(shards)))
            if got != expected:
                raise ValueError(
                    f"ngram embedding shards for {prefix} are not contiguous "
                    f"0..{len(shards) - 1}: {got}"
                )
            shapes = [tuple(handle.get_slice(key).get_shape()) for _, key in shards]
            first = handle.get_tensor(shards[0][1])
            dest = first.new_empty((sum(shape[0] for shape in shapes),) + shapes[0][1:])
            dest[0:shapes[0][0]].copy_(first)
            del first
            offset = shapes[0][0]
            for (_, key), shape in zip(shards[1:], shapes[1:]):
                piece = handle.get_tensor(key)
                dest[offset:offset + shape[0]].copy_(piece)
                offset += shape[0]
                del piece
            rest[f'{prefix}.weight'] = dest
    return rest


def ngram_mmap_exists(saving_path, layer_name):
    return (os.path.exists(str(saving_path / (layer_name + 'mmap')))
            and os.path.exists(str(saving_path / (layer_name + 'mmap.json')))
            and os.path.exists(str(saving_path / (layer_name + 'mmap.done'))))


def _ngram_shard_items(layer_prefix, index):
    items = []
    for key, filename in index.items():
        if not key.startswith(layer_prefix):
            continue
        match = _NGRAM_SHARD_RE.match(key)
        if not match:
            continue
        items.append((int(match.group('idx')), key, filename))
    items.sort(key=lambda item: item[0])
    got = [idx for idx, _, _ in items]
    if got != list(range(len(items))):
        raise ValueError(
            f"ngram embedding shards for {layer_prefix} are not contiguous "
            f"0..{len(items) - 1}: {got}"
        )
    return items


_SAFETENSORS_DTYPES = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32}


class _ShardReader:
    """Opens checkpoint shards on demand (downloading missing ones) and keeps them open."""

    def __init__(self, checkpoint_path, repo_id, hf_token, stack):
        self.checkpoint_path, self.repo_id, self.hf_token, self.stack = Path(checkpoint_path), repo_id, hf_token, stack
        self.handles = {}

    def handle(self, filename):
        if filename not in self.handles:
            src = self.checkpoint_path / filename
            if not src.exists():
                assert self.repo_id is not None
                huggingface_hub.snapshot_download(self.repo_id, allow_patterns=os.path.basename(src),
                                                  token=self.hf_token)
            self.handles[filename] = self.stack.enter_context(safe_open(str(src), framework='pt'))
        return self.handles[filename]

    def shape_and_dtype(self, filename, key):
        piece = self.handle(filename).get_slice(key)
        return tuple(piece.get_shape()), _SAFETENSORS_DTYPES.get(piece.get_dtype())


def persist_ngram_mmap(layer_prefix, index, checkpoint_path, saving_path,
                       repo_id=None, hf_token=None):
    """Stream n-gram shards into a file-backed mmap without holding the full table in RAM.

    Only one shard is in memory at a time; shape and dtype come from the shard headers.
    """
    if ngram_mmap_exists(saving_path, layer_prefix):
        return
    items = _ngram_shard_items(layer_prefix, index)
    if not items:
        return
    saving_path = Path(saving_path)
    with ExitStack() as stack:
        reader = _ShardReader(checkpoint_path, repo_id, hf_token, stack)
        shapes = [reader.shape_and_dtype(filename, key) for _, key, filename in items]
        dtype = shapes[0][1]
        if dtype not in _DTYPE_TAGS:
            raise ValueError(f"unsupported n-gram mmap dtype {dtype}")
        rows, cols = sum(shape[0] for shape, _ in shapes), shapes[0][0][1]
        nbytes = rows * cols * torch.empty(0, dtype=dtype).element_size()
        raw_path = saving_path / (layer_prefix + 'mmap')
        print(f"writing n-gram mmap ({rows} x {cols} {_DTYPE_TAGS[dtype]}, "
              f"{nbytes / (1024 ** 3):.1f}GB) to {raw_path}")
        tmp_path = saving_path / (layer_prefix + 'mmap.tmp')
        with open(tmp_path, 'wb') as out:
            for (_, key, filename), (shape, _) in zip(items, shapes):
                piece = reader.handle(filename).get_tensor(key)
                if tuple(piece.shape) != shape:
                    raise ValueError(f"{key} shape {tuple(piece.shape)} != {shape}")
                out.write(piece.contiguous().view(torch.uint8).numpy().tobytes())
                del piece
        tmp_path.replace(raw_path)
    meta = {'rows': rows, 'cols': cols, 'dtype': _DTYPE_TAGS[dtype], 'nbytes': nbytes}
    with open(saving_path / (layer_prefix + 'mmap.json'), 'w') as f:
        json.dump(meta, f)
    (saving_path / (layer_prefix + 'mmap.done')).touch()
    print(f"saved n-gram mmap as: {raw_path}")


def open_ngram_mmap_table(local_path, layer_name):
    """Map a persisted n-gram table without copying it into anonymous RAM."""
    meta_path = Path(local_path) / (layer_name + '.mmap.json')
    raw_path = Path(local_path) / (layer_name + '.mmap')
    meta = json.loads(meta_path.read_text())
    dtype = _TAG_DTYPES[meta['dtype']]
    rows, cols, nbytes = meta['rows'], meta['cols'], meta['nbytes']
    storage = torch.UntypedStorage.from_file(str(raw_path), shared=True, nbytes=nbytes)
    table = torch.tensor([], dtype=dtype)
    table.set_(storage, 0, (rows, cols))
    return table


class MmapEmbedding(nn.Module):
    """``nn.Embedding`` stand-in over a file-backed CPU table.

    The table is a plain attribute, not a parameter or buffer, so a parent decoder layer's
    ``module.to('meta')`` cannot evict it. A zero-size ``weight`` lives on CPU so modeling code
    that gathers on ``self.ngram_embedding.weight.device`` still runs the lookup on the host.
    """

    def __init__(self, table):
        super().__init__()
        if table.dim() != 2:
            raise ValueError(f"mmap embedding table must be 2D, got {tuple(table.shape)}")
        self.num_embeddings, self.embedding_dim = table.shape
        self._table = table
        self.weight = nn.Parameter(
            torch.empty(0, table.shape[1], dtype=table.dtype), requires_grad=False)

    def forward(self, input):
        return nn.functional.embedding(input, self._table)


@contextmanager
def _force_meta_embeddings():
    """Construct ``nn.Embedding`` on meta so huge tables never materialize on CPU.

    ``accelerate.init_empty_weights`` allocates each Parameter on CPU and then ``.to('meta')``.
    That is fine for ordinary weights, but Flash-Next's PLE table is
    ``nn.Embedding(~320M, 160)`` -- ~191GB of empty fp32 -- which OOMs a 64GB host before the
    move. Creating the module with ``device=meta`` keeps construction memory-free; AirLLM later
    replaces this module with a file-backed ``MmapEmbedding``.
    """
    orig = torch.nn.Embedding.__init__

    def _init(self, num_embeddings, embedding_dim, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault('device', torch.device('meta'))
        orig(self, num_embeddings, embedding_dim, *args, **kwargs)

    torch.nn.Embedding.__init__ = _init
    try:
        yield
    finally:
        torch.nn.Embedding.__init__ = orig
