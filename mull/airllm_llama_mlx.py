"""MLX runtime for macOS: streams Llama-style models block by block (see mlx_llama.py)."""
import mlx.core as mx
import mlx.nn as nn
import psutil
from mlx.utils import tree_flatten
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from .checkpoint import find_or_create_local_splitted_path
from .memory import collect_garbage
from .mlx_llama import (  # noqa: F401  (ModelArgs, Attention, ... re-exported for existing imports)
    Attention, FeedForward, ModelArgs, RMSNorm, TransformerBlock, _coerce_to_mlx_array,
    get_model_args_from_config, sample, sanitize_config,
)
from .persist.mlx_model_persister import QUANT_BITS, QUANT_GROUP_SIZE, MlxModelPersister
from .prefetch import PrefetchWindow
from .sliding import DEFAULT_MEMORY_PERCENT, MAX_PREFETCH_WINDOW, plan_sliding, validate_memory_percent

# Read-ahead reads layer files in chunks of this size, warming the OS page cache.
_WARM_CHUNK_BYTES = 64 << 20


class AirLLMLlamaMlx:
    """Streams a Llama-style model through MLX one block at a time (the macOS runtime).

    Parameters (beyond the PyTorch runtime's; device and dtype are ignored: MLX runs fp16 on the
    unified-memory GPU)
    ----------
    compression : str, optional
        '4bit' or '8bit': store and run the model MLX-quantized. Every token reads ~3.5x
        (4bit) or ~1.9x (8bit) fewer bytes, and resident layers take that much less memory.
        8bit is close to lossless; 4bit costs a little quality. Needs a one-time re-split
        (kept in its own folder). Default None: fp16.
    memory_percent : float, optional
        percentage [0-100] of the system RAM available at load time that resident layers may
        use (Apple silicon shares it with the GPU). Layers that fit are loaded once and kept;
        the rest are re-read from disk every token. Default 25. When nothing fits, or with
        0/None, one layer is loaded at a time. ``test_nonlayered=True`` keeps every layer.
    """


    # customize layer names here
    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'model.embed_tokens',
                       'layer_prefix': 'model.layers',
                       'norm': 'model.norm',
                       'lm_head': 'lm_head',}


    def record_memory(self, msg=None):
        if not self.show_memory_util:
            return

        available = psutil.virtual_memory().available / 1024 / 1024
        if self.least_available is None:
            self.least_available = available
        else:
            self.least_available = min(available, self.least_available)

        consumed = self.initial_available - available
        max_consumed = self.initial_available - self.least_available

        print(f"[{msg}] - available mem: {available:.02f}mb, consumed: {consumed:.02f}mb, least available:{available:.02f}mb, max consumed: {max_consumed:.02f}mb")

    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=None, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, test_nonlayered=False, show_memory_util=False,
                 delete_original=False, memory_percent=DEFAULT_MEMORY_PERCENT):
        """See the class docstring for the parameters."""
        if compression is not None and compression not in QUANT_BITS:
            raise ValueError(f"compression must be one of {sorted(QUANT_BITS)} or None, got {compression!r}")
        self.compression = compression
        self._quant_bits = QUANT_BITS.get(compression)
        self.hf_token = hf_token
        self.set_layer_names_dict()
        self.test_nonlayered = test_nonlayered
        self.show_memory_util = show_memory_util
        self.least_available = None
        self.initial_available = psutil.virtual_memory().available / 1024 / 1024
        self.max_seq_len = max_seq_len
        self.memory_percent = validate_memory_percent(memory_percent)

        self.persister = MlxModelPersister()
        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(
            model_local_path_or_repo_id, layer_shards_saving_path, compression=compression,
            layer_names=self.layer_names_dict, hf_token=hf_token, delete_original=delete_original,
            persister=self.persister)
        # Only in-tree Llama-style architectures run here, so the checkpoint's own Python code is
        # never needed (or executed).
        token_kwargs = {'token': hf_token} if hf_token is not None else {}
        self.config = AutoConfig.from_pretrained(self.model_local_path, trust_remote_code=False, **token_kwargs)
        self.model_args = get_model_args_from_config(self.config)
        names = self.layer_names_dict
        self.layer_names = ([names['embed']] + [f'{names["layer_prefix"]}.{i}' for i in range(self.model_args.n_layers)]
                            + [names['norm'], names['lm_head']])
        self.tokenizer = self.get_tokenizer(hf_token=hf_token)
        self._init_streaming(prefetching)

    def _init_streaming(self, prefetching):
        """Plan which layers stay loaded and set up the read-ahead for the rest."""
        self._resident = {}          # layer name -> loaded module, filled on first use
        self._window = 0
        self.prefetching = prefetching
        self.sliding_plan = None
        self._resident_names = self._plan_resident()
        self.prefetch_window = PrefetchWindow(
            [n for n in self.layer_names if n not in self._resident_names],
            load=self._warm_layer_file, load_now=lambda name: name,
            size=self._window, size_of=self._layer_file_bytes)

    def get_tokenizer(self, hf_token=None):
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

    def _plan_resident(self):
        """Sliding adaptive processing: plan residency and the prefetch window from the budget."""
        if self.test_nonlayered:
            return set(self.layer_names)
        if self.memory_percent is None:
            # No budget: one layer at a time, still reading the next one ahead (as the PyTorch
            # path always has), guarded by the free-RAM floor.
            self._window = 1 if self.prefetching else 0
            return set()

        # Shards are stored as fp16, which is also the dtype they run in, so file size ~= memory.
        sizes = {i: self.persister.layer_file(name, self.checkpoint_path).stat().st_size
                 for i, name in enumerate(self.layer_names)}
        available = psutil.virtual_memory().available
        budget = int(available * self.memory_percent / 100)
        a = self.model_args
        kv_cache = 2 * a.n_layers * a.n_kv_heads * a.head_dim * self.max_seq_len * 2
        plan = plan_sliding(sizes, budget, kv_cache_bytes=kv_cache, unified_memory=True,
                            max_window=MAX_PREFETCH_WINDOW if self.prefetching else 0,
                            free_ram_bytes=available)
        self.sliding_plan = plan
        # When every streamed byte already fits in the OS page cache, read-ahead only adds work:
        # measured on Llama-3.1-8B 4-bit, 0.12 s/token without it vs 0.18 s with it.
        self._window = 0 if plan.os_cache else plan.window
        print(plan.describe(self.memory_percent, available))
        if self._window != plan.window:
            print("MLX: read-ahead off, since the streamed layers are served from the OS page cache.")
        return {self.layer_names[i] for i in plan.resident}

    def _load_weights(self, name):
        return self.persister.load_model(name, self.checkpoint_path)

    def _warm_layer_file(self, name):
        """Read-ahead: read the layer's file so the main thread's mx.load finds it in the page cache.

        The worker deliberately creates no MLX arrays: MLX work on a second thread slowed every
        GPU op on the main thread (measured 7x slower decode on Llama-3.1-8B 4-bit). Plain reads
        overlap the SSD with compute (fp16 2.59 -> 2.18 s/token) and hold one reusable buffer.
        """
        path = self.persister.layer_file(name, self.checkpoint_path)
        buf = bytearray(_WARM_CHUNK_BYTES)
        with open(path, 'rb', buffering=0) as f:
            while f.readinto(buf):
                pass
        return name

    def _layer_file_bytes(self, name):
        """On-disk size of a layer's shard, cached: asked before every read-ahead."""
        cache = self.__dict__.setdefault('_layer_file_bytes_cache', {})
        if name not in cache:
            path = self.persister.layer_file(name, self.checkpoint_path)
            cache[name] = path.stat().st_size if path else 0
        return cache[name]

    def _quantize_to_match(self, module, weights):
        """Swap in quantized layers wherever the shard holds quantized weights.

        Only layers with ``scales`` in the shard are converted, so norms and any weight the
        splitter left in fp16 load as before. The fresh module's random init is lazy and replaced
        by update(), so converting it costs nothing.
        """
        if not self._quant_bits:
            return module
        names = {key for key, _ in tree_flatten(weights)}
        if 'scales' in names:
            # The module is itself one quantized Linear / Embedding (lm_head, embeddings).
            return module.to_quantized(group_size=QUANT_GROUP_SIZE, bits=self._quant_bits)
        nn.quantize(module, group_size=QUANT_GROUP_SIZE, bits=self._quant_bits,
                    class_predicate=lambda path, _m: f'{path}.scales' in names)
        return module

    def _run(self, name, build, select, fn):
        """Run ``fn`` on layer ``name``: the kept copy if resident, else load it and drop it after."""
        module = self._resident.get(name)
        if module is None:
            self.record_memory(f'before loading {name}')
            weights = self._load_weights(self.prefetch_window.take(name))
            selected = select(weights)
            module = self._quantize_to_match(build(), selected)
            module.update(selected)
            del weights, selected
            if name in self._resident_names:
                mx.eval(module.parameters())
                self._resident[name] = module
        out = fn(module)
        mx.eval(out)
        if name not in self._resident_names:
            del module
            collect_garbage()
        self.record_memory(f'after {name}')
        return out

    def generate(self, x, temperature=0, max_new_tokens=None, **kwargs):
        """Generate up to ``max_new_tokens`` (default 20) and return the decoded text.

        Stops early at the tokenizer's end-of-sequence token.
        """
        if max_new_tokens is None:
            max_new_tokens = 20
        x = _coerce_to_mlx_array(x)
        eos = getattr(self.tokenizer, 'eos_token_id', None)
        tokens = []
        if max_new_tokens > 0:
            for token in self.model_generate(x, temperature=temperature):
                token_id = token.item()
                if token_id == eos:
                    break
                tokens.append(token_id)
                if len(tokens) >= max_new_tokens:
                    break
        return self.tokenizer.decode(tokens)

    def model_generate(self, x, temperature=0, max_new_tokens=None):
        a = self.model_args
        names = self.layer_names_dict
        cache = [None] * a.n_layers

        # Additive causal mask for the prompt; single-token decode steps need none.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])

        while True:
            h = self._run(names['embed'], lambda: nn.Embedding(a.vocab_size, a.dim),
                          lambda w: w['tok_embeddings'], lambda m: m(x))
            if mask is not None:
                mask = mask.astype(h.dtype)

            for i in tqdm(range(a.n_layers), desc='running layers'):
                def block(layer, i=i):
                    return layer(h, mask=mask, cache=cache[i])
                h, cache[i] = self._run(f'{names["layer_prefix"]}.{i}', lambda: TransformerBlock(args=a),
                                        lambda w, i=i: w['layers'][i], block)

            h = self._run(names['norm'], lambda: RMSNorm(a.dim, eps=a.norm_eps),
                          lambda w: w['norm'], lambda m: m(h))
            y = self._run(names['lm_head'], lambda: nn.Linear(a.dim, a.vocab_size, bias=False),
                          lambda w: w['output'], lambda m: sample(m(h[:, -1]), temperature))
            yield y

            x = y[:, None]
            mask = None
