"""Sliding adaptive processing on a live PyTorch model: measure layers, plan, apply the plan.

The planning arithmetic itself lives in sliding.py and is shared with the MLX runtime.
"""
import math
from pathlib import Path

import torch
from safetensors import safe_open

from .memory import available_memory_bytes, clean_memory
from .sliding import MAX_PREFETCH_WINDOW, plan_sliding

# Bytes per element for the safetensors dtype tags a shard can hold.
_SAFETENSORS_ITEMSIZE = {
    'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1,
    'U64': 8, 'U32': 4, 'U16': 2, 'U8': 1, 'BOOL': 1, 'F8_E4M3': 1, 'F8_E5M2': 1, 'F8_E8M0': 1,
}
_HIGH_PRECISION_FLOATS = {'F64', 'F32', 'F16', 'BF16'}


class SlidingAdaptiveMixin:
    """Mixed into AirLLMBaseModel. Expects ``memory_percent``, ``prefetching``, ``device``,
    ``running_device``, ``running_dtype``, ``compression``, ``config``, ``max_seq_len``,
    ``layer_names``, ``checkpoint_path`` and ``_streamed_indices``; sets ``sliding_plan``,
    ``_window`` and ``_resident_layer_indices``."""

    def _layer_file_bytes(self, idx):
        """On-disk size of a streamed layer's shard (cached): what prefetching it will hold."""
        cache = self.__dict__.setdefault('_layer_file_bytes_cache', {})
        if idx not in cache:
            path = Path(self.checkpoint_path) / (self.layer_names[idx] + '.safetensors')
            try:
                cache[idx] = path.stat().st_size
            except OSError:
                cache[idx] = 0
        return cache[idx]

    def _layer_device_bytes(self, idx):
        """Bytes a streamed module occupies once loaded, from shard metadata (no tensor reads).

        Plain floats are cast to the runtime dtype on load; pre-quantized payloads are placed
        verbatim at their stored size (see _should_load_verbatim). Returns None when the size
        can't be determined, e.g. a non-safetensors persister.
        """
        path = Path(self.checkpoint_path) / (self.layer_names[idx] + '.safetensors')
        if not path.exists():
            return None
        run_itemsize = torch.empty(0, dtype=self.running_dtype).element_size()
        if self.compression is not None:
            # Shards hold a bitsandbytes payload that load_layer() expands to the runtime dtype:
            # 8bit stores one byte per element, nf4 packs two elements per byte.
            per_byte = 2 if self.compression == '4bit' else 1
            return path.stat().st_size * per_byte * run_itemsize
        total = 0
        with safe_open(str(path), framework='pt') as f:
            for key in f.keys():
                sl = f.get_slice(key)
                tag = sl.get_dtype()
                itemsize = run_itemsize if tag in _HIGH_PRECISION_FLOATS else _SAFETENSORS_ITEMSIZE.get(tag, 1)
                total += math.prod(sl.get_shape()) * itemsize
        return total

    def _kv_cache_bytes(self):
        """Upper bound on the KV cache at max_seq_len for batch size 1; 0 if the config is unusual."""
        cfg = getattr(self.config, 'text_config', None) or self.config
        n_layers = getattr(cfg, 'num_hidden_layers', None)
        heads = getattr(cfg, 'num_attention_heads', None)
        kv_heads = getattr(cfg, 'num_key_value_heads', None) or heads
        head_dim = getattr(cfg, 'head_dim', None)
        if head_dim is None and heads and getattr(cfg, 'hidden_size', None):
            head_dim = cfg.hidden_size // heads
        if not (n_layers and kv_heads and head_dim):
            return 0
        itemsize = torch.empty(0, dtype=self.running_dtype).element_size()
        return 2 * n_layers * kv_heads * head_dim * self.max_seq_len * itemsize

    def _keep_layers_resident(self):
        """Sliding adaptive processing: plan from layer sizes and the budget, then apply it.

        Loads the modules the plan keeps resident (once, for good) and sets the prefetch window.
        """
        if self.memory_percent is None:
            return
        sizes = {idx: self._layer_device_bytes(idx) for idx in self._streamed_indices}
        if any(size is None for size in sizes.values()):
            print('sliding adaptive processing: layer sizes unavailable for this checkpoint format; '
                  'streaming every layer.')
            return

        available = available_memory_bytes(self.running_device)
        budget = int(available * self.memory_percent / 100)
        unified = self.device.type != 'cuda'
        host_budget = None
        if not unified:
            # On a discrete GPU the window waits in host RAM: give it the same share of that.
            host_budget = int(available_memory_bytes('cpu') * self.memory_percent / 100)
        plan = plan_sliding(sizes, budget, kv_cache_bytes=self._kv_cache_bytes(),
                            unified_memory=unified, host_budget_bytes=host_budget,
                            max_window=MAX_PREFETCH_WINDOW if self.prefetching else 0,
                            free_ram_bytes=available_memory_bytes('cpu'))
        self.sliding_plan = plan

        for idx in plan.resident:
            self.move_layer_to_device(self.load_layer_to_cpu(self.layer_names[idx]))
        resident = set(plan.resident)
        self._streamed_indices = [idx for idx in self._streamed_indices if idx not in resident]
        self._resident_layer_indices = plan.resident
        self._window = plan.window
        if plan.resident:
            # Free the staging copies made while loading the resident layers.
            clean_memory(full=True)
        print(plan.describe(self.memory_percent, available))
