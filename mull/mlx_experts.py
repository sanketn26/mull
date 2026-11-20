"""Opt-in Qwen3 MoE expert streaming with a byte-bounded cache."""

from collections import OrderedDict
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import SwitchGLU

from .persist.mlx_model_persister import load_packed_rows


class ExpertCache:
    def __init__(self, model, limit, retain_routers):
        self.model = model
        self.limit = limit
        self.retain_routers = retain_routers
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.routers = {}
        self.hits = self.misses = 0
        self.router_sync_s = 0.0
        self.keys = {}
        self.row_bytes = {}
        a = model.decoder.args
        for layer in model.decoder.layers:
            prefix = layer + ".mlp.switch_mlp."
            keys = [key for key in model.keys[layer] if key.startswith(prefix)]
            if not keys:
                continue
            if any(model.manifest[key][2][0] != a.num_experts for key in keys):
                raise ValueError(f"Invalid packed expert axes in {layer}")
            self.keys[layer] = keys
            self.row_bytes[layer] = sum(model.manifest[key][3] for key in keys) // a.num_experts
            model.keys[layer] = [key for key in model.keys[layer] if key not in keys]
        self.router_bytes = sum(model.manifest[key][3] for layer in self.keys
                                for key in model.keys[layer] if ".mlp.gate." in key)

    def use_whole_layer(self, length):
        a = self.model.decoder.args
        return length * a.num_experts_per_tok > a.num_experts // 2

    def reserve(self, length):
        if not self.row_bytes:
            return 0
        a = self.model.decoder.args
        selected = a.num_experts if self.use_whole_layer(length) else 3 * length * a.num_experts_per_tok
        return self.limit + self.router_bytes + max(self.row_bytes.values()) * selected

    def read(self, layer, rows):
        model = self.model
        costs = self.row_bytes[layer]
        selected, missing = {}, []
        for row in rows:
            key = (layer, row)
            if key in self.cache:
                selected[row] = self.cache[key]
                self.cache.move_to_end(key)
                self.hits += 1
            else:
                missing.append(row)
                self.misses += 1
        if not missing:
            return selected
        keys = {model.source_keys[key]: key for key in self.keys[layer]}
        fetched = load_packed_rows(model.path, model.source_manifest, keys, missing)
        model.weight_bytes_loaded += len(missing) * costs
        for row, values in fetched.items():
            entry = {keys[key]: value for key, value in values.items()}
            selected[row] = entry
            if costs <= self.limit:
                while self.cache_bytes + costs > self.limit:
                    (evicted_layer, _), _values = self.cache.popitem(last=False)
                    self.cache_bytes -= self.row_bytes[evicted_layer]
                self.cache[layer, row] = entry
                self.cache_bytes += costs
        return selected

    def apply(self, layer, x, indices):
        model = self.model
        a = model.decoder.args
        if self.use_whole_layer(x.shape[1]):
            weights = model.read_weights(self.keys[layer])
            count = a.num_experts
        else:
            start = perf_counter()
            host_indices = np.asarray(indices.tolist(), dtype=np.int32)
            self.router_sync_s += perf_counter() - start
            rows, inverse = np.unique(host_indices, return_inverse=True)
            rows = rows.tolist()
            selected = self.read(layer, rows)
            weights = {key: mx.stack([selected[row][key] for row in rows]) for key in self.keys[layer]}
            indices = mx.array(inverse.astype(np.int32).reshape(host_indices.shape))
            count = len(rows)

        prefix = layer + ".mlp.switch_mlp."
        expert = SwitchGLU(a.hidden_size, a.moe_intermediate_size, count)
        local = {key[len(prefix):]: value for key, value in weights.items()}
        model.quantize_module(expert, prefix, local)
        expert.load_weights(list(local.items()), strict=True)
        out = expert.eval()(x, indices)
        mx.eval(out)
        return out


class StreamedExperts(nn.Module):
    def __init__(self, store, layer):
        super().__init__()
        self._store = store
        self._layer = layer

    def __call__(self, x, indices):
        return self._store.apply(self._layer, x, indices)
