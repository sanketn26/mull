"""Block streaming for native packed MLX-LM text checkpoints."""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import mlx.core as mx
import mlx.nn as nn
import psutil
from huggingface_hub import hf_hub_download, snapshot_download
from mlx_lm.models import gpt_oss, llama, qwen3, qwen3_5, qwen3_moe
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx_lm.utils import load_tokenizer

from .checkpoint import safetensor_manifest
from .mlx_experts import ExpertCache, StreamedExperts
from .mlx_llama import _coerce_to_mlx_array, sample
from .persist.mlx_model_persister import load_packed_weights, load_row_matrix, warm_packed_weights
from .memory import AUTO_KEEP_FREE, auto_memory_budget
from .prefetch import PrefetchWindow
from .sliding import (
    DEFAULT_MEMORY_PERCENT, MAX_PREFETCH_WINDOW, minimum_execution_budget, plan_sliding, validate_memory_percent,
)


class Decoder:
    embed = "model.embed_tokens"
    norm = "model.norm"
    head = "lm_head"
    media_ids = frozenset()

    def build(self, name):
        a = self.args
        if name == self.embed:
            return nn.Embedding(a.vocab_size, a.hidden_size)
        if name == self.head:
            return nn.Linear(a.hidden_size, a.vocab_size, bias=False)
        if name == self.norm:
            return nn.RMSNorm(a.hidden_size, eps=a.rms_norm_eps)
        return self.build_block(a, name)

    def mask(self, hidden, state):
        return create_attention_mask(hidden, state)

    def source_key(self, key):
        return key

    def forward_block(self, module, hidden, state):
        return module(hidden, self.mask(hidden, state), state)

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    def request_memory(self, prompt_length, output_limit):
        a = self.args
        capacity = math.ceil((prompt_length + max(output_limit - 1, 0)) / KVCache.step) * KVCache.step
        itemsize = self.activation_bytes
        cache = 2 * len(self.layers) * a.num_key_value_heads * a.head_dim * capacity * itemsize
        workspace = prompt_length * (3 * a.intermediate_size + 4 * a.hidden_size) * itemsize
        return cache, workspace


class Qwen3Decoder(Decoder):
    def __init__(self, config):
        if config.get("model_type") != "qwen3" or config.get("use_sliding_window"):
            raise ValueError("Expected a full-attention Qwen3 text configuration")
        self.args = a = qwen3.ModelArgs.from_dict(config)
        self.layers = [f"model.layers.{i}" for i in range(a.num_hidden_layers)]
        self.tied = a.tie_word_embeddings

    def build_block(self, args, name):
        return qwen3.TransformerBlock(args)


class LlamaDecoder(Decoder):
    """Llama and Mistral (mlx-lm runs Mistral as Llama), full attention in every layer."""

    def __init__(self, config):
        if config.get("model_type") not in ("llama", "mistral"):
            raise ValueError("Expected a Llama or Mistral configuration")
        self.args = a = llama.ModelArgs.from_dict(config)
        if any(kind != "full_attention" for kind in a.layer_types):
            raise ValueError("Llama/Mistral with sliding-window layers is not supported by the packed runtime")
        if a.head_dim is None:
            a.head_dim = a.hidden_size // a.num_attention_heads
        self.layers = [f"model.layers.{i}" for i in range(a.num_hidden_layers)]
        self.tied = a.tie_word_embeddings

    def source_key(self, key):
        return None if "rotary_emb.inv_freq" in key else key

    def build_block(self, args, name):
        return llama.TransformerBlock(args)


class Qwen35Decoder(Decoder):
    embed = "language_model.model.embed_tokens"
    norm = "language_model.model.norm"
    head = "language_model.lm_head"

    def __init__(self, config):
        text = dict(config["text_config"])
        text.setdefault("tie_word_embeddings", config.get("tie_word_embeddings", False))
        self.args = a = qwen3_5.TextModelArgs.from_dict(text)
        if a.num_experts:
            raise ValueError("Expected a dense Qwen3.5 text decoder")
        self.layers = [f"language_model.model.layers.{i}" for i in range(a.num_hidden_layers)]
        self.tied = a.tie_word_embeddings
        schedule = text.get("layer_types")
        if schedule is None:
            schedule = ["full_attention" if (i + 1) % a.full_attention_interval == 0
                        else "linear_attention" for i in range(len(self.layers))]
        if len(schedule) != len(self.layers) or set(schedule) - {"full_attention", "linear_attention"}:
            raise ValueError("Invalid Qwen3.5 layer schedule")
        self.schedule = dict(zip(self.layers, schedule))
        self.media_ids = {config[k] for k in ("image_token_id", "video_token_id") if k in config}

    def source_key(self, key):
        if key.startswith(("vision_tower.", "model.visual.", "visual.")) or "mtp." in key:
            return None
        if key.startswith("model.language_model."):
            return key.replace("model.language_model.", "language_model.model.", 1)
        if key.startswith("language_model."):
            return key
        return "language_model." + key

    def build_block(self, args, name):
        # The pinned block constructor uses the index only to select its attention implementation.
        index = args.full_attention_interval - 1 if self.schedule[name] == "full_attention" else 0
        if args.full_attention_interval == 1 and self.schedule[name] == "linear_attention":
            from dataclasses import replace

            return qwen3_5.DecoderLayer(replace(args, full_attention_interval=2), 0)
        return qwen3_5.DecoderLayer(args, index)

    def make_cache(self):
        return [KVCache() if kind == "full_attention" else ArraysCache(size=2)
                for kind in self.schedule.values()]

    def mask(self, hidden, state):
        if isinstance(state, ArraysCache):
            return create_ssm_mask(hidden, state)
        return create_attention_mask(hidden, state)

    def forward_block(self, module, hidden, state):
        out = super().forward_block(module, hidden, state)
        if isinstance(state, ArraysCache) and hidden.shape[1] > 1:
            # A contiguous suffix can still share the full convolution input allocation.
            # Gather into an independent small array and release each layer's prefill input.
            conv = state[0]
            state[0] = mx.take(conv, mx.arange(conv.shape[1]), axis=1)
            mx.eval(out, state[0])
        return out

    def request_memory(self, prompt_length, output_limit):
        a = self.args
        attention, workspace = super().request_memory(prompt_length, output_limit)
        full = list(self.schedule.values()).count("full_attention")
        linear = len(self.layers) - full
        conv_dim = (2 * a.linear_num_key_heads * a.linear_key_head_dim
                    + a.linear_num_value_heads * a.linear_value_head_dim)
        conv = (a.linear_conv_kernel_dim - 1) * conv_dim * self.activation_bytes
        recurrent = a.linear_num_value_heads * a.linear_value_head_dim * a.linear_key_head_dim * 4
        delta_workspace = prompt_length * (2 * conv_dim + 2 * a.linear_num_value_heads
                                            * a.linear_value_head_dim) * self.activation_bytes
        return attention * full // len(self.layers) + linear * (conv + recurrent), workspace + delta_workspace


class GptOssDecoder(Decoder):
    def __init__(self, config):
        self.args = a = gpt_oss.ModelArgs.from_dict(config)
        self.layers = [f"model.layers.{i}" for i in range(a.num_hidden_layers)]
        self.tied = False
        self.schedule = a.layer_types or ["sliding_attention", "full_attention"] * (len(self.layers) // 2)
        if len(self.schedule) != len(self.layers) or set(self.schedule) - {"full_attention", "sliding_attention"}:
            raise ValueError("Invalid gpt-oss attention schedule")

    def build_block(self, args, name):
        return gpt_oss.TransformerBlock(args)

    def make_cache(self):
        return [KVCache() if kind == "full_attention" else RotatingKVCache(max_size=self.args.sliding_window)
                for kind in self.schedule]

    def mask(self, hidden, state):
        window = state.max_size if isinstance(state, RotatingKVCache) else None
        return create_attention_mask(hidden, state, window_size=window)

    def validate_experts(self, manifest, quantization):
        a = self.args
        dims = {"gate_proj": (a.intermediate_size, a.hidden_size),
                "up_proj": (a.intermediate_size, a.hidden_size),
                "down_proj": (a.hidden_size, a.intermediate_size)}
        for layer in self.layers:
            for name, (outputs, inputs) in dims.items():
                prefix = f"{layer}.mlp.experts.{name}"
                quant = quantization.get(prefix, quantization)
                expected = {"weight": ("U32", (a.num_local_experts, outputs, inputs // 8)),
                            "scales": ("U8", (a.num_local_experts, outputs, inputs // 32))}
                if (quant.get("mode"), quant.get("bits"), quant.get("group_size")) != ("mxfp4", 4, 32):
                    raise ValueError(f"Expected native MXFP4 quantization for {prefix}")
                for suffix, metadata in expected.items():
                    key = f"{prefix}.{suffix}"
                    if manifest.get(key, (None, None, None))[1:3] != metadata:
                        raise ValueError(f"Unexpected native MXFP4 tensor metadata for {key}")
                bias = manifest.get(prefix + ".bias", (None, None, None))
                if bias[1] not in {"BF16", "F16", "F32"} or bias[2] != (a.num_local_experts, outputs):
                    raise ValueError(f"Unexpected expert bias metadata for {prefix}")

    def request_memory(self, prompt_length, output_limit):
        a = self.args
        attention, workspace = super().request_memory(prompt_length, output_limit)
        full = self.schedule.count("full_attention")
        sliding = len(self.layers) - full
        steps = max(output_limit - 1, 0)
        window = a.sliding_window
        if prompt_length == 1:
            rotating_capacity = min(window, math.ceil((1 + steps) / RotatingKVCache.step) * RotatingKVCache.step)
        elif steps:
            rotating_capacity = min(window, prompt_length + math.ceil(steps / RotatingKVCache.step)
                                    * RotatingKVCache.step)
        else:
            rotating_capacity = prompt_length
        bytes_per_slot = 2 * a.num_key_value_heads * a.head_dim * self.activation_bytes
        extra_prefill = sliding * max(0, prompt_length - rotating_capacity) * bytes_per_slot
        cache = attention * full // len(self.layers) + sliding * rotating_capacity * bytes_per_slot
        mask = prompt_length ** 2 if sliding and prompt_length > window else 0
        experts = prompt_length * a.num_experts_per_tok * (a.hidden_size + 3 * a.intermediate_size)
        projections = prompt_length * a.head_dim * (a.num_attention_heads + 2 * a.num_key_value_heads)
        workspace += mask + extra_prefill + (experts + projections) * self.activation_bytes
        return cache, workspace


class Qwen3MoeDecoder(Decoder):
    def __init__(self, config):
        self.args = a = qwen3_moe.ModelArgs.from_dict(config)
        self.layers = [f"model.layers.{i}" for i in range(a.num_hidden_layers)]
        self.tied = a.tie_word_embeddings

    def build_block(self, args, name):
        return qwen3_moe.Qwen3MoeDecoderLayer(args, int(name.rsplit(".", 1)[1]))

    def request_memory(self, prompt_length, output_limit):
        a = self.args
        cache, workspace = super().request_memory(prompt_length, output_limit)
        experts = prompt_length * a.num_experts_per_tok * (a.hidden_size + 3 * a.moe_intermediate_size)
        return cache, workspace + experts * self.activation_bytes


DECODERS = {"llama": LlamaDecoder, "mistral": LlamaDecoder, "qwen3": Qwen3Decoder, "qwen3_5": Qwen35Decoder, "gpt_oss": GptOssDecoder,
            "qwen3_moe": Qwen3MoeDecoder}


def packed_configuration(config):
    model_type = config.get("model_type")
    if model_type not in DECODERS:
        raise ValueError(f"Unsupported native MLX model type: {model_type!r}")
    decoder = DECODERS[model_type](config)
    quantization = config.get("quantization", {})
    expected_mode = "mxfp4" if model_type == "gpt_oss" else "affine"
    if quantization.get("bits") != 4 or quantization.get("mode", "affine") != expected_mode:
        raise ValueError(
            f"{model_type} runs on macOS from pre-quantized MLX checkpoints ({expected_mode} 4-bit), "
            f"for example an mlx-community/...-4bit repository. This checkpoint has "
            f"quantization={quantization or None}; plain Hugging Face weights are not supported here.")
    return decoder, quantization


class AirLLMPackedMlx:
    """Generate decoded text from native MLX weights, keeping a budgeted set of modules."""

    def __init__(self, checkpoint, *, memory_percent=DEFAULT_MEMORY_PERCENT, test_nonlayered=False,
                 prefetching=True, revision=None, hf_token=None, compression=None,
                 expert_streaming=False, expert_cache_bytes=256 << 20, keep_free_fraction=AUTO_KEEP_FREE,
                 **kwargs):
        if compression is not None or kwargs:
            raise ValueError(f"Unsupported native MLX options: {sorted(kwargs)} compression={compression!r}")
        path = Path(checkpoint)
        if path.is_dir():
            config_path = path / "config.json"
        else:
            config_path = Path(hf_hub_download(str(checkpoint), "config.json", revision=revision, token=hf_token))
        self.config = config = json.loads(config_path.read_text())
        self.decoder, self.quantization = packed_configuration(config)
        if not path.is_dir():
            path = Path(snapshot_download(
                str(checkpoint), revision=config_path.parent.name, token=hf_token,
                allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
            ))
        self.path = path
        self.inspect_weights()
        self.experts = None
        if expert_streaming:
            if not isinstance(self.decoder, Qwen3MoeDecoder):
                raise ValueError("Expert streaming requires Qwen3 MoE")
            if not isinstance(expert_cache_bytes, int) or expert_cache_bytes < 0:
                raise ValueError("expert_cache_bytes must be a non-negative integer")
            retain_routers = bool(memory_percent) or test_nonlayered
            self.experts = ExpertCache(self, expert_cache_bytes if retain_routers else 0, retain_routers)
        self.sizes = {i: sum(self.manifest[key][3] for key in self.keys[name])
                      for i, name in enumerate(self.names)}
        self.load_tokenizer()
        self.adaptive = memory_percent == "auto"
        if self.adaptive and not 0 <= keep_free_fraction < 1:
            raise ValueError("keep_free_fraction must be in [0, 1)")
        self.keep_free_fraction = keep_free_fraction
        self.memory_percent = 100.0 if self.adaptive else validate_memory_percent(memory_percent)
        self.available = None          # set to override free RAM; otherwise measured per request
        self.request_available = None
        self.force_resident = test_nonlayered
        self.prefetching = prefetching
        self.prefetch_window = None
        self.executor = None
        self.first_load_executor = None
        self.first_load = {}           # resident module name -> pending read-ahead of its bytes
        self.request_lock = Lock()
        self.resident = {}
        self.resident_names = set()
        self.weight_bytes_loaded = 0
        self.reserved_bytes = 0        # held outside the plan, e.g. by a prefix cache
        self.batch_rows = 1            # sequences decoded together (set by the server's batch)

    def inspect_weights(self):
        decoder = self.decoder
        source = safetensor_manifest(self.path)
        self.source_keys = {}
        for key in source:
            mapped = decoder.source_key(key)
            if mapped is not None:
                if mapped in self.source_keys:
                    raise ValueError(f"Duplicate text checkpoint tensor: {mapped}")
                self.source_keys[mapped] = key
        self.manifest = {mapped: source[key] for mapped, key in self.source_keys.items()}
        self.source_manifest = source
        self.shift_norm_weights = False
        if isinstance(decoder, Qwen35Decoder):
            self.shift_norm_weights = any("mtp." in key for key in source)
            if any(key.endswith("conv1d.weight") and item[2][-1] != 1
                   for key, item in self.manifest.items()):
                raise ValueError("Expected native MLX Qwen3.5 convolution weights")
        self.names = [decoder.embed, *decoder.layers, decoder.norm]
        if not decoder.tied:
            self.names.append(decoder.head)
        self.keys = {name: [key for key in self.manifest if key.startswith(name + ".")]
                     for name in self.names}
        for name, keys in self.keys.items():
            if not keys:
                raise ValueError(f"No checkpoint tensors for {name}")
            for key in keys:
                if key.endswith(".scales"):
                    weight = key.removesuffix("scales") + "weight"
                    if weight not in self.manifest or self.manifest[weight][1] != "U32":
                        raise ValueError(f"Expected packed uint32 weights for {key}")
        if isinstance(decoder, GptOssDecoder):
            decoder.validate_experts(self.manifest, self.quantization)
        embed_scales = decoder.embed + ".scales"
        activation = self.manifest.get(embed_scales, self.manifest[decoder.embed + ".weight"])
        decoder.activation_bytes = 4 if activation[1] == "F32" else 2

    def load_tokenizer(self):
        generation_file = self.path / "generation_config.json"
        generation = json.loads(generation_file.read_text()) if generation_file.is_file() else {}
        self.eos = set()
        for source in (self.config, self.config.get("text_config", {}), generation):
            ids = source.get("eos_token_id")
            if ids is not None:
                self.eos.update(ids if isinstance(ids, list) else [ids])
        self.tokenizer = load_tokenizer(self.path, eos_token_ids=self.eos or None)
        self.eos.update(self.tokenizer.eos_token_ids)
        self.eos.add(self.tokenizer.eos_token_id)
        self.eos.discard(None)

    def plan_request(self, prompt_length, output_limit):
        self.finish_request()
        self.cache_bytes, self.workspace_bytes = self.decoder.request_memory(prompt_length, output_limit)
        self.plan = None
        if self.force_resident:
            indices = self.sizes
        elif self.memory_percent:
            available, budget = self.request_budget()
            self.request_available = available
            # Prefill routes the prompt; a batched decode step routes one token per row. reserve() is not
            # monotonic (sparse reads can cost more than loading the whole layer), so take the larger.
            expert_reserve = (max(self.experts.reserve(prompt_length), self.experts.reserve(self.batch_rows))
                              if self.experts else 0)
            inference = (self.cache_bytes + self.workspace_bytes + expert_reserve + self.reserved_bytes
                         + self.embed_lookup_bytes(prompt_length))
            sizes = self.planned_sizes()
            self.plan = plan_sliding(sizes, budget, inference,
                                     max_window=MAX_PREFETCH_WINDOW if self.prefetching else 0,
                                     free_ram_bytes=available, prefetch_cached_layers=False)
            indices = self.plan.resident
            if self.plan.reserve_bytes > budget:
                # The floor is one module at a time (as in the Llama runtime), never a refusal.
                minimum = minimum_execution_budget(sizes, inference)
                if minimum > available:
                    # Hardware safety outranks the floor: even one module would not fit in free RAM.
                    raise MemoryError(f"Estimated minimum model memory is {minimum} bytes; "
                                      f"only {available} bytes of RAM are free")
                print(self.below_minimum_note(budget, minimum))
        else:
            indices = ()
        self.resident_names = {self.names[i] for i in indices}
        if self.room_for_whole_embedding():
            self.resident_names.add(self.decoder.embed)
        evicted = self.resident.keys() - self.resident_names
        for name in evicted:
            del self.resident[name]
        if evicted:
            mx.clear_cache()
        if self.plan and self.plan.window:
            streamed = [name for name in self.names
                        if name not in self.resident_names and not self.embeds_by_rows(name)]
            sizes = dict(zip(self.names, self.sizes.values()))
            self.executor = ThreadPoolExecutor(max_workers=self.plan.window)
            self.prefetch_window = PrefetchWindow(streamed, self.warm_module, self.plan.window,
                                                  size_of=sizes.__getitem__, load_now=lambda _: None,
                                                  executor=self.executor)
        self.start_first_load()

    @property
    def embed_by_rows(self):
        """Untied input embeddings are read row by row (only the prompt's tokens), never loaded whole.

        A token needs one row (a few KB) of a table that is hundreds of MB, and nothing else reads the
        table when the output projection has its own weights. So the planner leaves it out of residency
        and spends that RAM on decoder layers, which every token reads in full.
        """
        return not self.decoder.tied and not self.force_resident

    def embeds_by_rows(self, name):
        return name == self.decoder.embed and self.embed_by_rows

    def room_for_whole_embedding(self):
        """Keep the untied embedding in RAM only as a last use of spare budget: everything else is resident."""
        plan = self.plan
        if not (self.embed_by_rows and plan and plan.strategy == "all-resident"):
            return False
        embed = self.sizes[self.names.index(self.decoder.embed)]
        return plan.resident_bytes + plan.reserve_bytes + embed <= plan.budget_bytes

    @property
    def embed_from_rows(self):
        return self.embed_by_rows and self.decoder.embed not in self.resident_names

    def planned_sizes(self):
        return {i: size for i, size in self.sizes.items() if not self.embeds_by_rows(self.names[i])}

    def embed_lookup_bytes(self, prompt_length):
        """RAM to read the prompt's embedding rows: the packed bytes plus their MLX copy."""
        if not self.embed_by_rows:
            return 0
        row = sum(self.manifest[key][3] // self.manifest[key][2][0] for key in self.keys[self.decoder.embed])
        return 2 * row * prompt_length

    def embed_rows(self, tokens):
        """Embed ``tokens`` by reading and dequantizing only their rows of the table."""
        flat = tokens.reshape(-1).tolist()
        rows = sorted(set(flat))
        vocabulary = self.manifest[self.decoder.embed + ".weight"][2][0]
        if rows[0] < 0 or rows[-1] >= vocabulary:
            raise ValueError(f"Token IDs must be in [0, {vocabulary}); got {rows[0]}..{rows[-1]}")
        prefix = self.decoder.embed + "."
        sources = {key[len(prefix):]: self.source_keys[key] for key in self.keys[self.decoder.embed]}
        parts = {name: load_row_matrix(self.path, self.source_manifest, key, rows) for name, key in sources.items()}
        self.weight_bytes_loaded += sum(part.nbytes for part in parts.values())
        table = parts["weight"]
        if "scales" in parts:
            quant = self.quantization
            options = {key: quant[key] for key in ("group_size", "bits", "mode") if key in quant}
            options = quant.get(self.decoder.embed, options)
            table = mx.dequantize(table, parts["scales"], parts.get("biases"), **options)
        index = {row: i for i, row in enumerate(rows)}
        return table[mx.array([index[t] for t in flat]).reshape(tokens.shape)]

    def start_first_load(self):
        """Read resident modules not yet loaded ahead, in run order, so the first pass waits less on disk.

        Only file bytes are read (into the OS page cache), by one thread so reads stay sequential.
        These modules are planned to stay in RAM, so the read-ahead needs no extra budget.
        """
        missing = [name for name in self.names if name in self.resident_names and name not in self.resident]
        if not (self.prefetching and missing):
            return
        self.first_load_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mull-first-load")
        self.first_load = {name: self.first_load_executor.submit(self.warm_module, name) for name in missing}

    def wait_first_load(self, name):
        future = self.first_load.pop(name, None)
        if future is not None and not future.cancelled():
            try:
                future.result()
            except Exception:  # noqa: BLE001  the real load below reads it again and reports errors
                pass

    def below_minimum_note(self, budget, minimum):
        gib = 2**30
        if self.adaptive:
            fix = (f"keeping {self.keep_free_fraction:.0%} of RAM free for other apps; lower keep_free_fraction "
                   "(server: --keep-free-percent) or free some memory")
            setting = "memory_percent=auto"
        else:
            fix = "raise memory_percent or free some memory"
            setting = f"memory_percent={self.memory_percent:g}"
        return (f"{setting} gives {budget / gib:.2f} GiB, below the {minimum / gib:.2f} GiB needed to keep "
                f"anything resident: streaming one module at a time (slowest). The model is {fix}.")

    def request_budget(self):
        """Free RAM for this request and the share the model may use.

        A fixed memory_percent takes that share of free RAM. "auto" uses free RAM down to
        keep_free_fraction of total, so the model stays loaded when the machine is idle and
        streams when it is busy. Its own resident weights count as free: they would be if released.
        """
        if self.available is not None:
            available = self.available
        else:
            memory = psutil.virtual_memory()
            available = memory.available
            if self.adaptive:
                available += sum(self.sizes[i] for i, name in enumerate(self.names) if name in self.resident)
        if not self.adaptive:
            return available, int(available * self.memory_percent / 100)
        return available, auto_memory_budget(available, self.keep_free_fraction)

    def finish_request(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
            self.prefetch_window = None
        if self.first_load_executor is not None:
            self.first_load_executor.shutdown(wait=True, cancel_futures=True)
            self.first_load_executor = None
            self.first_load = {}

    def warm_module(self, name):
        keys = [self.source_keys[k] for k in self.keys[name]]
        warm_packed_weights(self.path, self.source_manifest, keys)

    def read_weights(self, keys):
        weights = load_packed_weights(self.path, self.source_manifest, [self.source_keys[k] for k in keys])
        self.weight_bytes_loaded += sum(self.manifest[key][3] for key in keys)
        return {key: weights[self.source_keys[key]] for key in keys}

    def quantize_module(self, module, prefix, weights):
        quant = self.quantization
        options = {key: quant[key] for key in ("group_size", "bits", "mode") if key in quant}
        if "scales" in weights:
            return module.to_quantized(**quant.get(prefix.rstrip("."), options))

        def predicate(path, child):
            return quant.get(prefix + path, hasattr(child, "to_quantized") and f"{path}.scales" in weights)

        nn.quantize(module, **options, class_predicate=predicate)
        return module

    def load_module(self, name):
        decoder = self.decoder
        prefix = name + "."
        module = decoder.build(name)
        keys = self.keys[name]
        sparse = self.experts is not None and name in self.experts.keys
        if sparse:
            module.mlp.switch_mlp = StreamedExperts(self.experts, name)
            router = self.experts.routers.get(name, {})
            weights = self.read_weights([key for key in keys if key not in router])
            weights.update(router)
            if not router and self.experts.retain_routers:
                router = {key: val for key, val in weights.items() if key.startswith(prefix + "mlp.gate.")}
                mx.eval(router)
                self.experts.routers[name] = router
        else:
            weights = self.read_weights(keys)
        local = {key[len(prefix):]: value for key, value in weights.items()}
        if self.shift_norm_weights:
            norms = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight")
            local = {key: value + 1.0 if value.ndim == 1 and
                     (name == decoder.norm or key.endswith(norms)) else value for key, value in local.items()}
        module = self.quantize_module(module, prefix, local)
        module.load_weights(list(local.items()), strict=True)
        return module.eval()

    def run_module(self, name, fn):
        module = self.resident.get(name)
        if module is None:
            if self.prefetch_window is not None:
                self.prefetch_window.take(name)
            self.wait_first_load(name)
            module = self.load_module(name)
            if name in self.resident_names:
                mx.eval(module.parameters())
                self.resident[name] = module
        out = fn(module)
        if name not in self.resident_names:
            mx.eval(out)
        return out

    def last_logits(self, tokens, cache, *, full_projection=False):
        decoder = self.decoder
        if self.embed_from_rows:
            h = self.embed_rows(tokens)
        else:
            h = self.run_module(decoder.embed, lambda module: module(tokens))
        for name, state in zip(decoder.layers, cache):
            h = self.run_module(name, lambda module: decoder.forward_block(module, h, state))
        if not full_projection:
            h = h[:, -1]
        h = self.run_module(decoder.norm, lambda module: module(h))
        if decoder.tied:
            logits = self.run_module(decoder.embed, lambda module: module.as_linear(h))
        else:
            logits = self.run_module(decoder.head, lambda module: module(h))
        return logits[:, -1] if full_projection else logits

    def generate(self, x, temperature=0, max_new_tokens=None, **kwargs):
        if kwargs:
            raise ValueError(f"Unsupported text generation inputs: {sorted(kwargs)}")
        limit = 20 if max_new_tokens is None else max_new_tokens
        if limit <= 0:
            return self.tokenizer.decode([])
        x = _coerce_to_mlx_array(x)
        if x.ndim != 2 or x.shape[0] != 1 or not x.shape[1]:
            raise ValueError("Expected one non-empty sequence of token IDs")
        if self.decoder.media_ids and self.decoder.media_ids.intersection(x.tolist()[0]):
            raise ValueError("Image and video inputs are unsupported for Qwen3.5 text inference")
        with self.request_lock:
            generated = self.generate_tokens(x, temperature, limit)
        return self.tokenizer.decode(generated)

    def generate_tokens(self, x, temperature, limit):
        self.plan_request(x.shape[1], limit)
        cache = self.decoder.make_cache()
        generated = []
        try:
            for _ in range(limit):
                token = sample(self.last_logits(x, cache), temperature)
                token_id = token.item()
                if token_id in self.eos:
                    break
                generated.append(token_id)
                x = token.reshape(1, 1)
        finally:
            self.finish_request()
        return generated
