
from pathlib import Path
import copy
from concurrent.futures import ThreadPoolExecutor

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.generation import GenerationMixin
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer

from .profiler import LayeredProfiler

from .checkpoint import find_or_create_local_splitted_path, load_layer_subset
from .memory import clean_memory, default_device
from .prefetch import PrefetchWindow
from .adaptive import SlidingAdaptiveMixin
from .experts import ExpertStreamingMixin
from .weights import WeightPlacementMixin
from .ngram import MmapEmbedding, _force_meta_embeddings, load_merged_ngram_embedding, open_ngram_mmap_table
from .sliding import DEFAULT_MEMORY_PERCENT, MAX_PREFETCH_WINDOW, validate_memory_percent
from .persist.safetensor_model_persister import SafetensorModelPersister


try:
    import bitsandbytes as bnb  # noqa: F401 - availability check

    bitsandbytes_installed = True
    print('>>>> bitsandbytes installed')
except ImportError:
    bitsandbytes_installed = False


# Helpers that transformers 5.0 moved out of transformers.utils.generic. Remote model code is
# routinely written against an older transformers and still imports them from the old location,
# which makes such models fail to import at all. Re-exporting is enough to load them.
_RELOCATED_TRANSFORMERS_SYMBOLS = {
    'OutputRecorder': 'transformers.utils.output_capturing',
    'check_model_inputs': 'transformers.utils.output_capturing',
}


def restore_relocated_transformers_symbols():
    """Re-export moved transformers helpers under their old names, where they are missing."""
    import importlib
    import transformers.utils.generic as generic

    for name, new_home in _RELOCATED_TRANSFORMERS_SYMBOLS.items():
        if hasattr(generic, name):
            continue
        try:
            module = importlib.import_module(new_home)
        except ImportError:
            continue
        symbol = getattr(module, name, None)
        if symbol is not None:
            setattr(generic, name, symbol)


class AirLLMBaseModel(WeightPlacementMixin, ExpertStreamingMixin, SlidingAdaptiveMixin):
    """
    Memory-frugal wrapper around a Hugging Face ``*ForCausalLM`` model.

    The checkpoint is split into per-layer shards on disk. The real transformers model is
    instantiated on the ``meta`` device (no memory used) and owns the full forward / generation
    logic. AirLLM only attaches forward hooks to each big module (embeddings, every decoder
    layer, the final norm and the lm_head) to stream that module's weights disk -> GPU right
    before it runs and free them right after, prefetching the next module on a worker thread.

    Because transformers drives the forward pass, AirLLM no longer needs to track per-architecture
    attention/rotary/cache details: new model architectures work as soon as transformers supports
    them.

    Parameters
    ----------
    model_local_path_or_repo_id : str or Path
        path to the local model checkpoint or huggingface repo id
    device : str, optional
        runtime device. By default the best available is picked: CUDA, else Apple MPS,
        else CPU. Pass an explicit value (e.g. "cuda:1", "cpu") to override.
    dtype : torch.dtype, optional
        runtime dtype; defaults to the model's own config.torch_dtype (usually bfloat16 for
        modern models). float16 has too narrow a range for very deep models and overflows to
        inf/NaN, which silently corrupts the output, so we don't force it.
    max_seq_len : int, optional
        max seq length, by default 512
    layer_shards_saving_path : str, optional
        optional path to save the splitted shards, by default next to the model cache
    profiling_mode : bool, optional
        whether to profile the model loading time, default False
    compression: str, optional
        '4bit' or '8bit' to enable block-wise quantization of the on-disk shards
    hf_token: str, optional
        huggingface api token
    prefetching: bool, optional
        overlap the next layer's disk load with the current layer's compute
    delete_original: bool, optional
        delete the original downloaded checkpoint after splitting to save disk space
    install_hooks: bool, optional
        attach inference streaming hooks. Training drives load/evict itself, so LoRA
        training passes False — the hooks call ``module.to('meta')`` after forward and
        that breaks backward.
    load_resident: bool, optional
        load ``resident`` modules (e.g. a vision tower) onto the GPU. Text-only LoRA
        training leaves them on meta.
    memory_percent: float, optional
        percentage (0-100] of the memory that is free when the model loads which AirLLM may
        use. Layers that fit are loaded once and kept, so they are not re-read from disk on
        every token; the rest stream as usual. Free VRAM on CUDA, available system RAM on
        CPU and Apple MPS. Default 25. When the budget can't fit a whole layer beyond what
        streaming itself needs, or with 0/None, only one layer is loaded at a time.
    """

    # Where the streamed modules live inside the transformers model. A model family with a
    # different layout declares its own LAYER_NAMES; no code is needed. Optional keys:
    #   resident            modules outside the streamed sequence, loaded once (vision tower, ...)
    #   expert_prefix       MoE experts under each layer, streamed per expert
    #   cpu_resident        modules kept on the host and never streamed to the GPU
    #   cpu_resident_marker name fragment of host-only tables nested inside a layer (n-gram PLE)
    # On-disk layer format this runtime reads and writes.
    persister_class = SafetensorModelPersister

    LAYER_NAMES = {'embed': 'model.embed_tokens',
                   'layer_prefix': 'model.layers',
                   'norm': 'model.norm',
                   'lm_head': 'lm_head'}

    def set_layer_names_dict(self):
        """Per-instance copy of LAYER_NAMES; subclasses may adjust it after calling super()."""
        self.layer_names_dict = copy.deepcopy(self.LAYER_NAMES)

    @property
    def persister(self):
        """This runtime's shard reader/writer: safetensors, on every platform."""
        if '_persister' not in self.__dict__:
            self._persister = self.persister_class()
        return self._persister

    def __init__(self, model_local_path_or_repo_id, device=None, dtype=None, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 install_hooks=True, load_resident=True, memory_percent=DEFAULT_MEMORY_PERCENT):
        """See the class docstring for the parameters."""
        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()
        self.total_disk_loading_time = None
        self.total_gpu_loading_time = None
        self.total_compression_overhead_time = None
        self.hf_quantizer = None

        if compression is not None and not bitsandbytes_installed:
            raise ImportError('WARNING: bitsandbytes not found. Compression needs bitsandbytes. '
                              'To use compression, please install bitsandbytes: `pip install bitsandbytes`')
        self.compression = compression
        self.hf_token = hf_token
        self.memory_percent = validate_memory_percent(memory_percent)
        self._resident_layer_indices = []
        restore_relocated_transformers_symbols()

        self.set_layer_names_dict()
        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(
            model_local_path_or_repo_id, layer_shards_saving_path, compression=compression,
            layer_names=self.layer_names_dict, hf_token=hf_token, delete_original=delete_original,
            persister=self.persister)

        self.running_device = device if device is not None else default_device()
        self.device = torch.device(self.running_device)
        self.config, self.trust_remote_code = self._load_config(hf_token)
        self.running_dtype = self._resolve_dtype(dtype)
        self.dtype = self.running_dtype
        self.generation_config = self.get_generation_config()
        self.tokenizer = self.get_tokenizer(hf_token=hf_token)
        self._init_prefetch_state(prefetching)

        self.init_model()
        self.max_seq_len = max_seq_len
        self._discover_layers()
        if load_resident:
            self._load_resident_modules()
        self._load_cpu_resident_modules()
        if install_hooks:
            self._install_streaming_hooks()

    def _load_config(self, hf_token):
        """(config, trust_remote_code). Prefer transformers' native implementation.

        The model's bundled remote code is only trusted when transformers doesn't recognize the
        architecture: vendored code is often pinned to an old transformers and breaks against the
        current cache/generation APIs (DeepSeek-V2 calls the long-removed DynamicCache.seen_tokens).
        """
        token_kwargs = {'token': hf_token} if hf_token is not None else {}
        try:
            return AutoConfig.from_pretrained(self.model_local_path, trust_remote_code=False, **token_kwargs), False
        except Exception:
            return AutoConfig.from_pretrained(self.model_local_path, trust_remote_code=True, **token_kwargs), True

    def _resolve_dtype(self, dtype):
        """The requested dtype, else the model's native one (bf16 for most modern models).

        Forcing fp16 overflows on deep models (Qwen3-235B's 94 layers) and produces garbage;
        bf16's wider range avoids it.
        """
        if dtype is not None:
            return dtype
        cfg_dtype = getattr(self.config, "torch_dtype", None) or getattr(self.config, "dtype", None)
        if cfg_dtype is None:
            text_cfg = getattr(self.config, "text_config", None)
            cfg_dtype = getattr(text_cfg, "torch_dtype", None) or getattr(text_cfg, "dtype", None)
        if isinstance(cfg_dtype, str):
            cfg_dtype = getattr(torch, cfg_dtype, None)
        return cfg_dtype if isinstance(cfg_dtype, torch.dtype) else torch.float16

    def _init_prefetch_state(self, prefetching):
        self.prefetching = prefetching
        if self.compression is not None and self.prefetching:
            print("prefetching is not supported together with compression for now; disabling prefetching.")
            self.prefetching = False
        # Sliding adaptive processing may keep up to MAX_PREFETCH_WINDOW reads in flight; the
        # planner sets the actual window. Idle worker threads cost nothing.
        self._executor = ThreadPoolExecutor(max_workers=MAX_PREFETCH_WINDOW) if self.prefetching else None
        self._window = 1 if self.prefetching else 0   # without a plan: classic one-layer prefetch
        self.prefetch_window = None                    # set once the streamed layers are known
        self.sliding_plan = None
        # Single-slot prefetch state, still used by the LoRA trainer's own streaming loop.
        self._prefetch_future = None
        self._prefetched_idx = None

    def _discover_layers(self):
        """Name and collect the streamed modules: embed, every decoder layer, norm, lm_head."""
        names = self.layer_names_dict
        layers_count = len(self._module_at(names["layer_prefix"]))
        self.layer_names = ([names['embed']] + [f'{names["layer_prefix"]}.{i}' for i in range(layers_count)]
                            + [names['norm'], names['lm_head']])
        self.set_layers_from_layer_names()

    def _module_at(self, dotted_path):
        module = self.model
        for attr_name in dotted_path.split("."):
            module = getattr(module, attr_name)
        return module

    # ---- customization hooks for subclasses -------------------------------------------------

    def get_generation_config(self):
        """The checkpoint's generation defaults, or the special tokens from its config.

        Falling back to an empty GenerationConfig would drop eos_token_id, and generate() would
        then run past the end of every answer.
        """
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig.from_model_config(self.config)

    def get_tokenizer(self, hf_token=None):
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

    # ---- model construction -----------------------------------------------------------------

    def _propagate_attn_implementation(self, impl):
        """Push the attention choice down into nested sub-configs.

        Multimodal wrappers keep the real decoder under a sub-config -- Kimi K3 uses ``text_config``
        -- and transformers records the request only on the config it was handed. The sub-model then
        reads an unset value and falls through to a flash-attention path, which fails outright on a
        machine without flash-attn installed.
        """
        from transformers import PretrainedConfig

        def walk(cfg, depth=0):
            if depth > 2:
                return
            for sub in vars(cfg).values():
                if isinstance(sub, PretrainedConfig):
                    sub._attn_implementation = impl
                    walk(sub, depth + 1)

        walk(self.config)

    def _model_matches_layer_names(self, model):
        """True when the instantiated class actually has the modules we plan to stream.

        Auto factories key off ``model_type``, so a VL checkpoint can land on a text-only or
        backbone class whose tree doesn't match ``layer_names_dict``. Reject those and keep trying.
        """
        try:
            for key in ('embed', 'layer_prefix', 'norm', 'lm_head'):
                mod = model
                for attr in self.layer_names_dict[key].split('.'):
                    mod = getattr(mod, attr)
            return True
        except AttributeError:
            return False

    def _auto_model_classes(self):
        """Auto* factories to try, in order, when building the empty model.

        ``AutoModelForCausalLM`` maps some VL model_types (notably ``qwen3_5``) onto a text-only
        ``*ForCausalLM`` class. That class expects a text config and either crashes or builds a
        model whose module names don't match the checkpoint. Conditional-generation architectures
        therefore try the image-text factories first.
        """
        archs = getattr(self.config, "architectures", None) or []
        arch = archs[0] if archs else ""
        names = []
        if any(tag in arch for tag in ("ConditionalGeneration", "ImageTextToText", "Multimodal")):
            names.extend(["AutoModelForImageTextToText", "AutoModelForMultimodalLM"])
        names.extend([
            "AutoModelForCausalLM",
            "AutoModelForImageTextToText",
            "AutoModelForMultimodalLM",
            "AutoModel",
        ])
        seen = set()
        classes = []
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            cls = getattr(transformers, name, None)
            if cls is not None:
                classes.append((name, cls))
        return classes

    def _instantiate_on_meta(self, attn_implementation):
        """Build the transformers model on the meta device for one attention implementation."""
        self._propagate_attn_implementation(attn_implementation)
        kwargs = {
            "attn_implementation": attn_implementation,
            "trust_remote_code": self.trust_remote_code,
        }
        errors = []
        with _force_meta_embeddings():
            for name, cls in self._auto_model_classes():
                try:
                    with init_empty_weights(include_buffers=False):
                        model = cls.from_config(self.config, **kwargs)
                except TypeError:
                    # Older Auto factories don't take attn_implementation.
                    try:
                        with init_empty_weights(include_buffers=False):
                            model = cls.from_config(self.config, trust_remote_code=self.trust_remote_code)
                    except Exception as e:  # noqa: BLE001 - try the next factory
                        errors.append(f"{name}: {type(e).__name__}: {e}")
                        continue
                except Exception as e:  # noqa: BLE001 - try the next factory
                    errors.append(f"{name}: {type(e).__name__}: {e}")
                    continue
                if not self._model_matches_layer_names(model):
                    errors.append(
                        f"{name}: built {type(model).__name__} but it is missing "
                        f"{self.layer_names_dict['layer_prefix']}"
                    )
                    continue
                print(f"built empty {type(model).__name__} via {name} (attn={attn_implementation})")
                return model
        raise RuntimeError(
            "Could not instantiate the model on meta from config. "
            f"architecture={getattr(self.config, 'architectures', None)} "
            f"model_type={getattr(self.config, 'model_type', None)}. "
            f"Tried: {'; '.join(errors)}"
        )

    def init_model(self):
        # Build the real model on meta (no memory). include_buffers=False so non-persistent
        # buffers such as rotary inv_freq are actually computed (they aren't in the checkpoint).
        try:
            self.model = self._instantiate_on_meta("sdpa")
        except Exception as e:
            print(f"attn_implementation='sdpa' not available ({e}), falling back to eager attention")
            # Some (often remote-code) architectures don't support sdpa and also default to it, so we
            # must request eager explicitly; otherwise transformers re-selects sdpa and errors again.
            self.model = self._instantiate_on_meta("eager")

        quantization_config = getattr(self.config, "quantization_config", None)
        if quantization_config is None:
            # Nested multimodal configs (Kimi K3) keep it under text_config.
            quantization_config = getattr(getattr(self.config, "text_config", None),
                                          "quantization_config", None)
        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)
            # compressed-tensors registers a hook that expands every packed module on the first
            # forward. That undoes per-expert streaming (a K3 layer becomes ~56GB) and is also
            # unnecessary: we decompress each expert ourselves as it loads. Remove the hook.
            hook = getattr(self.model, "ct_decompress_hook", None)
            if hook is not None:
                hook.remove()
                delattr(self.model, "ct_decompress_hook")

        self.model.eval()
        self.model.tie_weights()
        self.model.generation_config = self.generation_config

        # Move all (already-materialized) buffers to the running device, preserving their dtype.
        # This includes rotary inv_freq, which transformers computes once at the model level and
        # passes down to every decoder layer.
        for buffer_name, buffer in self.model.named_buffers():
            if buffer is not None and buffer.device.type != 'meta':
                set_module_tensor_to_device(self.model, buffer_name, self.running_device, value=buffer)

        # Force the model to report the running (cuda) device even though its parameters live on
        # meta between layer executions, so transformers' generation utilities place inputs/cache
        # tensors on the right device.
        self._patch_device_property()

    def _patch_device_property(self):
        running_device = torch.device(self.running_device)
        running_dtype = self.running_dtype
        base_cls = type(self.model)

        # transformers >= 4.50 removed GenerationMixin from PreTrainedModel, so model classes that
        # predate that change (or ship as remote code, like Kimi K3's multimodal wrapper) no longer
        # have .generate(). They still define prepare_inputs_for_generation, so mixing the class
        # back in restores generation. It must come after the model class, per transformers.
        extra_bases = () if isinstance(self.model, GenerationMixin) else (GenerationMixin,)
        if extra_bases:
            print(f"{base_cls.__name__} does not inherit GenerationMixin; mixing it in so "
                  f"generate() works.")

        class _AirLLMRuntimeModel(base_cls, *extra_bases):
            @property
            def device(self):
                return running_device

            @property
            def dtype(self):
                return running_dtype

        self.model.__class__ = _AirLLMRuntimeModel

    def set_layers_from_layer_names(self):
        names = self.layer_names_dict
        self.layers = ([self._module_at(names["embed"])] + list(self._module_at(names["layer_prefix"]))
                       + [self._module_at(names["norm"]), self._module_at(names["lm_head"])])

    # ---- modules outside the streamed sequence ----------------------------------------------

    def _load_resident_modules(self):
        """Load modules that sit outside the streamed embed -> layers -> norm -> lm_head sequence.

        Multimodal checkpoints carry a vision tower and projector, and some architectures add
        extra top-level norms. They never get a streaming hook, so without this they would stay on
        the meta device and fail the moment they run. They are small (well under a GB), so we load
        them once and leave them resident.
        """
        for name in self.layer_names_dict.get('resident', []):
            try:
                state_dict = self.load_layer_to_cpu(name)
            except FileNotFoundError:
                # Not every checkpoint of a given architecture ships every optional module.
                continue
            self.move_layer_to_device(state_dict)

    def _cpu_resident_names(self):
        """Module prefixes whose weights stay on CPU for the lifetime of the process.

        Flash-Next's n-gram table is the motivating case: transformers already gathers rows onto
        the embedding's own device, so keeping ~102GB of bf16 on the host is the designed path.
        Names come from ``cpu_resident`` and from split files matching ``cpu_resident_marker``.
        """
        names = list(self.layer_names_dict.get('cpu_resident', []))
        marker = self.layer_names_dict.get('cpu_resident_marker')
        if marker:
            for path in Path(self.checkpoint_path).glob('*.mmap.json'):
                stem = path.name[:-len('.mmap.json')]
                if stem.endswith(marker) and stem not in names:
                    names.append(stem)
            for path in Path(self.checkpoint_path).glob('*.safetensors'):
                stem = path.name[:-len('.safetensors')]
                if stem.endswith(marker) and stem not in names:
                    names.append(stem)
        return names

    def _install_mmap_embedding(self, module_name, table):
        parent_name, _, attr = module_name.rpartition('.')
        parent = self.model.get_submodule(parent_name)
        setattr(parent, attr, MmapEmbedding(table))

    def _load_cpu_resident_modules(self):
        """Load oversized lookup tables onto CPU and never stream them to the GPU.

        Flash-Next's n-gram table is ~102GB bf16. We mmap it from disk so a 64GB host can still
        run; transformers already gathers rows on ``ngram_embedding.weight.device``.
        ``module.to('meta')`` after a decoder layer would otherwise evict a child embedding that
        lives under that layer, so we also record the parameter names and evict selectively.
        """
        self._cpu_resident_params = set()
        for name in self._cpu_resident_names():
            mmap_meta = Path(self.checkpoint_path) / f"{name}.mmap.json"
            if mmap_meta.exists():
                table = open_ngram_mmap_table(self.checkpoint_path, name)
                self._install_mmap_embedding(name, table)
                self._cpu_resident_params.add(f"{name}.weight")
                print(f"cpu-resident mmap embedding: {name} "
                      f"{tuple(table.shape)} {table.dtype} (gathered from disk, not copied to RAM)")
                continue
            try:
                state_dict = load_merged_ngram_embedding(self.checkpoint_path, name)
            except FileNotFoundError:
                continue
            for param_name, value in state_dict.items():
                self._adopt_checkpoint_shape(param_name, value)
                if self._should_load_verbatim(param_name, value):
                    set_module_tensor_to_device(self.model, param_name, 'cpu', value=value)
                else:
                    set_module_tensor_to_device(
                        self.model, param_name, 'cpu', value=value, dtype=self.running_dtype)
                self._cpu_resident_params.add(param_name)
        if self._cpu_resident_params:
            n = len(self._cpu_resident_params)
            print(f"cpu-resident modules: left {n} tensors on host RAM "
                  f"(n-gram / PLE table is gathered on CPU, not streamed to the GPU).")

    def _install_streaming_hooks(self):
        # Modules execute in this order during a forward: embed -> layers -> norm -> lm_head.
        n = len(self.layer_names)

        # Detect tied input/output embeddings. When tied, lm_head shares the embedding weight, so
        # there is no separate lm_head shard. We keep the embedding resident on the GPU (it is the
        # only copy and such models are small) and re-tie lm_head to it, then stream only the
        # decoder layers and the final norm.
        self.tie_word_embeddings = bool(getattr(self.config, "tie_word_embeddings", False))

        if self.tie_word_embeddings:
            embed_state = self.load_layer_to_cpu(self.layer_names[0])
            self.move_layer_to_device(embed_state)
            self.model.tie_weights()
            self._streamed_indices = list(range(1, n - 1))  # decoder layers + final norm
        else:
            self._streamed_indices = list(range(n))

        self._keep_layers_resident()
        self._setup_expert_streaming()

        # Prefetched layers stay in host memory; the main thread places them on the device.
        # Copying to Apple MPS from a worker was tried: MPS queues the copy behind the running
        # layer anyway, so it was ~8% slower and doubled the GPU's live memory.
        self.prefetch_window = PrefetchWindow(
            self._streamed_indices, self._load_streamed_layer,
            size=self._window if self.prefetching else 0,
            size_of=self._layer_file_bytes, executor=self._executor)

        for idx in self._streamed_indices:
            module = self.layers[idx]
            module._airllm_idx = idx
            module.register_forward_pre_hook(self._pre_hook)
            module.register_forward_hook(self._post_hook)

    # ---- streaming hooks: load a layer just before it runs, release it right after --------

    def _load_streamed_layer(self, idx):
        """Load one streamed module's weights. Experts are excluded when they stream themselves."""
        keys = self._non_expert_keys.get(idx) if getattr(self, '_expert_streaming', False) else None
        if keys is None:
            return self.load_layer_to_cpu(self.layer_names[idx])
        return load_layer_subset(self.checkpoint_path, self.layer_names[idx], keys)

    def _pre_hook(self, module, args):
        idx = module._airllm_idx

        state_dict = self.prefetch_window.take(idx)
        module._airllm_moved = self.move_layer_to_device(state_dict)

    def _post_hook(self, module, args, output):
        # module.to('meta') would also evict experts (which manage their own lifetime) and any
        # cpu-resident child such as Flash-Next's n-gram table, which lives under a decoder layer
        # but must stay materialised on the host. When any of those are in play, only release the
        # tensors this hook actually placed.
        if (self.hf_quantizer is not None
                or getattr(self, '_expert_streaming', False)
                or getattr(self, '_cpu_resident_params', None)):
            for param_name in getattr(module, '_airllm_moved', []):
                set_module_tensor_to_device(self.model, param_name, 'meta')
        else:
            module.to('meta')
        clean_memory()
        return output

    # ---- delegation to the underlying transformers model ------------------------------------

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
