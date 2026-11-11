"""Placing checkpoint tensors onto the runtime device.

Handles the dtype rules (cast plain floats, keep pre-quantized payloads verbatim), bitsandbytes
and compressed-tensors (MXFP4) payloads, checkpoint/model shape mismatches, and pinned host memory
for faster CUDA copies.
"""
import time

import torch
from accelerate.utils.modeling import set_module_tensor_to_device

from .checkpoint import load_layer


class WeightPlacementMixin:
    """Mixed into AirLLMBaseModel. Expects ``model``, ``running_device``, ``running_dtype``,
    ``hf_quantizer``, ``persister``, ``checkpoint_path``, ``layer_names_dict`` and ``profiler``."""

    # Upper bound on the pinned (page-locked) host memory all in-flight prefetched layers may use
    # together. When the window's layers would exceed it they load into pageable memory instead.
    max_pinned_inflight_bytes = 4 * 1024 ** 3

    def load_layer_to_cpu(self, layer_name):
        t = time.time()
        load_layer_output = load_layer(self.checkpoint_path, layer_name, self.profiling_mode,
                                       persister=self.persister)
        elapsed_time = time.time() - t

        if self.profiling_mode:
            state_dict, compression_time = load_layer_output
            disk_loading_time = elapsed_time - compression_time
            self.profiler.add_profiling_time('load_safe_tensor', disk_loading_time)
            self.profiler.add_profiling_time('compression_time', compression_time)
        else:
            state_dict = load_layer_output

        if self.prefetching and torch.cuda.is_available():
            # pin_memory() returns a pinned copy rather than pinning in place, so the result has to
            # be kept for the faster host->device copy to actually happen. Pinned memory can't be
            # paged out, so we only spend it on layers small enough to be safe: a frontier MoE
            # checkpoint has ~17GB layers, and with prefetching two are in flight at once, which
            # would lock up ~34GB of RAM for a copy that is dwarfed by the disk read anyway.
            total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
            if total_bytes * max(1, self._window) <= self.max_pinned_inflight_bytes:
                try:
                    for k in state_dict.keys():
                        state_dict[k] = state_dict[k].pin_memory()
                except RuntimeError:
                    # Out of pinned memory: fall back to pageable, which is slower but always works.
                    pass

        return state_dict

    def _restore_plain_weight_modules(self, state_dict):
        """Undo CT's packed-parameter layout when the checkpoint still ships a plain ``weight``.

        Some checkpoints (Kimi K3) list residual/router Linears under the MXFP4 target list but
        store them as bf16. After ``preprocess_model`` those modules expose ``weight_packed`` and
        reject the real ``weight`` tensor. Restore a meta ``weight`` Parameter so the shard can load.
        """
        plain = {k[:-len('.weight')] for k in state_dict if k.endswith('.weight')}
        packed = {k[:-len('.weight_packed')] for k in state_dict if k.endswith('.weight_packed')}
        for prefix in plain - packed:
            try:
                module = self.model.get_submodule(prefix)
            except AttributeError:
                continue
            names = list(module._parameters.keys())
            if 'weight' in names or 'weight_packed' not in names:
                continue
            weight = state_dict[f'{prefix}.weight']
            for name in names:
                if name == 'weight' or name.startswith('weight_'):
                    module._parameters.pop(name, None)
            module.register_parameter(
                'weight',
                torch.nn.Parameter(torch.empty(weight.shape, device='meta', dtype=weight.dtype),
                                   requires_grad=False),
            )
            for attr in ('quantization_scheme', 'quantization_status', 'quantization_format'):
                if hasattr(module, attr):
                    delattr(module, attr)

    def _decompress_state_dict(self, state_dict):
        """Expand packed payloads into the plain weights the modules expect.

        compressed-tensors has no quantized compute kernels: it registers a hook that decompresses
        the whole model before the first forward, after which every quantized module wants a plain
        ``weight``. Our shards still hold ``weight_packed``/``weight_scale``, so we expand them here,
        on the GPU after transferring the *packed* bytes (4x less PCIe traffic for MXFP4).
        """
        if self.hf_quantizer is None:
            return state_dict
        packed_prefixes = {k[: -len('.weight_packed')] for k in state_dict if k.endswith('.weight_packed')}
        if not packed_prefixes:
            return state_dict
        out = dict(state_dict)
        for prefix in packed_prefixes:
            module = self._decompressible_module(prefix)
            if module is not None:
                self._decompress_module(prefix, module, state_dict, out)
        return out

    def _decompressible_module(self, prefix):
        """The module at ``prefix`` if its packed payload should be expanded here, else None."""
        try:
            from compressed_tensors.linear.compressed_linear import CompressedLinear
        except ImportError:
            CompressedLinear = None
        try:
            module = self.model.get_submodule(prefix)
        except AttributeError:
            return None
        # CompressedLinear's own forward consumes the packed payload, so leave it packed.
        if CompressedLinear is not None and isinstance(module, CompressedLinear):
            return None
        scheme = getattr(module, 'quantization_scheme', None)
        if scheme is None or getattr(scheme, 'format', None) is None:
            return None
        return module

    def _decompress_module(self, prefix, module, state_dict, out):
        """Replace ``prefix``'s packed tensors in ``out`` with the decompressed ones."""
        from compressed_tensors.compressors.base import BaseCompressor
        scheme = module.quantization_scheme
        local = {k[len(prefix) + 1:]: v.to(self.running_device)
                 for k, v in state_dict.items() if k.startswith(prefix + '.')}
        # scheme.format is an enum on some compressed-tensors versions and a plain str on others.
        fmt = getattr(scheme.format, 'value', scheme.format)
        decompressed = BaseCompressor.get_value_from_registry(fmt).decompress(local, scheme)
        for k in local:
            out.pop(f'{prefix}.{k}', None)
        if 'weight' in decompressed:
            # Once expanded the module reads only `weight`; the returned scales just describe how
            # the checkpoint stored it, and keeping them would waste VRAM.
            out[f'{prefix}.weight'] = decompressed['weight']
            self._expose_plain_weight(module, decompressed['weight'])
        else:
            for k, v in decompressed.items():
                out[f'{prefix}.{k}'] = v

    def _expose_plain_weight(self, module, weight):
        """Swap a module's packed parameters for the plain ``weight`` its forward reads.

        compressed-tensors patches a ``quantized_forward`` onto quantized Linears that reads
        ``self.weight``; the packed parameters only describe how the checkpoint stores the value.
        Marking the module COMPRESSED tells that forward the weight is already on the quantization
        grid, so it skips a fake-quantize that would only reproduce what we just decompressed.
        """
        existing = module._parameters.get('weight')
        if existing is None or existing.shape != weight.shape:
            for name in [n for n in list(module._parameters) if n.startswith('weight')]:
                module._parameters.pop(name, None)
            module.register_parameter(
                'weight',
                torch.nn.Parameter(torch.empty(weight.shape, device='meta', dtype=weight.dtype),
                                   requires_grad=False),
            )
        try:
            from compressed_tensors.quantization import QuantizationStatus
        except ImportError:
            return
        module.quantization_status = QuantizationStatus.COMPRESSED

    def _adopt_checkpoint_shape(self, param_name, value):
        """Resize a meta placeholder whose shape disagrees with the checkpoint.

        A model class builds its parameters from the config, and that construction can disagree
        with the weights actually shipped. Kimi K3 sizes ``A_log`` from ``num_heads`` while its
        checkpoint stores one entry per head channel, which would abort the load. For a parameter
        still on meta -- i.e. one we have never materialised -- the checkpoint is the source of
        truth, so adopt its shape.
        """
        module_path, _, attr = param_name.rpartition('.')
        try:
            module = self.model.get_submodule(module_path) if module_path else self.model
        except AttributeError:
            return
        current = module._parameters.get(attr)
        if current is None or current.device.type != 'meta' or current.shape == value.shape:
            return

        if not hasattr(self, '_shape_adoption_warned'):
            self._shape_adoption_warned = set()
        if attr not in self._shape_adoption_warned:
            self._shape_adoption_warned.add(attr)
            print(f"{attr}: checkpoint ships {tuple(value.shape)} but the model class builds "
                  f"{tuple(current.shape)}; using the checkpoint shape.")

        module.register_parameter(
            attr,
            torch.nn.Parameter(torch.empty(value.shape, device='meta', dtype=current.dtype),
                               requires_grad=False),
        )

    def move_layer_to_device(self, state_dict):
        self._restore_plain_weight_modules(state_dict)
        state_dict = self._decompress_state_dict(state_dict)
        marker = self.layer_names_dict.get('cpu_resident_marker')
        moved = []
        for param_name in self._param_names_from_state_dict(state_dict):
            # A nested CPU table (Flash-Next PLE) must never be placed on the GPU, even if a
            # decoder-layer shard still contains it.
            if param_name in getattr(self, '_cpu_resident_params', ()):
                continue
            if marker and marker in param_name:
                continue
            if self.hf_quantizer is not None and self._needs_quantization(param_name):
                # On-the-fly-quantizing schemes (e.g. bitsandbytes) reconstruct the param from the
                # weight plus companion quant-state tensors carried in state_dict.
                self.hf_quantizer.create_quantized_param(self.model, state_dict[param_name], param_name,
                                                         self.running_device, state_dict)
            else:
                # Normal load. Only ordinary high-precision tensors get cast to the runtime dtype;
                # pre-quantized payloads must be placed verbatim (see _should_load_verbatim).
                value = state_dict[param_name]
                self._adopt_checkpoint_shape(param_name, value)
                # clear_cache=False: accelerate otherwise empties the device allocator's cache
                # after every single tensor (once per weight, ~7 per layer). clean_memory()
                # decides when that is worth doing.
                if self._should_load_verbatim(param_name, value):
                    set_module_tensor_to_device(self.model, param_name, self.running_device, value=value,
                                                clear_cache=False)
                else:
                    set_module_tensor_to_device(self.model, param_name, self.running_device,
                                                value=value, dtype=self.running_dtype, clear_cache=False)
            moved.append(param_name)
        return moved

    # Suffixes of the companion tensors that pre-quantized checkpoints ship alongside a weight
    # (fp8 block scales, compressed-tensors/MXFP4 packed payloads and their scales, GPTQ indices).
    _QUANT_COMPANION_SUFFIXES = ("_scale", "_scale_inv", "_packed", "_zero_point", "_g_idx", "_shape")

    def _should_load_verbatim(self, param_name, value):
        """Whether a checkpoint tensor must be placed on the device without a dtype cast.

        Casting a pre-quantized payload to the runtime dtype destroys it: an fp8 weight loses its
        quantization, and a 4-bit MXFP4 ``weight_packed`` tensor (stored as packed integers) becomes
        meaningless floats. Equally important for very large models, decompressing on load would
        multiply a layer's footprint by ~4x, which is what keeps Kimi-K3-class checkpoints from
        fitting on a single GPU. So we keep anything that isn't a plain high-precision float as-is.
        """
        if not value.is_floating_point():
            # Packed 4-bit payloads, zero points, g_idx, shape metadata.
            return True
        if value.element_size() == 1:
            # Any 8-bit float: fp8 e4m3/e5m2 weights, and the e8m0 scales MXFP4 uses.
            return True
        return param_name.endswith(self._QUANT_COMPANION_SUFFIXES)

    def _needs_quantization(self, param_name):
        q = self.hf_quantizer
        # transformers renamed check_quantized_param -> param_needs_quantization.
        if hasattr(q, "param_needs_quantization"):
            return q.param_needs_quantization(self.model, param_name)
        return q.check_quantized_param(self.model, param_value=None, param_name=param_name, state_dict={})

    def _param_names_from_state_dict(self, state_dict):
        names = []
        for param_name in state_dict.keys():
            # bitsandbytes stores a weight plus companion quant-state tensors named
            # "<weight>.4bit.*" / "<weight>.8bit.*"; those are reconstructed together via
            # create_quantized_param, so collapse them down to the base weight name. Everything
            # else (including fp8 weight + weight_scale_inv pairs) is kept as distinct params.
            if '.4bit.' in param_name or '.8bit.' in param_name:
                base = param_name.split('.4bit.')[0].split('.8bit.')[0]
                if base not in names:
                    names.append(base)
            elif param_name not in names:
                names.append(param_name)
        return names
