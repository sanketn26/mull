"""Per-expert streaming for sparse MoE layers: load only the experts a token routes to."""
from accelerate.utils.modeling import set_module_tensor_to_device

from .checkpoint import layer_tensor_names, load_layer_subset
from .persist.safetensor_model_persister import SafetensorModelPersister


class ExpertStreamingMixin:
    """Mixed into AirLLMBaseModel. Expects ``layers``, ``layer_names``, ``layer_names_dict``,
    ``_streamed_indices``, ``persister``, ``checkpoint_path``, ``compression`` and
    ``move_layer_to_device``."""

    def _setup_expert_streaming(self):
        """Stream individual MoE experts instead of whole decoder layers, where that is possible.

        A sparse MoE layer holds hundreds of experts but routes each token to a handful of them.
        For Kimi K3 a layer's experts are ~55GB expanded, of which a token touches ~1GB. The model
        calls each selected expert as its own module, so a forward hook per expert loads exactly
        the experts that run. Needs `expert_prefix` in layer_names_dict and safetensors shards
        (individual tensors are read out of a shard).
        """
        self._expert_streaming = False
        self._expert_keys = {}
        self._non_expert_keys = {}
        if not self._expert_streaming_possible():
            return
        hooked = sum(self._hook_layer_experts(idx) for idx in self._streamed_indices)
        if hooked:
            self._expert_streaming = True
            print(f"per-expert streaming enabled: {hooked} experts across {len(self._expert_keys)} "
                  f"layers load on demand, so only the experts a token routes to are materialised.")

    def _expert_streaming_possible(self):
        if not self.layer_names_dict.get('expert_prefix'):
            return False
        if not isinstance(self.persister, SafetensorModelPersister):
            return False
        if self.compression is not None:
            # Per-expert loading reads tensors straight out of the shard with load_layer_subset(),
            # which -- unlike load_layer() -- does not run uncompress_layer_state_dict(). With
            # compression the shard holds a quantized payload plus .4bit./.8bit. quant-state
            # tensors, so subsets would reach move_layer_to_device() still quantized. Whole-layer
            # streaming does decompress.
            print("per-expert streaming is not supported together with compression for now; "
                  "streaming whole layers instead.")
            return False
        return True

    def _hook_layer_experts(self, idx):
        """Hook every expert of decoder layer ``idx``; returns how many were hooked."""
        layer_name = self.layer_names[idx]
        if not layer_name.startswith(self.layer_names_dict['layer_prefix'] + '.'):
            return 0
        try:
            names = layer_tensor_names(self.checkpoint_path, layer_name)
        except Exception:
            return 0
        expert_prefix = self.layer_names_dict['expert_prefix']
        per_expert, others = _group_tensors_by_expert(names, expert_prefix)
        if not per_expert:
            return 0
        try:
            experts_container = self.layers[idx]
            for attr in expert_prefix.split('.'):
                experts_container = getattr(experts_container, attr)
        except AttributeError:
            return 0

        self._non_expert_keys[idx] = others
        self._expert_keys[idx] = per_expert
        hooked = 0
        for expert_idx in per_expert:
            if expert_idx >= len(experts_container):
                continue
            expert_module = experts_container[expert_idx]
            expert_module._airllm_expert = (idx, expert_idx)
            expert_module.register_forward_pre_hook(self._expert_pre_hook)
            expert_module.register_forward_hook(self._expert_post_hook)
            hooked += 1
        return hooked

    def _expert_pre_hook(self, module, args):
        layer_idx, expert_idx = module._airllm_expert
        keys = self._expert_keys[layer_idx][expert_idx]
        state_dict = load_layer_subset(self.checkpoint_path, self.layer_names[layer_idx], keys)
        module._airllm_moved = self.move_layer_to_device(state_dict)

    def _expert_post_hook(self, module, args, output):
        for param_name in getattr(module, '_airllm_moved', []):
            set_module_tensor_to_device(self.model, param_name, 'meta')
        module._airllm_moved = []
        return output


def _group_tensors_by_expert(names, expert_prefix):
    """({expert index: its tensor names}, other tensor names) for one layer shard."""
    marker = f'.{expert_prefix}.'
    per_expert, others = {}, []
    for key in names:
        pos = key.find(marker)
        head = key[pos + len(marker):].split('.', 1)[0] if pos != -1 else ''
        if head.isdigit():
            per_expert.setdefault(int(head), []).append(key)
        else:
            others.append(key)
    return per_expert, others
