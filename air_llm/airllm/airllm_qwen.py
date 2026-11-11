from .airllm_base import AirLLMBaseModel


class AirLLMQWen(AirLLMBaseModel):
    """Qwen (v1, ``QWenLMHeadModel``, remote code): a GPT-2-style ``transformer.h`` layout."""

    LAYER_NAMES = {'embed': 'transformer.wte',
                   'layer_prefix': 'transformer.h',
                   'norm': 'transformer.ln_f',
                   'lm_head': 'lm_head'}
