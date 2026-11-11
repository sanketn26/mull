from .airllm_base import AirLLMBaseModel


class AirLLMInternLM(AirLLMBaseModel):
    """InternLM (``InternLMForCausalLM``, remote code). Standard Llama layout.

    Kept as its own class so ``AutoModel`` and existing imports resolve; it needs nothing beyond
    the base streaming model.
    """
