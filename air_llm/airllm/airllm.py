from .airllm_base import AirLLMBaseModel


class AirLLMLlama2(AirLLMBaseModel):
    """Backward-compatible name for Llama models. ``AutoModel`` now uses ``AirLLMBaseModel`` directly,
    since transformers drives the forward pass for every standard ``*ForCausalLM`` layout."""
