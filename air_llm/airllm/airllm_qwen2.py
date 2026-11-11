from .airllm_base import AirLLMBaseModel


class AirLLMQWen2(AirLLMBaseModel):
    """Backward-compatible name for Qwen2. ``AutoModel`` now uses ``AirLLMBaseModel`` directly,
    since transformers drives the forward pass for every standard ``*ForCausalLM`` layout."""
