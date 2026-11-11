from .airllm_base import AirLLMBaseModel


class AirLLMMistral(AirLLMBaseModel):
    """Backward-compatible name for Mistral. ``AutoModel`` now uses ``AirLLMBaseModel`` directly,
    since transformers drives the forward pass for every standard ``*ForCausalLM`` layout."""
