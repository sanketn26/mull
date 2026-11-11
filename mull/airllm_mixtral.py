from .airllm_base import AirLLMBaseModel


class AirLLMMixtral(AirLLMBaseModel):
    """Backward-compatible name for Mixtral. ``AutoModel`` now uses ``AirLLMBaseModel`` directly,
    since transformers drives the forward pass for every standard ``*ForCausalLM`` layout."""
