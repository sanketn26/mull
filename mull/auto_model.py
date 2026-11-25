import importlib
from transformers import AutoConfig, PretrainedConfig
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from mull import AirLLMLlamaMlx

# Architectures that need a dedicated AirLLM subclass because of a non-standard module layout
# (custom remote-code models). Everything else uses the generic AirLLMBaseModel, which streams any
# standard *ForCausalLM (model.model.layers + lm_head / norm) and lets transformers own the
# forward pass, so newly released architectures work without code changes.
ARCH_OVERRIDES = {
    "ChatGLMModel": "AirLLMChatGLM",
    "ChatGLMForConditionalGeneration": "AirLLMChatGLM",
    "QWenLMHeadModel": "AirLLMQWen",
    "BaichuanForCausalLM": "AirLLMBaichuan",
    "BaiChuanForCausalLM": "AirLLMBaichuan",
    "InternLMForCausalLM": "AirLLMInternLM",
    "KimiK3ForConditionalGeneration": "AirLLMKimiK3",
    "Qwen3_5ForConditionalGeneration": "AirLLMQwen3_5",
    "Qwen4ExpForConditionalGeneration": "AirLLMQwen4Exp",
}

# Llama and Mistral share the legacy MLX block and weight layout.
MLX_ARCHITECTURES = {"LlamaForCausalLM", "MistralForCausalLM"}
PACKED_MLX_ARCHITECTURES = {
    "Qwen3ForCausalLM", "Qwen3_5ForConditionalGeneration", "GptOssForCausalLM", "Qwen3MoeForCausalLM",
}


class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def get_architecture(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if 'hf_token' in kwargs:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True,
                                                token=kwargs['hf_token'])
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        architectures = getattr(config, "architectures", None) or []
        return architectures[0] if architectures else ""

    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        arch = AutoModel.get_architecture(pretrained_model_name_or_path, *inputs, **kwargs)

        cls_name = ARCH_OVERRIDES.get(arch)
        if cls_name is None:
            print(f"using generic AirLLM streaming model for architecture: {arch or 'unknown'}")
            cls_name = "AirLLMBaseModel"
        return "mull", cls_name

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):

        if is_on_mac_os:
            # Check this before AirLLMLlamaMlx downloads and splits the whole checkpoint, otherwise an
            # unsupported model only fails (or gives garbage) after that is done.
            config, _ = PretrainedConfig.get_config_dict(
                pretrained_model_name_or_path, token=kwargs.get("hf_token"), revision=kwargs.get("revision"),
            )
            architectures = config.get("architectures") or [""]
            arch = architectures[0]
            # Pre-quantized MLX Llama/Mistral checkpoints use the packed runtime (batching, prefix cache);
            # plain Hugging Face weights stay on the Llama runtime, which can compress them while loading.
            prequantized = arch in MLX_ARCHITECTURES and config.get("quantization", {}).get("bits") == 4
            if arch in PACKED_MLX_ARCHITECTURES or prequantized:
                try:
                    from .airllm_packed_mlx import AirLLMPackedMlx
                except ModuleNotFoundError as error:
                    if error.name != "mlx_lm":
                        raise
                    raise ImportError("Native packed MLX models require the mac extra on Python 3.10+") from error

                return AirLLMPackedMlx(pretrained_model_name_or_path, *inputs, **kwargs)
            if arch and arch not in MLX_ARCHITECTURES:
                raise ValueError(
                    f"{arch} is not supported on macOS. Supported architectures include "
                    f"{', '.join(sorted(MLX_ARCHITECTURES | PACKED_MLX_ARCHITECTURES))}."
                )
            return AirLLMLlamaMlx(pretrained_model_name_or_path, *inputs, **kwargs)

        module, class_name = AutoModel.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)
        module = importlib.import_module(module)
        class_ = getattr(module, class_name)
        return class_(pretrained_model_name_or_path, *inputs, **kwargs)
