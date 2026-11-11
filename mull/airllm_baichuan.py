from .airllm_base import AirLLMBaseModel
from .tokenization_baichuan import BaichuanTokenizer


class AirLLMBaichuan(AirLLMBaseModel):
    """Baichuan / Baichuan2 (``BaichuanForCausalLM``, remote code). Standard Llama layout."""

    def get_tokenizer(self, hf_token=None):
        # The repo's own tokenizer fails to load with AutoTokenizer; use the vendored copy.
        # https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/discussions/2
        return BaichuanTokenizer.from_pretrained(self.model_local_path, use_fast=False, trust_remote_code=True)
