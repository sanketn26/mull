"""AutoModel picks the right AirLLM class from a checkpoint's architecture name (offline)."""
import unittest
from unittest import mock

from airllm.auto_model import ARCH_OVERRIDES, AutoModel


class TestAutoModel(unittest.TestCase):
    def _class_for(self, architecture):
        with mock.patch.object(AutoModel, "get_architecture", return_value=architecture):
            return AutoModel.get_module_class("any/repo")

    def test_custom_architectures_get_their_dedicated_class(self):
        expected = {
            "QWenLMHeadModel": "AirLLMQWen",
            "InternLMForCausalLM": "AirLLMInternLM",
            "ChatGLMModel": "AirLLMChatGLM",
            "BaichuanForCausalLM": "AirLLMBaichuan",
            "KimiK3ForConditionalGeneration": "AirLLMKimiK3",
            "Qwen3_5ForConditionalGeneration": "AirLLMQwen3_5",
            "Qwen4ExpForConditionalGeneration": "AirLLMQwen4Exp",
        }
        for arch, cls in expected.items():
            self.assertEqual(self._class_for(arch), ("airllm", cls), arch)

    def test_standard_architectures_use_the_generic_streaming_model(self):
        # Since 4.0, transformers drives the forward pass, so standard layouts need no subclass.
        for arch in ("LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM", "", "SomethingNew"):
            self.assertEqual(self._class_for(arch), ("airllm", "AirLLMBaseModel"), arch)

    def test_every_override_names_a_class_that_exists(self):
        import airllm
        import importlib
        for cls_name in set(ARCH_OVERRIDES.values()):
            module = importlib.import_module(f"airllm.{_MODULE_FOR[cls_name]}")
            self.assertTrue(hasattr(module, cls_name), cls_name)
        self.assertIsNotNone(airllm)


_MODULE_FOR = {
    "AirLLMChatGLM": "airllm_chatglm",
    "AirLLMQWen": "airllm_qwen",
    "AirLLMBaichuan": "airllm_baichuan",
    "AirLLMInternLM": "airllm_internlm",
    "AirLLMKimiK3": "airllm_kimi_k3",
    "AirLLMQwen3_5": "airllm_qwen3_5",
    "AirLLMQwen4Exp": "airllm_qwen4_exp",
}


if __name__ == "__main__":
    unittest.main()
