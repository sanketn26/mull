"""End to end on a tiny random Llama: AirLLM must generate exactly what plain transformers does.

Covers the PyTorch streaming runtime in every mode the planner can pick (layer at a time, with and
without prefetch, all resident), tied embeddings, and the MLX runtime streamed vs. resident and
quantized. Runs in a few seconds; no downloads.
"""
import sys
import tempfile
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

PROMPT = "w5 w6 w7 w8"
NEW_TOKENS = 6


def make_tiny_llama(path, tie_word_embeddings=False):
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab_size = 128
    cfg = LlamaConfig(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=tie_word_embeddings, bos_token_id=1, eos_token_id=2)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    with torch.no_grad():   # larger weights so greedy choices are decisive, not near-ties
        for p in model.parameters():
            p.mul_(8.0) if p.dim() > 1 else None
    model.save_pretrained(path)
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2} | {f"w{i}": i for i in range(3, vocab_size)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>",
                            eos_token="</s>").save_pretrained(path)
    return model


class _TinyModelCase(unittest.TestCase):
    tie = False

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = cls.tmp.name
        reference = make_tiny_llama(cls.path, tie_word_embeddings=cls.tie).eval()
        tok = PreTrainedTokenizerFast.from_pretrained(cls.path)
        cls.ids = tok([PROMPT], return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = reference.generate(cls.ids, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS,
                                     do_sample=False, eos_token_id=None)
        cls.expected = out[0, -NEW_TOKENS:].tolist()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TestTorchRuntime(_TinyModelCase):
    """The PyTorch streaming path (Linux/CUDA default), forced onto CPU here."""


    def _generate(self, **kwargs):
        from airllm.airllm_base import AirLLMBaseModel
        model = AirLLMBaseModel(self.path, device="cpu", dtype=torch.float32,
                                layer_shards_saving_path=self.path + "/torch", **kwargs)
        out = model.generate(self.ids, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS,
                             do_sample=False, eos_token_id=None)
        return model, out[0, -NEW_TOKENS:].tolist()

    def test_one_layer_at_a_time_with_prefetch(self):
        model, tokens = self._generate(memory_percent=0)
        self.assertEqual(tokens, self.expected)
        self.assertEqual(model._window, 1)

    def test_one_layer_at_a_time_without_prefetch(self):
        _, tokens = self._generate(memory_percent=0, prefetching=False)
        self.assertEqual(tokens, self.expected)

    def test_all_resident(self):
        model, tokens = self._generate(memory_percent=50)
        self.assertEqual(tokens, self.expected)
        self.assertEqual(model.sliding_plan.strategy, "all-resident")

    def test_generates_twice_in_a_row(self):
        # The prefetch window carries over between generate() calls.
        from airllm.airllm_base import AirLLMBaseModel
        model = AirLLMBaseModel(self.path, device="cpu", dtype=torch.float32, memory_percent=0,
                                layer_shards_saving_path=self.path + "/torch")
        for _ in range(2):
            out = model.generate(self.ids, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS,
                                 do_sample=False, eos_token_id=None)
            self.assertEqual(out[0, -NEW_TOKENS:].tolist(), self.expected)


class TestTorchRuntimeTiedEmbeddings(TestTorchRuntime):
    tie = True


@unittest.skipUnless(sys.platform == "darwin", "MLX backend is macOS-only")
class TestMlxRuntime(_TinyModelCase):


    def _generate(self, **kwargs):
        import mlx.core as mx
        from airllm.airllm_llama_mlx import AirLLMLlamaMlx
        split = self.path + "/mlx-" + str(kwargs.get("compression"))
        model = AirLLMLlamaMlx(self.path, layer_shards_saving_path=split, **kwargs)
        model.tokenizer.eos_token_id = None
        text = model.generate(mx.array(self.ids.numpy()), max_new_tokens=NEW_TOKENS)
        return model, text

    def test_streamed_matches_resident(self):
        _, streamed = self._generate(memory_percent=0)
        _, resident = self._generate(test_nonlayered=True)
        self.assertEqual(streamed, resident)
        self.assertEqual(len(streamed.split()), NEW_TOKENS)

    def test_streamed_without_prefetch_matches(self):
        _, streamed = self._generate(memory_percent=0, prefetching=False)
        _, resident = self._generate(test_nonlayered=True)
        self.assertEqual(streamed, resident)

    def test_8bit_runs_streamed_and_resident_alike(self):
        _, streamed = self._generate(memory_percent=0, compression="8bit")
        _, resident = self._generate(memory_percent=50, compression="8bit")
        self.assertEqual(streamed, resident)


if __name__ == "__main__":
    unittest.main()
