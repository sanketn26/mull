"""AirLLMLoRA end to end on a tiny random Qwen3.5: trains, loss falls, adapters round-trip."""
import tempfile
import unittest

import torch
import transformers

if not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
    raise unittest.SkipTest("transformers build without in-tree Qwen3.5")

from transformers import PreTrainedTokenizerFast

VOCAB = 256


def make_tiny_qwen35(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    config = transformers.Qwen3_5Config(
        text_config=dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                         linear_value_head_dim=16, max_position_embeddings=256,
                         layer_types=["linear_attention"] * 3 + ["full_attention"],
                         eos_token_id=2, pad_token_id=2, bos_token_id=1),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64),
        image_token_id=VOCAB - 1, video_token_id=VOCAB - 2, vision_start_token_id=VOCAB - 3,
        vision_end_token_id=VOCAB - 4, tie_word_embeddings=False)
    torch.manual_seed(0)
    transformers.Qwen3_5ForConditionalGeneration(config).save_pretrained(path)
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2} | {f"w{i}": i for i in range(3, VOCAB)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>",
                            eos_token="</s>").save_pretrained(path)


class TestAirLLMLoRA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        make_tiny_qwen35(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


    def _trainer(self):
        from airllm.airllm_lora import AirLLMLoRA
        torch.manual_seed(0)
        return AirLLMLoRA(self.tmp.name, device="cpu", dtype=torch.float32, lora_r=2, lora_alpha=4,
                          lr=1e-2, layer_shards_saving_path=self.tmp.name + "/split")

    def test_trains_and_loss_falls(self):
        trainer = self._trainer()
        ids = torch.randint(3, VOCAB - 8, (2, 12), generator=torch.Generator().manual_seed(0))
        losses = [trainer.train_step(ids) for _ in range(4)]
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        self.assertLess(losses[-1], losses[0])

    def test_adapter_round_trip(self):
        trainer = self._trainer()
        ids = torch.randint(3, VOCAB - 8, (1, 8), generator=torch.Generator().manual_seed(1))
        trainer.train_step(ids)
        path = self.tmp.name + "/adapter.pt"
        trainer.save_adapter(path)
        fresh = self._trainer()
        fresh.load_adapter(path)
        from airllm.lora_linear import lora_state_dict
        saved, loaded = lora_state_dict(trainer.model), lora_state_dict(fresh.model)
        self.assertEqual(set(saved), set(loaded))
        for k in saved:
            self.assertTrue(torch.equal(saved[k].cpu(), loaded[k].cpu()), k)

    def test_rejects_packed_experts_on_a_dense_model(self):
        from airllm.airllm_lora import AirLLMLoRA
        with self.assertRaises(ValueError):
            AirLLMLoRA(self.tmp.name, device="cpu", packed_experts=True)


if __name__ == "__main__":
    unittest.main()
