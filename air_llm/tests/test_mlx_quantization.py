"""MLX-native 4/8-bit compression: shards shrink, layers load quantized, outputs stay close."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

if sys.platform != "darwin":
    raise unittest.SkipTest("MLX backend is macOS-only")

import mlx.core as mx
import mlx.nn as nn
import torch

from airllm.airllm_llama_mlx import AirLLMLlamaMlx, ModelArgs, TransformerBlock
from airllm.persist.mlx_model_persister import MlxModelPersister

ARGS = ModelArgs(dim=256, n_layers=1, head_dim=32, hidden_dim=512, n_heads=8, n_kv_heads=2,
                 norm_eps=1e-5, vocab_size=1000, rope_theta=10000.0, rope_traditional=False)


def llama_layer_state_dict():
    torch.manual_seed(0)
    d, h, kv = ARGS.dim, ARGS.hidden_dim, ARGS.n_kv_heads * ARGS.head_dim
    p = "model.layers.0."
    shapes = {"self_attn.q_proj.weight": (d, d), "self_attn.k_proj.weight": (kv, d),
              "self_attn.v_proj.weight": (kv, d), "self_attn.o_proj.weight": (d, d),
              "mlp.gate_proj.weight": (h, d), "mlp.up_proj.weight": (h, d), "mlp.down_proj.weight": (d, h)}
    sd = {p + k: torch.randn(*s) * 0.05 for k, s in shapes.items()}
    sd[p + "input_layernorm.weight"] = torch.ones(d)
    sd[p + "post_attention_layernorm.weight"] = torch.ones(d)
    return sd


def model_with_bits(bits):
    """An AirLLMLlamaMlx just far enough built to use _quantize_to_match."""
    m = AirLLMLlamaMlx.__new__(AirLLMLlamaMlx)
    m._quant_bits = bits
    return m


def relative_error(a, b):
    return (mx.linalg.norm((a - b).astype(mx.float32)) / mx.linalg.norm(b.astype(mx.float32))).item()


class TestMlxQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        persister = MlxModelPersister()
        sd = llama_layer_state_dict()
        cls.layers, cls.sizes = {}, {}
        for compression in (None, "8bit", "4bit"):
            path = Path(cls.tmp.name) / str(compression)
            path.mkdir()
            persister.persist_model(sd, "model.layers.0.", path, compression=compression)
            cls.sizes[compression] = os.path.getsize(path / "model.layers.0.mlx.safetensors")
            cls.layers[compression] = persister.load_model("model.layers.0", path)["layers"][0]
        mx.random.seed(0)
        cls.x = mx.random.normal((1, 6, ARGS.dim)).astype(mx.float16)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(6).astype(mx.float16)
        ref = TransformerBlock(ARGS)
        ref.update(cls.layers[None])
        cls.mask = mask
        cls.reference = ref(cls.x, mask=mask)[0]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run_quantized(self, compression, bits):
        block = model_with_bits(bits)._quantize_to_match(TransformerBlock(ARGS), self.layers[compression])
        block.update(self.layers[compression])
        self.assertIsInstance(block.attention.wq, nn.QuantizedLinear)
        self.assertIsInstance(block.feed_forward.w2, nn.QuantizedLinear)
        self.assertNotIsInstance(block.attention_norm, nn.QuantizedLinear)
        return block(self.x, mask=self.mask)[0]

    def test_shards_shrink(self):
        self.assertLess(self.sizes["8bit"], 0.6 * self.sizes[None])
        self.assertLess(self.sizes["4bit"], 0.35 * self.sizes[None])

    def test_norms_stay_full_precision(self):
        self.assertNotIn("scales", self.layers["4bit"]["attention_norm"])
        self.assertIn("scales", self.layers["4bit"]["attention"]["wq"])

    def test_8bit_output_is_close_to_fp16(self):
        self.assertLess(relative_error(self._run_quantized("8bit", 8), self.reference), 0.01)

    def test_4bit_output_is_close_to_fp16(self):
        self.assertLess(relative_error(self._run_quantized("4bit", 4), self.reference), 0.1)

    def test_lone_linear_and_embedding_are_quantized(self):
        # lm_head and the token embedding are built as a single Linear / Embedding.
        w = mx.random.normal((ARGS.vocab_size, ARGS.dim)).astype(mx.float16) * 0.05
        packed, scales, biases = mx.quantize(w, group_size=64, bits=4)
        weights = {"weight": packed, "scales": scales, "biases": biases}
        m = model_with_bits(4)
        head = m._quantize_to_match(nn.Linear(ARGS.dim, ARGS.vocab_size, bias=False), weights)
        emb = m._quantize_to_match(nn.Embedding(ARGS.vocab_size, ARGS.dim), weights)
        self.assertIsInstance(head, nn.QuantizedLinear)
        self.assertIsInstance(emb, nn.QuantizedEmbedding)
        head.update(weights)
        emb.update(weights)
        h = mx.random.normal((1, ARGS.dim)).astype(mx.float16)
        self.assertLess(relative_error(head(h), h @ w.T), 0.15)
        self.assertLess(relative_error(emb(mx.array([3, 7])), w[mx.array([3, 7])]), 0.15)

    def test_rejects_unknown_compression(self):
        with self.assertRaises(ValueError):
            MlxModelPersister().persist_model({}, "x.", Path(self.tmp.name), compression="2bit")


if __name__ == "__main__":
    unittest.main()
