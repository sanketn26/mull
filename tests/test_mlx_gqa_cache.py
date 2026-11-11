"""MLX attention caches unexpanded KV heads but must compute exactly what it did before."""
import sys
import unittest

if sys.platform != "darwin":
    raise unittest.SkipTest("MLX backend is macOS-only")

import mlx.core as mx
import mlx.nn as nn

from mull.airllm_llama_mlx import Attention, ModelArgs


def reference_attention(attn, x, mask, cache):
    """The previous implementation: expand KV heads first, then cache the expanded copy."""
    B, L, _ = x.shape
    q = attn.wq(x).reshape(B, L, attn.n_heads, -1).transpose(0, 2, 1, 3)
    k = attn.wk(x).reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    v = attn.wv(x).reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

    def repeat(a):
        a = mx.concatenate([mx.expand_dims(a, 2)] * attn.repeats, axis=2)
        return a.reshape([B, attn.n_heads, L, -1])

    k, v = map(repeat, (k, v))
    if cache is not None:
        kc, vc = cache
        q = attn.rope(q, offset=kc.shape[2])
        k = mx.concatenate([kc, attn.rope(k, offset=kc.shape[2])], axis=2)
        v = mx.concatenate([vc, v], axis=2)
    else:
        q, k = attn.rope(q), attn.rope(k)
    scores = (q * attn.scale) @ k.transpose(0, 1, 3, 2)
    if mask is not None:
        scores += mask
    scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
    return attn.wo((scores @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)), (k, v)


class TestGroupedQueryCache(unittest.TestCase):
    def test_matches_previous_implementation_over_prefill_and_decode(self):
        mx.random.seed(0)
        args = ModelArgs(dim=64, n_layers=1, head_dim=8, hidden_dim=128, n_heads=8, n_kv_heads=2,
                         norm_eps=1e-5, vocab_size=10, rope_theta=10000.0, rope_traditional=False)
        attn = Attention(args)
        x = mx.random.normal((1, 5, 64))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(5)

        out, cache = attn(x, mask=mask)
        ref, ref_cache = reference_attention(attn, x, mask, None)
        self.assertTrue(mx.allclose(out, ref, atol=1e-5).item())
        # Cache holds 2 KV heads instead of 8 expanded ones.
        self.assertEqual(cache[0].shape, (1, 2, 5, 8))
        self.assertEqual(ref_cache[0].shape, (1, 8, 5, 8))

        for _ in range(3):
            step = mx.random.normal((1, 1, 64))
            out, cache = attn(step, cache=cache)
            ref, ref_cache = reference_attention(attn, step, None, ref_cache)
            self.assertTrue(mx.allclose(out, ref, atol=1e-5).item())


if __name__ == "__main__":
    unittest.main()
