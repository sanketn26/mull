"""The direct-read loader must return exactly what safetensors' own loader returns."""
import os
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

from mull.persist.fast_safetensors import read_safetensors


class TestReadSafetensors(unittest.TestCase):
    def _roundtrip(self, tensors):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'layer.safetensors')
            save_file(tensors, path)
            expected = load_file(path)
            got = read_safetensors(path)
        self.assertEqual(set(got), set(expected))
        for name, value in expected.items():
            self.assertEqual(got[name].dtype, value.dtype, name)
            self.assertEqual(got[name].shape, value.shape, name)
            self.assertTrue(torch.equal(got[name], value), name)

    def test_matches_load_file_for_mixed_dtypes(self):
        torch.manual_seed(0)
        self._roundtrip({
            'w.f16': torch.randn(64, 32).half(),
            'w.bf16': torch.randn(16, 8).bfloat16(),
            'w.f32': torch.randn(7),
            'q.u8': torch.randint(0, 255, (13,), dtype=torch.uint8),
            'idx.i64': torch.arange(5),
            'scalar': torch.tensor(3.5),
            'empty': torch.empty(0, 4),
        })

    def test_odd_sized_tensor_before_a_wider_one(self):
        # A 3-byte tensor ahead of an fp32 one can leave it misaligned for a zero-copy view.
        self._roundtrip({'a': torch.ones(3, dtype=torch.uint8), 'b': torch.randn(5)})


if __name__ == '__main__':
    unittest.main()
