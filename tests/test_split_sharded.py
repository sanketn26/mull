"""Splitting a multi-shard safetensors checkpoint reads per tensor and can delete the originals."""
import glob
import os
import tempfile
import unittest

import torch
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

from mull.checkpoint import split_and_save_layers
from mull.persist.safetensor_model_persister import SafetensorModelPersister


def make_sharded_llama(path):
    cfg = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(path, max_shard_size="100KB")
    return {k: v.clone() for k, v in model.state_dict().items()}


class TestSplitShardedCheckpoint(unittest.TestCase):
    def split(self, delete_original):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        reference = make_sharded_llama(tmp.name)
        shards = glob.glob(os.path.join(tmp.name, "model-*.safetensors"))
        self.assertGreater(len(shards), 2, "test needs a checkpoint with several shards")
        out = split_and_save_layers(tmp.name, delete_original=delete_original,
                                    persister=SafetensorModelPersister())
        return reference, shards, out

    def assert_split_matches(self, reference, out):
        split = {}
        for f in glob.glob(os.path.join(out, "*.safetensors")):
            split.update(load_file(f))
        self.assertEqual(set(split), set(reference))
        for k, v in reference.items():
            self.assertTrue(torch.equal(split[k], v), k)

    def test_split_keeps_every_tensor(self):
        reference, shards, out = self.split(delete_original=False)
        self.assert_split_matches(reference, out)
        self.assertTrue(all(os.path.exists(s) for s in shards))

    def test_delete_original_removes_shards_only_after_their_last_reader(self):
        reference, shards, out = self.split(delete_original=True)
        self.assert_split_matches(reference, out)
        self.assertFalse(any(os.path.exists(s) for s in shards))


if __name__ == "__main__":
    unittest.main()
