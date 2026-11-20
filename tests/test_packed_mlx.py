"""Offline fixtures for native packed checkpoint dispatch and generation."""

import json
import sys

import pytest

if sys.platform != "darwin":
    pytest.skip("Apple Silicon MLX runtime required", allow_module_level=True)

pytest.importorskip("mlx_lm")

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from mull import AutoModel


def qwen3_fixture(path, tied, seed):
    path.mkdir()
    config = dict(architectures=["Qwen3ForCausalLM"], model_type="qwen3",
                  hidden_size=64, head_dim=48, intermediate_size=128,
                  num_attention_heads=2, num_key_value_heads=1,
                  num_hidden_layers=2, rms_norm_eps=1e-6, vocab_size=64,
                  max_position_embeddings=128, rope_theta=12345,
                  tie_word_embeddings=tied, eos_token_id=[62, 63],
                  quantization=dict(bits=4, group_size=32))
    mx.random.seed(seed)
    model = qwen3.Model(qwen3.ModelArgs.from_dict(config))
    nn.quantize(model, bits=4, group_size=32,
                class_predicate=lambda _, module: hasattr(module, "to_quantized")
                and module.weight.shape[-1] % 32 == 0)
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (path / "config.json").write_text(json.dumps(config))
    vocab = {str(i): i for i in range(64)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="0"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="0", eos_token="63").save_pretrained(path)


@pytest.mark.parametrize("tied", [False, True])
def test_qwen3_native_checkpoint(tmp_path, tied):
    path = tmp_path / "source"
    qwen3_fixture(path, tied, 13)
    ref, _ = load_model(path)
    prompt = mx.array([[1, 2, 3]])
    steps = [prompt, *(mx.array([[i]]) for i in range(4, 12))]
    for resident in (False, True):
        model = AutoModel.from_pretrained(path, memory_percent=0, test_nonlayered=resident)
        model.plan_request(3, 9)
        cache = model.decoder.make_cache()
        ref_cache = model.decoder.make_cache()
        for tokens in steps:
            expected = ref(tokens, cache=ref_cache)[:, -1]
            actual = model.last_logits(tokens, cache)
            assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()
        assert bool(model.resident) == resident
        ref_first = mx.argmax(ref(prompt)[:, -1], axis=-1).item()
        expected_text = model.tokenizer.decode([] if ref_first in model.eos else [ref_first])
        assert model.generate(prompt, max_new_tokens=1) == expected_text
        assert model.generate(prompt, max_new_tokens=1) == expected_text
        assert model.generate(prompt, max_new_tokens=0) == model.tokenizer.decode([])
        model.eos.add(ref_first)
        assert model.generate(prompt, max_new_tokens=4) == model.tokenizer.decode([])


def test_checkpoint_identity(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    qwen3_fixture(first, False, 13)
    qwen3_fixture(second, False, 23)
    a, b = (AutoModel.from_pretrained(path, memory_percent=0) for path in (first, second))
    tokens = mx.array([[1, 2]])
    scores = [model.last_logits(tokens, model.decoder.make_cache()) for model in (a, b)]
    assert not mx.allclose(*scores).item()


def qwen35_fixture(path, *, explicit_schedule=False):
    from mlx_lm.models import qwen3_5

    qwen3_fixture(path, False, 13)
    config = json.loads((path / "config.json").read_text())
    text = dict(config, model_type="qwen3_5_text", num_hidden_layers=3,
                full_attention_interval=2, layer_types=["linear_attention", "full_attention",
                                                       "linear_attention"],
                linear_num_key_heads=1, linear_num_value_heads=2,
                linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=3)
    config = dict(architectures=["Qwen3_5ForConditionalGeneration"], model_type="qwen3_5",
                  text_config=text, quantization=config["quantization"], image_token_id=61)
    if explicit_schedule:
        text["layer_types"] = ["full_attention", "linear_attention", "linear_attention"]
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(config))
    if explicit_schedule:
        decoder = model.language_model.model
        decoder.layers = [decoder.layers[1], decoder.layers[0], decoder.layers[2]]
    nn.quantize(model, bits=4, group_size=32,
                class_predicate=lambda _, module: hasattr(module, "to_quantized")
                and module.weight.shape[-1] % 32 == 0)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    vision = "vision_tower.sentinel"
    mx.save_safetensors(str(path / "model-vision.safetensors"), {vision: mx.ones((1024,))})
    index = {key: "model.safetensors" for key in weights}
    index[vision] = "model-vision.safetensors"
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    (path / "config.json").write_text(json.dumps(config))


def test_qwen35_mixed_cache(tmp_path, monkeypatch):
    from mull import airllm_packed_mlx as packed

    path = tmp_path / "mixed"
    qwen35_fixture(path)
    original = packed.load_packed_weights
    read = []

    def track(path, manifest, keys):
        read.extend(keys)
        return original(path, manifest, keys)

    monkeypatch.setattr(packed, "load_packed_weights", track)
    model = AutoModel.from_pretrained(path, memory_percent=0)
    ref, _ = load_model(path)
    original_load = mx.load

    def check_shard(filename, *args, **kwargs):
        assert "model-vision.safetensors" not in str(filename)
        return original_load(filename, *args, **kwargs)

    monkeypatch.setattr(mx, "load", check_shard)
    for length in (1, 257):
        cache = model.decoder.make_cache()
        ref_cache = ref.make_cache()
        for tokens in (mx.ones((1, length), dtype=mx.int32), mx.array([[2]])):
            expected = ref(tokens, cache=ref_cache)[:, -1]
            actual = model.last_logits(tokens, cache, full_projection=True)
            assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()
        estimated, workspace = model.decoder.request_memory(length, 2)
        assert estimated == sum(state.nbytes for state in cache)
        assert workspace > 0
    assert not any("vision" in key for key in read)
    assert model.generate(mx.array([[1]]), max_new_tokens=1) == model.generate(
        mx.array([[1]]), max_new_tokens=1)
    with pytest.raises(ValueError, match="Image and video"):
        model.generate(mx.array([[61]]), max_new_tokens=1)


def test_qwen35_explicit_schedule(tmp_path):
    from mlx_lm.models.cache import ArraysCache, KVCache

    path = tmp_path / "explicit"
    qwen35_fixture(path, explicit_schedule=True)
    model = AutoModel.from_pretrained(path, memory_percent=0)
    cache = model.decoder.make_cache()
    assert [type(state) for state in cache] == [KVCache, ArraysCache, ArraysCache]
    logits = model.last_logits(mx.array([[1, 2]]), cache)
    assert logits.shape == (1, 64)
    assert [state.nbytes for state in cache] == [256 * 48 * 8, 2 * 128 * 4 + 2 * 32 * 32 * 4,
                                                2 * 128 * 4 + 2 * 32 * 32 * 4]


def test_qwen35_request_replans_and_evicts(tmp_path):
    from mull.sliding import all_resident_budget, minimum_execution_budget

    path = tmp_path / "mixed"
    qwen35_fixture(path)
    model = AutoModel.from_pretrained(path, memory_percent=100)
    model.available = all_resident_budget(model.sizes, sum(model.decoder.request_memory(1, 1)))
    model.generate(mx.array([[1]]), max_new_tokens=1)
    assert len(model.resident) == len(model.names)
    model.generate(mx.ones((1, 16), dtype=mx.int32), max_new_tokens=1)
    assert len(model.resident) < len(model.names)
    minimum = minimum_execution_budget(model.sizes, sum(model.decoder.request_memory(1, 1)))
    model.memory_percent = 1
    model.available = minimum * 2      # 1% is below the minimum, but free RAM can hold it
    model.generate(mx.array([[1]]), max_new_tokens=1)   # floor: one module at a time
    assert model.resident == {} and model.plan.window == 0
    model.available = 1                 # not even one module fits in free RAM: refuse
    with pytest.raises(MemoryError, match="RAM are free"):
        model.generate(mx.array([[1]]), max_new_tokens=1)


def test_free_ram_is_measured_for_each_request(tmp_path, monkeypatch):
    import psutil

    path = tmp_path / "fresh"
    qwen35_fixture(path)
    model = AutoModel.from_pretrained(path, memory_percent=50)
    for free in (8 << 30, 2 << 30):
        monkeypatch.setattr(psutil, "virtual_memory", lambda free=free: type("M", (), {"available": free})())
        model.generate(mx.array([[1]]), max_new_tokens=1)
        assert model.request_available == free
        assert model.plan.budget_bytes == free // 2


def test_plain_checkpoint_error_names_the_mlx_format():
    from mull.airllm_packed_mlx import packed_configuration

    config = dict(model_type="qwen3", hidden_size=64, num_hidden_layers=1, intermediate_size=64,
                  num_attention_heads=2, num_key_value_heads=1, rms_norm_eps=1e-6, vocab_size=64,
                  max_position_embeddings=64, rope_theta=1e6, head_dim=32, tie_word_embeddings=True)
    with pytest.raises(ValueError, match="mlx-community"):
        packed_configuration(config)


def gpt_oss_fixture(path):
    from mlx_lm.models import gpt_oss

    qwen3_fixture(path, False, 13)
    config = dict(architectures=["GptOssForCausalLM"], model_type="gpt_oss",
                  hidden_size=64, head_dim=32, intermediate_size=64, vocab_size=64,
                  num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=2,
                  num_local_experts=4, num_experts_per_tok=2, sliding_window=4,
                  layer_types=["sliding_attention", "full_attention"],
                  quantization=dict(bits=4, group_size=32, mode="mxfp4"), eos_token_id=63)
    model = gpt_oss.Model(gpt_oss.ModelArgs.from_dict(config))
    router = model.layers[0].mlp.router
    router.weight = mx.zeros_like(router.weight)
    router.bias = mx.array([1.0, 4.0, 2.0, 3.0])
    nn.quantize(model, bits=4, group_size=32, mode="mxfp4",
                class_predicate=lambda name, module: ".experts." in name and hasattr(module, "to_quantized"))
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (path / "config.json").write_text(json.dumps(config))
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": [62, 63]}))


@pytest.mark.parametrize("resident", [False, True])
def test_gpt_oss_packed_experts_and_sliding_window(tmp_path, resident):
    from mlx_lm.models.gpt_oss import mlx_topk

    path = tmp_path / "gpt"
    gpt_oss_fixture(path)
    model = AutoModel.from_pretrained(path, memory_percent=0, test_nonlayered=resident)
    ref, _ = load_model(path)
    cache, ref_cache = model.decoder.make_cache(), make_prompt_cache(ref)
    model.plan_request(3, 4)
    for ids in ([1, 2, 3], [4], [5], [6]):
        tokens = mx.array([ids])
        actual = model.last_logits(tokens, cache, full_projection=True)
        expected = ref(tokens, cache=ref_cache)[:, -1]
        assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()
    assert sum(state.nbytes for state in cache) == model.cache_bytes
    block = model.load_module(model.decoder.layers[0])
    assert sum(array.nbytes for _, array in tree_flatten(block.parameters())) == model.sizes[1]
    assert block.mlp.experts.gate_proj.mode == "mxfp4"
    assert block.mlp.experts.gate_proj.weight.dtype == mx.uint32
    assert block.mlp.experts.gate_proj.scales.dtype == mx.uint8
    x = mx.ones((1, 1, 64))
    scores, indices = mlx_topk(block.mlp.router(x), k=2)
    assert sorted(indices.flatten().tolist()) == [1, 3]
    actual = block.mlp.experts(x, indices)
    expected = ref.layers[0].mlp.experts(x, indices)
    assert mx.allclose(actual, expected).item()
    assert mx.allclose(block.mlp(x), (actual * mx.softmax(scores)[..., None]).sum(-2)).item()
    assert model.eos == {62, 63}


def qwen3_moe_fixture(path):
    from mlx_lm.models import qwen3_moe

    qwen3_fixture(path, False, 13)
    config = json.loads((path / "config.json").read_text())
    config.update(architectures=["Qwen3MoeForCausalLM"], model_type="qwen3_moe",
                  num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1,
                  mlp_only_layers=[], moe_intermediate_size=64, norm_topk_prob=True)
    model = qwen3_moe.Model(qwen3_moe.ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, bits=4, group_size=32,
                class_predicate=lambda _, module: hasattr(module, "to_quantized")
                and module.weight.shape[-1] % 32 == 0)
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (path / "config.json").write_text(json.dumps(config))


def test_qwen3_moe_expert_streaming(tmp_path):
    path = tmp_path / "moe"
    qwen3_moe_fixture(path)
    ref, _ = load_model(path)
    whole = AutoModel.from_pretrained(path, memory_percent=0)
    sparse = AutoModel.from_pretrained(path, memory_percent=100, expert_streaming=True,
                                      expert_cache_bytes=64_000)
    for model in (whole, sparse):
        model.plan_request(3, 2)
        cache, ref_cache = model.decoder.make_cache(), make_prompt_cache(ref)
        for ids in ([1, 2, 3], [4], [4]):
            tokens = mx.array([ids])
            expected = ref(tokens, cache=ref_cache)[:, -1]
            actual = model.last_logits(tokens, cache, full_projection=True)
            assert mx.allclose(actual, expected, atol=0.015625, rtol=0).item()
    assert len(sparse.experts.routers) == len(sparse.decoder.layers)
    assert 0 < sparse.experts.cache_bytes <= 64_000
    prompt = mx.array([[4]])
    first = sparse.generate(prompt, max_new_tokens=1)
    prior_hits = sparse.experts.hits
    assert sparse.generate(prompt, max_new_tokens=1) == first == whole.generate(prompt, max_new_tokens=1)
    assert sparse.experts.hits > prior_hits
    assert sparse.weight_bytes_loaded < whole.weight_bytes_loaded



def test_qwen3_expert_cache_evicts_by_bytes(tmp_path):
    path = tmp_path / "moe"
    qwen3_moe_fixture(path)
    model = AutoModel.from_pretrained(path, memory_percent=100, expert_streaming=True)
    cache = model.experts
    first, second = model.decoder.layers
    cache.limit = 2 * cache.row_bytes[first]
    cache.read(first, [0, 1])
    cache.read(first, [0])
    cache.read(second, [0])
    assert list(cache.cache) == [(first, 0), (second, 0)]
    assert cache.cache_bytes == cache.limit
    assert cache.hits == 1



def test_packed_budget_reports_smallest_feasible_allocation():
    from mull.sliding import minimum_execution_budget, plan_sliding

    sizes = {0: 100, 1: 10}
    minimum = minimum_execution_budget(sizes, 0)
    assert minimum == 122
    assert plan_sliding(sizes, minimum).resident == [0]
    fallback = plan_sliding(sizes, minimum - 1)
    assert fallback.reserve_bytes > fallback.budget_bytes
