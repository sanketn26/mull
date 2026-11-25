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


def llama_fixture(path, tied=False, seed=5, model_type="llama"):
    """Tiny 4-bit Llama (or Mistral) in mlx-community layout: grouped-query attention, Llama 3 RoPE scaling."""
    from mlx_lm.models import llama

    path.mkdir()
    config = dict(architectures=["MistralForCausalLM" if model_type == "mistral" else "LlamaForCausalLM"],
                  model_type=model_type, hidden_size=64, intermediate_size=128,
                  num_attention_heads=4, num_key_value_heads=1, num_hidden_layers=2,
                  rms_norm_eps=1e-5, vocab_size=64, max_position_embeddings=256, rope_theta=500000,
                  rope_scaling=dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0,
                                    high_freq_factor=4.0, original_max_position_embeddings=64),
                  tie_word_embeddings=tied, eos_token_id=63, quantization=dict(bits=4, group_size=32))
    mx.random.seed(seed)
    model = llama.Model(llama.ModelArgs.from_dict(config))
    nn.quantize(model, bits=4, group_size=32,
                class_predicate=lambda _, module: hasattr(module, "to_quantized")
                and module.weight.shape[-1] % 32 == 0)
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (path / "config.json").write_text(json.dumps(config))
    vocab = {str(i): i for i in range(64)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="0"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="0", eos_token="63").save_pretrained(path)


@pytest.mark.parametrize("model_type,tied", [("llama", False), ("llama", True), ("mistral", False)])
def test_llama_runs_on_packed_runtime(tmp_path, model_type, tied):
    from mull.airllm_packed_mlx import AirLLMPackedMlx

    path = tmp_path / model_type
    llama_fixture(path, tied=tied, model_type=model_type)
    ref, _ = load_model(path)
    steps = [mx.array([[1, 2, 3, 4, 5]]), *(mx.array([[i]]) for i in range(6, 14))]
    for percent, resident in ((0, False), (100, False), (0, True)):
        model = AutoModel.from_pretrained(path, memory_percent=percent, test_nonlayered=resident)
        assert isinstance(model, AirLLMPackedMlx)
        model.plan_request(5, 9)
        cache, ref_cache = model.decoder.make_cache(), make_prompt_cache(ref)
        for tokens in steps:
            expected = ref(tokens, cache=ref_cache)[:, -1]
            assert mx.allclose(model.last_logits(tokens, cache), expected, atol=1e-5, rtol=1e-5).item()
        model.finish_request()


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
    planned = model.planned_sizes()    # the untied embedding is read by rows, never resident
    model.available = all_resident_budget(planned, sum(model.decoder.request_memory(1, 1))
                                          + model.embed_lookup_bytes(1))
    model.generate(mx.array([[1]]), max_new_tokens=1)
    assert len(model.resident) == len(planned) == len(model.names) - 1
    model.generate(mx.ones((1, 16), dtype=mx.int32), max_new_tokens=1)
    assert len(model.resident) < len(planned)
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


def test_auto_memory_keeps_a_reserve_and_counts_its_own_weights(tmp_path, monkeypatch):
    import psutil

    import mull.memory as memory

    path = tmp_path / "auto"
    qwen35_fixture(path)
    model = AutoModel.from_pretrained(path, memory_percent="auto", keep_free_fraction=0.25)
    total = 16 << 30
    monkeypatch.setattr(memory, "min_free_ram_bytes", lambda: 1 << 30)
    monkeypatch.setattr(memory, "system_ram_bytes", lambda: (total, 0))
    for free in (12 << 30, 3 << 30):
        memory = type("M", (), {"available": free, "total": total})()
        monkeypatch.setattr(psutil, "virtual_memory", lambda memory=memory: memory)
        model.resident.clear()
        model.generate(mx.array([[1]]), max_new_tokens=1)
        assert model.plan.budget_bytes == max(0, free - total // 4)
    model.resident = {name: None for name in model.names}
    available, _ = model.request_budget()
    assert available == (3 << 30) + sum(model.sizes.values())


def test_auto_memory_is_rejected_with_a_bad_reserve(tmp_path):
    path = tmp_path / "bad"
    qwen35_fixture(path)
    with pytest.raises(ValueError, match="keep_free_fraction"):
        AutoModel.from_pretrained(path, memory_percent="auto", keep_free_fraction=1.5)


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


def test_fallback_note_names_the_setting_to_change(tmp_path):
    path = tmp_path / "note"
    qwen35_fixture(path)
    auto = AutoModel.from_pretrained(path, memory_percent="auto", keep_free_fraction=0.3)
    note = auto.below_minimum_note(2 << 30, 3 << 30)
    assert "memory_percent=auto" in note and "30%" in note and "--keep-free-percent" in note
    fixed = AutoModel.from_pretrained(path, memory_percent=10)
    assert "raise memory_percent" in fixed.below_minimum_note(1 << 30, 3 << 30)


def test_first_pass_reads_resident_modules_ahead_in_order(tmp_path, monkeypatch):
    path = tmp_path / "qwen3"
    qwen3_fixture(path, tied=False, seed=3)
    model = AutoModel.from_pretrained(path, memory_percent=50)
    warmed = []
    original = model.warm_module
    monkeypatch.setattr(model, "warm_module", lambda name: (warmed.append(name), original(name)))
    expected = model.generate_tokens(mx.array([[5, 9, 14]]), 0, 3)
    assert model.resident_names and set(warmed) == model.resident_names and warmed == [n for n in model.names if n in warmed]
    assert model.first_load == {} and model.first_load_executor is None
    warmed.clear()
    assert model.generate_tokens(mx.array([[5, 9, 14]]), 0, 3) == expected
    assert warmed == []                                # already loaded: nothing read ahead again


@pytest.mark.parametrize("fixture", ["qwen3", "qwen35"])
def test_untied_embedding_reads_only_needed_rows(tmp_path, fixture):
    path = tmp_path / fixture
    qwen3_fixture(path, tied=False, seed=3) if fixture == "qwen3" else qwen35_fixture(path)
    model = AutoModel.from_pretrained(path, memory_percent=50)
    if model.decoder.tied:
        pytest.skip("fixture ties its embeddings")
    tokens = mx.array([[5, 9, 5, 14], [3, 3, 40, 2]])     # repeats and two rows
    module = model.load_module(model.decoder.embed)
    assert mx.array_equal(model.embed_rows(tokens), module(tokens))


def test_tied_embedding_keeps_the_module_path(tmp_path):
    path = tmp_path / "tied"
    qwen3_fixture(path, tied=True, seed=3)
    model = AutoModel.from_pretrained(path, memory_percent=50)
    assert not model.embed_by_rows and model.embed_lookup_bytes(10) == 0


def test_untied_embedding_resident_only_with_spare_budget(tmp_path):
    from mull.sliding import all_resident_budget

    path = tmp_path / "qwen3"
    qwen3_fixture(path, tied=False, seed=3)
    model = AutoModel.from_pretrained(path, memory_percent=100)
    expected = model.generate_tokens(mx.array([[5, 9, 14]]), 0, 4)
    inference = sum(model.decoder.request_memory(3, 4)) + model.embed_lookup_bytes(3)
    model.available = all_resident_budget(model.planned_sizes(), inference)   # just the rest fits
    assert model.generate_tokens(mx.array([[5, 9, 14]]), 0, 4) == expected
    assert model.decoder.embed not in model.resident_names and model.embed_from_rows
    model.available = 10 * all_resident_budget(model.sizes, inference)        # plenty of room
    assert model.generate_tokens(mx.array([[5, 9, 14]]), 0, 4) == expected
    assert model.decoder.embed in model.resident and not model.embed_from_rows


def test_moe_reserve_covers_batched_decode(tmp_path):
    path = tmp_path / "moe"
    qwen3_moe_fixture(path)
    model = AutoModel.from_pretrained(path, memory_percent=100, expert_streaming=True, expert_cache_bytes=64_000)
    seen = []
    original = model.experts.reserve
    model.experts.reserve = lambda length: (seen.append(length), original(length))[1]
    experts = model.experts
    assert not experts.use_whole_layer(2) and experts.use_whole_layer(4)   # phases on opposite sides
    assert experts.reserve(2) > experts.reserve(4)                         # reserve() is not monotonic
    model.batch_rows = 4
    model.plan_request(2, 4)                              # two-token prefill, then four decode rows
    model.finish_request()
    assert {2, 4} <= set(seen)                            # both phases were sized
    plain = model.plan.reserve_bytes
    model.batch_rows = 1
    model.experts.reserve = original
    model.plan_request(2, 4)                              # the prefill alone sets the larger reserve
    model.finish_request()
    assert model.plan.reserve_bytes == plain


def test_embedding_rows_reject_out_of_range_ids(tmp_path):
    from mull.persist.mlx_model_persister import load_row_matrix

    path = tmp_path / "qwen3"
    qwen3_fixture(path, tied=False, seed=3)
    model = AutoModel.from_pretrained(path, memory_percent=0)
    vocabulary = model.manifest[model.decoder.embed + ".weight"][2][0]
    for bad in (vocabulary, -1):
        with pytest.raises(ValueError, match="Token IDs"):
            model.embed_rows(mx.array([[1, bad]]))
    key = model.source_keys[model.decoder.embed + ".weight"]
    with pytest.raises(IndexError):
        load_row_matrix(model.path, model.source_manifest, key, [0, vocabulary])
