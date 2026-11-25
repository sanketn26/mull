"""Prefix cache: reused prompts must give exactly the tokens a fresh run gives."""

import sys

import pytest

if sys.platform != "darwin":
    pytest.skip("Apple Silicon MLX runtime required", allow_module_level=True)
pytest.importorskip("mlx_lm")

from mull import AutoModel  # noqa: E402
from mull.prefix_cache import PrefixCache, common_prefix  # noqa: E402
from mull.server.worker import PackedModelRunner  # noqa: E402
from tests.test_packed_mlx import qwen3_fixture, qwen35_fixture  # noqa: E402

PREFIX = [5, 9, 14, 3, 27, 8, 11, 40, 2, 7, 19, 33]


def run(runner, ids, n=12):
    return list(runner.tokens(ids, n))


@pytest.fixture(params=["qwen3", "qwen35"])
def model(request, tmp_path):
    path = tmp_path / request.param
    if request.param == "qwen3":
        qwen3_fixture(path, tied=False, seed=3)
    else:
        qwen35_fixture(path)
    return AutoModel.from_pretrained(path, memory_percent=50)


def test_extended_prompt_matches_a_fresh_run(model):
    fresh = run(PackedModelRunner(model), PREFIX + [21, 22])
    cached = PackedModelRunner(model, prefix_cache=PrefixCache())
    run(cached, PREFIX)
    assert run(cached, PREFIX + [21, 22]) == fresh
    assert cached.prefix_cache.tokens_reused == len(PREFIX)


def test_repeated_prompt_matches_a_fresh_run(model):
    fresh = run(PackedModelRunner(model), PREFIX)
    cached = PackedModelRunner(model, prefix_cache=PrefixCache())
    assert run(cached, PREFIX) == fresh
    assert run(cached, PREFIX) == fresh


def test_diverging_prompt_reuses_the_shared_part_when_trimmable(model):
    fresh = run(PackedModelRunner(model), PREFIX + [30, 31])
    cached = PackedModelRunner(model, prefix_cache=PrefixCache())
    run(cached, PREFIX + [44, 45, 46])
    assert run(cached, PREFIX + [30, 31]) == fresh
    trimmable = all(layer.is_trimmable() for layer in model.decoder.make_cache())
    assert cached.prefix_cache.tokens_reused == (len(PREFIX) if trimmable else 0)


def test_cache_respects_its_byte_limit(model):
    probe = PackedModelRunner(model, prefix_cache=PrefixCache())
    run(probe, PREFIX, n=1)
    one = probe.prefix_cache.nbytes
    cache = PrefixCache(limit_bytes=int(one * 1.5))
    runner = PackedModelRunner(model, prefix_cache=cache)
    run(runner, PREFIX, n=1)
    run(runner, [6] + PREFIX, n=1)
    assert len(cache.entries) == 1 and cache.nbytes <= cache.limit_bytes
    assert 0 < model.reserved_bytes <= cache.limit_bytes


def test_common_prefix():
    assert common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix([1, 2], [1, 2, 3]) == 2
    assert common_prefix([], [1]) == 0
