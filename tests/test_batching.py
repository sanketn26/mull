"""Batching: jobs decoded together must give exactly the tokens each gives alone."""

import sys

import pytest

if sys.platform != "darwin":
    pytest.skip("Apple Silicon MLX runtime required", allow_module_level=True)
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402

from mull import AutoModel  # noqa: E402
from mull.prefix_cache import PrefixCache  # noqa: E402
from mull.server.batch import NoRoom, PackedBatch  # noqa: E402
from mull.server.runners import PackedModelRunner  # noqa: E402
from mull.server.store import JobStore  # noqa: E402
from mull.server.worker import Worker  # noqa: E402
from tests.test_packed_mlx import llama_fixture, qwen3_fixture, qwen35_fixture  # noqa: E402

PROMPTS = [[5, 9, 14, 3, 27, 8, 11, 40, 2, 7], [12, 4], [30, 31, 32, 33, 34, 35]]
STEPS = 10


@pytest.fixture(params=["qwen3", "qwen35", "llama"])
def model(request, tmp_path):
    path = tmp_path / request.param
    if request.param == "qwen3":
        qwen3_fixture(path, tied=False, seed=3)
    elif request.param == "llama":
        llama_fixture(path)
    else:
        qwen35_fixture(path)
    return AutoModel.from_pretrained(path, memory_percent=50)


def alone(model, ids, steps=STEPS):
    """Reference: the runtime's own one-sequence generator (plain caches, no batch)."""
    model.eos = set()  # compare raw greedy streams
    return model.generate_tokens(mx.array([ids]), 0, steps)


def test_batch_matches_each_sequence_alone(model):
    expected = [alone(model, ids) for ids in PROMPTS]
    batch = PackedBatch(PackedModelRunner(model), max_batch=4)
    streams = [[batch.add(ids, STEPS, 0)] for ids in PROMPTS]
    for _ in range(STEPS - 1):
        for stream, token in zip(streams, batch.step()):
            stream.append(token)
    assert streams == expected


def test_rows_can_join_and_leave_mid_batch(model):
    expected = [alone(model, ids) for ids in PROMPTS]
    batch = PackedBatch(PackedModelRunner(model), max_batch=4)
    a = [batch.add(PROMPTS[0], STEPS, 0)]
    for _ in range(3):
        a.append(batch.step()[0])
    b = [batch.add(PROMPTS[1], STEPS, 0)]             # joins after 4 tokens of the first
    for _ in range(3):
        first, second = batch.step()
        a.append(first)
        b.append(second)
    batch.remove([0])                                  # the first leaves early
    c = [batch.add(PROMPTS[2], STEPS, 0)]
    while len(b) < STEPS:
        second, third = batch.step()
        b.append(second)
        c.append(third)
    assert a == expected[0][:7] and b == expected[1] and c == expected[2][:len(c)]


def test_full_batch_refuses_more(model):
    batch = PackedBatch(PackedModelRunner(model), max_batch=1)
    batch.add(PROMPTS[0], 4, 0)
    with pytest.raises(NoRoom):
        batch.add(PROMPTS[1], 4, 0)


def test_batched_prefix_cache_still_matches(model):
    expected = alone(model, PROMPTS[0] + [21, 22])
    runner = PackedModelRunner(model, prefix_cache=PrefixCache())
    batch = PackedBatch(runner, max_batch=2)
    batch.add(PROMPTS[0], 1, 0)
    batch.remove([0])
    stream = [batch.add(PROMPTS[0] + [21, 22], STEPS, 0)]
    stream += [batch.step()[0] for _ in range(STEPS - 1)]
    assert stream == expected and runner.prefix_cache.hits == 1


class TokenRunner(PackedModelRunner):
    """Real packed model, with token IDs as text so the worker can run on the fixture."""

    def encode(self, prompt, system=None, thinking=False):
        return [int(t) for t in prompt.split()]

    def decode(self, ids):
        return " ".join(map(str, ids))


def test_worker_runs_queued_jobs_together(model):
    expected = [alone(model, ids, 6) for ids in PROMPTS]
    store = JobStore(":memory:")
    jobs = [store.submit(" ".join(map(str, ids)), 6) for ids in PROMPTS]
    worker = Worker(store, TokenRunner(model), max_batch=3)
    worker.tick()
    assert sorted(worker.running) == sorted(jobs)      # all three admitted at once
    while worker.active:
        worker.advance()
    for job, tokens in zip(jobs, expected):
        done = store.get(job)
        assert done["state"] == "done" and done["output"] == " ".join(map(str, tokens))


def test_short_job_beside_long_one_is_budgeted_at_padded_length(model):
    batch = PackedBatch(PackedModelRunner(model), max_batch=2)
    long_ids, short = list(range(1, 41)), [5, 9]
    batch.add(long_ids, 300, 0)                                   # the long row runs first
    seen = {}
    original = model.plan_request
    model.plan_request = lambda n, limit: (seen.update(reserved=model.reserved_bytes), original(n, limit))[1]
    batch.add(short, 4, 0)
    request = model.decoder.request_memory
    padded, own = request(40 + 300, 1)[0], request(2, 4)[0]
    if padded == own:
        pytest.skip("this fixture's cache does not grow with length")   # Qwen3.5 fixture: no full attention
    # the long row's padded cache, plus the short row's padding up to that length
    assert padded > own                                           # the short row really is padded
    assert seen["reserved"] == padded + padded - own
    assert model.batch_rows == 2
    batch.remove([0, 1])
    assert model.batch_rows == 1


def test_worker_frees_the_model_after_the_idle_time(model):
    import mlx.core as mx

    now = [0.0]
    store = JobStore(":memory:")
    runner = TokenRunner(model, prefix_cache=PrefixCache())
    worker = Worker(store, runner, max_batch=2, unload_after=60, clock=lambda: now[0])
    store.submit("5 9 14 3", 4)
    worker.run_one()
    assert runner.loaded and model.resident
    assert not worker.maybe_unload()                   # idle clock starts here
    now[0] = 59
    assert not worker.maybe_unload() and runner.loaded  # not long enough
    now[0] = 61
    assert worker.maybe_unload()
    assert not model.resident and runner.prefix_cache.nbytes == 0 and not runner.loaded
    assert mx.get_cache_memory() == 0
    store.submit("5 9 14 3", 4)                         # the next job loads again and matches
    worker.run_one()
    assert store.get(2)["state"] == "done" and store.get(2)["output"] == store.get(1)["output"]


def test_worker_never_frees_while_running_or_when_disabled(model):
    now = [0.0]
    store = JobStore(":memory:")
    runner = TokenRunner(model)
    keep = Worker(store, runner, unload_after=None, clock=lambda: now[0])
    store.submit("5 9 14 3", 4)
    keep.run_one()
    now[0] = 10_000
    assert not keep.maybe_unload() and model.resident   # unload_after=None keeps it loaded
    busy = Worker(store, runner, max_batch=1, unload_after=0, clock=lambda: now[0])
    store.submit("5 9 14 3", 40)
    busy.tick()
    assert busy.active and not busy.maybe_unload() and model.resident   # a job is running


def test_repeated_load_and_unload_does_not_grow_memory(model):
    import gc

    import mlx.core as mx

    runner = TokenRunner(model, prefix_cache=PrefixCache())
    store = JobStore(":memory:")
    worker = Worker(store, runner, unload_after=0)
    readings = []
    for _ in range(6):
        store.submit("5 9 14 3 27 8", 8)
        worker.run_one()
        worker.maybe_unload()
        worker.maybe_unload()
        gc.collect()
        readings.append((mx.get_active_memory(), mx.get_cache_memory()))
    assert all(cache == 0 for _, cache in readings)
    assert max(active for active, _ in readings[1:]) <= readings[1][0]   # flat after the first cycle
