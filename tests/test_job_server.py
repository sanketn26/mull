"""Job server: queue, worker and HTTP API, with a fake model so the tests run offline."""

import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from mull.server.http import make_server
from mull.server.store import JobStore, QueueFull
from mull.server.worker import Worker, cut_at_stop, split_reasoning


class FakeRunner:
    """Emits the prompt's characters as tokens; can block, fail, or run out of memory."""

    def __init__(self):
        self.gate = threading.Event()
        self.gate.set()
        self.closed = 0

    def encode(self, prompt, system=None, thinking=False):
        if prompt == "boom":
            raise RuntimeError("bad input")
        self.last = {"system": system, "thinking": thinking}
        return [ord(c) for c in prompt]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def describe(self):
        return {"model": "fake"}

    def tokens(self, ids, max_new_tokens, temperature=0.0):
        if ids == [ord(c) for c in "huge"]:
            raise MemoryError("needs 99 GB")
        try:
            for token in (ids * 100)[:max_new_tokens]:
                self.gate.wait(5)
                yield token
        finally:
            self.closed += 1


def wait_for(predicate, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestJobStore(unittest.TestCase):
    def setUp(self):
        self.store = JobStore(":memory:", max_queued=2)

    def test_jobs_run_in_submission_order(self):
        first, second = self.store.submit("a", 5), self.store.submit("b", 5)
        self.assertEqual(self.store.claim_next()["id"], first)
        self.assertEqual(self.store.claim_next()["id"], second)
        self.assertIsNone(self.store.claim_next())

    def test_full_queue_is_refused(self):
        self.store.submit("a", 5)
        self.store.submit("b", 5)
        with self.assertRaises(QueueFull):
            self.store.submit("c", 5)

    def test_cancelling_a_queued_job_is_immediate(self):
        job = self.store.submit("a", 5)
        self.assertTrue(self.store.cancel(job))
        self.assertEqual(self.store.get(job)["state"], "cancelled")
        self.assertIsNone(self.store.claim_next())
        self.assertFalse(self.store.cancel(999))

    def test_interrupted_jobs_are_queued_again_on_restart(self):
        job = self.store.submit("a", 5)
        self.store.claim_next()
        self.store.progress(job, "partial", 3)
        self.assertEqual(self.store.requeue_interrupted(), 1)
        self.assertEqual(self.store.get(job)["state"], "queued")
        self.assertEqual(self.store.get(job)["output"], "")


class TestJobStoreFile(unittest.TestCase):
    def test_jobs_survive_reopening_the_database(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.sqlite3"
            job = JobStore(path).submit("kept", 5)
            self.assertEqual(JobStore(path).get(job)["prompt"], "kept")


class TestWorker(unittest.TestCase):
    def setUp(self):
        self.store = JobStore(":memory:")
        self.runner = FakeRunner()
        self.worker = Worker(self.store, self.runner, poll_seconds=0.01)

    def tearDown(self):
        self.runner.gate.set()
        self.worker.stop()

    def state(self, job):
        return self.store.get(job)["state"]

    def test_job_runs_to_completion(self):
        job = self.store.submit("hi", 5)
        self.assertTrue(self.worker.run_one())
        self.assertEqual(self.store.get(job)["output"], "hihih")
        self.assertEqual(self.store.get(job)["tokens"], 5)
        self.assertEqual(self.state(job), "done")

    def test_failures_are_recorded_and_the_worker_continues(self):
        bad, huge, good = self.store.submit("boom", 5), self.store.submit("huge", 5), self.store.submit("ok", 2)
        for _ in range(3):
            self.worker.run_one()
        self.assertEqual(self.state(bad), "failed")
        self.assertIn("bad input", self.store.get(bad)["error"])
        self.assertIn("Not enough free memory", self.store.get(huge)["error"])
        self.assertEqual(self.state(good), "done")

    def test_running_job_stops_when_cancelled(self):
        self.runner.gate.clear()
        job = self.store.submit("x", 1000)
        self.worker.start()
        self.assertTrue(wait_for(lambda: self.state(job) == "running"))
        self.store.cancel(job)
        self.runner.gate.set()
        self.assertTrue(wait_for(lambda: self.state(job) == "cancelled"))
        self.assertLess(self.store.get(job)["tokens"], 1000)
        self.assertEqual(self.runner.closed, 1)

    def test_shutdown_leaves_the_job_for_the_next_start(self):
        self.runner.gate.clear()
        job = self.store.submit("x", 1000)
        self.worker.start()
        self.assertTrue(wait_for(lambda: self.state(job) == "running"))
        self.worker.stopping.set()
        self.runner.gate.set()
        self.worker.stop()
        self.assertEqual(self.state(job), "running")
        self.assertEqual(self.store.requeue_interrupted(), 1)


class TestHttpApi(unittest.TestCase):
    def setUp(self):
        self.store = JobStore(":memory:", max_queued=3)
        self.worker = Worker(self.store, FakeRunner(), poll_seconds=0.01)
        self.server = make_server(self.store, self.worker, port=0, max_new_tokens_limit=50)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.stop()

    def call(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def test_submit_then_read_the_result(self):
        self.worker.start()
        status, body = self.call("/jobs", {"prompt": "ab", "max_new_tokens": 4})
        self.assertEqual(status, 202)
        job = json.loads(body)["id"]
        self.assertTrue(wait_for(lambda: json.loads(self.call(f"/jobs/{job}")[1])["state"] == "done"))
        self.assertEqual(json.loads(self.call(f"/jobs/{job}")[1])["output"], "abab")
        self.assertEqual(len(json.loads(self.call("/jobs")[1])), 1)

    def test_invalid_requests_are_rejected(self):
        self.assertEqual(self.call("/jobs", {"prompt": ""})[0], 400)
        self.assertEqual(self.call("/jobs", {"prompt": "a", "max_new_tokens": 51})[0], 400)
        self.assertEqual(self.call("/jobs/42")[0], 404)
        self.assertEqual(self.call("/jobs/42/cancel", {})[0], 404)

    def test_full_queue_returns_429(self):
        for _ in range(3):
            self.assertEqual(self.call("/jobs", {"prompt": "a"})[0], 202)
        self.assertEqual(self.call("/jobs", {"prompt": "a"})[0], 429)

    def test_no_web_interface_unless_asked(self):
        self.assertEqual(self.call("/")[0], 404)


if __name__ == "__main__":
    unittest.main()


class TestOutputHandling(unittest.TestCase):
    def test_reasoning_is_split_from_the_answer(self):
        self.assertEqual(split_reasoning("<think>\nhmm\n</think>\n\nParis"), ("Paris", "hmm"))
        self.assertEqual(split_reasoning("<think>still going"), ("", "still going"))
        self.assertEqual(split_reasoning("Paris"), ("Paris", ""))

    def test_stop_sequences_cut_at_the_earliest(self):
        self.assertEqual(cut_at_stop("a END b STOP", ["STOP", "END"]), "a ")
        self.assertIsNone(cut_at_stop("abc", ["x"]))


class TestJobOptions(unittest.TestCase):
    def setUp(self):
        self.store = JobStore(":memory:")
        self.runner = FakeRunner()
        self.worker = Worker(self.store, self.runner)

    def test_options_reach_the_runner_and_stop_ends_the_job(self):
        job = self.store.submit("ab;", 50, {"system": "be brief", "thinking": True, "stop": [";"]})
        self.worker.run_one()
        self.assertEqual(self.runner.last, {"system": "be brief", "thinking": True})
        done = self.store.get(job)
        self.assertEqual((done["state"], done["output"], done["tokens"]), ("done", "ab", 3))

    def test_delete_only_finished_jobs(self):
        job = self.store.submit("a", 2)
        self.assertFalse(self.store.delete(job))
        self.worker.run_one()
        self.assertTrue(self.store.delete(job))
        self.assertIsNone(self.store.delete(job))


class TestHttpApiV2(TestHttpApi):
    def test_health(self):
        health = json.loads(self.call("/health")[1])
        self.assertEqual((health["status"], health["model"], health["jobs"]["queued"]), ("ok", "fake", 0))
        self.assertGreater(health["free_ram_gb"], 0)

    def test_options_are_validated(self):
        for bad in ({"temperature": 3}, {"thinking": "yes"}, {"stop": ["a"] * 5}, {"system": 1}):
            self.assertEqual(self.call("/jobs", dict(prompt="a", **bad))[0], 400, bad)
        status, body = self.call("/jobs", {"prompt": "a", "stop": "x", "thinking": True})
        self.assertEqual(status, 202)
        options = self.store.get(json.loads(body)["id"])["options"]
        self.assertEqual(options, {"temperature": 0.0, "thinking": True, "stop": ["x"]})

    def test_filter_and_delete(self):
        first = json.loads(self.call("/jobs", {"prompt": "a"})[1])["id"]
        self.call(f"/jobs/{first}/cancel", {})
        self.call("/jobs", {"prompt": "b"})
        self.assertEqual([j["id"] for j in json.loads(self.call("/jobs?state=cancelled")[1])], [first])
        self.assertEqual(self.call("/jobs?state=nope")[0], 400)
        request = urllib.request.Request(f"{self.base}/jobs/{first}", method="DELETE")
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
        self.assertEqual(self.call(f"/jobs/{first}")[0], 404)

    def test_events_stream_until_the_job_finishes(self):
        self.worker.start()
        job = json.loads(self.call("/jobs", {"prompt": "ab", "max_new_tokens": 20})[1])["id"]
        with urllib.request.urlopen(f"{self.base}/jobs/{job}/events", timeout=10) as response:
            self.assertEqual(response.headers["Content-Type"], "text/event-stream")
            events = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                      if line.startswith("data: ")]
        self.assertEqual(events[-1]["state"], "done")
        self.assertEqual(events[-1]["output"], "ab" * 10)


class TestWebInterface(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        web = Path(self.tmp.name) / "web"
        web.mkdir()
        (web / "index.html").write_text("<title>Mull</title>")
        (Path(self.tmp.name) / "secret.txt").write_text("private")
        self.store = JobStore(":memory:")
        self.server = make_server(self.store, Worker(self.store, FakeRunner()), port=0, web_root=web)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def test_index_is_served(self):
        with urllib.request.urlopen(self.base + "/", timeout=5) as response:
            self.assertIn("text/html", response.headers["Content-Type"])
            self.assertIn(b"Mull", response.read())

    def test_files_outside_the_folder_are_not_served(self):
        for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/missing.js"):
            with self.assertRaises(urllib.error.HTTPError, msg=path) as caught:
                urllib.request.urlopen(self.base + path, timeout=5)
            self.assertEqual(caught.exception.code, 404)


class TestRunnerChoice(unittest.TestCase):
    def test_runner_matches_the_runtime(self):
        from mull.server.worker import LlamaMlxRunner, PackedModelRunner, runner_for

        packed = type("Packed", (), {"request_lock": None, "plan_request": None, "tokenizer": None})()
        llama = type("Llama", (), {"model_generate": None, "tokenizer": None})()
        self.assertIsInstance(runner_for(packed), PackedModelRunner)
        self.assertIs(type(runner_for(llama)), LlamaMlxRunner)
        self.assertIsNone(runner_for(object()))
