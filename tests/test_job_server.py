"""Job server: queue, worker and HTTP API, with a fake model so the tests run offline."""

import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from mull.server.http import make_server
from mull.server.store import JobStore, QueueFull
from mull.server.worker import Worker


class FakeRunner:
    """Emits the prompt's characters as tokens; can block, fail, or run out of memory."""

    def __init__(self):
        self.gate = threading.Event()
        self.gate.set()
        self.closed = 0

    def encode(self, prompt):
        if prompt == "boom":
            raise RuntimeError("bad input")
        return [ord(c) for c in prompt]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def tokens(self, ids, max_new_tokens):
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

    def test_status_page_is_served(self):
        status, body = self.call("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Mull jobs", body)


if __name__ == "__main__":
    unittest.main()
