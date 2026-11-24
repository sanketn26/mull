"""mull-mcp: the HTTP client against a live job server (fake model), and the MCP tool layer."""

import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path

from mull.mcp_server import MullClient, MullError, build_server, with_files
from mull.server.http import make_server
from mull.server.store import JobStore
from mull.server.worker import Worker
from tests.test_job_server import FakeRunner


class LiveServer(unittest.TestCase):
    def setUp(self):
        self.store = JobStore(":memory:", max_queued=5)
        self.worker = Worker(self.store, FakeRunner(), poll_seconds=0.01)
        self.server = make_server(self.store, self.worker, port=0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.client = MullClient(f"http://127.0.0.1:{self.server.server_port}")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.stop()


class TestClient(LiveServer):
    def test_submit_and_wait_for_the_result(self):
        self.worker.start()
        job = self.client.submit("ab", max_new_tokens=4)
        result = self.client.job(job, wait_seconds=5)
        self.assertEqual((result["state"], result["output"]), ("done", "abab"))

    def test_cancel_and_list(self):
        job = self.client.submit("queued only")
        self.assertTrue(self.client.cancel(job))
        self.assertEqual(self.client.jobs()[0]["state"], "cancelled")

    def test_server_errors_are_readable(self):
        with self.assertRaisesRegex(MullError, "400: 'prompt'"):
            self.client.submit("")
        with self.assertRaisesRegex(MullError, "not reachable"):
            MullClient("http://127.0.0.1:9").jobs()


class TestFiles(unittest.TestCase):
    def test_files_are_appended_with_their_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            path.write_text("hello")
            self.assertEqual(with_files("Summarise", [str(path)]), f"Summarise\n\n--- {path} ---\nhello")
            with self.assertRaisesRegex(MullError, "cannot read"):
                with_files("x", [str(Path(tmp) / "missing.txt")])


class TestMcpTools(LiveServer):
    def setUp(self):
        try:
            import mcp  # noqa: F401
        except ImportError:
            self.skipTest("mcp extra not installed")
        super().setUp()
        self.mcp = build_server(self.client)

    def call(self, name, arguments):
        result = asyncio.run(self.mcp.call_tool(name, arguments))
        self.assertFalse(getattr(result, "is_error", False), result)
        return json.loads(result.content[0].text)

    def test_tools_are_listed(self):
        names = {tool.name for tool in asyncio.run(self.mcp.list_tools())}
        self.assertEqual(names, {"mull_submit", "mull_result", "mull_cancel", "mull_list"})

    def test_submit_then_result_through_mcp(self):
        self.worker.start()
        job = self.call("mull_submit", {"prompt": "hi", "max_new_tokens": 3, "system": "terse"})["id"]
        self.assertEqual(self.store.get(job)["options"]["system"], "terse")
        result = self.call("mull_result", {"job_id": job, "wait_seconds": 5})
        self.assertEqual((result["state"], result["output"]), ("done", "hih"))
        self.assertNotIn("prompt", result)

    def test_expected_errors_reach_the_agent(self):
        from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

        with self.assertRaisesRegex(ToolError, "404: job 999 not found") as caught:
            asyncio.run(self.mcp.call_tool("mull_result", {"job_id": 999}))
        self.assertNotIsInstance(caught.exception, UnexpectedToolError)


if __name__ == "__main__":
    unittest.main()
