"""mull-mcp: the HTTP client against a live job server (fake model), and the MCP tool layer."""

import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path

from mull.mcp_server import MullClient, MullError, build_server, execution_result, with_files
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


class TestMcpPipelineTools(unittest.TestCase):
    """The definition tools exist only when the server runs with --pipelines."""

    def setUp(self):
        try:
            import mcp  # noqa: F401
        except ImportError:
            self.skipTest("mcp extra not installed")
        from mull.pipelines.api import PipelineApi
        from mull.pipelines.runner import Executor
        from mull.pipelines.spec import Policy
        from mull.pipelines.store import ExecutionStore

        self.root = tempfile.mkdtemp()
        self.store = JobStore(":memory:")
        self.worker = Worker(self.store, FakeRunner(), poll_seconds=0.01)
        self.worker.start()
        executions = ExecutionStore(self.store)
        policy = Policy(roots=(self.root,))
        self.executor = Executor(executions, self.worker, policy, poll_seconds=0.01)
        self.executor.start()
        self.plain = make_server(self.store, self.worker, port=0)
        self.piped = make_server(self.store, self.worker, port=0, pipelines=PipelineApi(policy, executions, self.executor))
        for server in (self.plain, self.piped):
            threading.Thread(target=server.serve_forever, daemon=True).start()

    def tearDown(self):
        for server in (self.plain, self.piped):
            server.shutdown()
            server.server_close()
        self.executor.stop()
        self.worker.stop()

    def tools(self, server):
        client = MullClient(f"http://127.0.0.1:{server.server_port}")
        return {tool.name for tool in asyncio.run(build_server(client).list_tools())}

    def test_tools_appear_only_with_the_feature(self):
        self.assertEqual(self.tools(self.plain), {"mull_submit", "mull_result", "mull_cancel", "mull_list"})
        extra = self.tools(self.piped) - self.tools(self.plain)
        self.assertEqual(extra, {"mull_steps", "mull_check", "mull_dry_run", "mull_execute", "mull_execution",
                                 "mull_cancel_execution"})

    def test_execute_and_read_the_result_through_the_client(self):
        client = MullClient(f"http://127.0.0.1:{self.piped.server_port}")
        (Path(self.root) / "a.md").write_text("abcdef")
        definition = {"job": "j", "reads": [self.root], "limits": {"time": "1m"},
                      "steps": [{"read": {"path": f"{self.root}/a.md"}}, {"prompt": {"text": "{steps.1}", "max_new_tokens": 3}}]}
        self.assertTrue(client.check(definition)["ok"])
        self.assertEqual(len(client.dry_run(definition)), 2)
        item = client.execution(client.execute(definition), wait_seconds=10)
        shown = execution_result(item)
        self.assertEqual((shown["state"], shown["steps"][1]["output"]), ("done", "abc"))

    def test_long_outputs_are_cut_for_the_agent(self):
        item = {"id": 1, "name": "j", "state": "done", "error": None,
                "steps": [{"kind": "read", "state": "done", "output": "x" * 5000, "error": None}]}
        shown = execution_result(item)["steps"][0]
        self.assertEqual((len(shown["output"]), shown["output_truncated"]), (4000, True))


if __name__ == "__main__":
    unittest.main()
