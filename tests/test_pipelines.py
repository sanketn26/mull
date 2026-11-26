"""Job definitions: checking, dry run, executions and their API, and that the feature stays out of the way when off."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

from mull.pipelines import spec
from mull.pipelines.api import PipelineApi
from mull.pipelines.runner import Executor
from mull.pipelines.store import ExecutionStore
from mull.server.http import make_server
from mull.server.store import JobStore
from mull.server.worker import Worker
from tests.test_job_server import FakeRunner, wait_for


def definition(root, **changes):
    base = {
        "job": "weekly-notes",
        "params": {"topic": "decoding"},
        "reads": [root],
        "writes": [root],
        "limits": {"time": "30m"},
        "steps": [
            {"read": {"path": f"{root}/week.md"}},
            {"prompt": {"text": "{steps.1}", "max_new_tokens": 5}},
            {"save": {"path": f"{root}/{{date}}.md", "content": "{steps.2}"}},
        ],
    }
    base.update(changes)
    return base


def errors(issues):
    return [i["text"] for i in issues if i["level"] == "error"]


def make_root():
    root = os.path.realpath(tempfile.mkdtemp())
    with open(os.path.join(root, "week.md"), "w") as handle:
        handle.write("hello world")
    return root


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.root = make_root()
        self.policy = spec.Policy(roots=(self.root,))

    def problems(self, **changes):
        return errors(spec.check(definition(self.root, **changes), self.policy))

    def test_valid_definition_passes(self):
        self.assertEqual(spec.check(definition(self.root), self.policy), [])

    def test_time_limit_is_required(self):
        self.assertTrue(any("limits.time" in e for e in self.problems(limits={})))

    def test_bad_duration_and_unknown_keys_are_refused(self):
        self.assertEqual(len(self.problems(limits={"time": "soon", "cpu": 1}, extra=1)), 3)

    def test_unknown_step_type_is_refused(self):
        self.assertIn("unknown step type", self.problems(steps=[{"shell": {"cmd": "ls"}}])[0])

    def test_reference_must_point_to_an_earlier_step(self):
        self.assertIn("earlier step", self.problems(steps=[{"prompt": {"text": "{steps.1}"}}])[0])

    def test_undeclared_param_and_unknown_reference_are_refused(self):
        self.assertEqual(len(self.problems(steps=[{"prompt": {"text": "{params.nope} {env.HOME}"}}])), 2)

    def test_read_outside_reads_is_refused(self):
        self.assertIn("not inside reads", self.problems(steps=[{"read": {"path": "/etc/hosts"}}])[0])

    def test_symlink_out_of_reads_is_refused(self):
        os.symlink(os.path.realpath(tempfile.mkdtemp()), os.path.join(self.root, "link"))
        self.assertIn("not inside reads", self.problems(steps=[{"read": {"path": f"{self.root}/link/secret"}}])[0])

    def test_dotdot_escape_is_refused(self):
        self.assertTrue(self.problems(steps=[{"read": {"path": f"{self.root}/../x"}}]))

    def test_save_outside_writes_is_refused(self):
        steps = [{"save": {"path": "/tmp-not-listed/x", "content": "hi"}}]
        self.assertIn("not inside writes", self.problems(steps=steps)[0])

    def test_path_may_not_start_with_a_reference(self):
        self.assertIn("not inside reads", self.problems(steps=[{"read": {"path": "{params.topic}/x"}}])[0])

    def test_server_roots_limit_what_a_definition_may_list(self):
        elsewhere = os.path.realpath(tempfile.mkdtemp())
        self.assertTrue(any("--pipeline-roots" in e for e in self.problems(reads=[elsewhere])))

    def test_no_roots_means_no_file_access(self):
        found = errors(spec.check(definition(self.root)))
        self.assertTrue(any("--pipeline-roots" in e for e in found))

    def test_server_caps_apply(self):
        tight = spec.Policy(roots=(self.root,), max_steps=2, max_seconds=60, max_new_tokens=3)
        found = errors(spec.check(definition(self.root), tight))
        self.assertEqual(len(found), 3, found)

    def test_parameters_can_describe_themselves(self):
        params = {"topic": {"type": "text", "label": "Topic", "default": "kv"}, "n": 3, "flag": {"type": "bool"}}
        info = {i["name"]: i for i in spec.params_info(definition(self.root, params=params))}
        self.assertEqual(info["topic"]["label"], "Topic")
        self.assertEqual((info["n"]["type"], info["n"]["default"]), ("number", 3))
        self.assertIsNone(info["flag"]["default"])
        self.assertTrue(self.problems(params={"x": {"type": "date"}, "y": {"bogus": 1}}))

    def test_missing_and_mistyped_parameters_are_refused(self):
        d = definition(self.root, params={"flag": {"type": "bool"}})
        with self.assertRaises(spec.SpecError):
            spec.merged_params(d, {})
        with self.assertRaises(spec.SpecError):
            spec.merged_params(d, {"flag": "yes"})
        self.assertEqual(spec.merged_params(d, {"flag": True}), {"flag": True})


class TestLoadAndDryRun(unittest.TestCase):
    def setUp(self):
        self.root = make_root()
        self.policy = spec.Policy(roots=(self.root,))

    def test_json_loads_without_pyyaml(self):
        with mock.patch.dict(sys.modules, {"yaml": None}):
            self.assertEqual(spec.load_definition('{"job": "a"}'), {"job": "a"})
            with self.assertRaises(spec.SpecError) as raised:
                spec.load_definition("job: a")
        self.assertIn("PyYAML", str(raised.exception))

    def test_yaml_loads_when_installed(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")
        self.assertEqual(spec.load_definition("job: a\nlimits: {time: 5m}")["limits"], {"time": "5m"})

    def test_dry_run_lists_steps_and_touches_nothing(self):
        before = os.listdir(self.root)
        lines = spec.dry_run(definition(self.root), None, today="2026-11-26", policy=self.policy)
        self.assertEqual(lines, [f"1. read {self.root}/week.md", "2. ask the model (up to 5 tokens)",
                                 f"3. write {self.root}/2026-11-26.md"])
        self.assertEqual(os.listdir(self.root), before)

    def test_dry_run_refuses_an_invalid_definition_and_unknown_params(self):
        with self.assertRaises(spec.SpecError):
            spec.dry_run(definition(self.root, limits={}), policy=self.policy)
        with self.assertRaises(spec.SpecError):
            spec.dry_run(definition(self.root), {"bogus": 1}, policy=self.policy)

    def test_expand_fills_every_reference(self):
        self.assertEqual(spec.expand("{params.a}-{steps.1}-{date}", {"a": "x"}, {1: "y"}, "d"), "x-y-d")


class Stack(unittest.TestCase):
    """A job server with a fake model, plus the executor and the optional API."""

    def build(self, clock=None, with_api=True):
        self.root = make_root()
        self.policy = spec.Policy(roots=(self.root,))
        self.store = JobStore(":memory:", max_queued=20)
        self.runner = FakeRunner()
        self.worker = Worker(self.store, self.runner, poll_seconds=0.01)
        self.executions = ExecutionStore(self.store)
        kwargs = {"clock": clock} if clock else {}
        self.executor = Executor(self.executions, self.worker, self.policy, poll_seconds=0.01, **kwargs)
        self.worker.start()
        self.executor.start()
        self.addCleanup(self.executor.stop)
        self.addCleanup(self.worker.stop)
        if with_api:
            self.serve(PipelineApi(self.policy, self.executions, self.executor))

    def serve(self, pipelines):
        server = make_server(self.store, self.worker, port=0, pipelines=pipelines)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.base = f"http://127.0.0.1:{server.server_port}"

    def call(self, path, body=None, method=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method or ("POST" if data is not None else "GET"))
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def execute(self, given=None, **changes):
        status, body = self.call("/pipelines/executions", {"definition": definition(self.root, **changes), "params": given})
        self.assertEqual(status, 202, body)
        return body["id"]

    def state(self, execution_id):
        return self.call(f"/pipelines/executions/{execution_id}")[1]

    def wait_state(self, execution_id, state):
        self.assertTrue(wait_for(lambda: self.state(execution_id)["state"] == state), self.state(execution_id))
        return self.state(execution_id)


class TestExecutions(Stack):
    def setUp(self):
        self.build()

    def test_steps_run_in_order_and_write_the_file(self):
        item = self.wait_state(self.execute(), "done")
        self.assertEqual([s["state"] for s in item["steps"]], ["done"] * 3)
        self.assertEqual(item["steps"][1]["output"], "hello")
        self.assertIsNotNone(item["steps"][1]["job"])
        saved = item["steps"][2]["output"]
        self.assertEqual(open(saved).read(), "hello")
        self.assertEqual(os.listdir(self.root).count("week.md"), 1)

    def test_prompt_step_is_an_ordinary_job(self):
        item = self.wait_state(self.execute(), "done")
        job = self.call(f"/jobs/{item['steps'][1]['job']}")[1]
        self.assertEqual((job["state"], job["output"]), ("done", "hello"))

    def test_cancel_stops_a_running_execution_and_its_job(self):
        self.runner.gate.clear()
        execution = self.execute()
        self.assertTrue(wait_for(lambda: self.state(execution)["steps"][1]["job"] is not None))
        self.assertEqual(self.call(f"/pipelines/executions/{execution}/cancel", {})[0], 200)
        self.runner.gate.set()
        item = self.wait_state(execution, "cancelled")
        self.assertEqual(item["steps"][2]["state"], "cancelled")
        job = self.call(f"/jobs/{item['steps'][1]['job']}")[1]
        self.assertTrue(wait_for(lambda: self.call(f"/jobs/{job['id']}")[1]["state"] in ("cancelled", "done")))

    def test_a_failed_step_stops_the_rest(self):
        steps = [{"read": {"path": f"{self.root}/missing.md"}}, {"save": {"path": f"{self.root}/x.md", "content": "a"}}]
        item = self.wait_state(self.execute(steps=steps), "failed")
        self.assertIn("cannot read", item["error"])
        self.assertEqual([s["state"] for s in item["steps"]], ["failed", "skipped"])
        self.assertNotIn("x.md", os.listdir(self.root))

    def test_a_symlink_swapped_in_after_submit_is_refused_at_run_time(self):
        outside = os.path.join(os.path.realpath(tempfile.mkdtemp()), "secret.txt")
        with open(outside, "w") as handle:
            handle.write("secret")
        self.executor.stopping.set()  # hold the executor so the file can be swapped before it runs
        self.executor.thread.join(5)
        execution = self.execute(steps=[{"read": {"path": f"{self.root}/week.md"}}])
        os.remove(f"{self.root}/week.md")
        os.symlink(outside, f"{self.root}/week.md")
        self.executor.stopping.clear()
        self.executor.start()
        item = self.wait_state(execution, "failed")
        self.assertIn("not inside reads", item["error"])  # refused when the run starts, before any step

    def test_a_parameter_cannot_steer_a_path_outside_at_run_time(self):
        outside = os.path.join(os.path.realpath(tempfile.mkdtemp()), "secret.txt")
        with open(outside, "w") as handle:
            handle.write("secret")
        steps = [{"read": {"path": f"{self.root}/{{params.name}}"}}]
        passes = self.execute(steps=steps, params={"name": "week.md"}, given={"name": "week.md"})
        self.assertEqual(self.wait_state(passes, "done")["steps"][0]["output"], "hello world")
        status, body = self.call("/pipelines/executions", {
            "definition": definition(self.root, steps=steps, params={"name": "week.md"}),
            "params": {"name": "../" + os.path.relpath(outside, self.root)}})
        item = self.wait_state(body["id"], "failed")
        self.assertIn("outside the allowed folders", item["error"])

    def test_the_time_limit_stops_an_execution(self):
        ticks = [0.0]
        self.executor.stop()
        self.build(clock=lambda: ticks[0])
        self.runner.gate.clear()
        execution = self.execute(limits={"time": "10s"})
        self.assertTrue(wait_for(lambda: self.state(execution)["steps"][1]["job"] is not None))
        ticks[0] = 100.0
        item = self.wait_state(execution, "failed")
        self.assertIn("time limit", item["error"])
        self.runner.gate.set()

    def test_running_executions_are_failed_after_a_restart(self):
        self.executor.stop()
        execution = self.executions.submit(definition(self.root), {"topic": "x"})
        self.executions.claim_next()
        self.executions.interrupted()
        item = self.executions.get(execution)
        self.assertEqual((item["state"], item["error"]), ("failed", "interrupted by a server restart"))

    def test_list_delete_and_events(self):
        execution = self.execute()
        self.wait_state(execution, "done")
        listed = self.call("/pipelines/executions")[1]
        self.assertEqual((listed[0]["id"], listed[0]["steps"][0]), (execution, {"kind": "read", "state": "done"}))
        self.assertNotIn("output", listed[0]["steps"][0])
        with urllib.request.urlopen(f"{self.base}/pipelines/executions/{execution}/events", timeout=5) as response:
            text = response.read().decode()
        self.assertIn("event: execution", text)
        self.assertEqual(self.call(f"/pipelines/executions/{execution}", method="DELETE")[0], 200)
        self.assertEqual(self.call(f"/pipelines/executions/{execution}")[0], 404)

    def test_active_executions_cannot_be_deleted(self):
        self.runner.gate.clear()
        execution = self.execute()
        self.assertTrue(wait_for(lambda: self.state(execution)["state"] == "running"))
        self.assertEqual(self.call(f"/pipelines/executions/{execution}", method="DELETE")[0], 409)
        self.call(f"/pipelines/executions/{execution}/cancel", {})
        self.runner.gate.set()

    def test_invalid_definitions_and_params_are_refused_at_submit(self):
        self.assertEqual(self.call("/pipelines/executions", {"definition": definition(self.root, limits={})})[0], 422)
        body = {"definition": definition(self.root), "params": {"nope": 1}}
        self.assertEqual(self.call("/pipelines/executions", body)[0], 422)
        self.assertEqual(self.call("/pipelines/executions", {"definition": definition(self.root), "params": 3})[0], 400)

    def test_check_dry_run_and_steps(self):
        status, body = self.call("/pipelines/check", {"definition": definition(self.root)})
        self.assertEqual((status, body["ok"], body["params"][0]["name"]), (200, True, "topic"))
        body = self.call("/pipelines/check", {"definition": json.dumps(definition(self.root, limits={}))})[1]
        self.assertFalse(body["ok"])
        self.assertEqual(len(self.call("/pipelines/dry-run", {"definition": definition(self.root)})[1]["steps"]), 3)
        self.assertEqual(self.call("/pipelines/check", {})[0], 400)
        self.assertIn("prompt", self.call("/pipelines/steps")[1])

    def test_health_reports_the_feature(self):
        self.assertTrue(self.call("/health")[1]["pipelines"])


class TestOffByDefault(Stack):
    def test_routes_do_not_exist_unless_enabled(self):
        self.build(with_api=False)
        self.serve(None)
        self.assertEqual(self.call("/pipelines/check", {"definition": {}})[0], 404)
        self.assertEqual(self.call("/pipelines/steps")[0], 404)
        self.assertEqual(self.call("/pipelines/executions")[0], 404)
        self.assertNotIn("pipelines", self.call("/health")[1])

    def test_check_works_without_an_execution_store(self):
        self.build(with_api=False)
        self.serve(PipelineApi(self.policy))
        self.assertTrue(self.call("/pipelines/check", {"definition": definition(self.root)})[1]["ok"])
        self.assertEqual(self.call("/pipelines/executions", {"definition": definition(self.root)})[0], 404)

    def test_importing_the_server_does_not_load_pipelines(self):
        code = ("import sys, mull.server.http, mull.server.worker, mull.server.store, mull.server.__main__; "
                "print('mull.pipelines' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.getcwd())
        self.assertEqual(out.stdout.strip(), "False", out.stderr)

    def test_no_tables_are_created_without_the_feature(self):
        store = JobStore(":memory:")
        tables = {row[0] for row in store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("executions", tables)


if __name__ == "__main__":
    unittest.main()
