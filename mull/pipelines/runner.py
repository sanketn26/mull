"""Runs queued executions one at a time, one step after another.

Prompt steps become ordinary jobs in the job queue, so batching, cancelling and the idle
unload all keep working. File steps run here, and every path is checked again at the moment
it is used: it must resolve inside the definition's folders and the server's allowed roots.
"""

import datetime
import os
import tempfile
import threading
import time

from ..server.store import FINISHED, QueueFull
from . import spec

MAX_READ_BYTES = 1 << 20
DEFAULT_NEW_TOKENS = 256


class StepError(Exception):
    pass


class Cancelled(Exception):
    pass


class Executor:
    def __init__(self, executions, worker, policy, poll_seconds=0.2, clock=time.monotonic):
        self.executions = executions
        self.jobs = executions.jobs
        self.worker = worker
        self.policy = policy
        self.poll_seconds = poll_seconds
        self.clock = clock
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.thread = None

    def start(self):
        for job_id in self.executions.interrupted():
            self.jobs.cancel(job_id)
        self.thread = threading.Thread(target=self.loop, daemon=True, name="executions")
        self.thread.start()

    def stop(self):
        self.stopping.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=10)

    def loop(self):
        while not self.stopping.is_set():
            execution = self.executions.claim_next()
            if execution is None:
                self.wake.wait(self.poll_seconds)
                self.wake.clear()
                continue
            self.run(execution)

    def run(self, execution):
        run = Run(self, execution)
        try:
            run.go()
            self.executions.finish(execution["id"], "done")
        except Cancelled:
            run.skip_rest("cancelled")
            self.executions.finish(execution["id"], "cancelled")
        except StepError as error:
            run.skip_rest("skipped")
            self.executions.finish(execution["id"], "failed", str(error))
        except Exception as error:  # a bug here must not kill the executor thread
            run.skip_rest("skipped")
            self.executions.finish(execution["id"], "failed", f"{type(error).__name__}: {error}")


class Run:
    """One execution in progress: its step records, results so far and deadline."""

    def __init__(self, executor, execution):
        self.executor = executor
        self.id = execution["id"]
        self.definition = execution["definition"]
        self.params = execution["params"]
        self.steps = execution["steps"]
        self.results = {}
        self.today = datetime.date.today().isoformat()
        self.deadline = executor.clock() + spec.parse_duration(self.definition["limits"]["time"])

    def go(self):
        issues = spec.check(self.definition, self.executor.policy)
        if not spec.ok(issues):
            raise StepError("; ".join(i["text"] for i in issues if i["level"] == "error"))
        for index, step in enumerate(self.definition["steps"], start=1):
            (kind, args), = step.items()
            self.stop_if_asked()
            self.record(index, state="running")
            try:
                output = getattr(self, f"do_{kind}")(index, self.expanded(args))
            except (StepError, Cancelled) as error:
                self.record(index, state="failed" if isinstance(error, StepError) else "skipped", error=str(error) or None)
                raise
            self.results[index] = output
            self.record(index, state="done", output=output)

    def expanded(self, args):
        return {k: spec.expand(v, self.params, self.results, self.today) if isinstance(v, str) else v
                for k, v in args.items()}

    def record(self, index, **changes):
        self.steps[index - 1].update(changes)
        self.executor.executions.save_steps(self.id, self.steps)

    def skip_rest(self, state):
        for item in self.steps:
            if item["state"] in ("pending", "running"):
                item["state"] = state
        self.executor.executions.save_steps(self.id, self.steps)

    def stop_if_asked(self):
        if self.executor.executions.cancel_requested(self.id) or self.executor.stopping.is_set():
            raise Cancelled()
        if self.executor.clock() > self.deadline:
            raise StepError("the time limit ran out")

    def allowed(self, path, key):
        """Resolve path and require it to sit inside the definition's folders and the server's roots."""
        folders = self.definition.get(key) or []
        if not (spec.inside(path, folders) and spec.inside(path, self.executor.policy.roots)):
            raise StepError(f"{path} is outside the allowed folders")
        return spec.resolve(path)

    def do_read(self, index, args):
        target = self.allowed(args["path"], "reads")
        try:
            with open(target, "rb") as handle:
                data = handle.read(MAX_READ_BYTES + 1)
        except OSError as error:
            raise StepError(f"cannot read {args['path']}: {error.strerror or error}") from None
        if len(data) > MAX_READ_BYTES:
            raise StepError(f"{args['path']} is larger than {MAX_READ_BYTES // 1024} KiB")
        return data.decode("utf-8", errors="replace")

    def do_save(self, index, args):
        target = self.allowed(args["path"], "writes")
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=os.path.dirname(target), prefix=".mull-")
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                out.write(args["content"])
            os.replace(temporary, target)
        except OSError as error:
            raise StepError(f"cannot write {args['path']}: {error.strerror or error}") from None
        return target

    def do_prompt(self, index, args):
        limits = self.definition["limits"]
        tokens = args.get("max_new_tokens") or limits.get("max_new_tokens") or min(DEFAULT_NEW_TOKENS, self.executor.policy.max_new_tokens)
        try:
            job_id = self.executor.jobs.submit(args["text"], tokens, {"temperature": 0.0, "thinking": False})
        except QueueFull as error:
            raise StepError(str(error)) from None
        self.record(index, job=job_id)
        self.executor.worker.wake.set()
        return self.wait_for(job_id)

    def wait_for(self, job_id):
        jobs = self.executor.jobs
        while True:
            job = jobs.get(job_id)
            if job["state"] in FINISHED:
                return self.finished_output(job)
            try:
                self.stop_if_asked()
            except (StepError, Cancelled):
                jobs.cancel(job_id)
                raise
            time.sleep(self.executor.poll_seconds)

    @staticmethod
    def finished_output(job):
        if job["state"] == "done":
            return job["output"]
        raise StepError(f"the model job {job['id']} {job['state']}" + (f": {job['error']}" if job["error"] else ""))
