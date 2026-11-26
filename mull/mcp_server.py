"""MCP server that lets Claude Code, Codex or any MCP client queue jobs on a running Mull job server.

Start the job server first (`make serve MODEL=...`), then register this with your agent:
    claude mcp add mull -- uv run --directory /path/to/mull --extra mcp mull-mcp
Set MULL_URL if the job server is not on http://127.0.0.1:8765.
"""

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8765"
MAX_FILES_BYTES = 900_000  # stays under the job server's 1 MiB request limit
MAX_WAIT_SECONDS = 60
FINISHED = ("done", "failed", "cancelled")


try:  # with the mcp extra, MCP passes the message to the agent instead of logging a crash
    from mcp.server.mcpserver.exceptions import ToolError as _Base
except ImportError:
    _Base = Exception


class MullError(_Base):
    """An expected failure (bad request, unknown job, server down) with a message for the agent."""


class MullClient:
    """Plain HTTP client for the job API; the MCP tools are thin wrappers around it."""

    def __init__(self, base_url=None):
        self.base_url = (base_url or os.environ.get("MULL_URL") or DEFAULT_URL).rstrip("/")

    def call(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base_url + path, data=data,
                                         method="GET" if body is None else "POST",
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise MullError(f"{error.code}: {_error_text(error)}") from None
        except urllib.error.URLError as error:
            raise MullError(f"Mull job server not reachable at {self.base_url} ({error.reason}). "
                            "Start it with `make serve MODEL=...`.") from None

    def submit(self, prompt, files=(), max_new_tokens=None, **options):
        """Queue a job. `options` are passed through when set: system, thinking, temperature, stop."""
        body = {"prompt": with_files(prompt, files), "max_new_tokens": max_new_tokens, **options}
        return self.call("/jobs", {key: value for key, value in body.items() if value is not None})["id"]

    def job(self, job_id, wait_seconds=0):
        """Return the job; with wait_seconds, poll until it finishes or the wait runs out."""
        deadline = time.monotonic() + min(max(wait_seconds, 0), MAX_WAIT_SECONDS)
        while True:
            job = self.call(f"/jobs/{int(job_id)}")
            if job["state"] in FINISHED or time.monotonic() >= deadline:
                return job
            time.sleep(1)

    def cancel(self, job_id):
        return self.call(f"/jobs/{int(job_id)}/cancel", {})["cancelled"]

    def jobs(self, limit=20):
        return [summary(job) for job in self.call("/jobs")[:limit]]

    def pipelines_enabled(self):
        """True when the server was started with --pipelines (or MULL_PIPELINES=1 forces the tools on)."""
        if os.environ.get("MULL_PIPELINES") == "1":
            return True
        try:
            return bool(self.call("/health").get("pipelines"))
        except MullError:
            return False

    def check(self, definition):
        return self.call("/pipelines/check", {"definition": definition})

    def dry_run(self, definition, params=None):
        return self.call("/pipelines/dry-run", {"definition": definition, "params": params})["steps"]

    def execute(self, definition, params=None):
        return self.call("/pipelines/executions", {"definition": definition, "params": params})["id"]

    def execution(self, execution_id, wait_seconds=0):
        """Return the execution; with wait_seconds, poll until it finishes or the wait runs out."""
        deadline = time.monotonic() + min(max(wait_seconds, 0), MAX_WAIT_SECONDS)
        while True:
            item = self.call(f"/pipelines/executions/{int(execution_id)}")
            if item["state"] in FINISHED or time.monotonic() >= deadline:
                return item
            time.sleep(1)

    def cancel_execution(self, execution_id):
        return self.call(f"/pipelines/executions/{int(execution_id)}/cancel", {})["cancelled"]


def _error_text(error):
    try:
        return json.loads(error.read()).get("error", error.reason)
    except (ValueError, AttributeError):
        return error.reason


def with_files(prompt, files):
    """Append local text files to the prompt, so the agent doesn't pass them through its own context."""
    parts, total = [prompt], 0
    for name in files:
        path = Path(name).expanduser()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            raise MullError(f"cannot read {path}: {error.strerror}") from None
        total += len(text.encode())
        if total > MAX_FILES_BYTES:
            raise MullError(f"files exceed {MAX_FILES_BYTES} bytes; send fewer or smaller files")
        parts.append(f"\n\n--- {path} ---\n{text}")
    return "".join(parts)


def summary(job):
    keys = ("id", "state", "tokens", "error")
    return dict({key: job[key] for key in keys}, prompt=job["prompt"][:80])


def result(job):
    """What the agent sees: state, output, and timing; the prompt is left out to save context."""
    started, finished = job.get("started"), job.get("finished")
    seconds = round(finished - started, 1) if started and finished else None
    shown = {key: job[key] for key in ("id", "state", "tokens", "output", "error")} | {"seconds": seconds}
    if job.get("reasoning"):
        shown["reasoning"] = job["reasoning"]
    return shown


OUTPUT_CHARS = 4000


def execution_result(item):
    """What the agent sees of an execution: state and each step's result, long outputs cut."""
    steps = []
    for number, step in enumerate(item["steps"], start=1):
        shown = {"step": number, "kind": step["kind"], "state": step["state"], "output": step["output"][:OUTPUT_CHARS]}
        if len(step["output"]) > OUTPUT_CHARS:
            shown["output_truncated"] = True
        if step["error"]:
            shown["error"] = step["error"]
        steps.append(shown)
    return {"id": item["id"], "name": item["name"], "state": item["state"], "error": item["error"], "steps": steps}


INSTRUCTIONS = """Mull runs a local language model in the background, slowly (a few tokens per second when
the model is larger than free RAM). Use it for work that can wait or must stay on this machine:
summarising many files, overnight reviews, private documents. Submit with mull_submit, then check
back with mull_result. Pass file paths in `files` instead of pasting file contents."""


def build_server(client=None):
    from mcp.server.mcpserver import MCPServer

    client = client or MullClient()
    server = MCPServer("mull", instructions=INSTRUCTIONS)

    @server.tool()
    def mull_submit(prompt: str, files: list[str] | None = None, max_new_tokens: int | None = None,
                    system: str | None = None, thinking: bool = False, stop: list[str] | None = None) -> dict:
        """Queue a job on the local Mull model and return its id.

        files: local paths appended to the prompt. thinking: let the model reason first (slower).
        stop: up to 4 strings that end the answer."""
        return {"id": client.submit(prompt, files or (), max_new_tokens, system=system, thinking=thinking, stop=stop)}

    @server.tool()
    async def mull_result(job_id: int, wait_seconds: int = 0) -> dict:
        """Get a job's state and output so far. wait_seconds (up to 60) waits for it to finish first."""
        return result(await asyncio.to_thread(client.job, job_id, wait_seconds))

    @server.tool()
    def mull_cancel(job_id: int) -> dict:
        """Cancel a queued or running job."""
        return {"cancelled": client.cancel(job_id)}

    @server.tool()
    def mull_list(limit: int = 20) -> list[dict]:
        """List recent jobs, newest first, with their state."""
        return client.jobs(limit)

    if client.pipelines_enabled():
        add_pipeline_tools(server, client)
    return server


def add_pipeline_tools(server, client):
    """Tools for job definitions. Registered only when the Mull server runs with --pipelines."""

    @server.tool()
    def mull_steps() -> dict:
        """List the step types a definition may use, with their required and optional arguments."""
        return client.call("/pipelines/steps")

    @server.tool()
    def mull_check(definition: dict) -> dict:
        """Check a job definition without running it. Returns ok, issues and its parameters."""
        return client.check(definition)

    @server.tool()
    def mull_dry_run(definition: dict, params: dict | None = None) -> dict:
        """List what the definition would read, write and ask the model, and do none of it."""
        return {"steps": client.dry_run(definition, params)}

    @server.tool()
    def mull_execute(definition: dict, params: dict | None = None) -> dict:
        """Start a job definition in the background and return its execution id."""
        return {"id": client.execute(definition, params)}

    @server.tool()
    async def mull_execution(execution_id: int, wait_seconds: int = 0) -> dict:
        """Get an execution's state and each step's result. wait_seconds (up to 60) waits for it to finish."""
        return execution_result(await asyncio.to_thread(client.execution, execution_id, wait_seconds))

    @server.tool()
    def mull_cancel_execution(execution_id: int) -> dict:
        """Cancel a queued or running execution."""
        return {"cancelled": client.cancel_execution(execution_id)}


def main():
    build_server().run("stdio")


if __name__ == "__main__":
    main()
