"""JSON API and a minimal status page. Standard library only; binds to localhost by default."""

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .store import QueueFull

MAX_BODY_BYTES = 1 << 20
_JOB = re.compile(r"^/jobs/(\d+)$")
_CANCEL = re.compile(r"^/jobs/(\d+)/cancel$")

PAGE = """<!doctype html><meta charset="utf-8"><title>Mull jobs</title>
<style>body{font:14px system-ui;margin:16px;max-width:900px}textarea{width:100%;height:6em}
pre{white-space:pre-wrap;background:#8881;padding:8px}</style>
<h1>Mull jobs</h1>
<form id="f"><textarea name="prompt" placeholder="Prompt"></textarea>
<label>Max new tokens <input name="max_new_tokens" type="number" value="256"></label>
<button>Queue</button></form><div id="jobs"></div>
<script>
const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
async function refresh() {
  const jobs = await (await fetch("/jobs")).json();
  document.getElementById("jobs").innerHTML = jobs.map(j =>
    `<h3>#${j.id} ${j.state} (${j.tokens} tokens)</h3><p>${esc(j.prompt)}</p>
     <pre>${esc(j.output)}${j.error ? "\\n" + esc(j.error) : ""}</pre>`).join("");
}
document.getElementById("f").onsubmit = async e => {
  e.preventDefault();
  const d = new FormData(e.target);
  await fetch("/jobs", {method: "POST", body: JSON.stringify(
    {prompt: d.get("prompt"), max_new_tokens: +d.get("max_new_tokens")})});
  refresh();
};
refresh(); setInterval(refresh, 2000);
</script>"""


def make_handler(store, worker, max_new_tokens_limit):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body, content_type="application/json"):
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                return self.reply(HTTPStatus.OK, PAGE, "text/html; charset=utf-8")
            if self.path == "/jobs":
                return self.reply(HTTPStatus.OK, store.list())
            match = _JOB.match(self.path)
            job = store.get(int(match.group(1))) if match else None
            if job is None:
                return self.reply(HTTPStatus.NOT_FOUND, {"error": "not found"})
            self.reply(HTTPStatus.OK, job)

        def do_POST(self):
            match = _CANCEL.match(self.path)
            if match:
                found = store.cancel(int(match.group(1)))
                return self.reply(HTTPStatus.OK if found else HTTPStatus.NOT_FOUND, {"cancelled": found})
            if self.path != "/jobs":
                return self.reply(HTTPStatus.NOT_FOUND, {"error": "not found"})
            self.submit()

        def submit(self):
            try:
                prompt, max_new_tokens = parse_job(read_body(self), max_new_tokens_limit)
            except ValueError as error:
                return self.reply(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            try:
                job_id = store.submit(prompt, max_new_tokens)
            except QueueFull as error:
                return self.reply(HTTPStatus.TOO_MANY_REQUESTS, {"error": str(error)})
            worker.wake.set()
            self.reply(HTTPStatus.ACCEPTED, {"id": job_id})

    return Handler


def read_body(handler):
    length = int(handler.headers.get("Content-Length") or 0)
    if length > MAX_BODY_BYTES:
        raise ValueError("request body too large")
    return handler.rfile.read(length)


def parse_job(body, max_new_tokens_limit):
    try:
        data = json.loads(body or b"{}")
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {error}") from None
    prompt = data.get("prompt") if isinstance(data, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("'prompt' must be a non-empty string")
    max_new_tokens = data.get("max_new_tokens", min(256, max_new_tokens_limit))
    if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= max_new_tokens_limit:
        raise ValueError(f"'max_new_tokens' must be an integer from 1 to {max_new_tokens_limit}")
    return prompt, max_new_tokens


def make_server(store, worker, host="127.0.0.1", port=8765, max_new_tokens_limit=4096):
    return ThreadingHTTPServer((host, port), make_handler(store, worker, max_new_tokens_limit))
