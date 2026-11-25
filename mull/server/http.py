"""JSON API, server-sent events, and (optionally) the web interface. Standard library only."""

import json
import mimetypes
import re
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import psutil

from .store import FINISHED, STATES, QueueFull

MAX_BODY_BYTES = 1 << 20
EVENT_POLL_SECONDS = 0.5
_JOB = re.compile(r"^/jobs/(\d+)$")
_ACTION = re.compile(r"^/jobs/(\d+)/(cancel|events)$")


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Api:
    """Request handling that doesn't depend on the HTTP plumbing, so each route stays small."""

    def __init__(self, store, worker, max_new_tokens_limit=4096, web_root=None):
        self.store = store
        self.worker = worker
        self.max_new_tokens_limit = max_new_tokens_limit
        self.web_root = Path(web_root).resolve() if web_root else None

    def health(self):
        memory = psutil.virtual_memory()
        info = getattr(self.worker.runner, "describe", dict)()
        return dict(info, status="ok", jobs=self.store.counts(), running_job=self.worker.current,
                    running_jobs=getattr(self.worker, "running", []), max_batch=self.worker.batch.max_batch,
                    free_ram_gb=round(memory.available / 2**30, 2), total_ram_gb=round(memory.total / 2**30, 2))

    def list_jobs(self, query):
        state = query.get("state", [None])[0]
        if state and state not in STATES:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"'state' must be one of {', '.join(STATES)}")
        try:
            limit = min(max(int(query.get("limit", ["50"])[0]), 1), 500)
        except ValueError:
            raise ApiError(HTTPStatus.BAD_REQUEST, "'limit' must be an integer") from None
        return self.store.list(limit, state)

    def job(self, job_id):
        job = self.store.get(job_id)
        if job is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"job {job_id} not found")
        return job

    def submit(self, body):
        prompt, max_new_tokens, options = parse_job(body, self.max_new_tokens_limit)
        try:
            job_id = self.store.submit(prompt, max_new_tokens, options)
        except QueueFull as error:
            raise ApiError(HTTPStatus.TOO_MANY_REQUESTS, str(error)) from None
        self.worker.wake.set()
        return {"id": job_id}

    def cancel(self, job_id):
        if not self.store.cancel(job_id):
            raise ApiError(HTTPStatus.NOT_FOUND, f"job {job_id} not found")
        return {"cancelled": True}

    def delete(self, job_id):
        deleted = self.store.delete(job_id)
        if deleted is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"job {job_id} not found")
        if not deleted:
            raise ApiError(HTTPStatus.CONFLICT, "only finished jobs can be deleted; cancel it first")
        return {"deleted": True}

    def static_file(self, path):
        """Resolve a path under the web folder, or None. Never serves anything outside it."""
        if self.web_root is None:
            return None
        target = (self.web_root / (path.lstrip("/") or "index.html")).resolve()
        if target != self.web_root and self.web_root not in target.parents:
            return None
        return target if target.is_file() else None


def parse_job(body, max_new_tokens_limit):
    try:
        data = json.loads(body or b"{}")
    except json.JSONDecodeError as error:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid JSON: {error}") from None
    if not isinstance(data, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ApiError(HTTPStatus.BAD_REQUEST, "'prompt' must be a non-empty string")
    max_new_tokens = data.get("max_new_tokens", min(256, max_new_tokens_limit))
    if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= max_new_tokens_limit:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"'max_new_tokens' must be an integer from 1 to {max_new_tokens_limit}")
    return prompt, max_new_tokens, parse_options(data)


def parse_options(data):
    options = {}
    system = data.get("system")
    if system is not None:
        if not isinstance(system, str):
            raise ApiError(HTTPStatus.BAD_REQUEST, "'system' must be a string")
        options["system"] = system
    temperature = data.get("temperature", 0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ApiError(HTTPStatus.BAD_REQUEST, "'temperature' must be a number from 0 to 2")
    options["temperature"] = float(temperature)
    thinking = data.get("thinking", False)
    if not isinstance(thinking, bool):
        raise ApiError(HTTPStatus.BAD_REQUEST, "'thinking' must be true or false")
    options["thinking"] = thinking
    stop = data.get("stop", [])
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or len(stop) > 4 or not all(isinstance(s, str) and s for s in stop):
        raise ApiError(HTTPStatus.BAD_REQUEST, "'stop' must be up to 4 non-empty strings")
    if stop:
        options["stop"] = stop
    return options


def make_handler(api):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.dispatch(get_route)

        def do_POST(self):
            self.dispatch(post_route)

        def do_DELETE(self):
            self.dispatch(delete_route)

        def dispatch(self, route):
            url = urlsplit(self.path)
            try:
                result = route(api, self, url.path, parse_qs(url.query))
            except ApiError as error:
                return send_json(self, error.status, {"error": str(error)})
            if result is not None:
                send_json(self, HTTPStatus.ACCEPTED if route is post_route and "id" in result else HTTPStatus.OK,
                          result)

    return Handler


def get_route(api, handler, path, query):
    if path == "/health":
        return api.health()
    if path == "/jobs":
        return api.list_jobs(query)
    if match := _JOB.match(path):
        return api.job(int(match.group(1)))
    if (match := _ACTION.match(path)) and match.group(2) == "events":
        stream_events(api, handler, int(match.group(1)))
        return None
    target = api.static_file(path)
    if target is None:
        raise ApiError(HTTPStatus.NOT_FOUND, "not found")
    send_file(handler, target)
    return None


def post_route(api, handler, path, query):
    if path == "/jobs":
        return api.submit(read_body(handler))
    if (match := _ACTION.match(path)) and match.group(2) == "cancel":
        return api.cancel(int(match.group(1)))
    raise ApiError(HTTPStatus.NOT_FOUND, "not found")


def delete_route(api, handler, path, query):
    if match := _JOB.match(path):
        return api.delete(int(match.group(1)))
    raise ApiError(HTTPStatus.NOT_FOUND, "not found")


def read_body(handler):
    length = int(handler.headers.get("Content-Length") or 0)
    if length > MAX_BODY_BYTES:
        raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body too large")
    return handler.rfile.read(length)


def send_json(handler, status, body):
    send_bytes(handler, status, json.dumps(body).encode(), "application/json")


def send_file(handler, target):
    content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    if content_type.startswith("text/") or content_type.endswith("javascript"):
        content_type += "; charset=utf-8"
    send_bytes(handler, HTTPStatus.OK, target.read_bytes(), content_type)


def send_bytes(handler, status, data, content_type):
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def stream_events(api, handler, job_id):
    """Server-sent events: send the job whenever it changes, then end once it has finished."""
    job = api.job(job_id)
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.close_connection = True
    last = None
    try:
        while job is not None:
            if job["updated"] != last:
                last = job["updated"]
                handler.wfile.write(f"event: job\ndata: {json.dumps(job)}\n\n".encode())
                handler.wfile.flush()
            if job["state"] in FINISHED or api.worker.stopping.is_set():
                break
            time.sleep(EVENT_POLL_SECONDS)
            job = api.store.get(job_id)
    except (BrokenPipeError, ConnectionResetError):
        pass


def make_server(store, worker, host="127.0.0.1", port=8765, max_new_tokens_limit=4096, web_root=None):
    server = ThreadingHTTPServer((host, port), make_handler(Api(store, worker, max_new_tokens_limit, web_root)))
    server.daemon_threads = True
    return server
