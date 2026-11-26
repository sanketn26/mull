"""HTTP routes for job definitions and their executions. Mounted only when the server starts with --pipelines.

POST   /pipelines/check                  {"definition"}                -> {"ok", "issues", "params"}
POST   /pipelines/dry-run                {"definition", "params"}      -> {"steps": [...]}
GET    /pipelines/steps                                                -> the step vocabulary
POST   /pipelines/executions             {"definition", "params"}      -> {"id"}
GET    /pipelines/executions             ?state=&limit=                -> summaries, newest first
GET    /pipelines/executions/{id}                                      -> state, per-step results
GET    /pipelines/executions/{id}/events                               -> live updates (server-sent events)
POST   /pipelines/executions/{id}/cancel
DELETE /pipelines/executions/{id}                                      -> finished executions only
"""

import json
import re
import threading
from http import HTTPStatus

from ..server.http import ApiError, stream_updates
from ..server.store import FINISHED, STATES, QueueFull
from . import spec

_EXECUTION = re.compile(r"^/pipelines/executions/(\d+)$")
_ACTION = re.compile(r"^/pipelines/executions/(\d+)/(cancel|events)$")


class PipelineApi:
    def __init__(self, policy=None, executions=None, executor=None):
        self.policy = policy or spec.Policy()
        self.executions = executions
        self.executor = executor
        self.stopping = executor.stopping if executor else threading.Event()

    # routing -----------------------------------------------------------
    def get(self, path, query, handler):
        if path == "/pipelines/steps":
            return {name: {"required": sorted(req), "optional": sorted(opt)} for name, (req, opt) in spec.STEPS.items()}
        if path == "/pipelines/executions":
            return self.list_executions(query)
        if match := _EXECUTION.match(path):
            return self.execution(int(match.group(1)))
        if (match := _ACTION.match(path)) and match.group(2) == "events":
            self.stream(handler, int(match.group(1)))
            return None
        raise ApiError(HTTPStatus.NOT_FOUND, "not found")

    def post(self, path, body):
        if (match := _ACTION.match(path)) and match.group(2) == "cancel":
            return self.cancel(int(match.group(1)))
        data = self.parse(body)
        definition = self.definition(data)
        if path == "/pipelines/check":
            issues = spec.check(definition, self.policy)
            info = spec.params_info(definition) if isinstance(definition, dict) else []
            return {"ok": spec.ok(issues), "issues": issues, "params": info}
        if path == "/pipelines/dry-run":
            return {"steps": self.dry_run(definition, data.get("params"))}
        if path == "/pipelines/executions":
            return self.submit(definition, data.get("params"))
        raise ApiError(HTTPStatus.NOT_FOUND, "not found")

    def delete(self, path):
        match = _EXECUTION.match(path)
        if not match or self.executions is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "not found")
        deleted = self.executions.delete(int(match.group(1)))
        if deleted is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"execution {match.group(1)} not found")
        if not deleted:
            raise ApiError(HTTPStatus.CONFLICT, "only finished executions can be deleted; cancel it first")
        return {"deleted": True}

    # definitions ---------------------------------------------------------
    def dry_run(self, definition, params):
        self.require_object(params, "params")
        try:
            return spec.dry_run(definition, params, policy=self.policy)
        except spec.SpecError as error:
            raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(error)) from None

    @staticmethod
    def require_object(value, name):
        if value is not None and not isinstance(value, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, f"'{name}' must be an object")

    @staticmethod
    def parse(body):
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid JSON: {error}") from None
        if not isinstance(data, dict) or "definition" not in data:
            raise ApiError(HTTPStatus.BAD_REQUEST, "body must be an object with a 'definition'")
        return data

    @staticmethod
    def definition(data):
        value = data["definition"]
        if not isinstance(value, str):
            return value
        try:
            return spec.load_definition(value)
        except spec.SpecError as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from None

    # executions ----------------------------------------------------------
    def require_store(self):
        if self.executions is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "executions are not enabled on this server")

    def submit(self, definition, params):
        self.require_store()
        self.require_object(params, "params")
        self.dry_run(definition, params)  # refuses invalid definitions and bad parameters
        try:
            execution_id = self.executions.submit(definition, spec.merged_params(definition, params))
        except QueueFull as error:
            raise ApiError(HTTPStatus.TOO_MANY_REQUESTS, str(error)) from None
        if self.executor:
            self.executor.wake.set()
        return {"id": execution_id}

    def execution(self, execution_id):
        self.require_store()
        item = self.executions.get(execution_id)
        if item is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"execution {execution_id} not found")
        return item

    def list_executions(self, query):
        self.require_store()
        state = query.get("state", [None])[0]
        if state and state not in STATES:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"'state' must be one of {', '.join(STATES)}")
        try:
            limit = min(max(int(query.get("limit", ["50"])[0]), 1), 500)
        except ValueError:
            raise ApiError(HTTPStatus.BAD_REQUEST, "'limit' must be an integer") from None
        return [summary(item) for item in self.executions.list(limit, state)]

    def cancel(self, execution_id):
        self.require_store()
        if not self.executions.cancel(execution_id):
            raise ApiError(HTTPStatus.NOT_FOUND, f"execution {execution_id} not found")
        return {"cancelled": True}

    def stream(self, handler, execution_id):
        first = self.execution(execution_id)
        stream_updates(handler, "execution", first, lambda: self.executions.get(execution_id),
                       lambda item: item["state"] in FINISHED, self.stopping)


def summary(item):
    """The list view: everything but the definition and the step outputs."""
    shown = {key: item[key] for key in ("id", "name", "state", "error", "created", "started", "finished", "updated")}
    return dict(shown, steps=[{"kind": s["kind"], "state": s["state"]} for s in item["steps"]])
