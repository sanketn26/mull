"""Load, check and dry-run a job definition. No model, no disk writes, no network.

A definition is a mapping (JSON, or YAML when PyYAML is installed):

    job: weekly-notes
    params: {topic: speculative decoding}
    reads: [~/notes]
    writes: [~/Briefs]
    limits: {time: 30m}
    steps:
      - read:   {path: ~/notes/week.md}
      - prompt: {text: "Summarise about {params.topic}: {steps.1}", max_new_tokens: 300}
      - save:   {path: "~/Briefs/{date}.md", content: "{steps.2}"}
"""

import datetime
import json
import os
import re
from dataclasses import dataclass

TOP_KEYS = {"job", "params", "reads", "writes", "limits", "steps"}
# step type -> (required keys, optional keys)
STEPS = {
    "read": ({"path"}, set()),
    "prompt": ({"text"}, {"max_new_tokens"}),
    "save": ({"path", "content"}, set()),
}
LIMIT_KEYS = {"time", "max_new_tokens"}
NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
DURATION = re.compile(r"^(\d+)\s*([smh])$")
BRACES = re.compile(r"\{([^{}]*)\}")
REF = re.compile(r"^(steps\.(\d+)|params\.([A-Za-z_]\w*)|date)$")
UNITS = {"s": 1, "m": 60, "h": 3600}


PARAM_KEYS = {"type", "label", "default"}
PARAM_TYPES = {"text": str, "number": (int, float), "bool": bool}
PARAM_NAME = re.compile(r"^[A-Za-z_]\w*$")


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class Policy:
    """What this server allows, whatever a definition asks for. No roots means no file access."""

    roots: tuple = ()
    max_steps: int = 20
    max_seconds: int = 4 * 3600
    max_new_tokens: int = 4096


def parse_duration(text):
    match = DURATION.match(str(text).strip())
    if not match:
        raise SpecError(f"time {text!r} must look like 30s, 90m or 2h")
    return int(match.group(1)) * UNITS[match.group(2)]


def load_definition(text):
    """Parse JSON, or YAML if PyYAML is installed. PyYAML is only imported here, only for YAML."""
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return _json(text)
    try:
        import yaml
    except ImportError:
        raise SpecError("this looks like YAML, and PyYAML is not installed (pip install 'mull[jobs]'); JSON works without it")
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise SpecError(f"cannot parse the definition: {error}")


def _json(text):
    try:
        return json.loads(text)
    except ValueError as error:
        raise SpecError(f"cannot parse the definition: {error}")


def resolve(path):
    return os.path.realpath(os.path.expanduser(path))


def inside(path, roots):
    """True when path, after resolving symlinks and .., lies in or under one of roots."""
    target = resolve(path)
    for root in roots:
        base = resolve(root)
        if target == base or target.startswith(base.rstrip(os.sep) + os.sep):
            return True
    return False


def static_prefix(text):
    return text.split("{", 1)[0]


def refs_in(text):
    return BRACES.findall(text)


class Checker:
    """Collects problems. Errors stop a job from starting; warnings are advice."""

    def __init__(self, definition, policy):
        self.definition = definition
        self.policy = policy
        self.issues = []

    def error(self, text):
        self.issues.append({"level": "error", "text": text})

    def warn(self, text):
        self.issues.append({"level": "warning", "text": text})

    def run(self):
        if not isinstance(self.definition, dict):
            self.error("the definition must be a mapping with job, limits and steps")
            return self.issues
        for key in sorted(set(self.definition) - TOP_KEYS):
            self.error(f"unknown key {key!r}; allowed: {', '.join(sorted(TOP_KEYS))}")
        self.check_name()
        self.check_limits()
        self.check_params()
        self.check_paths("reads")
        self.check_paths("writes")
        self.check_steps()
        return self.issues

    def check_name(self):
        name = self.definition.get("job")
        if not isinstance(name, str) or not NAME.match(name):
            self.error("job needs a name of lowercase letters, digits and dashes")

    def check_limits(self):
        limits = self.definition.get("limits")
        if not isinstance(limits, dict) or "time" not in limits:
            self.error("limits.time is required; every job needs a time limit")
            return
        for key in sorted(set(limits) - LIMIT_KEYS):
            self.error(f"unknown limit {key!r}; allowed: {', '.join(sorted(LIMIT_KEYS))}")
        try:
            if parse_duration(limits["time"]) > self.policy.max_seconds:
                self.error(f"limits.time is above what this server allows ({self.policy.max_seconds // 60} minutes)")
        except SpecError as error:
            self.error(str(error))
        self.check_token_count(limits.get("max_new_tokens"), "limits.max_new_tokens")

    def check_token_count(self, value, where):
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= self.policy.max_new_tokens:
            self.error(f"{where} must be a whole number from 1 to {self.policy.max_new_tokens}")

    def check_params(self):
        params = self.definition.get("params", {})
        if not isinstance(params, dict):
            self.error("params must be a mapping of name to default, or to {type, label, default}")
            return
        for name, value in params.items():
            if not PARAM_NAME.match(str(name)):
                self.error(f"parameter name {name!r} must be letters, digits and underscores")
            elif isinstance(value, dict):
                self.check_param_detail(name, value)

    def check_param_detail(self, name, value):
        for key in sorted(set(value) - PARAM_KEYS):
            self.error(f"parameter {name}: unknown key {key!r}; allowed: {', '.join(sorted(PARAM_KEYS))}")
        kind = value.get("type", "text")
        if kind not in PARAM_TYPES:
            self.error(f"parameter {name}: type must be one of {', '.join(PARAM_TYPES)}")
        elif "default" in value and not fits(kind, value["default"]):
            self.error(f"parameter {name}: default is not a {kind}")

    def check_paths(self, key):
        value = self.definition.get(key, [])
        if not isinstance(value, list) or not all(isinstance(p, str) for p in value):
            self.error(f"{key} must be a list of folders")
            return
        for folder in value:
            if not inside(folder, self.policy.roots):
                self.error(f"{key}: {folder} is outside the folders this server allows (--pipeline-roots)")

    def steps(self):
        steps = self.definition.get("steps")
        return steps if isinstance(steps, list) else []

    def check_steps(self):
        if not self.steps():
            self.error("steps must be a non-empty list")
            return
        if len(self.steps()) > self.policy.max_steps:
            self.error(f"at most {self.policy.max_steps} steps are allowed")
        for index, step in enumerate(self.steps(), start=1):
            self.check_step(index, step)

    def check_step(self, index, step):
        if not isinstance(step, dict) or len(step) != 1:
            self.error(f"step {index}: write one step type per list item, such as - read: {{path: ...}}")
            return
        (kind, args), = step.items()
        if kind not in STEPS:
            self.error(f"step {index}: unknown step type {kind!r}; allowed: {', '.join(sorted(STEPS))}")
        elif not isinstance(args, dict):
            self.error(f"step {index} ({kind}): arguments must be a mapping")
        else:
            self.check_args(index, kind, args)

    def check_args(self, index, kind, args):
        required, optional = STEPS[kind]
        for key in sorted(required - set(args)):
            self.error(f"step {index} ({kind}): {key} is required")
        for key in sorted(set(args) - required - optional):
            self.error(f"step {index} ({kind}): unknown argument {key!r}")
        for key, value in args.items():
            if isinstance(value, str):
                self.check_refs(index, value)
        self.check_step_path(index, kind, args)
        if kind == "prompt":
            self.check_token_count(args.get("max_new_tokens"), f"step {index}: max_new_tokens")

    def check_refs(self, index, text):
        declared = self.definition.get("params") or {}
        for ref in refs_in(text):
            match = REF.match(ref)
            if not match:
                self.error(f"step {index}: {{{ref}}} is not a known reference (use steps.N, params.name or date)")
            elif match.group(2) and not 1 <= int(match.group(2)) < index:
                self.error(f"step {index}: {{{ref}}} must point to an earlier step")
            elif match.group(3) and match.group(3) not in declared:
                self.error(f"step {index}: {{{ref}}} is not declared under params")

    def check_step_path(self, index, kind, args):
        roots_key = {"read": "reads", "save": "writes"}.get(kind)
        path = args.get("path")
        if roots_key is None or not isinstance(path, str):
            return
        roots = self.definition.get(roots_key)
        roots = roots if isinstance(roots, list) else []
        prefix = static_prefix(path)
        if not prefix or not inside(prefix, roots):
            self.error(f"step {index}: {path} is not inside {roots_key}; add its folder there")


def check(definition, policy=None):
    """Return a list of {level, text}. A definition may start only when no item is an error."""
    return Checker(definition, policy or Policy()).run()


def fits(kind, value):
    if kind == "number" and isinstance(value, bool):
        return False
    return isinstance(value, PARAM_TYPES[kind])


def params_info(definition):
    """Each parameter as {name, type, label, default}; default is None when the caller must supply it."""
    info = []
    for name, value in (definition.get("params") or {}).items():
        detail = value if isinstance(value, dict) else {"default": value}
        kind = detail.get("type") or next((k for k in ("bool", "number", "text") if fits(k, detail.get("default"))), "text")
        info.append({"name": name, "type": kind, "label": detail.get("label", name), "default": detail.get("default")})
    return info


def ok(issues):
    return not any(issue["level"] == "error" for issue in issues)


def expand(text, params, results, today):
    """Fill {params.x}, {steps.N} and {date}. Anything else was refused by check()."""
    def fill(match):
        ref = match.group(1)
        if ref == "date":
            return today
        kind, _, name = ref.partition(".")
        return str(params[name] if kind == "params" else results[int(name)])
    return BRACES.sub(fill, text)


def merged_params(definition, given):
    """Defaults, overridden by what the caller gave. Unknown, missing and wrongly typed values are refused."""
    given = given or {}
    info = {item["name"]: item for item in params_info(definition)}
    unknown = set(given) - set(info)
    if unknown:
        raise SpecError(f"unknown parameter: {', '.join(sorted(unknown))}")
    params = {name: item["default"] for name, item in info.items()}
    params.update(given)
    for name, item in info.items():
        if params[name] is None:
            raise SpecError(f"parameter {name} needs a value")
        if not fits(item["type"], params[name]):
            raise SpecError(f"parameter {name} must be a {item['type']}")
    return params


def dry_run(definition, given=None, today=None, policy=None):
    """List what a run would read, write and ask the model. Touches nothing."""
    issues = check(definition, policy)
    if not ok(issues):
        raise SpecError("; ".join(i["text"] for i in issues if i["level"] == "error"))
    params = merged_params(definition, given)
    today = today or datetime.date.today().isoformat()
    lines = []
    for index, step in enumerate(definition["steps"], start=1):
        (kind, args), = step.items()
        lines.append(describe(index, kind, args, params, today))
    return lines


def describe(index, kind, args, params, today):
    shown = {k: (v if k in ("max_new_tokens",) or not isinstance(v, str) else expand_shallow(v, params, today)) for k, v in args.items()}
    if kind == "read":
        return f"{index}. read {shown['path']}"
    if kind == "save":
        return f"{index}. write {shown['path']}"
    return f"{index}. ask the model (up to {shown.get('max_new_tokens', 'default')} tokens)"


def expand_shallow(text, params, today):
    """Expand params and date, but leave {steps.N} as it is: their values do not exist yet."""
    def fill(match):
        ref = match.group(1)
        if ref == "date":
            return today
        if ref.startswith("params."):
            return str(params[ref[7:]])
        return match.group(0)
    return BRACES.sub(fill, text)
