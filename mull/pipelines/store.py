"""Executions of job definitions, kept in the job database (tables exist only when --pipelines is on)."""

import json
import time

from ..server.store import FINISHED, QueueFull

_SCHEMA = """
CREATE TABLE IF NOT EXISTS executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    definition TEXT NOT NULL,
    params TEXT NOT NULL DEFAULT '{}',
    state TEXT NOT NULL DEFAULT 'queued',
    steps TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    started REAL,
    finished REAL,
    updated REAL NOT NULL DEFAULT 0
)
"""


def _execution(row):
    if row is None:
        return None
    item = dict(row)
    for key in ("definition", "params", "steps"):
        item[key] = json.loads(item[key])
    item.pop("cancel_requested")
    return item


def blank_steps(definition):
    return [{"kind": next(iter(step)), "state": "pending", "output": "", "job": None, "error": None}
            for step in definition["steps"]]


class ExecutionStore:
    """Shares the job store's connection and lock, so one SQLite file holds everything."""

    def __init__(self, jobs, max_queued=20):
        self.jobs = jobs
        self.max_queued = max_queued
        self._run(_SCHEMA)

    def _run(self, sql, args=()):
        return self.jobs._run(sql, args)

    def submit(self, definition, params):
        with self.jobs.lock:
            queued = self.jobs.db.execute("SELECT COUNT(*) FROM executions WHERE state='queued'").fetchone()[0]
            if queued >= self.max_queued:
                raise QueueFull(f"{queued} executions already queued (limit {self.max_queued})")
            now = time.time()
            cursor = self.jobs.db.execute(
                "INSERT INTO executions (name, definition, params, steps, created, updated) VALUES (?, ?, ?, ?, ?, ?)",
                (definition["job"], json.dumps(definition), json.dumps(params), json.dumps(blank_steps(definition)),
                 now, now))
            return cursor.lastrowid

    def claim_next(self):
        with self.jobs.lock:
            row = self.jobs.db.execute("SELECT * FROM executions WHERE state='queued' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return None
            now = time.time()
            self.jobs.db.execute("UPDATE executions SET state='running', started=?, updated=? WHERE id=?",
                                 (now, now, row["id"]))
            return dict(_execution(row), state="running")

    def save_steps(self, execution_id, steps):
        self._run("UPDATE executions SET steps=?, updated=? WHERE id=?", (json.dumps(steps), time.time(), execution_id))

    def finish(self, execution_id, state, error=None):
        now = time.time()
        self._run("UPDATE executions SET state=?, error=?, finished=?, updated=? WHERE id=?",
                  (state, error, now, now, execution_id))

    def cancel(self, execution_id):
        """Cancel a queued execution at once; ask a running one to stop. False if unknown."""
        with self.jobs.lock:
            row = self.jobs.db.execute("SELECT state FROM executions WHERE id=?", (execution_id,)).fetchone()
            if row is None:
                return False
            now = time.time()
            if row["state"] == "queued":
                self.jobs.db.execute("UPDATE executions SET state='cancelled', finished=?, updated=? WHERE id=?",
                                     (now, now, execution_id))
            elif row["state"] == "running":
                self.jobs.db.execute("UPDATE executions SET cancel_requested=1, updated=? WHERE id=?",
                                     (now, execution_id))
            return True

    def cancel_requested(self, execution_id):
        row = self._run("SELECT cancel_requested FROM executions WHERE id=?", (execution_id,)).fetchone()
        return bool(row and row[0])

    def delete(self, execution_id):
        """True if deleted, False if still active, None if unknown."""
        with self.jobs.lock:
            row = self.jobs.db.execute("SELECT state FROM executions WHERE id=?", (execution_id,)).fetchone()
            if row is None:
                return None
            if row["state"] not in FINISHED:
                return False
            self.jobs.db.execute("DELETE FROM executions WHERE id=?", (execution_id,))
            return True

    def get(self, execution_id):
        return _execution(self._run("SELECT * FROM executions WHERE id=?", (execution_id,)).fetchone())

    def list(self, limit=50, state=None):
        query, args = "SELECT * FROM executions", []
        if state:
            query, args = query + " WHERE state=?", [state]
        rows = self._run(query + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        return [_execution(row) for row in rows]

    def interrupted(self):
        """Executions left running by a stopped server. Marked failed; returns their child job ids to cancel."""
        jobs = []
        for item in [e for e in self.list(500, "running")]:
            jobs += [s["job"] for s in item["steps"] if s["job"] and s["state"] == "running"]
            self.finish(item["id"], "failed", "interrupted by a server restart")
        return jobs
