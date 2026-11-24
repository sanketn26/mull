"""Durable job queue in SQLite: jobs survive restarts and the API reads progress from it."""

import json
import sqlite3
import threading
import time

STATES = ("queued", "running", "done", "failed", "cancelled")
FINISHED = ("done", "failed", "cancelled")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    max_new_tokens INTEGER NOT NULL,
    options TEXT NOT NULL DEFAULT '{}',
    state TEXT NOT NULL DEFAULT 'queued',
    output TEXT NOT NULL DEFAULT '',
    reasoning TEXT NOT NULL DEFAULT '',
    tokens INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    started REAL,
    finished REAL,
    updated REAL NOT NULL DEFAULT 0
)
"""


class QueueFull(Exception):
    pass


def _job(row):
    if row is None:
        return None
    job = dict(row)
    job["options"] = json.loads(job["options"])
    job.pop("cancel_requested")
    return job


class JobStore:
    """Thread-safe job table. One connection, guarded by a lock; SQLite does the persistence."""

    def __init__(self, path, max_queued=100):
        self.max_queued = max_queued
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(_SCHEMA)

    def _run(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args)

    def submit(self, prompt, max_new_tokens, options=None):
        with self.lock:
            queued = self.db.execute("SELECT COUNT(*) FROM jobs WHERE state='queued'").fetchone()[0]
            if queued >= self.max_queued:
                raise QueueFull(f"{queued} jobs already queued (limit {self.max_queued})")
            now = time.time()
            cursor = self.db.execute(
                "INSERT INTO jobs (prompt, max_new_tokens, options, created, updated) VALUES (?, ?, ?, ?, ?)",
                (prompt, max_new_tokens, json.dumps(options or {}), now, now))
            return cursor.lastrowid

    def claim_next(self):
        """Mark the oldest queued job as running and return it, or None if the queue is empty."""
        with self.lock:
            row = self.db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return None
            now = time.time()
            self.db.execute("UPDATE jobs SET state='running', started=?, updated=? WHERE id=?", (now, now, row["id"]))
            return dict(_job(row), state="running")

    def progress(self, job_id, output, tokens, reasoning=""):
        self._run("UPDATE jobs SET output=?, reasoning=?, tokens=?, updated=? WHERE id=?",
                  (output, reasoning, tokens, time.time(), job_id))

    def finish(self, job_id, state, error=None):
        assert state in FINISHED, state
        now = time.time()
        self._run("UPDATE jobs SET state=?, error=?, finished=?, updated=? WHERE id=?", (state, error, now, now, job_id))

    def cancel(self, job_id):
        """Cancel a queued job at once; ask a running one to stop at its next token. False if unknown."""
        with self.lock:
            row = self.db.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return False
            now = time.time()
            if row["state"] == "queued":
                self.db.execute("UPDATE jobs SET state='cancelled', finished=?, updated=? WHERE id=?",
                                (now, now, job_id))
            elif row["state"] == "running":
                self.db.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,))
            return True

    def cancel_requested(self, job_id):
        row = self._run("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
        return bool(row and row[0])

    def delete(self, job_id):
        """Remove a finished job. Returns None if unknown, False if it is still queued or running."""
        with self.lock:
            row = self.db.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return None
            if row["state"] not in FINISHED:
                return False
            self.db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            return True

    def requeue_interrupted(self):
        """After a restart, jobs left running never finished: queue them again from the start."""
        cursor = self._run("UPDATE jobs SET state='queued', output='', reasoning='', tokens=0, started=NULL, "
                           "cancel_requested=0, updated=? WHERE state='running'", (time.time(),))
        return cursor.rowcount

    def get(self, job_id):
        return _job(self._run("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def list(self, limit=50, state=None):
        if state:
            rows = self._run("SELECT * FROM jobs WHERE state=? ORDER BY id DESC LIMIT ?", (state, limit)).fetchall()
        else:
            rows = self._run("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [_job(row) for row in rows]

    def counts(self):
        rows = self._run("SELECT state, COUNT(*) FROM jobs GROUP BY state").fetchall()
        return {state: 0 for state in STATES} | {row[0]: row[1] for row in rows}
