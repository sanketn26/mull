"""Durable job queue in SQLite: jobs survive restarts and the API reads progress from it."""

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
    state TEXT NOT NULL DEFAULT 'queued',
    output TEXT NOT NULL DEFAULT '',
    tokens INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    started REAL,
    finished REAL
)
"""


class QueueFull(Exception):
    pass


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

    def submit(self, prompt, max_new_tokens):
        with self.lock:
            queued = self.db.execute("SELECT COUNT(*) FROM jobs WHERE state='queued'").fetchone()[0]
            if queued >= self.max_queued:
                raise QueueFull(f"{queued} jobs already queued (limit {self.max_queued})")
            cursor = self.db.execute("INSERT INTO jobs (prompt, max_new_tokens, created) VALUES (?, ?, ?)",
                                     (prompt, max_new_tokens, time.time()))
            return cursor.lastrowid

    def claim_next(self):
        """Mark the oldest queued job as running and return it, or None if the queue is empty."""
        with self.lock:
            row = self.db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return None
            self.db.execute("UPDATE jobs SET state='running', started=? WHERE id=?", (time.time(), row["id"]))
            return dict(row, state="running")

    def progress(self, job_id, output, tokens):
        self._run("UPDATE jobs SET output=?, tokens=? WHERE id=?", (output, tokens, job_id))

    def finish(self, job_id, state, error=None):
        assert state in FINISHED, state
        self._run("UPDATE jobs SET state=?, error=?, finished=? WHERE id=?", (state, error, time.time(), job_id))

    def cancel(self, job_id):
        """Cancel a queued job at once; ask a running one to stop at its next token. False if unknown."""
        with self.lock:
            row = self.db.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return False
            if row["state"] == "queued":
                self.db.execute("UPDATE jobs SET state='cancelled', finished=? WHERE id=?", (time.time(), job_id))
            elif row["state"] == "running":
                self.db.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,))
            return True

    def cancel_requested(self, job_id):
        row = self._run("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
        return bool(row and row[0])

    def requeue_interrupted(self):
        """After a restart, jobs left running never finished: queue them again from the start."""
        cursor = self._run("UPDATE jobs SET state='queued', output='', tokens=0, started=NULL WHERE state='running'")
        return cursor.rowcount

    def get(self, job_id):
        row = self._run("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list(self, limit=50):
        rows = self._run("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]
