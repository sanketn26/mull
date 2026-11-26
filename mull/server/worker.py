"""The worker that owns the model: admits queued jobs into a batch and steps them together."""

import threading
import time
import traceback

from .batch import NoRoom, SequentialBatch
from .runners import LlamaMlxRunner, PackedModelRunner, runner_for  # noqa: F401  (re-exported)

PROGRESS_EVERY = 8  # tokens between progress writes to the database
STOP_WINDOW = 64    # trailing tokens decoded to look for stop sequences


def split_reasoning(text):
    """Split '<think>…</think>answer' into (answer, reasoning). An unfinished block is all reasoning."""
    stripped = text.lstrip()
    if not stripped.startswith("<think>"):
        return text, ""
    body = stripped[len("<think>"):]
    reasoning, closed, answer = body.partition("</think>")
    if not closed:
        return "", reasoning.strip()
    return answer.lstrip(), reasoning.strip()


def cut_at_stop(text, stops):
    """Return text up to the first stop sequence, or None if no stop sequence occurs."""
    found = [text.find(stop) for stop in stops if stop and stop in text]
    return text[:min(found)] if found else None


class Active:
    """A job in the batch: its tokens so far and, once decided, how it ends."""

    def __init__(self, job):
        self.job = job
        self.id = job["id"]
        self.limit = job["max_new_tokens"]
        self.stops = job["options"].get("stop") or []
        self.generated = []
        self.outcome = None  # "done", "cancelled" or ("failed", message) once finished


class Worker:
    """Takes jobs from the store, runs them in a batch and writes output as it is produced."""

    def __init__(self, store, runner, poll_seconds=0.5, max_batch=1, unload_after=None, clock=time.monotonic):
        self.store = store
        self.unload_after = unload_after  # idle seconds before the model is freed (None: keep it loaded)
        self.clock = clock
        self.idle_since = None
        self.runner = runner
        self.poll_seconds = poll_seconds
        self.batch = runner.make_batch(max_batch) if hasattr(runner, "make_batch") else SequentialBatch(runner)
        self.active = []
        self.full = False  # a job didn't fit next to the running ones; wait for one to finish
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.thread = None

    @property
    def current(self):
        return self.active[0].id if self.active else None

    @property
    def running(self):
        return [entry.id for entry in self.active]

    def start(self):
        self.store.requeue_interrupted()
        self.thread = threading.Thread(target=self.loop, name="mull-worker", daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping.set()
        self.wake.set()
        if self.thread:
            self.thread.join()

    def loop(self):
        while not self.stopping.is_set():
            if not self.tick():
                self.maybe_unload()
                self.wake.wait(self.poll_seconds)
                self.wake.clear()
        self.active.clear()  # left running in the store: the next start queues them again

    def tick(self):
        """Admit what fits, then advance every running job by one token. False if idle."""
        self.admit()
        if not self.active:
            return False
        self.advance()
        return True

    def maybe_unload(self):
        """Free the model once nothing has run for ``unload_after`` seconds. True if it was freed."""
        if self.active or not getattr(self.runner, "loaded", False):
            self.idle_since = None
            return False
        if self.unload_after is None:
            return False
        now = self.clock()
        if self.idle_since is None:
            self.idle_since = now
        if now - self.idle_since < self.unload_after:
            return False
        try:
            self.runner.unload()
        except Exception:  # noqa: BLE001  keep serving; the next job loads again
            traceback.print_exc()
            return False
        self.idle_since = None
        return True

    def run_one(self):
        """Run until the batch is empty (tests and simple callers). False if nothing was queued."""
        if not self.tick():
            return False
        while self.active and not self.stopping.is_set():
            self.advance()
        return True

    def admit(self):
        while not self.full and self.batch.size < self.batch.max_batch:
            job = self.store.claim_next()
            if job is None:
                return
            entry = Active(job)
            try:
                options = job["options"]
                ids = self.runner.encode(job["prompt"], options.get("system"), options.get("thinking", False))
                token = self.batch.add(ids, entry.limit, options.get("temperature", 0.0))
            except NoRoom:
                self.store.requeue(job["id"])
                self.full = True
                return
            except Exception as error:  # a bad job must not take the worker down
                self.fail(entry, error)
                continue
            self.active.append(entry)
            self.accept(entry, token)
        self.drop_finished()

    def advance(self):
        try:
            tokens = self.batch.step()
        except Exception as error:
            for entry in self.active:
                if entry.outcome is None:
                    self.fail(entry, error, record=False)
            tokens = [None] * len(self.active)
        for entry, token in zip(self.active, tokens):
            if entry.outcome is None:
                self.accept(entry, token)
        self.drop_finished()

    def accept(self, entry, token):
        """Take one generated token for a job and decide whether it has finished."""
        if token is None:
            entry.outcome = entry.outcome or "done"
            return
        entry.generated.append(token)
        count = len(entry.generated)
        if count == 1:
            self.store.first_token(entry.id)
        if entry.stops and cut_at_stop(self.runner.decode(entry.generated[-STOP_WINDOW:]), entry.stops) is not None:
            entry.outcome = "done"
        elif count >= entry.limit:
            entry.outcome = "done"
        elif self.store.cancel_requested(entry.id):
            entry.outcome = "cancelled"
        elif count % PROGRESS_EVERY == 0:
            self.save(entry)

    def drop_finished(self):
        rows = [i for i, entry in enumerate(self.active) if entry.outcome is not None]
        if not rows:
            return
        for i in rows:
            entry = self.active[i]
            self.save(entry)
            if isinstance(entry.outcome, tuple):
                self.store.finish(entry.id, "failed", entry.outcome[1])
            else:
                self.store.finish(entry.id, entry.outcome)
        self.batch.remove(rows)
        self.active = [entry for entry in self.active if entry.outcome is None]
        self.full = False

    def fail(self, entry, error, record=True):
        if isinstance(error, MemoryError):
            message = f"Not enough free memory: {error}"
        else:
            traceback.print_exc()
            message = f"{type(error).__name__}: {error}"
        entry.outcome = ("failed", message)
        if record:  # not in the batch yet: finish it here
            self.save(entry)
            self.store.finish(entry.id, "failed", message)

    def save(self, entry):
        output, reasoning = split_reasoning(self.runner.decode(entry.generated))
        cut = cut_at_stop(output, entry.stops) if entry.stops else None
        self.store.progress(entry.id, output if cut is None else cut, len(entry.generated), reasoning)
