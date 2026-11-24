"""The single worker that owns the model. One job at a time for now; batching comes next."""

import threading
import traceback

PROGRESS_EVERY = 8  # tokens between progress writes to the database


class Cancelled(Exception):
    pass


class Interrupted(Exception):
    """The server is shutting down: leave the job running so the next start queues it again."""


class PackedModelRunner:
    """Adapts a packed MLX model to the worker: streams token IDs and stops when asked."""

    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer

    def encode(self, prompt):
        tokenizer = self.tokenizer
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                   tokenize=False, add_generation_prompt=True)
        return tokenizer.encode(prompt)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def tokens(self, ids, max_new_tokens):
        """Yield generated token IDs. Raises MemoryError if the request can never fit in RAM."""
        import mlx.core as mx

        from ..mlx_llama import sample

        model = self.model
        x = mx.array([ids], dtype=mx.int32)
        with model.request_lock:
            model.plan_request(x.shape[1], max_new_tokens)
            cache = model.decoder.make_cache()
            try:
                for _ in range(max_new_tokens):
                    token = sample(model.last_logits(x, cache), 0)
                    token_id = token.item()
                    if token_id in model.eos:
                        return
                    yield token_id
                    x = token.reshape(1, 1)
            finally:
                model.finish_request()


class Worker:
    """Takes jobs from the store, runs them on the runner and writes output as it is produced."""

    def __init__(self, store, runner, poll_seconds=0.5):
        self.store = store
        self.runner = runner
        self.poll_seconds = poll_seconds
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.thread = None

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
            if not self.run_one():
                self.wake.wait(self.poll_seconds)
                self.wake.clear()

    def run_one(self):
        """Run the next queued job to completion. Returns False if there was nothing to do."""
        job = self.store.claim_next()
        if job is None:
            return False
        try:
            self.generate(job)
            self.store.finish(job["id"], "done")
        except Cancelled:
            self.store.finish(job["id"], "cancelled")
        except Interrupted:
            pass
        except MemoryError as error:
            self.store.finish(job["id"], "failed", f"Not enough free memory: {error}")
        except Exception as error:  # a bad job must not take the worker down
            traceback.print_exc()
            self.store.finish(job["id"], "failed", f"{type(error).__name__}: {error}")
        return True

    def generate(self, job):
        job_id, generated = job["id"], []
        ids = self.runner.encode(job["prompt"])
        stream = self.runner.tokens(ids, job["max_new_tokens"])
        try:
            for token_id in stream:
                generated.append(token_id)
                if len(generated) % PROGRESS_EVERY == 0:
                    self.store.progress(job_id, self.runner.decode(generated), len(generated))
                if self.stopping.is_set():
                    raise Interrupted()
                if self.store.cancel_requested(job_id):
                    raise Cancelled()
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
            self.store.progress(job_id, self.runner.decode(generated), len(generated))
