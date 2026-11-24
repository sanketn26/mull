"""The single worker that owns the model. One job at a time for now; batching comes next."""

import threading
import traceback

PROGRESS_EVERY = 8  # tokens between progress writes to the database
STOP_WINDOW = 64    # trailing tokens decoded to look for stop sequences


class Cancelled(Exception):
    pass


class Interrupted(Exception):
    """The server is shutting down: leave the job running so the next start queues it again."""


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


class PackedModelRunner:
    """Adapts a packed MLX model to the worker: streams token IDs and stops when asked."""

    def __init__(self, model, name=None):
        self.model = model
        self.name = name
        self.tokenizer = model.tokenizer

    def encode(self, prompt, system=None, thinking=False):
        tokenizer = self.tokenizer
        if not getattr(tokenizer, "chat_template", None):
            return tokenizer.encode(prompt if not system else f"{system}\n\n{prompt}")
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=thinking)
        return tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def describe(self):
        model = self.model
        share = "auto" if getattr(model, "adaptive", False) else getattr(model, "memory_percent", None)
        return {"model": self.name, "memory_percent": share}

    def tokens(self, ids, max_new_tokens, temperature=0.0):
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
                    token = sample(model.last_logits(x, cache), temperature)
                    token_id = token.item()
                    if token_id in model.eos:
                        return
                    yield token_id
                    x = token.reshape(1, 1)
            finally:
                model.finish_request()


class LlamaMlxRunner(PackedModelRunner):
    """Adapts the layer-split Llama/Mistral MLX runtime (e.g. TinyLlama) to the worker."""

    def __init__(self, model, name=None):
        super().__init__(model, name)
        self.lock = threading.Lock()

    def tokens(self, ids, max_new_tokens, temperature=0.0):
        import mlx.core as mx

        eos = getattr(self.tokenizer, "eos_token_id", None)
        with self.lock:
            stream = self.model.model_generate(mx.array([ids], dtype=mx.int32), temperature=temperature)
            try:
                for _ in range(max_new_tokens):
                    token_id = next(stream).item()
                    if token_id == eos:
                        return
                    yield token_id
            finally:
                stream.close()


def runner_for(model, name=None):
    """Pick the adapter for a loaded model, or None if the job server can't drive it."""
    if hasattr(model, "request_lock") and hasattr(model, "plan_request"):
        return PackedModelRunner(model, name)
    if hasattr(model, "model_generate") and hasattr(model, "tokenizer"):
        return LlamaMlxRunner(model, name)
    return None


class Worker:
    """Takes jobs from the store, runs them on the runner and writes output as it is produced."""

    def __init__(self, store, runner, poll_seconds=0.5):
        self.store = store
        self.runner = runner
        self.poll_seconds = poll_seconds
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.thread = None
        self.current = None

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
        self.current = job["id"]
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
        finally:
            self.current = None
        return True

    def generate(self, job):
        options, generated = job["options"], []
        ids = self.runner.encode(job["prompt"], options.get("system"), options.get("thinking", False))
        stream = self.runner.tokens(ids, job["max_new_tokens"], options.get("temperature", 0.0))
        stops = options.get("stop") or []
        try:
            for token_id in stream:
                generated.append(token_id)
                if stops and cut_at_stop(self.runner.decode(generated[-STOP_WINDOW:]), stops) is not None:
                    break
                if len(generated) % PROGRESS_EVERY == 0:
                    self.save(job["id"], generated, stops)
                if self.stopping.is_set():
                    raise Interrupted()
                if self.store.cancel_requested(job["id"]):
                    raise Cancelled()
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
            self.save(job["id"], generated, stops)

    def save(self, job_id, generated, stops):
        output, reasoning = split_reasoning(self.runner.decode(generated))
        cut = cut_at_stop(output, stops) if stops else None
        self.store.progress(job_id, output if cut is None else cut, len(generated), reasoning)
