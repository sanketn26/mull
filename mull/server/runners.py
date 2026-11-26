"""Adapters between the worker and Mull's model runtimes."""

import threading

from .batch import PackedBatch, SequentialBatch


class PackedModelRunner:
    """Adapts a packed MLX model (Qwen3, Qwen3.5, gpt-oss, Qwen3 MoE): batches, prefix cache."""

    def __init__(self, model, name=None, prefix_cache=None):
        self.model = model
        self.name = name
        self.tokenizer = model.tokenizer
        self.prefix_cache = prefix_cache

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

    @property
    def loaded(self):
        """Whether anything is held in RAM: resident weights, the prefix cache or MLX's buffer cache."""
        import mlx.core as mx

        held = self.prefix_cache.nbytes if self.prefix_cache is not None else 0
        return bool(self.model.resident) or held > 0 or mx.get_cache_memory() > 0

    def unload(self):
        """Free the weights and the prefix cache; the next job reloads what it needs."""
        if self.prefix_cache is not None:
            self.prefix_cache.clear()
        self.model.release_weights()

    def describe(self):
        import mlx.core as mx

        model = self.model
        share = "auto" if getattr(model, "adaptive", False) else getattr(model, "memory_percent", None)
        info = {"model": self.name, "memory_percent": share, "loaded": self.loaded,
                "model_memory_gb": round(mx.get_active_memory() / 2**30, 2)}
        if share == "auto":
            info["keep_free_percent"] = round(model.keep_free_fraction * 100, 1)
        if self.prefix_cache is not None:
            info["prefix_cache"] = self.prefix_cache.stats()
        return info

    def make_batch(self, max_batch):
        return PackedBatch(self, max_batch)

    def prefix_reserve(self, prompt_length):
        """Room the prefix cache needs: what it holds plus the copy of this prompt it may add."""
        if self.prefix_cache is None:
            return 0
        prompt_bytes = self.model.decoder.request_memory(prompt_length, 1)[0]
        return min(self.prefix_cache.nbytes + prompt_bytes, self.prefix_cache.limit_bytes)

    def prefill(self, ids):
        """Process the prompt (reusing a cached prefix) and return (cache, logits of its last token)."""
        import mlx.core as mx

        model = self.model
        cache, reused = self.prefix_cache.lookup(ids) if self.prefix_cache else (None, 0)
        if cache is None:
            cache = model.decoder.make_cache()
        logits = model.last_logits(mx.array([ids[reused:]], dtype=mx.int32), cache)
        if self.prefix_cache is not None:
            self.prefix_cache.store(ids, cache)
        return cache, logits

    def pick(self, logits, temperature):
        """Sample one row of logits; None at end of sequence."""
        from ..mlx_llama import sample

        token_id = sample(logits, temperature).item()
        return None if token_id in self.model.eos else token_id

    def tokens(self, ids, max_new_tokens, temperature=0.0):
        """Yield generated token IDs for one prompt (a batch of one)."""
        batch = self.make_batch(1)
        try:
            token = batch.add(ids, max_new_tokens, temperature)
            for count in range(max_new_tokens):
                if token is None:
                    return
                yield token
                if count + 1 < max_new_tokens:
                    token = batch.step()[0]
        finally:
            batch.remove([0] if batch.size else [])


class LlamaMlxRunner(PackedModelRunner):
    """Adapts the layer-split Llama/Mistral MLX runtime (e.g. TinyLlama): one job at a time."""

    def __init__(self, model, name=None):
        super().__init__(model, name)
        self.lock = threading.Lock()

    @property
    def loaded(self):
        import mlx.core as mx

        return bool(self.model._resident) or mx.get_cache_memory() > 0

    def unload(self):
        self.model.release_weights()

    def make_batch(self, max_batch):
        return SequentialBatch(self)

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


def runner_for(model, name=None, prefix_cache=None):
    """Pick the adapter for a loaded model, or None if the job server can't drive it.

    The prefix cache is used with packed models; the Llama runtime keeps its own cache format."""
    if hasattr(model, "request_lock") and hasattr(model, "plan_request"):
        return PackedModelRunner(model, name, prefix_cache)
    if hasattr(model, "model_generate") and hasattr(model, "tokenizer"):
        return LlamaMlxRunner(model, name)
    return None
