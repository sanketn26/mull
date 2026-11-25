"""Batches: several jobs decoded together, so one pass over the weights serves every row.

Both classes share one interface used by the worker:
    add(ids, max_new_tokens, temperature) -> first token id, or None if the sequence ended at once
    step() -> one token id (or None at end of sequence) per row, in row order
    remove(rows) -> drop finished rows
    size, max_batch
"""


class NoRoom(Exception):
    """The job doesn't fit next to the rows already running; queue it again until one finishes."""


class SequentialBatch:
    """A batch of one, for runners that can only stream one sequence (e.g. the Llama runtime)."""

    max_batch = 1

    def __init__(self, runner):
        self.runner = runner
        self.stream = None

    @property
    def size(self):
        return 0 if self.stream is None else 1

    def add(self, ids, max_new_tokens, temperature):
        if self.stream is not None:
            raise NoRoom("one sequence at a time")
        self.stream = self.runner.tokens(ids, max_new_tokens, temperature)
        try:
            return self.step()[0]
        except BaseException:
            self.stream = None
            raise

    def step(self):
        return [next(self.stream, None)]

    def remove(self, rows):
        if rows and self.stream is not None:
            self.stream.close()
            self.stream = None


class _Row:
    __slots__ = ("context", "temperature", "token")

    def __init__(self, context, temperature, token):
        self.context, self.temperature, self.token = context, temperature, token


class PackedBatch:
    """Up to max_batch sequences on a packed MLX model. Prompts are processed one at a time
    (using the prefix cache), then merged into one batch cache; every decode step runs all rows."""

    def __init__(self, runner, max_batch=4):
        self.runner = runner
        self.model = runner.model
        self.max_batch = max_batch
        self.rows = []
        self.cache = None

    @property
    def size(self):
        return len(self.rows)

    def plan(self, ids, max_new_tokens):
        """Re-plan memory for the rows already running plus this one; NoRoom if it can't fit."""
        model = self.model
        context = len(ids) + max_new_tokens
        longest = max([row.context for row in self.rows] + [context])
        # Merged rows are left-padded to the longest context, the new one included: every existing row
        # needs a padded cache, and the new row needs padding beyond its own cache (planned below).
        padded = model.decoder.request_memory(longest, 1)[0]
        own = model.decoder.request_memory(len(ids), max_new_tokens)[0]
        padding = max(padded - own, 0) if self.rows else 0
        model.reserved_bytes = padded * len(self.rows) + padding + self.runner.prefix_reserve(len(ids))
        model.batch_rows = len(self.rows) + 1  # decode steps route every row's token through the experts
        try:
            model.plan_request(len(ids), max_new_tokens)
        except MemoryError:
            if self.rows:
                raise NoRoom("not enough free memory next to the running jobs") from None
            raise

    def add(self, ids, max_new_tokens, temperature):
        if len(self.rows) >= self.max_batch:
            raise NoRoom(f"batch is full ({self.max_batch})")
        with self.model.request_lock:
            self.plan(ids, max_new_tokens)
            cache, logits = self.runner.prefill(ids)
            token = self.runner.pick(logits, temperature)
            merged = [type(layer).merge([layer]) for layer in cache]
            if self.cache is None:
                self.cache = merged
            else:
                for batch_layer, new_layer in zip(self.cache, merged):
                    batch_layer.extend(new_layer)
        self.rows.append(_Row(len(ids) + max_new_tokens, temperature, token))
        return token

    def step(self):
        import mlx.core as mx

        x = mx.array([[row.token if row.token is not None else 0] for row in self.rows], dtype=mx.int32)
        with self.model.request_lock:
            logits = self.model.last_logits(x, self.cache)
            for i, row in enumerate(self.rows):
                row.token = self.runner.pick(logits[i:i + 1], row.temperature)
        return [row.token for row in self.rows]

    def remove(self, rows):
        if not rows:
            return
        import mlx.core as mx

        keep = [i for i in range(len(self.rows)) if i not in set(rows)]
        self.rows = [self.rows[i] for i in keep]
        if not keep:
            self.cache = None
            self.model.batch_rows = 1
            self.model.finish_request()
            return
        indices = mx.array(keep)
        for layer in self.cache:
            layer.filter(indices)
