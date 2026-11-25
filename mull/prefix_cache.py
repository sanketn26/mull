"""Prefix cache: reuse the processed prompt (KV cache) when a new prompt starts the same way.

Jobs that share a system prompt or the same files skip re-processing that part. Entries are
copies of the model cache taken right after prefill, kept in least-recently-used order under a
byte limit. Attention caches can be trimmed back to the shared part; state-space caches (Qwen3.5)
cannot, so for them only an earlier prompt that the new one fully extends is reused.
"""

from collections import OrderedDict

DEFAULT_LIMIT_BYTES = 1 << 30


def _copy_tree(value):
    import mlx.core as mx

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_tree(item) for item in value)
    return mx.array(value)


def snapshot(cache):
    """Independent copy of a list of mlx-lm cache objects."""
    import mlx.core as mx

    copied = [type(layer).from_state(_copy_tree(layer.state), layer.meta_state) for layer in cache]
    mx.eval([layer.state for layer in copied])
    return copied


def cache_bytes(cache):
    return sum(layer.nbytes for layer in cache)


def common_prefix(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


class PrefixCache:
    def __init__(self, limit_bytes=DEFAULT_LIMIT_BYTES):
        self.limit_bytes = limit_bytes
        self.entries = OrderedDict()  # tuple(token ids) -> (cache snapshot, nbytes)
        self.nbytes = 0
        self.hits = self.misses = self.tokens_reused = 0

    def usable(self, tokens, ids, cache):
        """Tokens of this entry the new prompt can reuse; at least one prompt token is always left
        to run, since generation needs its logits."""
        shared = common_prefix(tokens, ids)
        if shared == len(tokens) and len(tokens) < len(ids):
            return shared
        if all(layer.is_trimmable() for layer in cache):
            return min(shared, len(ids) - 1)
        return 0

    def lookup(self, ids):
        """Return (cache copy, tokens reused), or (None, 0) when nothing useful is stored."""
        best, best_len = None, 0
        for tokens, (cache, _size) in self.entries.items():
            length = self.usable(tokens, ids, cache)
            if length > best_len:
                best, best_len = tokens, length
        if best is None:
            self.misses += 1
            return None, 0
        self.entries.move_to_end(best)
        cache = snapshot(self.entries[best][0])
        excess = len(best) - best_len
        if excess:
            for layer in cache:
                layer.trim(excess)
        self.hits += 1
        self.tokens_reused += best_len
        return cache, best_len

    def store(self, ids, cache):
        """Keep a copy of the cache for prompt `ids`, if it fits under the limit."""
        key = tuple(ids)
        size = cache_bytes(cache)
        if size > self.limit_bytes or key in self.entries:
            return
        while self.entries and self.nbytes + size > self.limit_bytes:
            _tokens, (_cache, evicted) = self.entries.popitem(last=False)
            self.nbytes -= evicted
        self.entries[key] = (snapshot(cache), size)
        self.nbytes += size

    def clear(self):
        self.entries.clear()
        self.nbytes = 0

    def stats(self):
        return {"entries": len(self.entries), "bytes": self.nbytes, "limit_bytes": self.limit_bytes,
                "hits": self.hits, "misses": self.misses, "tokens_reused": self.tokens_reused}
