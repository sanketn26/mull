"""Sliding prefetch window: keep the next few streamed layers loading while the current one runs.

Shared by the PyTorch and MLX runtimes. The runtime says which layers stream (in execution order),
how to load one, and how big each is; the window does the rest:

* as each layer is taken, the next ``size`` layers after it are loading on worker threads;
* the order wraps round, so the next token's first layers load while the last ones run;
* loads that are no longer upcoming (generation stopped early) are dropped, so at most ``size``
  layers are ever held ahead;
* a read-ahead only starts if it keeps free RAM above the safety floor
  (memory.has_room_to_prefetch); otherwise that layer is read when it is needed.
"""
from concurrent.futures import ThreadPoolExecutor

from .memory import has_room_to_prefetch

LOW_RAM_MESSAGE = ('sliding adaptive processing: free RAM is low, so layers are read when needed '
                   'instead of ahead of time until memory frees up.')


class PrefetchWindow:
    """Prefetch the next ``size`` keys of a cyclic ``order``.

    Parameters
    ----------
    order : list
        streamed layer keys, in the order they run
    load : callable(key) -> value
        loads one layer; runs on a worker thread
    size : int
        how many layers to keep loading ahead (0 disables prefetching)
    size_of : callable(key) -> int, optional
        bytes a layer will occupy once loaded, for the free-RAM check
    load_now : callable(key) -> value, optional
        loader for a layer that was not prefetched, run on the caller's thread (default: load)
    executor : concurrent.futures.Executor, optional
        thread pool to use; one with ``size`` workers is created if omitted
    """

    def __init__(self, order, load, size, size_of=None, load_now=None, executor=None):
        self.order = list(order)
        self._pos = {key: i for i, key in enumerate(self.order)}
        self._load = load
        self._load_now = load_now or load
        self._size_of = size_of or (lambda key: 0)
        self.size = size if len(self.order) > 1 else 0
        self._executor = executor
        if self._executor is None and self.size:
            self._executor = ThreadPoolExecutor(max_workers=self.size)
        self._inflight = {}
        self.low_ram_events = 0

    def take(self, key):
        """Return ``key``'s loaded value, and top the window up behind it."""
        future = self._inflight.pop(key, None)
        if self.size and key in self._pos:
            self._fill_after(key)   # start the next reads before waiting on this one
        return future.result() if future is not None else self._load_now(key)

    def upcoming(self, key):
        """The ``size`` keys after ``key``, wrapping round into the next pass."""
        n = len(self.order)
        if n < 2 or not self.size:
            return []
        pos = self._pos[key]
        return [self.order[(pos + k) % n] for k in range(1, min(self.size, n - 1) + 1)]

    @property
    def inflight(self):
        return list(self._inflight)

    def _fill_after(self, key):
        upcoming = self.upcoming(key)
        keep = set(upcoming)
        for stale in [k for k in self._inflight if k not in keep]:
            self._inflight.pop(stale).cancel()
        for k in upcoming:   # nearest first: the executor runs them in submission order
            if k in self._inflight:
                continue
            if not has_room_to_prefetch(self._size_of(k)):
                if not self.low_ram_events:
                    print(LOW_RAM_MESSAGE)
                self.low_ram_events += 1
                break
            self._inflight[k] = self._executor.submit(self._load, k)
