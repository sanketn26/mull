"""Sliding adaptive processing: decide residency and prefetch depth from layer sizes and a budget.

Pure arithmetic with no torch or I/O, so both backends (PyTorch and MLX) share it and it can be
unit-tested in isolation.
"""
from dataclasses import dataclass
from typing import List


# Sliding adaptive processing: share of free memory AirLLM may use for resident layers.
DEFAULT_MEMORY_PERCENT = 25


def validate_memory_percent(memory_percent):
    """Return the percentage as a float, or None when residency is off (None or 0)."""
    if memory_percent is None:
        return None
    if isinstance(memory_percent, bool) or not isinstance(memory_percent, (int, float)) \
            or not 0 <= memory_percent <= 100:
        raise ValueError(f"memory_percent must be a number in [0, 100], got {memory_percent!r}")
    return float(memory_percent) or None


def plan_resident_layers(layer_bytes, budget_bytes, reserve_bytes):
    """Sliding adaptive processing: choose which streamed modules to keep loaded permanently.

    Every module costs the same disk read per byte, so any choice that fills the budget saves the
    same bandwidth. Smallest-first is used because each resident module also removes a fixed
    per-load cost (file open, hook, device copy), so fitting more of them is strictly better.

    Parameters
    ----------
    layer_bytes : dict[int, int]
        streamed module index -> bytes it occupies once loaded
    budget_bytes : int
        total memory this model may use
    reserve_bytes : int
        part of the budget kept free for streaming and inference (layers in flight, KV cache)

    Returns
    -------
    list[int]
        indices to keep resident, in ascending index order
    """
    room = budget_bytes - reserve_bytes
    chosen = []
    for idx, size in sorted(layer_bytes.items(), key=lambda kv: (kv[1], kv[0])):
        if size > room:
            break
        chosen.append(idx)
        room -= size
    return sorted(chosen)


# Sliding adaptive processing: measured throughput, relative to one 64MB-read stream from the SSD.
#
# Window = reads kept in flight. Measured cold (data written and read with caching off) on an
# Apple-silicon SSD: 5.8GB/s with one read in flight, ~6.4GB/s with two or more, because one large
# sequential read nearly saturates the drive. So the window's main job is overlapping reading with
# compute (0 -> 1); beyond 2 it only costs memory.
DISK_READ_SPEEDUP_BY_WINDOW = {0: 0.9, 1: 1.0, 2: 1.1, 3: 1.1, 4: 1.1}


MAX_PREFETCH_WINDOW = max(DISK_READ_SPEEDUP_BY_WINDOW)


# Layers served from the OS page cache instead of the SSD (18.3GB/s vs 6.2GB/s measured). The
# cache only helps while every streamed byte fits in it: generation reads layers in a fixed loop,
# and a least-recently-used cache on a loop larger than itself misses almost every time.
CACHE_READ_SPEEDUP = 3.0


_WORKING_MEMORY_SLACK = 1.1  # +10% for activations and allocator fragmentation


@dataclass
class SlidingPlan:
    """What sliding adaptive processing decided for one model on one machine."""
    strategy: str             # 'all-resident' | 'resident+window' | 'window' | 'single-layer'
    resident: List[int]       # module indices loaded once and kept
    window: int               # layers loading ahead of the one running (0 = no prefetch)
    budget_bytes: int
    reserve_bytes: int        # budget kept free for streaming and inference
    resident_bytes: int
    streamed_bytes: int       # still read on every token
    os_cache: bool            # True: streamed layers fit in the OS page cache
    est_speedup: float        # estimate vs. streaming everything from the SSD with a 1-layer window

    def describe(self, memory_percent, available_bytes):
        gb = 1024 ** 3
        what = {
            'all-resident': 'every layer fits, nothing streams',
            'resident+window': f'{len(self.resident)} modules resident, the rest slide through a '
                               f'{self.window}-layer prefetch window',
            'window': f'layers slide through a {self.window}-layer prefetch window',
            'single-layer': 'one layer at a time, no prefetch',
        }[self.strategy]
        source = ''
        if self.streamed_bytes:
            source = (' from the OS page cache (fits in free RAM)' if self.os_cache
                      else ' mostly from the SSD (more than free RAM can cache)')
        speed = 'all in memory' if self.est_speedup == float('inf') else f'~{self.est_speedup:.1f}x'
        return (f"sliding adaptive processing: {what}. Budget {self.budget_bytes / gb:.2f}GB "
                f"({memory_percent:g}% of {available_bytes / gb:.2f}GB free); "
                f"{self.resident_bytes / gb:.2f}GB resident, {self.streamed_bytes / gb:.2f}GB read per "
                f"token{source}; {speed} vs. plain streaming.")


class _Budget:
    """The memory arithmetic behind one plan_sliding() call."""

    def __init__(self, layer_bytes, budget_bytes, kv_cache_bytes, unified_memory, host_budget_bytes,
                 free_ram_bytes):
        self.sizes = dict(layer_bytes)
        self.total = sum(self.sizes.values())
        self.largest = max(self.sizes.values(), default=0)
        self.budget, self.kv_cache = budget_bytes, kv_cache_bytes
        self.unified, self.host_budget, self.free_ram = unified_memory, host_budget_bytes, free_ram_bytes

    def device_reserve(self, window):
        """Budget kept for streaming: the running layer, plus its staging copy and the window
        when they share this memory."""
        layers = 2 + window if self.unified else 1
        return int((layers * self.largest + self.kv_cache) * _WORKING_MEMORY_SLACK)

    def window_fits(self, window):
        if self.device_reserve(window) > self.budget:
            return False
        if not self.unified and self.host_budget is not None:
            return (window + 1) * self.largest * _WORKING_MEMORY_SLACK <= self.host_budget
        return True

    def fits_page_cache(self, streamed, window, budget_used):
        """Whether every streamed byte fits in the RAM left outside what the model holds."""
        if self.free_ram is None:
            return False
        ram_used = budget_used if self.unified else (window + 1) * self.largest
        return streamed * _WORKING_MEMORY_SLACK <= self.free_ram - ram_used

    def evaluate(self, window, resident):
        reserve = self.device_reserve(window)
        resident_bytes = sum(self.sizes[i] for i in resident)
        streamed = self.total - resident_bytes
        os_cache = self.fits_page_cache(streamed, window, resident_bytes + reserve)
        rate = DISK_READ_SPEEDUP_BY_WINDOW[window] * (CACHE_READ_SPEEDUP if os_cache else 1.0)
        strategy = 'single-layer' if window == 0 else ('resident+window' if resident else 'window')
        return SlidingPlan(strategy, resident, window, self.budget, reserve, resident_bytes, streamed,
                           os_cache, self.total / streamed * rate)


def plan_sliding(layer_bytes, budget_bytes, kv_cache_bytes=0, unified_memory=True,
                 host_budget_bytes=None, max_window=MAX_PREFETCH_WINDOW, free_ram_bytes=None):
    """Sliding adaptive processing: decide how to run a model within a memory budget.

    Per-token time is roughly (bytes still read per token) / (how fast they arrive). Resident
    layers remove bytes, a prefetch window overlaps reads with compute, and RAM left outside the
    budget serves streamed layers from the OS page cache if they all fit. Every feasible window is
    tried with the budget either full of resident layers or empty (largest page cache); the
    fastest estimate wins, ties going to the smaller window. Everything fits -> nothing streams;
    the floor is one layer at a time.

    layer_bytes: {streamed module index: bytes once loaded}. budget_bytes: memory the model may
    use on the device. kv_cache_bytes: KV cache at full length. unified_memory: device shares
    system RAM (CPU, Apple silicon); on a discrete GPU the window waits in host RAM, bounded by
    host_budget_bytes. max_window: largest window (0 = no prefetch). free_ram_bytes: RAM free now
    (None = don't count on the page cache).
    """
    b = _Budget(layer_bytes, budget_bytes, kv_cache_bytes, unified_memory, host_budget_bytes, free_ram_bytes)
    if b.total + b.device_reserve(0) <= budget_bytes:
        return SlidingPlan('all-resident', sorted(b.sizes), 0, budget_bytes, b.device_reserve(0),
                           b.total, 0, False, float('inf'))
    best = None
    for window in range(0, min(max_window, MAX_PREFETCH_WINDOW) + 1):
        if not b.window_fits(window):
            break
        for resident in (plan_resident_layers(b.sizes, budget_bytes, b.device_reserve(window)), []):
            plan = b.evaluate(window, resident)
            if best is None or plan.est_speedup > best.est_speedup * 1.0001:
                best = plan
    if best is None:
        # Budget below even one layer: stream one at a time anyway (the floor), no prefetch.
        os_cache = free_ram_bytes is not None and b.total * _WORKING_MEMORY_SLACK <= free_ram_bytes
        best = SlidingPlan('single-layer', [], 0, budget_bytes, b.device_reserve(0), 0, b.total, os_cache,
                           DISK_READ_SPEEDUP_BY_WINDOW[0] * (CACHE_READ_SPEEDUP if os_cache else 1.0))
    return best
