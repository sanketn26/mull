"""Memory housekeeping: picking a device, freeing memory between layers, and RAM safety checks."""
import ctypes
import gc
import os

import torch


# A full gc.collect() walks every live Python object, and a loaded transformers model has
# millions of them, so it costs ~30ms each time. The streaming loops used to run one after
# every layer, which made it 70-85% of per-token time in profiling. Freed layer weights don't
# need it: CPython releases a tensor as soon as its last reference goes away. It is only
# needed for reference cycles, which accumulate slowly. So the per-layer call collects just
# the youngest generation (cheap), and a full pass runs every GC_FULL_EVERY calls.
# 256 layers is a few tokens of a 70B model. Profiling showed the full pass costing ~10ms per
# token at an interval of 64 (7-13% of streaming time); the cycles it frees hold no layer
# weights, and peak GPU memory measured the same at intervals of 1, 64 and 256.
GC_FULL_EVERY = max(1, int(os.environ.get("MULL_GC_FULL_EVERY", "256")))


_gc_calls = 0


def collect_garbage(full=False):
    """Run the cheap young-generation gc, or a full pass every GC_FULL_EVERY calls.

    Returns True when a full pass ran, so callers can do other periodic cleanup with it.
    """
    global _gc_calls
    _gc_calls += 1
    if full or _gc_calls % GC_FULL_EVERY == 0:
        gc.collect()
        return True
    gc.collect(0)
    return False


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        # Not glibc (macOS, musl): nothing to trim.
        pass


# Function to clean RAM & vRAM
def clean_memory(full=False):
    """Release memory after a streamed layer.

    The heavy steps (full gc, malloc_trim, emptying the device allocator's cache) run only
    on the periodic full pass, or when ``full=True``. Emptying the cache after every layer
    gains nothing: the next layer has the same shapes, so the allocator would reuse those
    exact blocks, and handing them back to the driver only to request them again is slow.
    """
    if not collect_garbage(full=full):
        return
    _malloc_trim()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def default_device():
    """Pick the best device available, instead of assuming CUDA.

    AirLLM's public API defaulted to ``cuda:0``, so a plain
    ``AutoModel.from_pretrained(repo_id)`` raised on any machine without an NVIDIA GPU -- Apple
    silicon, a CPU-only box, or a container without the runtime -- even though the layer-streaming
    path itself is device-agnostic. Prefer CUDA (where the VRAM savings matter), then Apple's MPS
    backend, then CPU.
    """
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def available_memory_bytes(device):
    """Memory free for this process on ``device`` right now, in bytes.

    CUDA reports its own free VRAM. CPU and Apple MPS both draw from system RAM (on Apple silicon
    the GPU shares it), so they use the OS's "available" figure, which counts reclaimable cache.
    """
    device = torch.device(device)
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        # Linux without psutil. macOS has no SC_AVPHYS_PAGES, but psutil is installed there.
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


def system_ram_bytes():
    """(total, available) system RAM in bytes."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except ImportError:
        page = os.sysconf("SC_PAGE_SIZE")
        return (int(os.sysconf("SC_PHYS_PAGES") * page), int(os.sysconf("SC_AVPHYS_PAGES") * page))


def min_free_ram_bytes():
    """Free RAM that prefetching must always leave: 10% of total, at least 1GB.

    Override with MULL_MIN_FREE_RAM_GB. Below this the OS starts compressing and swapping,
    which slows the whole machine and wears the SSD for no gain.
    """
    override = os.environ.get("MULL_MIN_FREE_RAM_GB")
    if override:
        return int(float(override) * 1024 ** 3)
    total, _available = system_ram_bytes()
    return max(1024 ** 3, total // 10)


def has_room_to_prefetch(nbytes):
    """Whether reading ``nbytes`` ahead of time keeps free RAM above min_free_ram_bytes().

    Checked before every read-ahead, not just at load time, so memory taken by other programs
    after the model loaded is respected too. When it fails, the layer is simply read when it is
    needed instead: slower, never unsafe.
    """
    _total, available = system_ram_bytes()
    return available - nbytes >= min_free_ram_bytes()
