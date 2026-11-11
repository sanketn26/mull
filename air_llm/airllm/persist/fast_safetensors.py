"""Read a whole safetensors file with large sequential reads.

``safetensors.torch.load_file`` memory-maps the file and copies each tensor out of the map, so
the data arrives through page faults a few KB at a time. On a fast NVMe drive that tops out
around 1.3GB/s, and extra threads don't help. Reading the data section in 64MB ``pread`` calls
into one buffer instead reaches the drive's real speed (6.4GB/s on one thread, 10GB/s with four
in flight, measured on an Apple-silicon SSD), and parallel reads from prefetch threads now add up
because ``pread`` releases the GIL.
"""
import json
import os
import struct

import torch
from safetensors.torch import load_file

_CHUNK = 64 << 20

_DTYPES = {
    'F64': 'float64', 'F32': 'float32', 'F16': 'float16', 'BF16': 'bfloat16',
    'I64': 'int64', 'I32': 'int32', 'I16': 'int16', 'I8': 'int8',
    'U64': 'uint64', 'U32': 'uint32', 'U16': 'uint16', 'U8': 'uint8', 'BOOL': 'bool',
    'F8_E4M3': 'float8_e4m3fn', 'F8_E5M2': 'float8_e5m2', 'F8_E8M0': 'float8_e8m0fnu',
}


def _torch_dtype(tag):
    name = _DTYPES.get(tag)
    return getattr(torch, name, None) if name else None


# Reads go through the OS page cache on purpose. Bypassing it (F_NOCACHE) was tried for models
# too big to cache: on macOS it also disables read-ahead and halved throughput (2.8 vs 6.3GB/s).


def read_safetensors(path):
    """Load every tensor in ``path`` onto the CPU. Same result as ``load_file(path)``.

    The tensors are views into one buffer, so no per-tensor allocation or copy is made. Falls
    back to ``load_file`` for anything unusual (unknown dtype, misaligned tensor).
    """
    path = str(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        header_len = struct.unpack('<Q', os.pread(fd, 8, 0))[0]
        header = json.loads(os.pread(fd, header_len, 8))
        header.pop('__metadata__', None)
        if any(_torch_dtype(v['dtype']) is None for v in header.values()):
            return load_file(path)

        data_start = 8 + header_len
        size = os.fstat(fd).st_size - data_start
        buf = torch.empty(size, dtype=torch.uint8)
        view = memoryview(buf.numpy())
        offset = 0
        while offset < size:
            got = os.preadv(fd, [view[offset:offset + _CHUNK]], data_start + offset)
            if got <= 0:
                raise OSError(f'short read from {path} at byte {data_start + offset}')
            offset += got
    finally:
        os.close(fd)

    out = {}
    for name, info in header.items():
        begin, end = info['data_offsets']
        raw = buf[begin:end]
        try:
            out[name] = raw.view(_torch_dtype(info['dtype'])).view(info['shape'])
        except RuntimeError:
            # Not aligned to the element size: viewing needs aligned storage, so copy this one.
            return load_file(path)
    return out
