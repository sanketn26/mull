import math


import os
from pathlib import Path
import mlx.core as mx
from .model_persister import ModelPersister
from mlx.utils import tree_unflatten
import torch


# compression= values the MLX backend quantizes natively, and the quantization group size. Groups
# of 64 are MLX's default and what mlx-lm ships: one fp16 scale and bias per 64 weights, so 4-bit
# costs ~4.5 bits per weight and 8-bit ~8.5.
QUANT_BITS = {'4bit': 4, '8bit': 8}
QUANT_GROUP_SIZE = 64


def quantizable(name, shape):
    """Matrix weights whose rows split into whole groups. Norms and odd shapes stay fp16."""
    return name.endswith('.weight') and len(shape) == 2 and shape[-1] % QUANT_GROUP_SIZE == 0



def map_torch_to_mlx(model):

    # things to change
    # 1. there's no "model." in the weight names
    model = {k.replace("model.", ""): v for k, v in model.items()}

    # 2. mlp is called feed_forward
    model = {k.replace("mlp", "feed_forward"): v for k, v in model.items()}

    # 3. up_proj, down_proj, gate_proj
    model = {k.replace("down_proj", "w2"): v for k, v in model.items()}
    model = {k.replace("up_proj", "w3"): v for k, v in model.items()}
    model = {k.replace("gate_proj", "w1"): v for k, v in model.items()}

    # 4. layernorms
    model = {
        k.replace("input_layernorm", "attention_norm"): v for k, v in model.items()
    }
    model = {
        k.replace("post_attention_layernorm", "ffn_norm"): v for k, v in model.items()
    }

    # 5. lm head
    model = {k.replace("lm_head", "output"): v for k, v in model.items()}

    # 6. token emb
    model = {k.replace("embed_tokens", "tok_embeddings"): v for k, v in model.items()}

    # 7. attention
    model = {k.replace("self_attn", "attention"): v for k, v in model.items()}
    model = {k.replace("q_proj", "wq"): v for k, v in model.items()}
    model = {k.replace("k_proj", "wk"): v for k, v in model.items()}
    model = {k.replace("v_proj", "wv"): v for k, v in model.items()}
    model = {k.replace("o_proj", "wo"): v for k, v in model.items()}


    #weights = {k: v.to(torch.float16).numpy() for k, v in model.items()}


    return model

class MlxModelPersister(ModelPersister):
    """Layer shards for the MLX backend.

    New splits are written as ``<layer>.mlx.safetensors``: fp16, or with ``compression='4bit'``
    / ``'8bit'`` every matrix weight stored MLX-quantized (packed ``weight`` plus ``scales`` and
    ``biases``, the names ``nn.QuantizedLinear`` uses), so the shard can be read straight into
    quantized layers that multiply without expanding the weights. ``mx.load`` reads safetensors
    directly into MLX arrays; the older ``.mlx.npz`` format goes through Python's zipfile,
    including a CRC check over every byte, and loaded ~3x slower in profiling (13ms vs 4ms for a
    100MB layer, warm). Splits made before the change are still found and loaded as ``.npz``.
    """

    FORMATS = ('mlx.safetensors', 'mlx.npz')
    # Tells the splitter to hand compression= to persist_model instead of using bitsandbytes.
    supports_native_compression = True

    def __init__(self, *args, **kwargs):
        super(MlxModelPersister, self).__init__(*args, **kwargs)

    def layer_file(self, layer_name, saving_path):
        """The shard file for ``layer_name`` (``layer_name`` without its trailing dot), or None."""
        for fmt in self.FORMATS:
            path = Path(saving_path) / f'{layer_name}.{fmt}'
            if path.exists():
                return path
        return None

    def model_persist_exist(self, layer_name, saving_path):
        # The splitter passes layer names with a trailing '.', hence no separator here.
        done = os.path.exists(str(Path(saving_path) / (layer_name + 'mlx.done')))
        return done and any(os.path.exists(str(Path(saving_path) / (layer_name + fmt)))
                            for fmt in self.FORMATS)

    def persist_model(self, state_dict, layer_name, saving_path, compression=None):
        if compression is not None and compression not in QUANT_BITS:
            raise ValueError(f"MLX compression must be one of {sorted(QUANT_BITS)}, got {compression!r}")
        bits = QUANT_BITS.get(compression)
        arrays = {}
        for name, value in state_dict.items():
            array = mx.array(value.to(torch.float16).contiguous().numpy())
            if bits and quantizable(name, array.shape):
                packed, scales, biases = mx.quantize(array, group_size=QUANT_GROUP_SIZE, bits=bits)
                base = name[:-len('weight')]
                arrays[name], arrays[base + 'scales'], arrays[base + 'biases'] = packed, scales, biases
            else:
                arrays[name] = array
        target = Path(saving_path) / (layer_name + 'mlx.safetensors')
        mx.save_safetensors(str(target), arrays, metadata={'airllm_compression': compression or 'none'})
        print(f"saved as: {target}")
        (Path(saving_path) / (layer_name + 'mlx.done')).touch()

    def load_model(self, layer_name, path):
        to_load_path = self.layer_file(layer_name, path)
        if to_load_path is None:
            raise FileNotFoundError(f"no MLX shard for {layer_name} under {path}")
        layer_state_dict = mx.load(str(to_load_path))
        layer_state_dict = map_torch_to_mlx(layer_state_dict)
        return tree_unflatten(list(layer_state_dict.items()))


def load_packed_weights(path, manifest, keys):
    """Read selected arrays from native MLX shards, retaining their packed dtype."""
    by_file = {}
    for key in keys:
        by_file.setdefault(manifest[key][0], []).append(key)
    selected = {}
    for filename, names in by_file.items():
        arrays = mx.load(str(path / filename))
        selected.update((key, arrays[key]) for key in names)
    return selected


def warm_packed_weights(path, manifest, keys):
    """Read the selected tensor ranges into the OS page cache without creating MLX arrays."""
    by_file = {}
    for key in keys:
        filename, _, _, size, offset = manifest[key]
        by_file.setdefault(filename, []).append((offset, size))
    buffer = memoryview(bytearray(1 << 20))
    for filename, ranges in by_file.items():
        with (path / filename).open("rb", buffering=0) as shard:
            for offset, size in sorted(ranges):
                shard.seek(offset)
                while size:
                    count = shard.readinto(buffer[:size])
                    if not count:
                        raise EOFError(f"Incomplete tensor range in {filename}")
                    size -= count


def load_row_matrix(path, manifest, key, rows):
    """Read the given rows of one tensor (sorted, unique) into a single array, coalescing adjacent rows."""
    from itertools import groupby

    import numpy as np

    filename, dtype, shape, size, offset = manifest[key]
    if rows and (rows[0] < 0 or rows[-1] >= shape[0]):
        raise IndexError(f"Rows {rows[0]}..{rows[-1]} out of range for {key} with {shape[0]} rows")
    row_bytes = size // shape[0]
    data = bytearray(row_bytes * len(rows))
    view = memoryview(data)
    done = 0
    with (path / filename).open("rb", buffering=0) as shard:
        for _, group in groupby(enumerate(rows), lambda item: item[1] - item[0]):
            group = [row for _, row in group]
            want = row_bytes * len(group)
            shard.seek(offset + row_bytes * group[0])
            while want:
                count = shard.readinto(view[done:done + want])
                if not count:
                    raise EOFError(f"Incomplete tensor range for {key}")
                done += count
                want -= count
    dtypes = {"U32": "<u4", "U8": "u1", "BF16": "<u2", "F16": "<f2", "F32": "<f4"}
    array = mx.array(np.frombuffer(data, dtype=dtypes[dtype]).reshape(len(rows), *shape[1:]))
    return array.view(mx.bfloat16) if dtype == "BF16" else array


def load_packed_rows(path, manifest, keys, rows):
    """Load sorted expert rows while keeping their packed bits and coalescing adjacent reads."""
    from itertools import groupby

    import numpy as np

    groups = [tuple(row for _, row in group)
              for _, group in groupby(enumerate(rows), lambda item: item[1] - item[0])]
    dtypes = {"U32": "<u4", "U8": "u1", "BF16": "<u2", "F16": "<f2", "F32": "<f4"}
    result = {row: {} for row in rows}
    for key in keys:
        filename, dtype, shape, size, offset = manifest[key]
        row_bytes = size // shape[0]
        with (path / filename).open("rb") as shard:
            for group in groups:
                shard.seek(offset + row_bytes * group[0])
                data = shard.read(row_bytes * len(group))
                if len(data) != row_bytes * len(group):
                    raise EOFError(f"Incomplete expert tensor range for {key}")
                for i, row in enumerate(group):
                    values = np.frombuffer(data, dtype=dtypes[dtype], count=math.prod(shape[1:]),
                                           offset=i * row_bytes)
                    array = mx.array(values).reshape(shape[1:])
                    result[row][key] = array.view(mx.bfloat16) if dtype == "BF16" else array
    return result
