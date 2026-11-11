

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
