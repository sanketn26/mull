# Mull

Mull runs language models that are larger than your machine's memory. It streams them from disk one layer at a time, so a model is never fully loaded. It is being built into a local tool for background work: you queue a job, let it think, and come back for the answer. Apple-silicon Macs are the first target.

## An adaptation of AirLLM

Mull is an adaptation of [AirLLM](https://github.com/lyogavin/airllm) by Gavin Li, under the Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The core technique and most of the model support come from AirLLM:

- The model is built empty.
- Each layer is loaded, run and dropped in turn.

Mull does not sync with upstream.

**For anything this README doesn't cover, use [AirLLM's documentation](https://github.com/lyogavin/airllm#readme):**
- the full option list;
- CUDA usage;
- the supported-model list;
- the training API;
- the FAQ.

Those results were measured by AirLLM, and Mull has not re-verified them. Mull is developed on a Mac without an NVIDIA GPU.

The package is imported as `mull`; classes keep their AirLLM names (for example `AirLLMLlamaMlx`).

## What Mull adds

All numbers below were measured on a 24 GB Apple-silicon Mac.

| Change | Result |
| --- | --- |
| Selective garbage collection (was a full collection after every layer) | 3–7× faster per token |
| Direct-read safetensors loader | Reads 1.3 → 5.8 GB/s from the SSD; a 32 GB model went from 8.7 to 5.1 s/token |
| Sliding adaptive processing (`memory_percent`) | 32 GB model: 4.2 s/token at 50%. 16 GB model: 935 → 733 ms |
| MLX 4-bit / 8-bit quantization | 3.5× fewer bytes read per token (4-bit) |
| MLX safetensors layer files | 155 → 70 ms/token |
| Prefetch memory guard | Read-ahead stops before free RAM falls below 10% |
| Read-ahead without MLX in the worker thread | Llama-3.1-8B 4-bit: 1.67 → 0.17 s/token |
| Splitter reads per tensor, not per shard | Peak memory while splitting Llama-3.1-8B: 7.6 → 4.0 GB |

### Llama-3.1-8B-Instruct on a 24 GB Mac

The fp16 model is 15 GB, more than the 10–14 GB of RAM that was free, so it streams from the SSD. Each figure is one run of 15 tokens, including the prompt.

| Mode | s/token | Peak memory | Output |
| --- | --- | --- | --- |
| fp16, `memory_percent=0` (one layer at a time) | 3.16 | 2.0 GB | "The vast expanse of the sea stretches out to the horizon, its waves" |
| fp16, 25% | 2.56 | 3.0 GB | same |
| fp16, 50% | 1.95 | 4.5 GB | same |
| 4-bit, 25% | **0.17** | 3.0 GB | "The vast and mysterious ocean covers over 70% of the Earth's surface" |
| 4-bit, 50% (whole model in memory) | **0.06** | 5.0 GB | same as 4-bit, 25% |

fp16 is limited by the SSD (about 6 GB/s). 4-bit reads 3.5× fewer bytes, and those bytes fit in the OS page cache. The planner adapts to the RAM that is free when the model loads, so results at a given percentage vary with what else is running.

The code was also split into smaller modules with offline end-to-end tests, and no function is longer than 50 lines.

## Where it's going

1. Qwen and coding models on the Mac.
2. A background job server with a web interface.
3. Batching of queued jobs, so one read of the weights serves several prompts.
4. Speculative decoding.
5. Audio to text and video to text.

Video generation is out of scope. See [docs/PLAN.md](docs/PLAN.md) for details.

## Quick start

Mull is not on PyPI. Install it from the repository with [uv](https://docs.astral.sh/uv/) (`brew install uv` on macOS):

```bash
git clone https://github.com/sanketn26/mull.git && cd mull
make setup          # uv sync: Python 3.12, Mull (editable), MLX on macOS, dev tools
make test           # offline tests, about 10 s
make run MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 PROMPT="What is the capital of France?"
```

`make run` also accepts `MAX_NEW_TOKENS`, `COMPRESSION=4bit|8bit` and `MEMORY_PERCENT`.

From Python:

```python
from mull import AutoModel

model = AutoModel.from_pretrained("meta-llama/Llama-3.3-70B-Instruct",
                                  compression="4bit", memory_percent=25)
```

On macOS, `AutoModel` returns the MLX runtime. Only Llama-style models (`LlamaForCausalLM` and `MistralForCausalLM`) run there for now; other architectures raise an error before anything is downloaded.

The first load splits the checkpoint into one file per layer. This needs about as much free disk space as the model itself. Set `delete_original=True` to remove the downloaded original after the split.

## Options Mull adds or changes

- **`memory_percent`** (default `25`; `0` turns it off): the share of free memory that sliding adaptive processing may use. On CUDA that memory is VRAM; on Apple silicon and CPU it is system RAM.
- **`compression`** (`'4bit'` or `'8bit'`): on macOS this uses MLX-native quantization. The quantized split is written once, to its own folder. On CUDA it uses bitsandbytes, as in AirLLM.

Every other option behaves as described in [AirLLM's documentation](https://github.com/lyogavin/airllm#configurations).

### Sliding adaptive processing

When the model loads, the planner reads three things: the layer sizes, the free memory, and `memory_percent`. It then estimates the time per token for each strategy and picks the fastest:

- **Everything fits:** load all layers once.
- **Some layers resident:** keep those loaded. The rest pass through a small prefetch window that reads the next layer while the current one runs.
- **Use the OS page cache:** if the streamed layers fit in the RAM left over, keep Mull's own memory use small so the operating system can cache them.
- **One layer at a time:** the fallback when the budget is too small for anything else.

The planner prints its decision:

```
sliding adaptive processing: 17 modules resident, the rest slide through a 2-layer prefetch window.
Budget 8.15GB (50% of 16.30GB free); 6.18GB resident, 23.56GB read per token mostly from the SSD ...
```

Read-ahead never pushes free RAM below `max(1 GB, 10% of total)`. Override that floor with `MULL_MIN_FREE_RAM_GB`.

### MLX compression

Measured with TinyLlama-1.1B on Apple silicon:

| | fp16 | 8bit | 4bit |
|---|---|---|---|
| layer files read per token | 2.20 GB | 1.17 GB | 0.62 GB |
| speed, layers kept in memory | 22.5 ms | 12.9 ms | 10.4 ms |
| greedy output vs. fp16 | – | same for 8–32 tokens, then different but sensible | same for 3–32 tokens, then different but sensible |

On a model larger than RAM the gain is larger than the bytes saved: Llama-3.1-8B went from 3.16 s/token (fp16) to 0.17 s/token (4-bit) at 25%, because the 4-bit layers also fit in the OS page cache. See [the table above](#llama-31-8b-instruct-on-a-24-gb-mac). 8-bit has not been measured on this model.

## Development

| Command | What it does |
| --- | --- |
| `make setup` | `uv sync`: create `.venv` from `uv.lock` with Mull, MLX (macOS) and dev tools |
| `make lock` | Refresh `uv.lock` after changing dependencies in `pyproject.toml` |
| `make test` | Run the test suite |
| `make lint` | Run ruff |
| `make clean` | Remove caches and build files |
| `make clean-prof` | Remove profiling and benchmark data in `.prof/` |

The guiding rules, set out in [docs/PLAN.md](docs/PLAN.md):
- **Hardware safety first:** never push the machine into swap.
- **Measure before claiming a speed-up.**
- **Keep code in small, tested units.**

## Credits

- [AirLLM](https://github.com/lyogavin/airllm) by Gavin Li, which Mull is adapted from. If you cite this work, cite AirLLM:

  ```
  @software{airllm2023,
    author = {Gavin Li},
    title = {AirLLM: scaling large language models on low-end commodity computers},
    url = {https://github.com/lyogavin/airllm/},
    year = {2023},
  }
  ```

- [SimJeg's Kaggle work](https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag), which AirLLM credits as its starting point.
