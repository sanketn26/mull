# Mull

Mull runs language models that are larger than your machine's memory. It streams layers from disk and keeps weights resident when the memory budget allows. It is being built into a local tool for background work: you queue a job, let it think, and come back for the answer. Apple-silicon Macs are the first target.

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

AirLLM's own performance figures are in its documentation. Every number in this README was measured on Mull, on a 24 GB Apple-silicon Mac without an NVIDIA GPU.

The package is imported as `mull`; classes keep their AirLLM names (for example `AirLLMLlamaMlx`).

## What Mull adds


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

model = AutoModel.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct",
                                  compression="4bit", memory_percent=25)
```

On macOS, `AutoModel` supports Llama and Mistral checkpoints through the layer splitter. The first load needs roughly one additional copy of the weights on disk. Set `delete_original=True` to remove the downloaded original after the split.

The macOS extra also supports native packed MLX checkpoints for Qwen3, Qwen3.5 text, gpt-oss, and Qwen3 MoE. These models read the packed weights from the original shards. Use a local checkpoint directory and leave `compression` unset.

```python
model = AutoModel.from_pretrained("/path/to/Qwen3-8B-4bit", memory_percent=25)
ids = model.tokenizer.encode("The capital of France is")
text = model.generate([ids], max_new_tokens=32)
```

Packed models generate decoded continuations for one request at a time. The request plan estimates model weight, cache and prefill workspace bytes. Free RAM is measured again for each request. An allocation below the estimated execution footprint streams one module at a time instead of failing; only when even that would not fit in free RAM does it raise `MemoryError`. Set `memory_percent=0` to stream without a residency allocation. Qwen3.5 accepts text tokens. Qwen3 MoE uses whole-block loading by default and accepts `expert_streaming=True` to try the experimental expert cache.

## Background job server

Queue prompts over HTTP and let one worker run them on a loaded packed MLX model. Jobs are kept in SQLite, so they survive a restart: a job that was running when the server stopped is queued again.

```bash
make serve MODEL=/path/to/Qwen3-8B-4bit          # API only, on http://127.0.0.1:8765
make serve MODEL=/path/to/Qwen3-8B-4bit WEB=1    # also serve the web interface from web/
```

| Method | Path | What it does |
| --- | --- | --- |
| `GET` | `/health` | Loaded model, memory share, job counts, free RAM |
| `POST` | `/jobs` | Queue a job; returns `202 {"id": ...}`. 400 for bad input, 429 when the queue is full |
| `GET` | `/jobs?state=&limit=` | Recent jobs, newest first |
| `GET` | `/jobs/{id}` | State, tokens so far, `output`, `reasoning`, error, timings |
| `GET` | `/jobs/{id}/events` | Server-sent events: the job each time it changes, until it finishes |
| `POST` | `/jobs/{id}/cancel` | Cancel now if queued, at the next token if running |
| `DELETE` | `/jobs/{id}` | Remove a finished job |

`POST /jobs` fields:

| Field | Default | |
| --- | --- | --- |
| `prompt` | required | |
| `max_new_tokens` | 256 | Up to the server's limit (`--max-new-tokens-limit`, 4096) |
| `system` | none | System prompt |
| `thinking` | `false` | Let a reasoning model (Qwen3) think before answering. The thinking goes to `reasoning`, not `output` |
| `temperature` | 0 | 0 to 2; 0 is greedy |
| `stop` | none | Up to 4 strings; the answer ends before the first one |

```bash
curl -X POST localhost:8765/jobs -d '{"prompt": "Explain this stack trace ...", "max_new_tokens": 512}'
curl localhost:8765/jobs/1
```

Thinking costs time. On Qwen3-8B-4bit, "What is 17 × 23?" took 3 tokens and 1.0 s with thinking off, and 523 tokens and 53.6 s with it on. Both answered 391.

**Batching.** On packed models, up to `--max-batch` queued jobs (default 4) run together: each prompt is processed on its own, then every generation step reads the weights once for all of them. Jobs join and leave the batch as they arrive and finish. A job that doesn't fit in memory next to the running ones waits in the queue instead of failing. Measured on Qwen3-8B-4bit, four 128-token jobs queued at once:

| | batch 1 | batch 2 | batch 4 |
| --- | --- | --- | --- |
| Fully loaded: total throughput | 29.9 tok/s | 57.8 tok/s | **108.8 tok/s** |
| Fully loaded: wait for the last job's first token | 13.1 s | 4.8 s | 0.5 s |
| Streamed (`--memory-percent 0`): total throughput | 4.5 tok/s | 8.4 tok/s | **15.1 tok/s** |

Batched answers are valid but not always identical to running each job alone. Batched and single-row GPU kernels round differently in bf16 (logits differed by up to 0.375 here), which can flip a choice between two nearly tied tokens: in one job, "dictionary" instead of "hash" at token 11. Use `--max-batch 1` when you need runs to be reproducible. The Llama runtime (TinyLlama, Llama, Mistral) runs one job at a time.

**Prefix cache.** Jobs that start with the same text (a shared system prompt, the same files followed by a different question) reuse the already-processed part of the prompt. On Qwen3-8B-4bit with a 4,200-token source file and three different questions, the time to the first token fell from about 4 s to 0.41–0.50 s for the second and third jobs, with identical answers. The cache holds up to 1 GB by default (`--prefix-cache-gb`, `0` turns it off), evicts the least recently used prompt first, and the memory planner counts what it holds. It is used with packed models (Qwen3, Qwen3.5, gpt-oss, Qwen3 MoE). Qwen3.5's state-space layers can't be rewound, so for Qwen3.5 only a prompt that fully extends an earlier one is reused.

Jobs run one at a time, in order. The server uses `memory_percent="auto"` unless you pass `--memory-percent` (or `MEMORY_PERCENT=`). A job that needs more memory than is free fails with a message instead of pushing the machine into swap. The server listens on localhost only and has no authentication. The web interface in `web/` is plain HTML, CSS and JavaScript that uses only the API above; it is served only with `--web` (or `WEB=1`).

### Use from Claude Code or Codex (MCP)

`mull-mcp` is an MCP server that lets a coding agent hand slow or private work to Mull: summarising many files, overnight reviews, transcripts. The agent queues a job and checks back later; it can pass file paths, so large files don't go through its own context. Tools: `mull_submit`, `mull_result` (can wait up to 60 s), `mull_cancel`, `mull_list`.

Start the job server (`make serve MODEL=...`), then register `mull-mcp` once. `make setup` installs it into `.venv`.

```bash
claude mcp add mull -- /path/to/mull/.venv/bin/mull-mcp     # Claude Code
```

```toml
# Codex: ~/.codex/config.toml
[mcp_servers.mull]
command = "/path/to/mull/.venv/bin/mull-mcp"
```

Set `MULL_URL` if the job server isn't on `http://127.0.0.1:8765`. Mull is too slow to be the agent's main model; it is meant for work the agent can leave running.

## Options Mull adds or changes

- **`memory_percent`** (default `25`; `0` turns it off): the share of free memory that sliding adaptive processing may use. On CUDA that memory is VRAM; on Apple silicon and CPU it is system RAM.
- **`memory_percent="auto"`** (packed MLX models; the job server's default): use free RAM but always leave 30% of total for other apps (`keep_free_fraction=0.3`). The model stays fully loaded while the machine is idle and streams when it is busy. Its own resident weights count as free, so it doesn't shrink after the first request. On Qwen3-8B-4bit (200 tokens, two runs): 32.1 and 28.7 tok/s with `auto`, against 14.5 and 9.5 tok/s at 25%, for 4.7 GB of RAM instead of 3.1 GB.
- **`compression`** (`'4bit'` or `'8bit'`): on macOS this uses MLX-native quantization. The quantized split is written once, to its own folder. On CUDA it uses bitsandbytes, as in AirLLM.

For the layer-splitting backend, other options are described in [AirLLM's documentation](https://github.com/lyogavin/airllm#configurations).

### Sliding adaptive processing

On Apple silicon, when the streamed weights fit in the OS page cache, the planner uses the read-ahead allowance for resident layers. For a partially resident model, startup also reports the estimated RAM allocation needed to keep every layer resident.

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

Two refinements on packed MLX models:

- **First load reads ahead.** On the first request, a background thread reads the files of the layers that will stay resident, in the order they run, while earlier layers are set up. It needs no extra memory, since those layers are planned to stay loaded. Qwen3-8B-4bit read cold from a USB drive: first token in 10.0 and 9.8 s, against 12.4 and 12.7 s without it.
- **Untied embeddings are read by row.** A token needs one row (a few KB) of an input table that can be hundreds of MB. When the output layer has its own weights, Mull reads just the prompt's rows, so the planner spends that RAM on decoder layers instead, which every token reads in full. The table stays in RAM only when everything else already fits. Qwen3-8B-4bit at 25% of 12 GB (512-token prompt, 64 tokens, three runs each): 9.99 / 9.96 / 9.94 tok/s before, 11.36 / 11.27 / 11.27 after, with identical tokens. Fully loaded is unchanged (about 32 tok/s).

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
