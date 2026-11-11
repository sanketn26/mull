# Mull: next-steps plan

Last updated 2026-11-11. Collaborative copy: https://claude.ai/artifact/VKVsAfoqwNgWHvVEikuK4f

## Goal

Run models larger than RAM — Qwen and coding models first — on a 24 GB Apple-silicon Mac, as a standalone local tool with a web interface for background work and delayed processing. Inputs are text, audio and video; output is text. Mull is a fork of AirLLM (lyogavin/airllm, Apache 2.0); it does not sync with upstream and is not published to PyPI. The import package is `mull`; class names keep their AirLLM prefixes.

For models bigger than RAM, speed is set by bytes read from the SSD per token, about 6 GB/s on this machine. The levers are therefore: read fewer bytes, read them once for several tokens, or read them once for several prompts.

Three rules apply to every workstream:

- **Hardware safety first.** Never push the machine into swap or out-of-memory, and never exceed the planned memory budget. A speed-up that raises peak memory is rejected.
- **Measure, don't estimate.** Claim a gain only after measuring it, with the data on disk, not in cache.
- **SOLID and small units.** Shared abstractions over copies, per-model layouts declared as data, functions of 50 lines or fewer (enforced by `tests/test_function_length.py`). End-to-end tests come before any refactor.

## Where things stand

The fork is 2–8× faster per token than the original code and passes 120 tests.

| Change | Measured result |
| --- | --- |
| Selective garbage collection (was a full GC after every layer) | 3–7× faster per token on every backend; peak memory unchanged |
| Direct-read safetensors loader (was memory-mapped) | 1.3 → 5.8 GB/s from the SSD; 32 GB model 8.7 → 5.1 s/token |
| Sliding adaptive processing (`memory_percent`, default 25%) | 32 GB model 5.1 → 4.2 s/token at 50%; 16 GB model 935 → 733 ms at 25% |
| MLX 4-bit / 8-bit compression | Bytes read per token 2.20 → 0.62 GB (4-bit); kept-in-memory speed 22.5 → 10.4 ms (TinyLlama-1.1B) |
| MLX safetensors layer files (were `.npz`) | 155 → 70 ms/token streamed from cache |
| Prefetch memory guard | Read-ahead stops before free RAM drops below 10% of total |
| Refactor (focused modules, shared `PrefetchWindow`, per-runtime persister, declarative model layouts) | No speed change; all functions ≤ 50 lines |

Gaps:

- **Not yet measured on a model bigger than RAM** for the MLX path or 4-bit; those gains are estimates.
- **Untested here:** CUDA (no NVIDIA GPU), the compressed-tensors path used by Kimi K3, and the example training scripts' loop.
- **Mac model support:** MLX runs only Llama and Mistral. Kimi K3 cannot run on this machine (needs CUDA and ~1.6 TB of disk).

## Workstreams

Effort is S (a session), M (a few sessions) or L (a project). Gains are estimates until measured.

| # | Workstream | Expected gain | Effort | Done when |
| --- | --- | --- | --- | --- |
| 1 | ~~Commit the work; drop the PyPI release workflow~~ (done) | Work is safe and reviewable | S | Commits on `main`; `release.yml` removed |
| 2 | CI and disk-bound benchmark | Regressions caught; key estimates confirmed | S | GitHub Actions runs `make test` and `make lint`; measured s/token and peak memory for a model about 2× free RAM, fp16 vs 4-bit |
| 3 | Qwen and coding models on MLX (reuse `mlx-lm` definitions) | Qwen2.5/3, Qwen-Coder run on the Mac | M | Qwen-Coder streams end to end and matches `mlx-lm`'s own output |
| 4 | Background job server with web interface | Submit jobs, come back for results | M | Local server with a persistent job queue, web UI and HTTP API; one model loaded, jobs run in turn |
| 5 | Batched prompts | Throughput rises with batch size | M | Queued jobs on the same model share one weight read; planner reserves KV cache per batch |
| 6 | Speculative decoding | ~2–4× per token for disk-bound models | M–L | Small Qwen draft proposes k tokens; big model verifies in one pass; output identical to greedy |
| 7 | Audio to text | Transcribe and reason over audio | M | Speech-to-text (Whisper on MLX) as a job type; optionally Qwen-Audio for questions about the audio |
| 8 | Video to text | Describe and answer questions about video | M–L | Qwen-VL job: frames sampled, small vision encoder resident, language model streamed; planner sizes KV cache for video tokens |
| 9 | MLX per-expert streaming | MoE models read only routed experts | M | Qwen MoE streams experts on demand, matching full-layer output |

**3. Qwen via `mlx-lm`.** Start with a spike: can `mlx-lm`'s layers be built and run one at a time with AirLLM's streaming and KV cache? If yes, AirLLM keeps streaming, planner and quantization, and `mlx-lm` supplies the networks. If no, port Qwen by hand; it is close to Llama.

**4–5. Job server and batching.** Background work doesn't need fast replies, so the target is tokens per hour across jobs. Jobs waiting in the queue for the same model are batched automatically: one read of the weights serves all of them.

**7–8. Audio and video to text.** Both put a small encoder (resident) in front of a language model (streamed). Whisper models are small (≤ 1.6 B) and run fully in memory with `mlx-whisper`, so transcription needs no streaming; AirLLM's role is the larger model that reasons over the transcript or the audio/video tokens. A minute of video can be thousands of tokens, so frame rate and resolution are capped by the planner.

**Out of scope: video generation.** Diffusion models run the whole network 30–50 times per clip and are compute-bound, so disk streaming would take hours per clip. Use a separate tool (ComfyUI or diffusers with a small model) if ever needed.

## Sequencing

1. **Foundation:** 1 → 2 (a session or two).
2. **Models and service:** 3 → 4 → 5. Gate: Qwen-Coder matches `mlx-lm` before the server is built on it.
3. **Speed and inputs:** 6, 7, 8 in any order; 9 when a MoE model is wanted.

CUDA verification is parked until an NVIDIA GPU is available.

## Risks and guardrails

| Risk | Guardrail |
| --- | --- |
| A speed-up raises peak memory and pushes the Mac into swap | Every change reports peak memory beside speed; the prefetch RAM floor stays on |
| Speculative decoding, batching or video tokens grow the KV cache past the budget | The planner reserves KV cache for draft tokens, batch size and input length before keeping layers |
| Benchmarks look good only because data sits in the OS cache | Disk-bound runs use a model about 2× free RAM, or files written and read with caching off |
| `mlx-lm` changes its internals and breaks streaming | Pin its version; end-to-end tests compare against `mlx-lm`'s own output |
| Running code downloaded with a model (`trust_remote_code`) | MLX loads configs with it off; family checks use JSON metadata only |
| Untested CUDA paths regress unseen | CI covers CPU paths; CUDA-only changes are flagged in the PR until a GPU run confirms them |
| Large test models fill the disk | Test data lives in `.prof/` or the scratchpad and is deleted after each run (`make clean-prof`); benchmarks use at most 50 GB |

## Decisions (2026-11-11)

- Model priority: Qwen family and coding models.
- Use: programmatic and background jobs with batching, not chat latency.
- Distribution: standalone tool with a web interface; no PyPI.
- Video: video-to-text yes; video generation out of scope. Audio-to-text added.
- NVIDIA GPU: not available yet; CUDA checks parked.
- Disk for benchmarks: up to 50 GB.
