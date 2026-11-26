"""Run the job server: python -m mull.server --model /path/to/Qwen3-8B-4bit"""

import argparse
import os
import signal
from pathlib import Path

from ..prefix_cache import PrefixCache
from .http import make_server
from .store import JobStore
from .worker import Worker, runner_for

DEFAULT_WEB = Path(__file__).resolve().parents[2] / "web"


def memory_share(value):
    if value == "auto":
        return value
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('expected "auto" or a number from 0 to 100') from None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="MLX model: packed checkpoint or Llama/Mistral (local path or repo id)")
    parser.add_argument("--memory-percent", type=memory_share, default="auto",
                        help='"auto" (default): use free RAM minus --keep-free-percent of total; or a fixed 0-100')
    parser.add_argument("--db", type=Path, default=Path("mull-jobs.sqlite3"))
    parser.add_argument("--host", default="127.0.0.1", help="keep localhost unless you add authentication")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-queued", type=int, default=100)
    parser.add_argument("--max-new-tokens-limit", type=int, default=4096)
    parser.add_argument("--keep-free-percent", type=float, default=30,
                        help="with --memory-percent auto: share of total RAM always left for other apps (default 30)")
    parser.add_argument("--max-batch", type=int, default=4,
                        help="jobs decoded together on packed models (each step reads the weights once for all)")
    parser.add_argument("--unload-after-seconds", type=float, default=120,
                        help="free the model's RAM after this long without a job (0: at once; negative: never)")
    parser.add_argument("--prefix-cache-gb", type=float, default=1.0,
                        help="RAM for reusing prompts that start the same way (packed models; 0 turns it off)")
    parser.add_argument("--web", nargs="?", const=DEFAULT_WEB, type=Path, metavar="DIR",
                        help=f"also serve the web interface (default folder: {DEFAULT_WEB})")
    args = parser.parse_args()
    if args.max_batch < 1:
        parser.error("--max-batch must be at least 1")
    if not 0 <= args.keep_free_percent < 100:
        parser.error("--keep-free-percent must be from 0 to below 100")
    if args.keep_free_percent != 30 and args.memory_percent != "auto":
        parser.error("--keep-free-percent only applies with --memory-percent auto")
    if args.web and not (args.web / "index.html").is_file():
        parser.error(f"--web: no index.html in {args.web}")
    return args


def _raise_interrupt(signum, frame):
    """Treat SIGTERM like Ctrl-C so `kill` stops the server cleanly and running jobs are kept."""
    raise KeyboardInterrupt


def main():
    args = parse_args()
    os.environ.setdefault("TQDM_DISABLE", "1")  # per-layer progress bars would flood the server log
    from .. import AutoModel

    memory = {"memory_percent": args.memory_percent}
    if args.memory_percent == "auto":
        memory["keep_free_fraction"] = args.keep_free_percent / 100
    model = AutoModel.from_pretrained(args.model, **memory)
    prefix_cache = PrefixCache(int(args.prefix_cache_gb * 2**30)) if args.prefix_cache_gb > 0 else None
    runner = runner_for(model, name=Path(str(args.model)).name, prefix_cache=prefix_cache)
    if runner is None:
        raise SystemExit("The job server runs MLX models on macOS: packed checkpoints (Qwen3, Qwen3.5, gpt-oss, "
                         "Qwen3 MoE) and Llama/Mistral checkpoints.")
    store = JobStore(args.db, max_queued=args.max_queued)
    unload_after = args.unload_after_seconds if args.unload_after_seconds >= 0 else None
    worker = Worker(store, runner, max_batch=args.max_batch, unload_after=unload_after)
    worker.start()
    server = make_server(store, worker, args.host, args.port, args.max_new_tokens_limit, args.web)
    where = f"http://{args.host}:{server.server_port}/"
    print(f"Mull job server: API at {where}jobs, web interface {'at ' + where if args.web else 'off (--web)'}; "
          f"jobs in {args.db}", flush=True)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        worker.stop()


if __name__ == "__main__":
    main()
