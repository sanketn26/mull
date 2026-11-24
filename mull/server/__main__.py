"""Run the job server: python -m mull.server --model /path/to/Qwen3-8B-4bit"""

import argparse
import signal
from pathlib import Path

from .http import make_server
from .store import JobStore
from .worker import PackedModelRunner, Worker


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="packed MLX checkpoint (local path or repo id)")
    parser.add_argument("--memory-percent", type=int, default=25)
    parser.add_argument("--db", type=Path, default=Path("mull-jobs.sqlite3"))
    parser.add_argument("--host", default="127.0.0.1", help="keep localhost unless you add authentication")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-queued", type=int, default=100)
    parser.add_argument("--max-new-tokens-limit", type=int, default=4096)
    return parser.parse_args()


def _raise_interrupt(signum, frame):
    """Treat SIGTERM like Ctrl-C so `kill` stops the server cleanly and running jobs are kept."""
    raise KeyboardInterrupt


def main():
    args = parse_args()
    from .. import AutoModel

    model = AutoModel.from_pretrained(args.model, memory_percent=args.memory_percent)
    if not hasattr(model, "request_lock"):
        raise SystemExit("The job server supports packed MLX checkpoints (Qwen3, Qwen3.5, gpt-oss, Qwen3 MoE).")
    store = JobStore(args.db, max_queued=args.max_queued)
    worker = Worker(store, PackedModelRunner(model))
    worker.start()
    server = make_server(store, worker, args.host, args.port, args.max_new_tokens_limit)
    print(f"Mull job server on http://{args.host}:{server.server_port}/ (jobs in {args.db})", flush=True)
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
