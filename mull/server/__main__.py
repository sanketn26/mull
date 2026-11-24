"""Run the job server: python -m mull.server --model /path/to/Qwen3-8B-4bit"""

import argparse
import os
import signal
from pathlib import Path

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
                        help='"auto" (default): use free RAM but keep 30%% of total for other apps; or 0-100')
    parser.add_argument("--db", type=Path, default=Path("mull-jobs.sqlite3"))
    parser.add_argument("--host", default="127.0.0.1", help="keep localhost unless you add authentication")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-queued", type=int, default=100)
    parser.add_argument("--max-new-tokens-limit", type=int, default=4096)
    parser.add_argument("--web", nargs="?", const=DEFAULT_WEB, type=Path, metavar="DIR",
                        help=f"also serve the web interface (default folder: {DEFAULT_WEB})")
    args = parser.parse_args()
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

    model = AutoModel.from_pretrained(args.model, memory_percent=args.memory_percent)
    runner = runner_for(model, name=Path(str(args.model)).name)
    if runner is None:
        raise SystemExit("The job server runs MLX models on macOS: packed checkpoints (Qwen3, Qwen3.5, gpt-oss, "
                         "Qwen3 MoE) and Llama/Mistral checkpoints.")
    store = JobStore(args.db, max_queued=args.max_queued)
    worker = Worker(store, runner)
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
