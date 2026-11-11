"""Shared command-line plumbing for the streamed LoRA example scripts."""
import argparse
import sys
import time

import torch

from .lora_data import encode_record, iter_train_items, load_records


def lora_arg_parser(description, default_model):
    """Arguments every streamed LoRA script takes; add model-specific ones to the result."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", default=None,
                   help="JSONL/JSON/TXT dataset (see README Training). Omit to overfit a built-in snippet.")
    p.add_argument("--model", default=default_model)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=None,
                   help="stop after N steps (default: all examples × epochs; 1 if --data is omitted)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ce-chunk-size", type=int, default=4096)
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="shard/runtime cap; defaults to --seq-len")
    p.add_argument("--save-adapter", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-csv", default=None, help="write step,loss,peak_vram_gb,elapsed_s")
    return p


def require_cuda():
    if not torch.cuda.is_available():
        print("CUDA is required for this trainer.")
        sys.exit(1)


def training_plan(args, fallback_text):
    """(records, epochs, steps): the dataset, or a built-in snippet to overfit when none is given."""
    if args.data:
        return load_records(args.data), args.epochs, args.steps
    return [{"text": fallback_text, "mask_prefix": None}], (args.steps or 1), None


def trainer_kwargs(args):
    return dict(device=args.device, max_seq_len=args.max_seq_len or args.seq_len, lora_r=args.lora_r,
                lora_alpha=args.lora_alpha, lr=args.lr, ce_chunk_size=args.ce_chunk_size)


def run_training(trainer, records, epochs, steps, args):
    """Train, log each step (and to --log-csv), then save the adapter if asked."""
    tok = trainer.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"examples={len(records)} epochs={epochs} seq_len={args.seq_len}", flush=True)
    csv_f = open(args.log_csv, "w", encoding="utf-8") if args.log_csv else None
    if csv_f:
        csv_f.write("step,loss,peak_vram_gb,elapsed_s\n")
    torch.cuda.reset_peak_memory_stats(trainer.device)
    t0, n = time.time(), 0
    try:
        for step, epoch, rec in iter_train_items(records, epochs, steps):
            input_ids, attention_mask, labels = encode_record(tok, rec, args.seq_len, trainer.device)
            loss = trainer.train_step(input_ids, labels=labels, attention_mask=attention_mask,
                                      verbose=args.verbose)
            peak, elapsed = torch.cuda.max_memory_allocated(trainer.device) / 1024 ** 3, time.time() - t0
            print(f"step {step} epoch {epoch} loss={loss:.6f} peak_vram={peak:.2f}GB "
                  f"seq={input_ids.shape[1]} elapsed_s={elapsed:.1f}", flush=True)
            if csv_f:
                csv_f.write(f"{step},{loss:.8f},{peak:.4f},{elapsed:.2f}\n")
                csv_f.flush()
            n += 1
    finally:
        if csv_f:
            csv_f.close()
    if n == 0:
        print("no training steps ran")
        sys.exit(1)
    if args.save_adapter:
        trainer.save_adapter(args.save_adapter)
        print(f"saved adapter to {args.save_adapter}")
