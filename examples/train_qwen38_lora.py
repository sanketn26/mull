"""Streamed LoRA on Qwen3.8-27B.

v1 is text-only, batch=1. Start at ``--seq-len 512`` on a 4GB card; 2048 is the
8GB target. This is not Hugging Face Trainer / bitsandbytes QLoRA.

Example::

    python examples/train_qwen38_lora.py --data my_data.jsonl --save-adapter out.pt
"""

import os
import sys

_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_ROOT))

from mull.airllm_lora import AirLLMLoRA
from mull.lora_cli import lora_arg_parser, require_cuda, run_training, trainer_kwargs, training_plan

_FALLBACK = "AirLLM trains LoRA with one decoder layer on the GPU at a time. " * 256


def main():
    args = lora_arg_parser("AirLLM streamed LoRA (Qwen3.8-27B)", "Qwen/Qwen3.8-27B").parse_args()
    require_cuda()
    records, epochs, steps = training_plan(args, _FALLBACK)
    trainer = AirLLMLoRA(args.model, **trainer_kwargs(args))
    run_training(trainer, records, epochs, steps, args)


if __name__ == "__main__":
    main()
