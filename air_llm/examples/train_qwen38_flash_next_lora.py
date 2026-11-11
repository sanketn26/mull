"""Streamed LoRA on Qwen3.8-Flash-Next (125B MoE + 51B PLE).

Shared-expert / attention / hyper-connection / PLE Linears train; packed-expert
LoRA is off by default (48 layers of 512-expert adapters plus Adam will not
fit). Measured peak is under 6GB at seq 512 (RTX 3060 Ti / 4090). The n-gram
table stays mmap'd.

Needs a transformers build with in-tree ``qwen4_exp``::

    pip install git+https://github.com/huggingface/transformers.git
    python air_llm/examples/train_qwen38_flash_next_lora.py --data my_data.jsonl --save-adapter out.pt
"""

import os
import sys

_AIRLLM = os.path.join(os.path.dirname(__file__), "..")
if _AIRLLM not in sys.path:
    sys.path.insert(0, os.path.abspath(_AIRLLM))

from airllm.airllm_lora import AirLLMLoRAQwen4Exp
from airllm.lora_cli import lora_arg_parser, require_cuda, run_training, trainer_kwargs, training_plan

_FALLBACK = (
    "AirLLM trains LoRA with one decoder layer on the GPU at a time. "
    "Frozen Qwen3.8-Flash-Next experts stay on disk until a layer is needed. "
    "The n-gram PLE table is file-mapped on the host and never copied into VRAM. "
    "Hyper-connections mix four residual streams before the language-model head. "
    "Gated DeltaNet layers use a recurrent state; sparse-attention layers use QSA. "
    "Adam keeps only the adapter matrices resident, about 168 megabytes at rank 16. "
    "A packed MoE layer is five gigabytes of bfloat16; streaming it is the whole trick. "
    "Loss should fall as the adapters fit this short supervised sequence. "
) * 32


def main():
    parser = lora_arg_parser("AirLLM streamed LoRA (Qwen3.8-Flash-Next)", "Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--packed-expert-lora", action="store_true",
                        help="also LoRA the 512 packed experts (VRAM-heavy)")
    parser.add_argument("--keep-original", action="store_true",
                        help="keep the Hugging Face shard files after splitting")
    args = parser.parse_args()
    require_cuda()
    records, epochs, steps = training_plan(args, _FALLBACK)
    trainer = AirLLMLoRAQwen4Exp(args.model, delete_original=not args.keep_original,
                                 packed_experts=args.packed_expert_lora, **trainer_kwargs(args))
    run_training(trainer, records, epochs, steps, args)


if __name__ == "__main__":
    main()
