"""Generate a few tokens with AirLLM. Used by `make run`.

Works on both backends: MLX on macOS (returns the decoded text directly) and the PyTorch
streaming path elsewhere (returns token ids).
"""
import argparse
import time

from airllm import AutoModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local checkpoint path or Hugging Face repo id")
    parser.add_argument("--prompt", default="What is the capital of the United States?")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=128, help="prompt truncation length")
    parser.add_argument("--compression", choices=["4bit", "8bit"], default=None,
                        help="quantize the stored layers (MLX natively on macOS; bitsandbytes on CUDA)")
    parser.add_argument("--memory-percent", type=float, default=None,
                        help="share of free memory for sliding adaptive processing (default 25; 0 = off)")
    args = parser.parse_args()

    kwargs = {"compression": args.compression}
    if args.memory_percent is not None:
        kwargs["memory_percent"] = args.memory_percent
    model = AutoModel.from_pretrained(args.model, **kwargs)

    start = time.time()
    if type(model).__name__ == "AirLLMLlamaMlx":
        import mlx.core as mx

        tokens = model.tokenizer([args.prompt], return_tensors="np", return_attention_mask=False,
                                 truncation=True, max_length=args.max_length)
        text = model.generate(mx.array(tokens["input_ids"]), max_new_tokens=args.max_new_tokens)
    else:
        tokens = model.tokenizer([args.prompt], return_tensors="pt", return_attention_mask=False,
                                 truncation=True, max_length=args.max_length)
        output = model.generate(tokens["input_ids"].to(model.device),
                                max_new_tokens=args.max_new_tokens, use_cache=True,
                                return_dict_in_generate=True)
        text = model.tokenizer.decode(output.sequences[0])
    elapsed = time.time() - start

    print(text)
    print(f"\n{args.max_new_tokens} new tokens in {elapsed:.1f}s "
          f"({elapsed / args.max_new_tokens:.2f}s/token)")


if __name__ == "__main__":
    main()
