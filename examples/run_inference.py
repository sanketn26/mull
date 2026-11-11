"""Generate a few tokens with Mull. Used by `make run`.

Works on both backends: MLX on macOS (returns the decoded text directly) and the PyTorch
streaming path elsewhere (returns token ids).
"""
import argparse
import time

from mull import AutoModel


def as_chat(tokenizer, prompt):
    """Wrap the prompt in the model's chat template; chat models often stop at once without it."""
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local checkpoint path or Hugging Face repo id")
    parser.add_argument("--prompt", default="What is the capital of the United States?")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=128, help="prompt truncation length")
    parser.add_argument("--compression", choices=["4bit", "8bit"], default=None,
                        help="quantize the stored layers (MLX natively on macOS; bitsandbytes on CUDA)")
    parser.add_argument("--raw", action="store_true",
                        help="send the prompt as-is instead of wrapping it in the model's chat template")
    parser.add_argument("--memory-percent", type=float, default=None,
                        help="share of free memory for sliding adaptive processing (default 25; 0 = off)")
    args = parser.parse_args()

    kwargs = {"compression": args.compression}
    if args.memory_percent is not None:
        kwargs["memory_percent"] = args.memory_percent
    model = AutoModel.from_pretrained(args.model, **kwargs)

    prompt = args.prompt if args.raw else as_chat(model.tokenizer, args.prompt)
    start = time.time()
    if type(model).__name__ == "AirLLMLlamaMlx":
        import mlx.core as mx

        tokens = model.tokenizer([prompt], return_tensors="np", return_attention_mask=False,
                                 truncation=True, max_length=args.max_length, add_special_tokens=args.raw)
        text = model.generate(mx.array(tokens["input_ids"]), max_new_tokens=args.max_new_tokens)
    else:
        tokens = model.tokenizer([prompt], return_tensors="pt", return_attention_mask=False,
                                 truncation=True, max_length=args.max_length, add_special_tokens=args.raw)
        output = model.generate(tokens["input_ids"].to(model.device),
                                max_new_tokens=args.max_new_tokens, use_cache=True,
                                return_dict_in_generate=True)
        text = model.tokenizer.decode(output.sequences[0][tokens["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
    elapsed = time.time() - start

    n = max(1, len(model.tokenizer.encode(text, add_special_tokens=False)))
    print(text)
    print(f"\n{n} new tokens in {elapsed:.1f}s ({elapsed / n:.2f}s/token)")


if __name__ == "__main__":
    main()
