"""Opt-in chat-template and Harmony stop-boundary test for the pinned local checkpoint."""

import os
import sys

import pytest

if sys.platform != "darwin":
    pytest.skip("Apple Silicon MLX runtime required", allow_module_level=True)


def test_gpt_oss_real_chat(monkeypatch):
    checkpoint = os.environ.get("MULL_GPT_OSS_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set MULL_GPT_OSS_CHECKPOINT to the pinned local snapshot")

    from mull import AutoModel, airllm_packed_mlx

    model = AutoModel.from_pretrained(checkpoint, memory_percent=0)
    messages = [{"role": "user", "content": "What is 2 + 2? Reply using only the digit."}]
    prompt = model.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, reasoning_effort="low",
    )
    assert model.tokenizer.decode(prompt).endswith("<|start|>assistant")
    original = airllm_packed_mlx.sample
    sampled = []

    def record(logits, temperature):
        token = original(logits, temperature)
        sampled.append(token.item())
        return token

    monkeypatch.setattr(airllm_packed_mlx, "sample", record)
    output = model.generate([prompt], max_new_tokens=128)
    assert sampled[-1] in model.eos
    assert not model.eos.intersection(sampled[:-1])
    assert output == model.tokenizer.decode(sampled[:-1])
    assert output.rsplit("<|channel|>final<|message|>", 1)[-1].strip() == "4"
    assert "<|channel|>final<|message|>" in output
    print(f"prompt_tokens={len(prompt)} sampled_tokens={len(sampled)} eos={sampled[-1]}")
