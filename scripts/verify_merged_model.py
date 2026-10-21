#!/usr/bin/env python3
"""Acceptance test for the merged, published-format model (P5).

Proves two things about `training/merge_lora_for_hub.py`'s output:

1. It loads in a FRESH process with plain `transformers` — no peft, no re-key helper —
   which is the whole point of publishing a merged model instead of an adapter.
2. It reproduces the tuned arm's character *under the benchmark's sampler*, and how
   much of that character is the sampler rather than the weights.  This is why the
   test runs both arms: with the Qwen3.5 card's `presence_penalty=2.0` and without it.

Why the second check exists: `transformers` 5.20 removed `presence_penalty` from
`GenerationConfig`, and llama.cpp applied it in the benchmark.  Dropping it silently
would make the published model look far more verbose than the model that was measured,
so it is implemented here as a `LogitsProcessor` with llama.cpp's semantics (subtract
the penalty from the logits of tokens already present in the sequence).

Run inside the training image, with the merged directory mounted:

    docker run --rm --gpus all -e HF_HOME=/hf-cache \
      -v <hf-cache>:/hf-cache -v /path/to/merged:/merged:ro \
      --entrypoint python3 <training-image> /work/verify_merged_model.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

PROMPTS = [
    "Write a Python function that returns the sum of the squares of the even numbers in a list.",
    "Implement a fibonacci sequence generator.",
]
MAX_NEW_TOKENS = 220
SEED = 43507


class PresencePenalty(LogitsProcessor):
    """llama.cpp semantics: subtract `penalty` from the logits of tokens already present."""

    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        if self.penalty == 0.0:
            return scores
        for row in range(input_ids.shape[0]):
            scores[row, torch.unique(input_ids[row])] -= self.penalty
        return scores


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="/merged", help="merged model directory")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    weights = os.path.join(args.merged, "model.safetensors")
    if os.path.exists(weights):
        print(f"# merged weights   : {os.path.getsize(weights)} bytes")
        print(f"# merged sha256    : {sha256(weights)}")

    tok = AutoTokenizer.from_pretrained(args.merged)
    model = AutoModelForCausalLM.from_pretrained(args.merged, dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.to("cuda")
    print(f"# class            : {type(model).__name__}")
    print(f"# dtype            : {next(model.parameters()).dtype}")
    print(f"# config model_type: {model.config.model_type}")
    print(f"# parameters       : {sum(p.numel() for p in model.parameters()):,}")
    # The claim under test: nothing from the training stack is needed to load this.
    print(f"# peft imported    : {'peft' in sys.modules}   (must be False)")

    def generate(prompt: str, penalty: float):
        torch.manual_seed(SEED)
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        ids = tok(text, return_tensors="pt").to(model.device)
        kw = {"logits_processor": [PresencePenalty(penalty)]} if penalty else {}
        out = model.generate(
            **ids, do_sample=True, temperature=1.0, top_p=1.0, top_k=20, min_p=0.0,
            repetition_penalty=1.0, max_new_tokens=args.max_new_tokens,
            pad_token_id=tok.pad_token_id or tok.eos_token_id, **kw,
        )
        n_new = out.shape[1] - ids["input_ids"].shape[1]
        body = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return n_new, body

    for penalty in (0.0, 2.0):
        print()
        print("=" * 72)
        print(f"# sampler: presence_penalty = {penalty}"
              f"{'  (the benchmark setting)' if penalty else '  (penalty omitted)'}")
        print("=" * 72)
        for prompt in PROMPTS:
            n_new, body = generate(prompt, penalty)
            flat = " ".join(body.split())
            print(f"\n--- {prompt}")
            print(f"--- {n_new} new tokens ---")
            print(body[:600])
            print(f"--- first 110 chars flattened: {flat[:110]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
