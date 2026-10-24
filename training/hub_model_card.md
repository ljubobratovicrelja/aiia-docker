---
base_model: Qwen/Qwen3.5-2B-Base
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
tags:
- text-generation
- lora
- peft
- sft
- code
- qwen3.5
- merged
datasets:
- ise-uiuc/Magicoder-OSS-Instruct-75K
---

# Qwen3.5-2B-Base — LoRA SFT, merged (text-only)

A text-only, standalone build of `Qwen/Qwen3.5-2B-Base` (revision `b1485b2f`,
Apache-2.0) with a LoRA SFT adapter merged into the base weights. It loads with a
plain `AutoModelForCausalLM.from_pretrained(...)`: no `peft`, no adapter
attachment, no re-key helper.

## Why a separate build

The Base repo is a composite vision-language checkpoint
(`Qwen3_5ForConditionalGeneration`: `model.language_model.*` + `model.visual.*` +
`mtp.*`). The text class `Qwen3_5ForCausalLM` expects plain `model.*`, so loading
Base as a text model matches no weights. This repo drops the vision tower and the
MTP head, re-keys the text weights once and folds the LoRA in: 320 tensors,
1 881 825 088 parameters, one 3.5 GiB bf16 shard. Loading it in a fresh process
reports 320/320 tensors, 0 missing, 0 unexpected.

## What it is

| | |
|---|---|
| Base | `Qwen/Qwen3.5-2B-Base` @ `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`, text-only |
| Method | LoRA SFT, `r=16 / alpha=32 / dropout 0.05`, explicit `target_modules` (10 names incl. the Gated DeltaNet projections), bf16, no quantization, then `merge_and_unload()` |
| Corpus | `ise-uiuc/Magicoder-OSS-Instruct-75K` @ `5f839b1f`; 3000 records drawn (seed 43506) from a 20 000-record scan, 2998 usable pairs |
| Schedule | 2 epochs = 374 optimizer steps; effective batch 2 × 8 = 16 sequences of ≤ 1024 tokens; lr 1e-4 cosine, 20 warm-up steps |
| Loss | cross-entropy on answer tokens only (prompt masked with `-100`) |
| Trainable | 15 630 336 parameters, 0.83 % of the text tower |
| Training loss | 0.4070 → 0.3359, per-step minimum 0.1763 |

## What it is not

**Not a coding improvement, and this card does not claim one.** The project's
benchmark (10 fixed prompts × 5 sampled runs, base vs adapter) found:

- **No gain in coding correctness.** On the one code-review prompt with a single
  verifiable answer: base 0/5, tuned 0/5.
- **A style change, not quality.** Replies ~45 % shorter (6 594 vs 11 908
  completion tokens over 50 runs) and no longer hitting the token cap (0/50 vs
  7/50) — but the terseness also dropped docstrings and requested examples.
- **One win**, short arithmetic (`reasoning-1`: 2/5 vs 0/5).
- **One regression**, instruction-following (`general-context-2`: 4/5 tuned
  replies describe the requested email instead of writing it).
- **Perplexity slightly worse** on both held-out corpora (+1.3 % general prose,
  +2.1 % code), the expected cost of a narrow fine-tune.

The length is sampler-sensitive: with `presence_penalty=2.0` the two acceptance
prompts answer in 112 / 116 tokens; without it, 121 / 220. Both benchmark arms
used the penalty, so the comparison is fair — but serve it with the sampler below
to get the measured behaviour.

## Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor

mid = "ljubobratovicrelja/qwen3.5-2b-magicoder-lora"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).eval()


class PresencePenalty(LogitsProcessor):
    """llama.cpp semantics: penalise every token already in the sequence."""

    def __init__(self, penalty):
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        present = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, input_ids, True)
        return scores - present.to(scores.dtype) * self.penalty


text = tok.apply_chat_template(
    [{"role": "user", "content": "Write a Python function that returns the sum of the squares of the even numbers in a list."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**ids, do_sample=True, temperature=1.0, top_p=1.0, top_k=20,
                     min_p=0.0, repetition_penalty=1.0, max_new_tokens=220,
                     logits_processor=[PresencePenalty(2.0)],
                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Provenance

- Base: `Qwen/Qwen3.5-2B-Base` @ `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` — Apache-2.0.
- Dataset: `ise-uiuc/Magicoder-OSS-Instruct-75K` @ `5f839b1f368a76b161028bb9edff055db34022b2`.
- Merged with `transformers 5.20.2`, `torch 2.14.0a0`, `peft 0.21.2`; the merge script and the full benchmark are in the `aiia-docker` repo.
