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

# Qwen3.5-2B-Base — stage-4 LoRA SFT, merged (text-only)

This is a **text-only, standalone** build of `Qwen/Qwen3.5-2B-Base` (revision
`b1485b2f`, Apache-2.0) with a stage-4 LoRA SFT adapter **merged into the base
weights**. It loads with a plain `AutoModelForCausalLM.from_pretrained(...)`:
no `peft`, no adapter attachment, and no re-key helper.

## Why a separate, text-only build

`Qwen/Qwen3.5-2B-Base` is published as a **composite vision-language**
checkpoint (`Qwen3_5ForConditionalGeneration`, weight names
`model.language_model.*` + `model.visual.*` + `mtp.*`). The text-only class
`Qwen3_5ForCausalLM` expects plain `model.*`, so a plain `from_pretrained` of the
Base repo matches **zero** weights and reports every key as missing.

The adapter here was trained against the text-only layout, which means the same
problem blocks the composite repo as a base for the adapter. This model removes
it: the vision tower and the multi-token-prediction head are dropped, the text
weights are re-keyed once, the LoRA is folded in, and what you get is a plain
text causal LM. 320 tensors, 1 881 825 088 parameters, `model_type:
qwen3_5_text`, single `model.safetensors` shard (3.51 GiB, bfloat16).

Verified in a fresh process: loading this directory with plain
`AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)` reports
**320/320 tensors loaded, 0 missing, 0 unexpected, 0 mismatched** keys — with
`peft` never imported. The merge is deterministic: a repeat run produced a
byte-identical `model.safetensors`
(sha256 `f8afd3b04fac0023dc9f99890e62e94f662526c95f44c3e1606eb284c03d90d8`,
3 763 687 248 bytes).

## What it is

| | |
|---|---|
| Base | `Qwen/Qwen3.5-2B-Base` @ `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`, text-only |
| Method | LoRA SFT, `r=16 / alpha=32 / dropout 0.05`, explicit `target_modules` (10 names incl. the Gated DeltaNet projections), bf16, **no** quantization — then merged with `merge_and_unload()` |
| Corpus | `ise-uiuc/Magicoder-OSS-Instruct-75K` @ `5f839b1f`; 3000 records drawn (seeded, seed 43506) from a 20 000-record scan → 2998 usable pairs |
| Schedule | 2 epochs = **374 optimizer steps**; effective batch 2 × 8 = 16 sequences of ≤ 1024 tokens; lr 1e-4 cosine, 20 warm-up steps |
| Loss | cross-entropy on **answer tokens only** (prompt masked with `-100`); prompts rendered with the tokenizer's own chat template |
| Trainable | 15 630 336 parameters = 0.82 % of 1.897 B |
| Training loss | 0.4070 → 0.3359, per-step minimum 0.1763 |

## What it is **not**

**It is not a coding improvement, and this card does not claim one.** The
project's own benchmark ([`docs/BENCHMARK_REPORT.md`](../docs/BENCHMARK_REPORT.md)
in the `aiia-docker` repo) compared base vs adapter on 10 fixed prompts × 5
sampled runs and found:

- **No demonstrated gain in coding correctness.** On the one code-review prompt
  with a single verifiable answer, base **0/5**, tuned **0/5**. The adapter was
  trained to write and review code and did not make either task reliably better.
- **A consistent style/length change, not quality.** Tuned replies were ~**45 %
  shorter** (6 594 vs 11 908 completion tokens over 50 runs) and stopped hitting
  the token cap (0/50 vs 7/50). Shorter is not better: the terseness also dropped
  docstrings and requested examples.
  **The brevity is sampler-sensitive**, which matters if you reproduce it: with the
  card's `presence_penalty=2.0` applied, this merged model answers the two acceptance
  prompts in **112 / 116 tokens**; with the penalty omitted, the same weights answer in
  **121 / 220 tokens**, the second hitting the cap with a docstring and a trailing prose
  explanation. Both arms of the benchmark had the penalty applied, so the comparison is
  fair — but do not read the shortness as a property of the weights alone. Serve it with
  the sampler below to get the behaviour that was measured.
- **One win on short arithmetic** (`reasoning-1`: 2/5 correct vs base 0/5 — but
  wrong in the other three runs).
- **One regression on instruction-following** (`general-context-2`: 4/5 tuned
  replies *describe* the requested email instead of writing it; base does not do
  this).
- **Held-out perplexity slightly worse** than base on both corpora (+1.3 %
  general prose, +2.1 % code) — the expected cost of a narrow fine-tune.

The one-line honest verdict from that report: *the adapter mostly changed style,
not correctness; it helped on short arithmetic and hurt on instruction-following,
and it did not solve the coding task it was trained for.* Read the report before
using this model, and note it measures 2B-scale behaviour on this adapter, not
LoRA fine-tuning in general.

## Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor

mid = "<this model>"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).eval()


class PresencePenalty(LogitsProcessor):
    """llama.cpp semantics: penalise every token already present in the sequence.

    `transformers` 5.20 dropped the built-in `presence_penalty`, so the sampling
    setting this model was benchmarked with has to be supplied explicitly.
    """

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

## Reproduce the merge

The merge is one script, `training/merge_lora_for_hub.py` in the `aiia-docker`
repo. It **reuses `train_lora.py::load_text_model` verbatim** for the composite →
text-only re-key (re-implementing that hop is how the two layouts drift apart),
attaches the adapter with `peft`, calls `merge_and_unload()`, saves in bfloat16,
and writes `merge.json` recording base repo + revision, adapter path + sha256,
resolved library versions, dtype, parameter count, and the sha256 + byte size of
the produced safetensors. It hard-errors if the re-key leaves a non-empty
`missing`/`unexpected` set, and again if the merge leaves the probed weight
unchanged.

```bash
# on a GPU host with Docker + the NVIDIA container runtime, from a copy of the repo
export HOME=/var/tmp/train-home DOCKER_CONFIG=/var/tmp/train-home/.docker   # host $HOME may be read-only
export HF_HOME=/var/tmp/aiia-hf-cache                                       # pinned base already cached

docker build -f training/Dockerfile -t aiia-docker/training:stage-4 training/

docker run --rm --gpus all -e HF_HOME=/hf-cache \
  -v "$HF_HOME":/hf-cache \
  -v "$PWD/training":/work \
  -v /var/tmp:/var/tmp \
  --entrypoint python3 aiia-docker/training:stage-4 \
  /work/merge_lora_for_hub.py \
    --adapter /var/tmp/aiia-fullrun/adapter \
    --output-dir /var/tmp/aiia-hub-merged \
    --device cpu
```

Notes for reproducing it exactly:

- `--gpus all` is required even though the merge itself runs on `--device cpu`:
  importing `peft` pulls in `transformer_engine`, which `dlopen`s `libcuda.so.1`,
  so a container without the driver library fails at `import peft`.
- The trainer's adapter directory is `training/out/adapter` in the repo
  (`adapter_model.safetensors`, 62 561 688 B). The container writes its output as
  `root`; copy it out with `docker cp` or make it world-readable.
- The produced model has `use_cache: true` (the trainer ran with it off — a
  training-only setting); that is the one config field the merge script changes.

## Provenance

- Base: `Qwen/Qwen3.5-2B-Base` @ `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` — Apache-2.0.
- Dataset: `ise-uiuc/Magicoder-OSS-Instruct-75K` @ `5f839b1f368a76b161028bb9edff055db34022b2`.
- Merged with `transformers 5.20.2`, `torch 2.14.0a0`, `peft 0.21.2`.
- Benchmark and reading: `docs/BENCHMARK_REPORT.md`; training record:
  `docs/TRAINING_REPORT.md` — both in the `aiia-docker` repo.
