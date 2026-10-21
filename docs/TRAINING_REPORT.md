# P4 — LoRA training report

The training stage of the assignment ("train a model for a few epochs"), which also carries the
max-grade bullet's training half. Everything below is measured; the numbers come from
[`evidence/train-run.txt`](../evidence/train-run.txt) (the full run) and
[`evidence/train-smoke.txt`](../evidence/train-smoke.txt) (the 10-step smoke run), and the figure is
[`evidence/loss-curve.png`](../evidence/loss-curve.png).

## What was trained, and on what

| | |
|---|---|
| Base model | `Qwen/Qwen3.5-2B-Base`, pinned at revision `b1485b2f…`, loaded **text-only** (composite checkpoint re-keyed: 320 tensors loaded, 312 dropped, `missing: []`) |
| Method | LoRA SFT, `r=16 / alpha=32 / dropout 0.05`, explicit `target_modules` (10 names incl. the GDN projections), bf16, **no** quantization |
| Corpus | `ise-uiuc/Magicoder-OSS-Instruct-75K`, pinned at revision `5f839b1f…`; 3000 records drawn (seeded) from a 20 000-record scan — **2998 usable** after tokenisation |
| Schedule | 2 epochs = **374 optimizer steps**; effective batch 2 × 8 = 16 sequences of ≤ 1024 tokens; lr 1e-4 cosine with 20 warm-up steps; seed 43506 |
| Loss | cross-entropy on **answer tokens only** (prompt masked with `-100`), prompt rendered with the tokenizer's own chat template |
| Hardware | RTX 3090 Ti 24 GB, off the serving host (reason in [`PROJECT_PLAN.md`](PROJECT_PLAN.md) P4 row) |
| Image | `aiia-docker/training:stage-4`, 38 248 955 946 B — a third, profile-gated Compose service; the serving stack is untouched |

## Result

| Measurement | Value |
|---|---|
| Outcome | **exit 0, `OOMKilled: false`**, all 374 steps logged |
| Wall clock | **1801 s (~30 min)** end to end including load; **4.76 s per optimizer step** measured between the first and last logged step (373 intervals) |
| Throughput | **2071.6 padded tokens/s** median, flat across the run |
| Loss | **0.4070 → 0.3359** first→last, per-step **minimum 0.1763** at step 293 |
| Trainable parameters | **15 630 336** = 0.82 % of 1.897 B |
| Adapter on disk | **62 561 688 B** (peft stores LoRA parameters in fp32) |
| Device memory | **13 249 MiB** peak in the sampled half; the smoke run, sampled from step 1, showed **18 063 MiB** in its device table (its own sampler peaked at 17 905 MiB) → **~18 GB of 24 GB, ~6 GB headroom** |
| Container | 2.1 GiB RSS, 99.98 % CPU, GPU at 100 % utilisation while stepping |

Loss trajectory by interval (from the record's summary — the shape matters more than the endpoints):

| Steps | Mean loss | Steps | Mean loss |
|---|---|---|---|
| 1–25 | 0.4071 | 201–225 | 0.2676 |
| 26–50 | 0.3465 | 226–250 | 0.2762 |
| 51–75 | 0.3394 | 251–275 | 0.2595 |
| 76–100 | 0.3256 | 276–300 | 0.2556 |
| 101–125 | 0.3253 | 301–325 | 0.2637 |
| 126–150 | 0.3329 | 326–350 | 0.2743 |
| 151–175 | 0.3286 | 351–374 | 0.2713 |
| 176–200 | 0.2971 | | |

## The figure, and how to read it

`evidence/loss-curve.png` has three panels over a shared x axis:

1. **Loss** — per-step (light) and the mean of the last 18 steps (dark). It rises to ~0.48 during
   the 20-step warm-up, falls to ~0.33 by step 50, plateaus there to ~step 175, drops again to ~0.26
   by step 300, and then flattens at ~0.27 for the last 75 steps while the LR decays to zero.
2. **Learning rate** — linear warm-up to 1e-4 at step 20, cosine decay to 0 at step 374.
3. **Padded tokens/s** — flat at ~2070 for the whole run.

The second and third panels are not decoration: they are what distinguishes "the schedule worked"
from "the schedule stalled while throughput collapsed". Neither happened here.

## What this does **not** show

**The loss curve is not evidence that the adapter is better.** The base model's loss on this corpus
is already ~0.41 at step 1 — the median answer is only 234 supervised code tokens of 524, and the
corpus is formulaic — so a 0.07 drop in mean loss is a schedule behaving, not a capability gained. The
discriminating measurement for this bullet is plan §6.1 R4's two metrics (`ast.parse` adherence and a
HumanEval pass@1 slice) measured **base vs adapter**. That comparison is deliberately a separate task
and is not claimed here.

## Reproduce

```bash
# on the GPU host, from a copy of this repo
export TRAIN_DOCKER_HOME=/var/tmp/train-home
export TRAIN_HF_CACHE=/var/tmp/aiia-hf-cache
export TRAIN_OUT=/var/tmp/aiia-fullrun

bash scripts/train_lora.sh up          # build if needed, start detached
bash scripts/train_lora.sh tail        # live: the JSONL loss log
bash scripts/train_lora.sh status      # container state + last logged step

TRAIN_RUN_DIR=$TRAIN_OUT bash scripts/train_record.sh    # -> evidence/train-run.txt
```

```bash
# on the workstation: the figure, from the same JSONL
scp <host>:$TRAIN_OUT/loss.jsonl training/out/loss.jsonl
python3 scripts/plot_loss.py --log training/out/loss.jsonl --out evidence/loss-curve.png \
    --run-json training/out/adapter/run.json
```

The adapter is **not** committed (62.5 MB binary). It is written by the container's root user into the
mounted output directory, so retrieving it from the host needs `docker cp` rather than a plain copy:

```bash
docker cp aiia-docker-training-1:/out/adapter/. ./training/out/adapter/
```
