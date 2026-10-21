#!/usr/bin/env python3
"""LoRA SFT of the served Qwen3.5-2B base model — stage P4 of aiia-docker.

This runs *off* the serving path: it produces a peft adapter directory, nothing
else in the stack changes.  The hyperparameters below are the frozen run
configuration (documented in docs/PROJECT_PLAN.md §P4); scripts/train_lora.sh
launches it and only overrides the values a smoke run has to shrink.

Two decisions are worth stating up front, because both are deviations from the
first sketch in the plan and both are visible in this file:

* **No `trl.SFTTrainer`.**  A plain torch loop over the same `peft` adapters is
  used instead.  The plan itself flagged SFTTrainer as a risk on this checkpoint
  ("fall back to a hand-rolled peft loop if it fights"), and it does fight: the
  checkpoint is a composite VLM whose processor is the documented
  `processing_class`, `sample_packing` is silently ignored for this family, and
  the requirement here is a *per-step JSONL loss log* an external monitor can
  `tail -f` — which is a training-loop concern, not a trainer concern.  The loop
  below is ~60 lines and has no API churn to survive.

* **The model is loaded text-only by hand, not with `from_pretrained`.**  The
  `-Base` repo carries the composite layout (`Qwen3_5ForConditionalGeneration`,
  keys `model.language_model.*` + `model.visual.*` + `mtp.*`), while the
  text-only class `Qwen3_5ForCausalLM` expects `model.layers.*`
  (plan risk R1: "train and load the adapter against ONE layout").  A plain
  `from_pretrained` would therefore match *no* weight, report every key as
  missing and quietly train a randomly initialised model.  So the state dict is
  read and re-keyed explicitly, and a non-empty `missing` set is a hard error
  rather than a warning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch

# --------------------------------------------------------------------------
# Frozen run configuration.  These defaults ARE the committed config; the
# launcher passes nothing for them, a smoke run overrides a few.
# --------------------------------------------------------------------------
DEFAULTS = dict(
    model_repo="Qwen/Qwen3.5-2B-Base",
    # Pinned the same way inference/Dockerfile pins the GGUF: by revision, so a
    # later push to the repo cannot move the base under a finished adapter.
    model_revision="b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
    dataset_repo="ise-uiuc/Magicoder-OSS-Instruct-75K",
    dataset_revision="5f839b1f368a76b161028bb9edff055db34022b2",
    dataset_file="data-oss_instruct-decontaminated.jsonl",
    # Corpus slice: 3000 records drawn from the first 20000 (plan §6.1 R4).
    # The draw is seeded, not "the first 3000": the file is ordered by language,
    # so head-slicing would bias the adapter towards whichever language leads.
    train_size=3000,
    scan_limit=20000,
    epochs=2,
    # 1024 tokens covers the great majority of problem+solution pairs in this
    # corpus; the effective batch is therefore 2 x 8 = 16 sequences = ~16k
    # tokens per optimizer step, which is a normal SFT budget for a 2B.
    max_seq_len=1024,
    batch_size=2,
    grad_accum=8,
    lr=1e-4,
    warmup_steps=20,
    seed=43506,
    # Explicit target list, never `all-linear` (plan §6.1 R1): `all-linear` would
    # reach into the vision tower, silently skip names that do not match, and
    # behave differently on aarch64.  The GDN projections are named with their
    # parent module because `out_proj` alone is ambiguous here.
    target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,"
    "linear_attn.in_proj_qkv,linear_attn.in_proj_z,linear_attn.out_proj",
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    log_every=1,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for key, val in DEFAULTS.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            p.add_argument(flag, action="store_true", default=val)
        else:
            p.add_argument(flag, type=type(val), default=val)
    p.add_argument("--output-dir", default="/out/adapter", help="peft adapter destination")
    p.add_argument("--loss-log", default="/out/loss.jsonl", help="JSONL loss log (flushed per line)")
    p.add_argument("--max-steps", type=int, default=0, help="optimizer steps; 0 = full epochs. A smoke run sets 10")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def load_text_model(repo, revision, device):
    """Load the pinned `-Base` checkpoint text-only, re-keyed to `model.*`.

    Returns (model, tokenizer, report dict).  The report is printed and written
    into the output directory: it is the proof that the base weights actually
    landed in the text tower (the failure mode this guards against is silent).
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    cfg_path = hf_hub_download(repo, "config.json", revision=revision)
    idx_path = hf_hub_download(repo, "model.safetensors.index.json", revision=revision)
    full_cfg = json.load(open(cfg_path))
    index = json.load(open(idx_path))["weight_map"]

    # text_config is the whole architecture; dropping vision_config is the point
    # of loading text-only (plan D6) — the vision tower is ~300 of the 632
    # tensors and none of it is wanted in the training graph.
    text_cfg = Qwen3_5TextConfig(**full_cfg["text_config"])
    if text_cfg.model_type != "qwen3_5_text":
        raise SystemExit(f"unexpected text model_type {text_cfg.model_type!r}")

    # One file, but resolve the list rather than hard-coding the name.
    shards = sorted({os.path.basename(f) for f in index.values()})
    weights = {}
    for shard in shards:
        weights.update(load_file(hf_hub_download(repo, shard, revision=revision)))

    # The layout hop.  `model.language_model.<rest>` -> `model.<rest>`; `mtp.*`
    # (the multi-token-prediction head) and `model.visual.*` are dropped, which
    # is exactly what the shipped class declares it ignores on load.
    rekeyed, dropped = {}, []
    for key, tensor in weights.items():
        if key.startswith("model.language_model."):
            rekeyed["model." + key[len("model.language_model.") :]] = tensor
        elif key.startswith(("mtp.", "model.visual.")):
            dropped.append(key)
        else:
            rekeyed[key] = tensor
    del weights

    n_rekeyed = len(rekeyed)
    model = Qwen3_5ForCausalLM(text_cfg)
    missing, unexpected = model.load_state_dict(rekeyed, strict=False)
    del rekeyed  # the 4.5 GB of checkpoint tensors are copied into the params now
    # `lm_head.weight` is tied to the embedding and is legitimately absent from
    # the checkpoint; every other missing name means a layout mistake.
    missing = [m for m in missing if m != "lm_head.weight"]
    if missing or unexpected:
        raise SystemExit(
            f"checkpoint/re-key mismatch — {len(missing)} missing, "
            f"{len(unexpected)} unexpected (first missing: {missing[:5]})"
        )
    model.tie_weights()

    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    report = {
        "layout": "composite -> text-only",
        "text_config.model_type": text_cfg.model_type,
        "tensors_loaded": n_rekeyed,
        "tensors_dropped": len(dropped),
        "dropped_prefixes": sorted({k.split(".")[0] for k in dropped}),
        "missing": missing,
        "unexpected": unexpected,
        "base_parameters": sum(p.numel() for p in model.parameters()),
    }
    model.to(dtype=torch.bfloat16)
    model.config.use_cache = False
    return model, tokenizer, report


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def read_records(repo, revision, filename, scan_limit):
    """Stream the corpus from the HF cache and yield up to `scan_limit` records.

    The file is a single 194 MiB JSONL, not the parquet the plan assumed — see
    JOURNAL (2026-10-20).  It is streamed line by line so the process never
    holds the whole corpus.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if n >= scan_limit:
                return


def render_prefix(tokenizer, problem, chat_template: bool):
    """The prompt side of one training pair, in the served format.

    The corpus carries `problem` / `solution` and no template of its own, so the
    model's own chat template is used: the GGUF we serve carries the same
    template, and rendering the corpus any other way would train on a format the
    server never sees at inference time.
    """
    if chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"


def build_samples(records, tokenizer, max_seq_len):
    """Tokenize and split prompt/answer.  Loss is taken on the answer only.

    Masking the prompt is not cosmetic: the corpus `problem` field is long, and
    without the mask most of the gradient would go into learning to reproduce
    the dataset's phrasing of problems.
    """
    eos = tokenizer.eos_token_id
    # The template is chosen ONCE, for the whole corpus: falling back per record
    # would train on two different prompt formats within one run.
    chat_template = True
    try:
        render_prefix(tokenizer, "probe", True)
    except Exception as exc:
        chat_template = False
        print(f"WARNING: chat template unusable ({exc.__class__.__name__}: {exc}); "
              "falling back to the literal ChatML prefix", flush=True)
    samples, stats = [], {"too_long": 0, "degenerate": 0}
    for rec in records:
        problem = (rec.get("problem") or "").strip()
        solution = (rec.get("solution") or "").strip()
        if not problem or not solution:
            stats["degenerate"] += 1
            continue
        prefix = render_prefix(tokenizer, problem, chat_template)
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        if len(prefix_ids) >= max_seq_len - 8:
            stats["too_long"] += 1
            continue
        answer_ids = tokenizer(solution, add_special_tokens=False)["input_ids"] + [eos]
        answer_ids = answer_ids[: max_seq_len - len(prefix_ids)]
        ids = prefix_ids + answer_ids
        samples.append((ids, [-100] * len(prefix_ids) + answer_ids))
    stats["used_chat_template"] = chat_template
    return samples, stats


def make_batch(samples, indices, pad_id, device):
    """Right-pad to the longest sequence in the micro-batch."""
    picked = [samples[i] for i in indices]
    width = max(len(ids) for ids, _ in picked)
    input_ids, labels, mask = [], [], []
    for ids, labs in picked:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(labs + [-100] * pad)
        mask.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(input_ids, device=device),
        torch.tensor(labels, device=device),
        torch.tensor(mask, device=device),
    )


def main(argv=None):
    args = parse_args(argv)
    t_run = time.time()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device.startswith("cuda"):
        # The seed fixes the data order and the LoRA initialisation.  Bitwise
        # reproducibility of the forward pass is NOT claimed — stage 3 measured
        # the same request giving different completions on two backends, for the
        # same reason (bf16 arithmetic is not associative).
        torch.cuda.manual_seed_all(args.seed)

    print(f"# device: {args.device} "
          f"({torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'cpu'})", flush=True)

    model, tokenizer, load_report = load_text_model(
        args.model_repo, args.model_revision, args.device
    )
    print(f"# base loaded: {load_report['base_parameters']/1e6:.1f} M params, "
          f"{load_report['tensors_loaded']} tensors re-keyed, "
          f"{load_report['tensors_dropped']} dropped "
          f"({','.join(load_report['dropped_prefixes'])})", flush=True)

    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
    )
    model = get_peft_model(model, lora)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    # A `target_modules` typo is silent in peft — the model simply stays frozen.
    # So the count is asserted, not printed.
    if n_train == 0:
        raise SystemExit("no trainable parameters: check --target-modules")
    model.print_trainable_parameters()
    print(f"# adapter attached to target_modules: {args.target_modules}", flush=True)

    # Gradient checkpointing: a 2B in bf16 fits without it, but the recompute is
    # what keeps the *sequence length* (not the batch) the tunable that measures
    # against the 3090 Ti's 24 GB.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    records = list(read_records(args.dataset_repo, args.dataset_revision,
                               args.dataset_file, args.scan_limit))
    if len(records) < args.train_size:
        raise SystemExit(f"corpus scan produced {len(records)} records, need {args.train_size}")
    drawn = random.Random(args.seed).sample(range(len(records)), args.train_size)
    drawn.sort()
    samples, data_stats = build_samples([records[i] for i in drawn], tokenizer, args.max_seq_len)
    if len(samples) < args.batch_size:
        raise SystemExit(f"only {len(samples)} usable samples after tokenisation")
    print(f"# corpus: {args.dataset_repo}@{args.dataset_revision[:8]} "
          f"{len(records)} scanned -> {len(samples)} usable samples "
          f"(too_long={data_stats['too_long']}, degenerate={data_stats['degenerate']}, "
          f"chat_template={data_stats['used_chat_template']})", flush=True)

    model.to(args.device)
    model.train()

    pad_id = tokenizer.pad_token_id
    accum = max(1, args.grad_accum)
    per_epoch = max(1, len(samples) // args.batch_size)
    steps_per_epoch = max(1, per_epoch // accum)
    total_steps = args.max_steps or (steps_per_epoch * args.epochs)
    print(f"# schedule: {steps_per_epoch} steps/epoch, {args.epochs} epochs, "
          f"running {total_steps} steps; effective batch = {args.batch_size} x {accum} = "
          f"{args.batch_size * accum} sequences of <= {args.max_seq_len} tokens", flush=True)

    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    optim = AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    os.makedirs(os.path.dirname(args.loss_log) or ".", exist_ok=True)
    # Truncated, not appended: the file is the record of THIS container run, and
    # appending two runs into one JSONL would silently concatenate two schedules.
    log_fh = open(args.loss_log, "w", encoding="utf-8")

    step = 0
    micro = 0
    running = 0.0
    step_tokens = 0
    t_step = time.time()
    epoch = 0
    order = list(range(len(samples)))
    while step < total_steps:
        if epoch >= args.epochs and not args.max_steps:
            break
        random.Random(args.seed + epoch).shuffle(order)
        epoch += 1
        for start in range(0, len(order) - args.batch_size + 1, args.batch_size):
            if step >= total_steps:
                break
            blocks = order[start : start + args.batch_size]
            input_ids, labels, mask = make_batch(samples, blocks, pad_id, args.device)
            # Counted after padding: the pads are tokens the GPU really computes.
            step_tokens += int(input_ids.numel())
            out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            (out.loss / accum).backward()
            running += out.loss.item()
            micro += 1
            if micro % accum:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            elapsed = time.time() - t_step
            lr = optim.param_groups[0]["lr"]
            loss = running / accum
            running = 0.0

            if step % max(1, args.log_every) == 0 or step == total_steps:
                rec = {
                    "step": step,
                    "loss": round(loss, 6),
                    "lr": round(lr, 10),
                    "tokens_s": round(step_tokens / elapsed, 2),
                    "elapsed_s": round(time.time() - t_run, 3),
                }
                # one JSON object per line, flushed immediately, so an external
                # `tail -f` on the mounted file is a live second monitor next to
                # `docker logs` — which is what the full run is watched with.
                log_fh.write(json.dumps(rec) + "\n")
                log_fh.flush()
                print(f"  step {step:>4}/{total_steps}  loss {loss:8.4f}  lr {lr:.3e}  "
                      f"{rec['tokens_s']:8.1f} tok/s  {elapsed:5.2f} s/step  "
                      f"elapsed {rec['elapsed_s']:.0f}s", flush=True)
            step_tokens = 0
            t_step = time.time()

    log_fh.close()

    peak = (torch.cuda.max_memory_allocated() / 2**30) if args.device.startswith("cuda") else 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    summary = {
        "run_config": vars(args),
        "load_report": load_report,
        "trainable_parameters": n_train,
        "samples": len(samples),
        "data_stats": data_stats,
        "steps_run": step,
        "effective_batch_sequences": args.batch_size * accum,
        "max_seq_len": args.max_seq_len,
        "peak_vram_gib": round(peak, 3),
        "wall_clock_s": round(time.time() - t_run, 1),
        "loss_log": os.path.abspath(args.loss_log),
    }
    with open(os.path.join(args.output_dir, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(f"# adapter written to {args.output_dir} "
          f"(peak VRAM {peak:.2f} GiB, wall clock {summary['wall_clock_s']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
