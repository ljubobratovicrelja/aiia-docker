#!/usr/bin/env python3
"""Merge the stage-4 LoRA adapter into a STANDALONE, TEXT-ONLY Qwen3.5-2B.

Why this file exists
--------------------
The published artefact cannot be the peft adapter attached to the Hub `-Base`
checkpoint, because that checkpoint is a **composite VLM**: its weight names are
`model.language_model.*` plus `model.visual.*` (the vision tower) and `mtp.*`
(the multi-token-prediction head), while the text-only class
`Qwen3_5ForCausalLM` expects plain `model.*`.  A plain
`AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B-Base")` therefore matches
**zero** weights and reports every one of them missing — which is exactly why
`train_lora.py::load_text_model` loads the state dict by hand, re-keys
`model.language_model.*` -> `model.*`, drops the vision/mtp tensors, and hard-
errors on a mismatch.

The consequence for a consumer is that the adapter's weights are keyed for the
TEXT-ONLY layout, so `PeftModel.from_pretrained` on top of the *composite* repo
cannot load them either.  This script produces the thing a consumer can actually
use: one directory holding a text-only causal-LM with the adapter already folded
into the base weights, loadable with a plain

    AutoModelForCausalLM.from_pretrained(<dir>)

and no re-key helper, no peft, and no vision tower.

How it works
------------
The base is loaded text-only by **reusing `load_text_model` from
`train_lora.py` verbatim** — the same re-key, the same "a non-empty
missing/unexpected set is a hard error" discipline, and the same chat tokenizer.
Re-implementing the hop here is precisely how the two layouts drift apart, so it
is not re-implemented.  The adapter is attached with `peft`, folded in with
`merge_and_unload()`, and the merged model is written with `save_pretrained` in
bfloat16 together with the tokenizer.  A `merge.json` next to the weights records
base repo + revision, adapter path + sha256, the resolved library versions, the
dtype, the parameter count, the sha256 and byte size of the produced safetensors,
and two self-checks: that the merge actually moved weights, and that no
vision/mtp/composite key survived into the output.

Nothing here uploads anything.  Publishing is a separate, deliberate step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

# The re-key lives in train_lora.py and is imported, never copied.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lora import load_text_model  # noqa: E402

# Frozen, and identical to train_lora.py: the adapter was trained against this
# exact revision, so merging against any other revision is a different model.
DEFAULTS = dict(
    model_repo="Qwen/Qwen3.5-2B-Base",
    model_revision="b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
    adapter="/out/adapter",
    output_dir="/out/merged",
    device="cpu",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for key, val in DEFAULTS.items():
        p.add_argument("--" + key.replace("_", "-"), type=type(val), default=val)
    return p.parse_args(argv)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _probe_weight(model, needle="q_proj.weight"):
    """Return (name, tensor-clone) of the first parameter whose name matches."""
    for name, param in model.named_parameters():
        if name.endswith(needle) or needle in name:
            return name, param.detach().clone()
    raise SystemExit(f"no parameter matching {needle!r} to probe — refusing to merge blind")


def main(argv=None):
    args = parse_args(argv)
    t_run = time.time()

    import torch
    import transformers

    print(f"# torch {torch.__version__}  transformers {transformers.__version__}", flush=True)
    print(f"# device: {args.device}", flush=True)

    # ---- 1. Base, text-only, re-keyed by the trainer's own helper -----------
    model, tokenizer, load_report = load_text_model(
        args.model_repo, args.model_revision, args.device
    )
    # load_text_model raises SystemExit on a non-empty missing/unexpected set, so
    # reaching this line IS the proof that the composite -> text-only hop landed
    # every tensor.  The report is re-stated here because it goes into merge.json.
    print(f"# base loaded: {load_report['base_parameters']/1e6:.1f} M params, "
          f"{load_report['tensors_loaded']} tensors re-keyed, "
          f"{load_report['tensors_dropped']} dropped "
          f"({','.join(load_report['dropped_prefixes'])}), "
          f"missing={load_report['missing']}, unexpected={load_report['unexpected']}", flush=True)
    if load_report["missing"] or load_report["unexpected"]:
        raise SystemExit("re-key mismatch — refusing to merge an unverified base")

    adapter_safetensors = os.path.join(args.adapter, "adapter_model.safetensors")
    adapter_sha = sha256_file(adapter_safetensors)
    adapter_bytes = os.path.getsize(adapter_safetensors)

    # A named parameter is cloned before the adapter is attached; after the merge
    # it must differ, or the adapter silently attached to nothing.
    probe_name, probe_before = _probe_weight(model)

    # ---- 2. Attach and merge ------------------------------------------------
    import peft
    from peft import PeftModel

    print(f"# peft {peft.__version__}: attaching {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model = model.merge_and_unload()
    model.to(dtype=torch.bfloat16)
    # The trainer ran with use_cache off (it is a training concern).  A published
    # inference model wants the KV cache back on; this is the one config field
    # this script deliberately changes, and it is recorded in merge.json.
    model.config.use_cache = True
    model.eval()

    probe_after = dict(model.named_parameters())[probe_name].detach().to(torch.float32)
    delta = (probe_before.to(torch.float32) - probe_after).abs().max().item()
    print(f"# merge delta on {probe_name}: max|Δ| = {delta:.3e}", flush=True)
    if delta == 0.0:
        raise SystemExit(
            f"merge changed nothing on {probe_name!r} — the adapter did not apply"
        )

    param_count = sum(p.numel() for p in model.parameters())

    # ---- 3. Save ------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"# saved to {args.output_dir}", flush=True)

    # ---- 4. Verify the artefact on disk ------------------------------------
    from safetensors import safe_open

    weight_files = sorted(
        f for f in os.listdir(args.output_dir) if f.endswith(".safetensors")
    )
    if not weight_files:
        raise SystemExit(f"no safetensors written to {args.output_dir}")
    keys, merged_bytes, shards = [], 0, []
    for name in weight_files:
        path = os.path.join(args.output_dir, name)
        size = os.path.getsize(path)
        merged_bytes += size
        shards.append({"file": name, "bytes": size, "sha256": sha256_file(path)})
        with safe_open(path, framework="pt") as fh:
            keys.extend(fh.keys())

    # Text-only means no composite layout survived: no `model.language_model.*`
    # hop, no vision tower, no mtp head.
    bad = [k for k in keys if k.startswith(("mtp.", "model.visual."))]
    bad += [k for k in keys if ".language_model." in k or k.startswith("visual.")]
    if bad:
        raise SystemExit(f"merged files still carry {len(bad)} non-text keys, e.g. {bad[:5]}")
    print(f"# text-only check: {len(keys)} tensors, none under mtp/visual/"
          f"language_model", flush=True)

    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_repo": args.model_repo,
        "base_revision": args.model_revision,
        "base_load_report": load_report,
        "adapter_path": os.path.abspath(args.adapter),
        "adapter_safetensors": {
            "file": os.path.abspath(adapter_safetensors),
            "bytes": adapter_bytes,
            "sha256": adapter_sha,
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "dtype": str(next(model.parameters()).dtype),
        "parameter_count": param_count,
        "merge_probe": {"tensor": probe_name, "max_abs_delta": delta},
        "merged_safetensors": {
            "dir": os.path.abspath(args.output_dir),
            "tensors": len(keys),
            "bytes": merged_bytes,
            "shards": shards,
        },
        "config_overrides": {"use_cache": True},
        "wall_clock_s": round(time.time() - t_run, 1),
    }
    with open(os.path.join(args.output_dir, "merge.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    print(f"# merged model: {param_count/1e6:.1f} M params, "
          f"{merged_bytes/2**30:.2f} GiB bf16, {len(shards)} shard(s)", flush=True)
    print(f"# merge.json written; wall clock {report['wall_clock_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
