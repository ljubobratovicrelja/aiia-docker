#!/usr/bin/env python3
"""Read a training run's JSONL loss log and turn it into a figure + a summary.

The loss log is written by training/train_lora.py: one flushed JSON object per
logged step, `{step, loss, lr, tokens_s, elapsed_s}`.  That shape is not a
convenience for this script — it is why the log exists at all: with a flushed
line per step, `--follow` here is a live monitor over a bind mount, and the same
file, read again later, is the loss curve the report quotes.  One file, two
readers, no re-parsing of docker logs.

    scripts/plot_loss.py --log training/out/loss.jsonl
        -> prints the summary and writes evidence/loss-curve.png

    scripts/plot_loss.py --follow --log training/out/loss.jsonl
        -> live, one formatted line per step, until interrupted (launcher `tail`)

matplotlib is imported lazily and the Agg backend is forced: the plotting path
is usually run on the CPU host that produced the report, where there is no
display, and a `plot_loss.py` that only works on the training host would be the
wrong shape.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def read_rows(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A torn last line is expected while the run is still
                    # writing; skipping it keeps a live read honest rather than
                    # reporting a crash that is not there.
                    continue
    except FileNotFoundError:
        pass
    return rows


def fmt(row):
    return (
        f"step {row['step']:>5}  loss {row['loss']:8.4f}  lr {row['lr']:.3e}  "
        f"{row['tokens_s']:8.1f} tok/s  t={row['elapsed_s']:7.0f}s"
    )


def follow(path):
    print(f"following {path} — Ctrl-C to stop (the run itself keeps going)")
    while not os.path.exists(path):
        time.sleep(1)
    with open(path, encoding="utf-8") as fh:
        while True:
            line = fh.readline()
            if not line:
                time.sleep(0.5)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                print(fmt(json.loads(line)), flush=True)
            except json.JSONDecodeError:
                continue


def summarise(rows, interval):
    """Print a loss summary, at intervals, then over the whole run.

    The interval lines are the point: a single first-to-last number hides a run
    that lost nothing for 200 steps and then spiked, and the report should be
    able to show that it did not happen.
    """
    first, last = rows[0], rows[-1]
    print(f"steps logged        : {len(rows)}  (step {first['step']} … {last['step']})")
    print(f"loss first / last   : {first['loss']:.4f} / {last['loss']:.4f}")
    lo = min(rows, key=lambda r: r["loss"])
    print(f"loss minimum        : {lo['loss']:.4f} at step {lo['step']}")
    if len(rows) > 1:
        steps = last["step"] - first["step"]
        # Throughput from the step count and the wall clock, not from the mean of
        # the per-step rates: the per-step rate divides by an elapsed time that
        # includes the first step's CUDA warm-up and the optimizer, so averaging
        # the rates weights short steps more than they deserve.
        print(f"wall clock          : {last['elapsed_s'] - first['elapsed_s']:.0f} s "
              f"for {steps} steps = {(last['elapsed_s'] - first['elapsed_s']) / steps:.2f} s/step")
        print(f"tokens/s median     : {sorted(r['tokens_s'] for r in rows)[len(rows) // 2]:.1f}")

    print()
    print("interval means:")
    for start in range(0, len(rows), interval):
        chunk = rows[start : start + interval]
        mean = sum(r["loss"] for r in chunk) / len(chunk)
        rate = sum(r["tokens_s"] for r in chunk) / len(chunk)
        print(f"  steps {chunk[0]['step']:>5}-{chunk[-1]['step']:<5}  "
              f"mean loss {mean:8.4f}  mean {rate:8.1f} tok/s")


def plot(rows, out_path, run_json):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    lrs = [r.get("lr") for r in rows]
    toks = [r.get("tokens_s") for r in rows]
    cfg = {}
    if run_json and os.path.exists(run_json):
        with open(run_json, encoding="utf-8") as fh:
            cfg = json.load(fh)

    fig, axes = plt.subplots(
        3, 1, figsize=(9, 8.5), dpi=140, sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1, 1]},
    )
    ax = axes[0]
    ax.plot(steps, losses, lw=0.8, alpha=0.45, color="#888", label="per-step loss")
    # A trailing mean is what the eye wants on a noisy SFT curve; 5 % of the run
    # keeps the window scaled to the run rather than to a magic constant.
    window = max(2, len(losses) // 20)
    if len(losses) > window:
        smooth = [
            sum(losses[max(0, i - window) : i + 1]) / len(losses[max(0, i - window) : i + 1])
            for i in range(len(losses))
        ]
        ax.plot(steps, smooth, lw=1.8, color="#1f4e79", label=f"mean of last {window}")
    ax.set_ylabel("cross-entropy loss\n(answer tokens only)")
    ax.set_title("aiia-docker P4 — LoRA SFT of Qwen3.5-2B-Base")

    # The two panels below are the rest of what the JSONL records.  They are not
    # decoration: a loss curve alone cannot distinguish "the schedule worked"
    # from "the schedule stalled while throughput collapsed", and both of those
    # show up here.
    if any(v is not None for v in lrs):
        axes[1].plot(steps, lrs, lw=1.2, color="#a33")
        axes[1].set_ylabel("learning rate")
        axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if any(v is not None for v in toks):
        axes[2].plot(steps, toks, lw=1.0, color="#2a7")
        axes[2].set_ylabel("padded tokens/s")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[2].set_xlabel("optimizer step")
    ax.legend(loc="upper right", fontsize=8)

    run_cfg = cfg.get("run_config", {})
    note = []
    if run_cfg:
        note.append(
            f"effective batch {run_cfg.get('batch_size', '?')}x{run_cfg.get('grad_accum', '?')} "
            f"sequences, max {run_cfg.get('max_seq_len', '?')} tokens, "
            f"lr {run_cfg.get('lr', '?')}, r={run_cfg.get('lora_r', '?')}"
        )
    if steps:
        # `len(rows)` is how many steps were LOGGED and the span is one fewer
        # interval; printing those side by side as "10 steps, 9 optimizer steps"
        # reads like a step went missing.
        span = steps[-1] - steps[0]
        note.append(
            f"{len(rows)} logged steps spanning {span} step intervals"
            + (f", {cfg.get('peak_vram_gib')} GiB peak VRAM" if cfg.get("peak_vram_gib") else "")
        )
    if note:
        ax.annotate("\n".join(note), xy=(0.01, 0.02), xycoords="axes fraction",
                    fontsize=7, color="#333")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    print(f"\nfigure written      : {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--log", default="training/out/loss.jsonl")
    p.add_argument("--out", default="evidence/loss-curve.png")
    p.add_argument("--run-json", default=None,
                   help="adapter run.json, for the figure annotation (default: sibling of --log)")
    p.add_argument("--interval", type=int, default=10, help="steps per summary row")
    p.add_argument("--follow", action="store_true", help="live monitor instead of a summary")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    if args.follow:
        follow(args.log)
        return 0

    rows = read_rows(args.log)
    if not rows:
        print(f"no loss rows in {args.log}", file=sys.stderr)
        return 1
    summarise(rows, max(1, args.interval))
    if not args.no_plot:
        run_json = args.run_json or os.path.join(os.path.dirname(args.log), "adapter", "run.json")
        plot(rows, args.out, run_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
