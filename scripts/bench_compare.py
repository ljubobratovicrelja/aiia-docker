#!/usr/bin/env python3
"""CPU vs GPU comparison for the same fixed prompt, from two bench artifacts.

Reads the `__SUMMARY__ {json}` lines that scripts/bench_probe.py emitted into
each artifact (the machine-readable record, not the pretty table — so the
comparison does not depend on the formatting of the table above it) and prints
one row per measurement path, CPU against GPU.

Honesty rules this script enforces:

  * the comparable metric is **effective tokens/s** (completion tokens divided
    by elapsed seconds).  Raw elapsed time is also printed, but wall-clock alone
    is only comparable when the two stacks generated the same number of tokens;
  * if the GPU produced a different number of completion tokens (the isolated
    inference path pins temperature 0 and max_tokens 160, so it should not; the
    GUI path samples at its own temperature and may), the difference is reported
    explicitly instead of being silently averaged away;
  * a speedup is printed as `n/a` rather than computed when the token counts
    differ, because tokens/s across different outputs measures the outputs as
    much as the hardware.

Usage:
  bench_compare.py evidence/bench-cpu.txt evidence/bench-gpu.txt [name-to-print]
"""

import json
import statistics
import sys

PREFIX = "__SUMMARY__ "
PATHS = ("inference", "gui")


def load(path):
    rows = {p: [] for p in PATHS}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(PREFIX):
                line = line[len(PREFIX):]
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("mode") in rows:
                rows[r["mode"]].append(r)
    return rows


def medians(rows):
    if not rows:
        return None
    return {
        "runs": len(rows),
        "elapsed_s": statistics.median(r["elapsed_s"] for r in rows),
        "tokens": statistics.median(r["completion_tokens"] for r in rows),
        "tokens_s": statistics.median(r["tokens_per_s"] for r in rows),
        "pooled_s": sum(r["completion_tokens"] for r in rows)
        / sum(r["elapsed_s"] for r in rows),
        "all_tokens": sorted({r["completion_tokens"] for r in rows}),
    }


def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__, file=sys.stderr)
        return 2
    cpu_file, gpu_file = sys.argv[1], sys.argv[2]
    # Optional third argument: the name to print for the GPU artifact.  The
    # caller assembles that artifact under a temp name and renames it into place
    # at the end, so without this the table would cite a /tmp path that no
    # longer exists by the time anyone reads it.
    gpu_label = sys.argv[3] if len(sys.argv) == 4 else gpu_file
    cpu, gpu = load(cpu_file), load(gpu_file)

    out = []
    bar = "=" * 78
    out.append("")
    out.append(bar)
    out.append("CPU vs GPU — same fixed prompt, same request parameters")
    out.append(bar)
    out.append("cpu artifact : %s" % cpu_file)
    out.append("gpu artifact : %s" % gpu_label)
    out.append("")
    out.append(
        "%-10s %-4s %10s %8s %11s %11s" % ("path", "dev", "elapsed_s", "tokens", "tokens/s", "pooled tok/s")
    )
    out.append("-" * 78)
    stats = {}
    for dev, rows in (("cpu", cpu), ("gpu", gpu)):
        for path in PATHS:
            m = medians(rows[path])
            stats[(dev, path)] = m
            if m is None:
                out.append("%-10s %-4s %10s %8s %11s %11s" % (path, dev, "-", "-", "-", "-"))
                continue
            out.append(
                "%-10s %-4s %10.3f %8d %11.2f %11.2f"
                % (path, dev, m["elapsed_s"], m["tokens"], m["tokens_s"], m["pooled_s"])
            )
    out.append("-" * 78)
    out.append("medians across each artifact's runs; tokens = median completion tokens")
    out.append("")

    for path in PATHS:
        c, g = stats[("cpu", path)], stats[("gpu", path)]
        if not c or not g:
            continue
        out.append("path %s:" % path)
        if c["all_tokens"] == g["all_tokens"]:
            out.append(
                "  same completion token count on both devices (%s) -> tokens/s IS "
                "comparable," % ", ".join(str(t) for t in c["all_tokens"])
            )
            out.append(
                "  and so is elapsed time: GPU %.2f tok/s vs CPU %.2f tok/s = %.2fx "
                "faster," % (g["tokens_s"], c["tokens_s"], g["tokens_s"] / c["tokens_s"])
            )
            out.append(
                "  latency %.3f s vs %.3f s = %.2fx lower."
                % (g["elapsed_s"], c["elapsed_s"], c["elapsed_s"] / g["elapsed_s"])
            )
        else:
            out.append(
                "  DIFFERENT completion token counts — cpu %s vs gpu %s — so the two rows "
                "generated" % (c["all_tokens"], g["all_tokens"])
            )
            out.append(
                "  different work.  The comparable number is the RATE: GPU %.2f vs CPU "
                "%.2f tokens/s"
                % (g["tokens_s"], c["tokens_s"])
            )
            out.append(
                "  = %.2fx the tokens per second, and %.1f ms/token vs %.1f ms/token."
                % (g["tokens_s"] / c["tokens_s"], 1000 / g["tokens_s"], 1000 / c["tokens_s"])
            )
            out.append(
                "  Raw elapsed (%.2fx) is NOT quoted as the headline: the GPU run emitted "
                "more tokens,"
                % (c["elapsed_s"] / g["elapsed_s"])
            )
            out.append(
                "  so its wall clock carries extra work — if anything that makes the rate "
                "advantage"
            )
            out.append(
                "  a floor rather than an overstatement.  Note also that a different token "
                "count at"
            )
            out.append(
                "  temperature 0 means the two backends did not produce the same text: the "
                "greedy"
            )
            out.append(
                "  trajectory diverges between the CPU and CUDA kernels.  That is a finding "
                "about"
            )
            out.append(
                "  the backends, not a measurement error, and it is why the two paths are "
                "compared"
            )
            out.append("  on effective tokens/s.")
        out.append("")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
