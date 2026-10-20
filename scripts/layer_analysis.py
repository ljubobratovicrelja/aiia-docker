#!/usr/bin/env python3
"""Largest-layer breakdown of a docker image, derived from `docker history`.

Used by scripts/image-sizes.sh: it turns the human-readable SIZE column of
`docker history --no-trunc` into a sorted table and names the single largest
layer, which is the number worth quoting in the report.

Usage:  layer_analysis.py <image> [--top N]
"""

import re
import subprocess
import sys

UNITS = {"B": 1, "kB": 1000, "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4}
PAT = re.compile(r"^([\d.]+)\s*([A-Za-z]*)$")


def parse_size(text):
    m = PAT.match(text.strip())
    if not m:
        return None
    unit = m.group(2) or "B"
    if unit not in UNITS:
        return None
    return float(m.group(1)) * UNITS[unit]


def human(n):
    for unit, factor in (("GB", 1000**3), ("MB", 1000**2), ("kB", 1000)):
        if n >= factor:
            return f"{n / factor:.2f}{unit}"
    return f"{n:.0f}B"


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    image = sys.argv[1]
    top = 8
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])

    out = subprocess.run(
        ["docker", "history", "--no-trunc", "--format", "{{.Size}}\t{{.CreatedBy}}", image],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        print(f"docker history failed for {image}: {out.stderr.strip()}", file=sys.stderr)
        return 1

    layers = []
    for line in out.stdout.splitlines():
        size_text, _, created_by = line.partition("\t")
        size = parse_size(size_text)
        if size is None:
            continue
        cmd = " ".join(created_by.split())  # collapse the buildkit whitespace
        layers.append((size, cmd))

    insp = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Size}}", image],
        capture_output=True,
        text=True,
    )
    real = int(insp.stdout.strip()) if insp.returncode == 0 and insp.stdout.strip() else None

    total = sum(s for s, _ in layers)
    print(f"  {len(layers)} layers (base-image layers included)")
    line = f"  sum of layer sizes {human(total)}"
    if real is not None:
        line += f"  vs  docker image inspect size {human(real)}  (delta {human(abs(real - total))})"
    print(line)
    print()
    print(f"  top {top} layers by size:")
    for size, cmd in sorted(layers, reverse=True)[:top]:
        if len(cmd) > 96:
            cmd = cmd[:93] + "..."
        print(f"    {human(size):>9}  {cmd}")

    biggest = max(layers)
    print()
    print(f"  LARGEST LAYER: {human(biggest[0])} ({biggest[0]:.0f} bytes)")
    print(f"    instruction: {' '.join(biggest[1].split())[:200]}")
    share = 100 * biggest[0] / total if total else 0
    print(f"    it is {share:.1f}% of the sum of all layers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
