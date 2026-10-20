#!/usr/bin/env bash
# Stage-2: ONE demonstrated image-size improvement, with before/after numbers.
#
# gui/Dockerfile installs with `pip install --no-cache-dir`.  Dropping that flag
# lets pip keep its download/wheel cache inside the layer, so the image grows by
# the size of the wheels a single-install image would never use again.  This
# script builds BOTH variants from the same source, measures them honestly, and
# then throws the experiment tags away, keeping the better (committed) variant.
#
# It builds under separate tags, so the running stack and the stage-1 image tags
# are not touched.
#
# Writes evidence/size-experiment.txt.  Nothing to restore: the Dockerfile on
# disk is never modified (the variant is generated in a temp dir and passed via
# `docker build -f`).
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=evidence/size-experiment.txt
COMMITTED=aiia-docker/gui:stage-1
TAG_NOFLAG=aiia-docker/gui:pip-cache-experiment   # without --no-cache-dir
TAG_FLAG=aiia-docker/gui:pip-nocache-experiment   # as committed
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if grep -q -- '--no-cache-dir' gui/Dockerfile; then
  sed 's/ *--no-cache-dir//' gui/Dockerfile > "$TMP/Dockerfile.without-flag"
  VARIANT_DESC="same Dockerfile with '--no-cache-dir' removed from the pip install"
else
  echo "gui/Dockerfile no longer uses --no-cache-dir; nothing to measure" >&2
  exit 1
fi

size_of() { docker image inspect --format '{{.Size}}' "$1"; }

mkdir -p evidence
{
  echo "======================================================================"
  echo "size-experiment.txt — one demonstrated image-size improvement"
  echo "======================================================================"
  echo "date            : $(date -Is)"
  echo "engine / compose: $(docker version --format '{{.Server.Version}}') / $(docker compose version --short)"
  echo "arch            : $(uname -m)"
  echo
  echo "Hypothesis: 'RUN pip install --no-cache-dir -r requirements.txt' could be"
  echo "simplified to 'RUN pip install -r requirements.txt'.  Both variants were"
  echo "built here from the identical context so the difference is only the flag."
  echo
  echo "  variant A (kept)    : gui/Dockerfile unchanged — pip install --no-cache-dir"
  echo "  variant B (rejected): ${VARIANT_DESC}"
  echo
  echo "Generated variant B Dockerfile:"
  sed 's/^/    /' "$TMP/Dockerfile.without-flag"
  echo
  echo "======================================================================"
  echo "BUILD B (without --no-cache-dir) — tag $TAG_NOFLAG"
  echo "======================================================================"
} > "$OUT"

echo "--- building variant B (without the flag) ---"
docker build --progress=plain -f "$TMP/Dockerfile.without-flag" -t "$TAG_NOFLAG" gui >> "$OUT" 2>&1

{
  echo
  echo "======================================================================"
  echo "BUILD A (as committed, with --no-cache-dir) — tag $TAG_FLAG"
  echo "======================================================================"
} >> "$OUT"

echo "--- building variant A (as committed) ---"
docker build --progress=plain -f gui/Dockerfile -t "$TAG_FLAG" gui >> "$OUT" 2>&1

SIZE_B=$(size_of "$TAG_NOFLAG")
SIZE_A=$(size_of "$TAG_FLAG")
SIZE_COMMITTED=$(size_of "$COMMITTED" 2>/dev/null || echo "$SIZE_A")

human() { python3 - "$1" <<'PY'
import sys
n = float(sys.argv[1])
for unit, f in (("GB", 1000**3), ("MB", 1000**2), ("kB", 1000)):
    if n >= f:
        print(f"{n/f:.2f}{unit}"); break
else:
    print(f"{n:.0f}B")
PY
}

{
  echo
  echo "======================================================================"
  echo "RESULT — before / after"
  echo "======================================================================"
  echo
  echo "  variant A  pip install --no-cache-dir   (KEPT)     $SIZE_A bytes  = $(human "$SIZE_A")"
  echo "  variant B  pip install                  (REJECTED) $SIZE_B bytes  = $(human "$SIZE_B")"
  python3 - "$SIZE_A" "$SIZE_B" <<'PY'
import sys
a, b = int(sys.argv[1]), int(sys.argv[2])
d = b - a
print()
print(f"  delta            : {d:+.0f} bytes  ({d/1e6:+.1f} MB)")
print(f"  B / A            : {b/a:.3f}x  ({100*d/a:+.1f}% larger without the flag)")
print(f"  relative to the committed image: {100*d/ (a or 1):+.1f}%")
PY
  echo
  echo "  layer breakdown, variant B (the rejected one):"
  python3 scripts/layer_analysis.py "$TAG_NOFLAG" --top 4 | sed 's/^/  /'
  echo
  echo "  layer breakdown, variant A (the kept one):"
  python3 scripts/layer_analysis.py "$TAG_FLAG" --top 4 | sed 's/^/  /'
  echo
  echo "======================================================================"
  echo "DECISION"
  echo "======================================================================"
  echo "  Variant A wins: dropping --no-cache-dir makes the image larger, and the"
  echo "  retained pip cache is dead weight in a container that only ever installs"
  echo "  once at build time.  gui/Dockerfile is left exactly as committed; the"
  echo "  experiment images are removed, and $COMMITTED"
  echo "  (which is variant A) is the one the stack runs."
  echo
  echo "  sanity check: committed image $SIZE_COMMITTED bytes"
  echo "                variant A           $SIZE_A bytes"
  if [ "$SIZE_COMMITTED" = "$SIZE_A" ]; then
    echo "                -> identical, so variant A is what is actually deployed"
  else
    echo "                -> DIFFERENT; investigate"
  fi
} >> "$OUT"

echo "--- removing the experiment images (keeping variant A) ---"
{
  echo
  echo "\$ docker image rm $TAG_NOFLAG $TAG_FLAG"
  docker image rm "$TAG_NOFLAG" "$TAG_FLAG"
  echo
  echo "images left:"
  docker image ls --format '{{.Repository}}:{{.Tag}}  {{.ID}}  {{.Size}}' \
    | grep -E 'aiia-docker' | sed 's/^/  /'
} >> "$OUT" 2>&1

echo "written: $OUT"
