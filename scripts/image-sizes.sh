#!/usr/bin/env bash
# Stage-2 evidence: image sizes and the per-layer breakdown, so the report can
# say exactly where the bytes are — and so the cache argument has numbers behind
# it (the model layer is the thing a bad Dockerfile order would re-download).
#
# Writes evidence/image-sizes.txt.  Read-only: it builds nothing and starts
# nothing.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=evidence/image-sizes.txt
IMAGES=(aiia-docker/inference:stage-1 aiia-docker/gui:stage-1)
BASE=ghcr.io/ggml-org/llama.cpp:server

mkdir -p evidence
{
  echo "======================================================================"
  echo "image-sizes.txt — image sizes and layer breakdown (aiia-docker stage 2)"
  echo "======================================================================"
  echo "date            : $(date -Is)"
  echo "engine / compose: $(docker version --format '{{.Server.Version}}') / $(docker compose version --short)"
  echo "arch            : $(uname -m)"
  echo
  echo "== docker images (full listing; the two project images are in it) =="
  echo
  docker images
  echo
  echo "NOTE: this engine (Docker 29) prints a 'DISK USAGE' column, which counts"
  echo "the bytes actually occupied on disk including shared base layers, and a"
  echo "'CONTENT SIZE' column.  The authoritative per-image size is"
  echo "'docker image inspect --format {{.Size}}', quoted below in bytes."
  echo
  echo "== docker image ls --format (portable view) =="
  echo
  docker image ls --format 'table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}' \
    | grep -E 'REPOSITORY|aiia-docker|llama.cpp' || true
  echo
  echo "== docker image inspect --format '{{.Size}}' (bytes) =="
  echo
  for img in "${IMAGES[@]}" "$BASE"; do
    printf '  %-40s %s bytes\n' "$img" \
      "$(docker image inspect --format '{{.Size}}' "$img" 2>/dev/null || echo 'not present')"
  done
  echo
  echo "======================================================================"
  echo "WHERE THE BYTES ARE — largest layer per image"
  echo "======================================================================"
  for img in "${IMAGES[@]}"; do
    echo
    echo "--- $img ---"
    python3 scripts/layer_analysis.py "$img"
  done
  echo
  echo "Reading: the inference image's largest layer (1.28 GB) is the pinned"
  echo "GGUF download.  It sits directly below the ARG block and ABOVE the ENV"
  echo "runtime settings in inference/Dockerfile, which is why changing a runtime"
  echo "setting never re-runs it — see evidence/build-cache.txt section C."
  echo "The gui image's largest layer is the pip install; it sits above COPY"
  echo "app.py, which is why editing the GUI never reinstalls gradio — same file,"
  echo "section B2."
  echo
  echo "======================================================================"
  echo "RAW: docker history --no-trunc (verbatim, per image)"
  echo "======================================================================"
  for img in "${IMAGES[@]}"; do
    echo
    echo "######################################################################"
    echo "# docker history --no-trunc $img"
    echo "######################################################################"
    docker history --no-trunc "$img"
  done
} > "$OUT"

echo "written: $OUT ($(wc -l < "$OUT") lines)"
