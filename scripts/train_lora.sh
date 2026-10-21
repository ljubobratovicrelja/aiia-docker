#!/usr/bin/env bash
# Stage-4 (P4) launcher: bring the LoRA training container up detached, watch it,
# and take it down again.  It is deliberately the *only* documented way to start
# a training run, so that the run configuration has one home (the defaults in
# training/train_lora.py) and the smoke run and the full run cannot drift apart.
#
# Run this ON the machine with the GPU (the host-side preamble is in the
# README).  The environment such a host may need is passed through
# TRAIN_DOCKER_HOME below rather than hard-coded here, so the launcher stays
# runnable on a host with an ordinary, writable HOME.
#
#   scripts/train_lora.sh up             # build if needed, start detached
#   scripts/train_lora.sh tail           # follow the JSONL loss log (blocking)
#   scripts/train_lora.sh status         # container state + last loss line
#   scripts/train_lora.sh down           # remove the container (image stays)
#
# Smoke-sized and full-sized runs are the SAME command; only the knobs differ,
# and they come from the environment:
#
#   MAX_STEPS=10 TRAIN_SIZE=24 SCAN_LIMIT=200 scripts/train_lora.sh up
#   scripts/train_lora.sh up                      # committed full-run config
#
#   TRAIN_OUT         host dir mounted at /out  (default: training/out)
#   TRAIN_HF_CACHE    host dir mounted at /hf-cache, holds the 4.5 GB checkpoint
#                     and the corpus (default: /var/tmp/aiia-hf-cache)
#   TRAIN_DOCKER_HOME writable HOME for the Docker CLI, for hosts whose $HOME is
#                     not writable — docker needs it for its config dir, and its
#                     absence is a `mkdir … permission denied` at build time,
#                     well before anything to do with training.
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-up}"

# ---------------------------------------------------------------------------
section() {
  echo
  echo "======================================================================"
  echo "# $*"
  echo "======================================================================"
}

# ---------------------------------------------------------------------------
if [ -n "${TRAIN_DOCKER_HOME:-}" ]; then
  mkdir -p "$TRAIN_DOCKER_HOME/.docker"
  export HOME="$TRAIN_DOCKER_HOME"
  export DOCKER_CONFIG="$TRAIN_DOCKER_HOME/.docker"
fi

export TRAIN_HF_CACHE="${TRAIN_HF_CACHE:-/var/tmp/aiia-hf-cache}"
export TRAIN_OUT="${TRAIN_OUT:-$PWD/training/out}"
LOSS_LOG="$TRAIN_OUT/loss.jsonl"

PROFILE=(--profile train)
COMPOSE=(docker compose "${PROFILE[@]}")

mkdir -p "$TRAIN_HF_CACHE" "$TRAIN_OUT"

case "$MODE" in
# ---------------------------------------------------------------------------
up)
  section "0. Preflight — device, disk, and the cache the run will reuse"
  # The whole point of the mounted cache is that this is printed once, and then
  # never downloads again.  Printing the directory lets the reader see that the
  # smoke run and the full run point at the same bytes.
  echo "host          : $(uname -n) $(uname -m), docker $(docker version --format '{{.Server.Version}}'), compose $(docker compose version --short)"
  echo "gpu           : $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
  echo "hf cache      : $TRAIN_HF_CACHE  ($(du -sh "$TRAIN_HF_CACHE" 2>/dev/null | cut -f1 || echo empty))"
  echo "output dir    : $TRAIN_OUT"
  echo "disk on /     : $(df -h / | tail -1 | awk '{print $4" free of "$2}')"
  echo "schedule knobs: MAX_STEPS='${MAX_STEPS:-<default>}' TRAIN_SIZE='${TRAIN_SIZE:-<default>}' " \
       "SCAN_LIMIT='${SCAN_LIMIT:-<default>}' EPOCHS='${EPOCHS:-<default>}'"

  section "1. Services this invocation will start"
  # The graded property from stage 1, re-checked here rather than remembered:
  # the plain stack must still resolve to exactly inference + gui.
  echo "docker compose config --services        : $(docker compose config --services | tr '\n' ' ')"
  echo "docker compose --profile train --services: $(docker compose "${PROFILE[@]}" config --services | tr '\n' ' ')"

  section "2. Build the training image (cache-served after the first time)"
  docker compose "${PROFILE[@]}" build training

  section "3. Start the training container, detached"
  # Detached on purpose: a LoRA run outlives an ssh session, and a build or a
  # step can be silent for minutes.  Progress is read from `docker logs` and
  # from the JSONL loss log, never from this terminal staying open.
  docker compose "${PROFILE[@]}" up -d training
  echo
  echo "loss log      : $LOSS_LOG"
  echo "follow it with: scripts/train_lora.sh tail"
  echo "container log : docker compose --profile train logs -f training"
  ;;

# ---------------------------------------------------------------------------
tail)
  section "Following the JSONL loss log"
  # A second monitor next to `docker logs`: the trainer flushes one JSON object
  # per step, so this shows numbers while the run is live.
  exec python3 scripts/plot_loss.py --follow --log "$LOSS_LOG"
  ;;

# ---------------------------------------------------------------------------
status)
  section "Container state"
  docker compose "${PROFILE[@]}" ps training
  section "Progress"
  echo "container state/ image:"
  IMG_SIZE=$(docker image inspect aiia-docker/training:stage-4 --format '{{.Size}} bytes' 2>/dev/null) \
    || IMG_SIZE="not built"
  echo "image         : $IMG_SIZE"
  echo "loss log      : $LOSS_LOG"
  if [ -s "$LOSS_LOG" ]; then
    echo "logged steps  : $(wc -l < "$LOSS_LOG")"
    echo "last line     : $(tail -1 "$LOSS_LOG")"
  else
    echo "logged steps  : 0 (nothing yet — a first run also downloads ~4.7 GB)"
  fi
  ;;

# ---------------------------------------------------------------------------
down)
  section "Removing the training container"
  docker compose "${PROFILE[@]}" down
  echo "image kept: aiia-docker/training:stage-4 (rebuild only when training/ changes)"
  ;;

*)
  echo "usage: $0 {up|tail|status|down}" >&2
  exit 2
  ;;
esac
