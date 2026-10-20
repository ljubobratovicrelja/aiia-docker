#!/usr/bin/env bash
# Stage-4 (P4) smoke record: run the SMALLEST real training run the stage allows
# and write down everything a reader needs to believe it happened.
#
#   run ON the GPU host, from the repository root:   bash scripts/train_smoke.sh
#   -> evidence/train-smoke.txt
#
# It is a smoke test, not a substitute for the full run: 10 optimizer steps on a
# 24-record slice, one execution, then the container is torn down.  What it
# proves is that the image, the mounted cache, the LoRA wiring, the optimizer and
# the adapter save path all work on real hardware — the things that are cheap to
# get wrong and expensive to discover after a 6-hour run.
#
# The full run uses the same command with the knobs left at their committed
# defaults; see README "How to run the training".
#
# Deliberately not `set -e`: the point of the script is to record what happened,
# including a non-zero exit code from the trainer, and aborting on the first
# unexpected status would throw away the evidence of the failure.
set -uo pipefail

cd "$(dirname "$0")/.."

OUT=evidence/train-smoke.txt
MAX_STEPS="${MAX_STEPS:-10}"
TRAIN_SIZE="${TRAIN_SIZE:-24}"
SCAN_LIMIT="${SCAN_LIMIT:-200}"

export TRAIN_HF_CACHE="${TRAIN_HF_CACHE:-/var/tmp/aiia-hf-cache}"
export TRAIN_OUT="${TRAIN_OUT:-$PWD/training/out}"
LOSS_LOG="$TRAIN_OUT/loss.jsonl"
SAMPLES=$(mktemp /tmp/vram-samples.XXXXXX)
TMP=$(mktemp)
mkdir -p evidence "$TRAIN_HF_CACHE" "$TRAIN_OUT"

cleanup() { rm -f "$SAMPLES"; }
trap cleanup EXIT

section() {
  {
    echo
    echo "======================================================================"
    echo "# $*"
    echo "======================================================================"
  } >> "$TMP"
}

# ---------------------------------------------------------------------------
{
  echo "======================================================================"
  echo "train-smoke.txt — P4 LoRA training smoke run (aiia-docker stage 4)"
  echo "======================================================================"
  echo "date            : $(date -Is)"
  echo "host            : $(uname -srm), $(nproc) threads"
  echo "gpu             : $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
  echo "engine          : docker $(docker version --format '{{.Server.Version}}' | tr -d '\n'), compose $(docker compose version --short)"
  echo "docker HOME     : ${HOME:-<unset>}${DOCKER_CONFIG:+ (DOCKER_CONFIG=$DOCKER_CONFIG)}"
  echo "hf cache (host) : $TRAIN_HF_CACHE"
  echo "output dir      : $TRAIN_OUT"
  echo "disk on /       : $(df -h / | tail -1)"
  echo
  echo "smoke schedule  : MAX_STEPS=$MAX_STEPS TRAIN_SIZE=$TRAIN_SIZE SCAN_LIMIT=$SCAN_LIMIT"
  echo "full-run config : the defaults in training/train_lora.py — 2 epochs, 3000"
  echo "                  records drawn from a 20000-record scan, 1024 tokens,"
  echo "                  batch 2 x accum 8, lr 1e-4 cosine, LoRA r=16 alpha=32"
  echo
  echo "This is an EMBEDDED smoke test inside one script, so every number below"
  echo "comes from one execution on one machine at one time.  The commands are"
  echo "printed as they are run — this file is the record, not a transcript."
} > "$TMP"

# ---------------------------------------------------------------------------
section "1. Plain 'docker compose up' still starts ONLY inference + gui (stage-1 property)"
# ---------------------------------------------------------------------------
{
  echo "The training service is added under profiles: [\"train\"], so it is not part"
  echo "of the default stack.  Checked three ways, because 'it has a profile' is a"
  echo "claim and these are observations:"
  echo
  echo '$ docker compose config --services'
  docker compose config --services
  echo
  echo '$ docker compose --profile train config --services'
  docker compose --profile train config --services
  echo
  echo '$ docker compose config | grep -c "^  training:"   # plain stack must resolve 0'
  docker compose config | grep -c '^  training:' || true
  echo
  echo '$ docker compose up -d --dry-run        # what a plain up would create'
  docker compose up -d --dry-run 2>&1 | sed -n '1,20p'
} >> "$TMP"

# ---------------------------------------------------------------------------
section "2. Build the training image"
# ---------------------------------------------------------------------------
BUILD_LOG=$(mktemp)
docker compose --profile train build training > "$BUILD_LOG" 2>&1
BUILD_RC=$?
{
  echo "\$ docker compose --profile train build training   -> exit $BUILD_RC"
  echo
  echo "--- build log (tail; the base image is ~12 GB compressed) ---"
  tail -12 "$BUILD_LOG"
  echo
  echo "--- the pip layer, where the training stack lands ---"
  grep -E "transformers|peft|Successfully installed|ERROR" "$BUILD_LOG" | tail -4
  echo
  echo "\$ docker image inspect aiia-docker/training:stage-4 --format '{{.Size}}'"
  docker image inspect aiia-docker/training:stage-4 --format '{{.Size}}'
  echo "base image size (narwhal layer removed):"
  docker image inspect nvcr.io/nvidia/pytorch:26.09-py3 --format '{{.Size}}'
} >> "$TMP"
rm -f "$BUILD_LOG"

# ---------------------------------------------------------------------------
section "3. Start the smoke run, detached (one execution)"
# ---------------------------------------------------------------------------
{
  echo "Same launcher the full run uses, only the knobs differ:"
  echo
  echo "\$ MAX_STEPS=$MAX_STEPS TRAIN_SIZE=$TRAIN_SIZE SCAN_LIMIT=$SCAN_LIMIT \\"
  echo "    scripts/train_lora.sh up"
  echo
} >> "$TMP"

# The wait below keys on the first line appearing in the loss log, and a stale
# log from an earlier run makes that check return immediately — which is how an
# earlier version of this script captured its nvidia-smi table on an IDLE
# device and called it evidence of GPU use.  Truncate before launching.
rm -f "$LOSS_LOG"

MAX_STEPS="$MAX_STEPS" TRAIN_SIZE="$TRAIN_SIZE" SCAN_LIMIT="$SCAN_LIMIT" \
  bash scripts/train_lora.sh up >> "$TMP" 2>&1
UP_RC=$?
{
  echo
  echo "scripts/train_lora.sh up -> exit $UP_RC"
  echo
  echo "\$ docker compose --profile train ps training"
  docker compose --profile train ps training
} >> "$TMP"

CID=$(docker compose --profile train ps -q training)

# ---------------------------------------------------------------------------
section "4. nvidia-smi DURING the run"
# ---------------------------------------------------------------------------
{
  echo "A sampler runs for the container's whole lifetime.  It is started here so"
  echo "that the device reading is taken from inside the training window — a"
  echo "before/after pair would show an idle GPU and prove nothing."
  echo
} >> "$TMP"
(
  for _ in $(seq 1 2400); do
    # Appended per line, deliberately: a single redirect for the whole loop is
    # block-buffered, so a short run (the smoke finishes in ~70 s) ends with an
    # empty sample file and a VRAM section that proves nothing.  Opening the
    # file per write is the cheap way to stay unbuffered without stdbuf.
    printf '%s | mem.used %s | %s\n' \
      "$(date +%T)" \
      "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" \
      "$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')" >> "$SAMPLES" 2>/dev/null
    sleep 2
  done
) &
SAMPLER=$!

# Wait for the first logged step before sampling the device: the first minutes
# of the container's life are a 4.7 GB download and a CPU-side model init, and
# the GPU is genuinely idle through them.
for _ in $(seq 1 900); do
  [ -s "$LOSS_LOG" ] && break
  sleep 2
  [ "$(docker inspect -f '{{.State.Running}}' "$CID" 2>/dev/null)" = "true" ] || break
done

{
  echo "--- nvidia-smi (full table) once the first step had been logged ---"
  nvidia-smi 2>&1 | sed -n '1,12p'
  echo
  echo "--- device samples: peak and the first/last rows naming a process ---"
  grep -o 'mem.used [0-9]* MiB' "$SAMPLES" | awk '{print $2}' | sort -n | tail -1 \
    | sed 's/^/peak memory.used : /;s/$/ MiB/'
  echo "samples taken    : $(wc -l < "$SAMPLES")"
  echo "rows naming a compute process (last 3):"
  grep -v '| *$' "$SAMPLES" | tail -3
} >> "$TMP"

# ---------------------------------------------------------------------------
section "5. Wait for the run to finish"
# ---------------------------------------------------------------------------
WAITED=0
while [ "$(docker inspect -f '{{.State.Running}}' "$CID" 2>/dev/null)" = "true" ]; do
  sleep 5
  WAITED=$((WAITED + 5))
  if [ "$WAITED" -ge 1800 ]; then
    echo "TIMEOUT after ${WAITED}s — container still running" >> "$TMP"
    break
  fi
done
kill "$SAMPLER" 2>/dev/null

EXIT_CODE=$(docker inspect -f '{{.State.ExitCode}}' "$CID" 2>/dev/null)
{
  echo "container      : $CID"
  echo "waited         : ${WAITED}s"
  echo "exit code      : $EXIT_CODE"
  echo "oom killed     : $(docker inspect -f '{{.State.OOMKilled}}' "$CID" 2>/dev/null)"
} >> "$TMP"

# ---------------------------------------------------------------------------
section "6. The loss log — every line of it"
# ---------------------------------------------------------------------------
{
  echo "The JSONL the trainer writes, one flushed object per step.  This file is"
  echo "the live monitor during a run (scripts/train_lora.sh tail) and the loss"
  echo "curve afterwards (scripts/plot_loss.py); it is not a copy of stdout."
  echo
  echo "\$ cat $LOSS_LOG"
  cat "$LOSS_LOG" 2>/dev/null || echo "(missing)"
  echo
  echo "FIRST loss line:"
  head -1 "$LOSS_LOG" 2>/dev/null || echo "(missing)"
  echo "LAST loss line:"
  tail -1 "$LOSS_LOG" 2>/dev/null || echo "(missing)"
  echo "steps logged   : $(wc -l < "$LOSS_LOG" 2>/dev/null || echo 0)"
} >> "$TMP"

# ---------------------------------------------------------------------------
section "7. The adapter on disk"
# ---------------------------------------------------------------------------
{
  echo "\$ ls -l $TRAIN_OUT/adapter"
  ls -l "$TRAIN_OUT/adapter" 2>&1
  echo
  echo "\$ du -sh $TRAIN_OUT/adapter"
  du -sh "$TRAIN_OUT/adapter" 2>&1
  echo
  echo "--- adapter_config.json (the LoRA geometry actually used) ---"
  cat "$TRAIN_OUT/adapter/adapter_config.json" 2>/dev/null || echo "(missing)"
  echo
  echo "--- run.json (the effective run config of this container) ---"
  cat "$TRAIN_OUT/adapter/run.json" 2>/dev/null || echo "(missing)"
} >> "$TMP"

# ---------------------------------------------------------------------------
section "8. Container log (tail) and docker stats"
# ---------------------------------------------------------------------------
{
  echo '$ docker compose --profile train logs --tail 30 training'
  docker compose --profile train logs --tail 30 training 2>&1
  echo
  echo '$ docker stats --no-stream training'
  docker stats --no-stream training 2>&1 | sed -n '1,3p'
} >> "$TMP"

# ---------------------------------------------------------------------------
section "9. Teardown — nothing left running"
# ---------------------------------------------------------------------------
{
  echo '$ docker compose --profile train down'
  docker compose --profile train down 2>&1 | tail -5
  echo
  echo "remaining containers on this host (must not list training):"
  docker ps -a --format '{{.Names}}\t{{.Status}}' | sed -n '1,10p'
  echo
  echo "training containers remaining: $(docker ps -aq --filter 'name=training' | wc -l)"
  echo
  echo "cache kept for the full run — the smoke run's download is reused, not repeated:"
  du -sh "$TRAIN_HF_CACHE" 2>/dev/null
  find "$TRAIN_HF_CACHE" -name '*.safetensors' -o -name '*.jsonl' 2>/dev/null | sed -n '1,6p' | xargs -r ls -lh 2>/dev/null
} >> "$TMP"

mv "$TMP" "$OUT"
echo "written: $OUT"
