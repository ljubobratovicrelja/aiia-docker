#!/usr/bin/env bash
# Stage-4 (P4) FULL-run record: assemble evidence/train-run.txt from a completed
# run's output directory.
#
#   run ON the GPU host, from the repository root, after the run has finished:
#     TRAIN_RUN_DIR=/var/tmp/aiia-fullrun bash scripts/train_record.sh
#   -> evidence/train-run.txt
#
# scripts/train_smoke.sh *drives* its own 10-step run; this script documents a
# run that already happened, so it never starts, stops or restarts anything.  It
# also does not plot: the figure is made from the same JSONL on the workstation
# (scripts/plot_loss.py), which is where matplotlib is, and the header says so
# rather than pretending the plot came from here.
#
# Deliberately not `set -e`: recording what happened is the point, including a
# missing section that should have been there.
set -uo pipefail

cd "$(dirname "$0")/.."

OUT=evidence/train-run.txt
RUN_DIR="${TRAIN_RUN_DIR:-$PWD/training/out}"
LOSS_LOG="$RUN_DIR/loss.jsonl"
SAMPLES="$RUN_DIR/vram-samples.txt"
STATS="$RUN_DIR/docker-stats.txt"
ADAPTER="$RUN_DIR/adapter"
CONTAINER="${TRAIN_CONTAINER:-aiia-docker-training-1}"
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

section() {
  {
    echo
    echo "======================================================================"
    echo "# $*"
    echo "======================================================================"
  } >> "$TMP"
}

{
  echo "======================================================================"
  echo "train-run.txt — P4 LoRA training, FULL run (aiia-docker stage 4)"
  echo "======================================================================"
  echo "record written  : $(date -Is)"
  echo "host            : $(uname -srm), $(nproc) threads"
  echo "gpu             : $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
  echo "engine          : docker $(docker version --format '{{.Server.Version}}' | tr -d '\n'), compose $(docker compose version --short)"
  echo "run directory   : $RUN_DIR"
  echo
  echo "launched as (detached; the committed defaults ARE the configuration, no"
  echo "hyperparameter is overridden for this run):"
  echo
  echo "  scripts/train_lora.sh up"
  echo
  echo "figure          : NOT produced here — scripts/plot_loss.py reads the same"
  echo "                  loss.jsonl on the workstation and writes evidence/loss-curve.png"
} > "$TMP"

# ---------------------------------------------------------------------------
section "1. The run configuration actually used (adapter/run.json)"
# ---------------------------------------------------------------------------
{
  echo '$ python3 -c "import json;print(json.dumps(json.load(open(run.json))[run_config],indent=2))"'
  echo
  python3 - "$ADAPTER/run.json" <<'PY' 2>&1
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as exc:
    print(f"(unreadable: {exc})")
    raise SystemExit(0)
for k, v in sorted(d.get("run_config", {}).items()):
    print(f"  {k:24} {v}")
print()
for k in ("samples", "steps_run", "effective_batch_sequences", "max_seq_len",
          "trainable_parameters", "peak_vram_gib", "wall_clock_s", "data_stats"):
    if k in d:
        print(f"  {k:24} {d[k]}")
print()
print("  load_report:", json.dumps(d.get("load_report", {})))
PY
} >> "$TMP"

# ---------------------------------------------------------------------------
section "2. The loss log — every line of it"
# ---------------------------------------------------------------------------
{
  echo "One flushed JSON object per optimizer step, written by the trainer and"
  echo "monitored live during the run.  This is the raw data behind the figure."
  echo
  echo "\$ cat $LOSS_LOG"
  cat "$LOSS_LOG" 2>/dev/null || echo "(missing)"
  echo
  echo "steps logged    : $(wc -l < "$LOSS_LOG" 2>/dev/null || echo 0)"
} >> "$TMP"

# ---------------------------------------------------------------------------
section "3. Summary — trajectory, throughput, and the schedule"
# ---------------------------------------------------------------------------
{
  echo "Computed from the same JSONL, with the same tool that draws the figure:"
  echo
  echo "\$ python3 scripts/plot_loss.py --log $LOSS_LOG --no-plot --interval 25"
  echo
  python3 scripts/plot_loss.py --log "$LOSS_LOG" --no-plot --interval 25 2>&1
} >> "$TMP"

# ---------------------------------------------------------------------------
section "4. Device use during the run (nvidia-smi sampler)"
# ---------------------------------------------------------------------------
{
  if [ -s "$SAMPLES" ]; then
    echo "A sampler appended one line every 5 s while the container was running."
    echo "WINDOW: it was started after this run was already in flight (step 121 of"
    echo "374), so the peak below is a LOWER BOUND on the run's peak.  The first"
    echo "half is covered by the smoke record (evidence/train-smoke.txt §4), which"
    echo "sampled the identical configuration from the first step."
    echo
    echo "\$ grep -o 'mem.used [0-9]* MiB' $SAMPLES | awk '{print \$2}' | sort -n | tail -1"
    grep -o 'mem.used [0-9]* MiB' "$SAMPLES" | awk '{print $2}' | sort -n | tail -1 \
      | sed 's/^/peak memory.used : /;s/$/ MiB/'
    echo "samples taken    : $(wc -l < "$SAMPLES")"
    echo
    echo "first and last rows:"
    head -1 "$SAMPLES"
    tail -1 "$SAMPLES"
    echo
    echo "rows naming a compute process (last 2):"
    grep -v '| *$' "$SAMPLES" | tail -2
  else
    echo "(no sampler output at $SAMPLES)"
  fi
  echo
  echo "\$ docker stats --no-stream (snapshot taken while the run was live)"
  grep -E 'NAME|training' "$STATS" 2>/dev/null || echo "(no snapshot)"
} >> "$TMP"

# ---------------------------------------------------------------------------
section "5. What the container itself said (docker logs)"
# ---------------------------------------------------------------------------
{
  echo "\$ docker compose --profile train logs training   # startup lines"
  docker compose --profile train logs training 2>&1 \
    | grep -E 'device:|base loaded|trainable params|adapter attached|corpus:|schedule:|adapter written|WARNING' | head -20
  echo
  echo "\$ ... # first three and last three step lines"
  docker compose --profile train logs training 2>&1 | grep -E 'step +[0-9]+/' | head -3
  echo "  ..."
  docker compose --profile train logs training 2>&1 | grep -E 'step +[0-9]+/' | tail -3
} >> "$TMP"

# ---------------------------------------------------------------------------
section "6. The adapter on disk"
# ---------------------------------------------------------------------------
{
  echo "\$ ls -l $ADAPTER"
  ls -l "$ADAPTER" 2>&1
  echo
  echo "\$ du -sh $ADAPTER"
  du -sh "$ADAPTER" 2>&1
  echo
  echo "--- adapter_config.json (the LoRA geometry actually used) ---"
  cat "$ADAPTER/adapter_config.json" 2>/dev/null || echo "(missing)"
} >> "$TMP"

# ---------------------------------------------------------------------------
section "7. Container state at record time"
# ---------------------------------------------------------------------------
{
  echo "container       : $CONTAINER"
  echo "running         : $(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo gone)"
  echo "exit code       : $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER" 2>/dev/null || echo -)"
  echo "oom killed      : $(docker inspect -f '{{.State.OOMKilled}}' "$CONTAINER" 2>/dev/null || echo -)"
  echo "image           : $(docker image inspect aiia-docker/training:stage-4 --format '{{.Size}} bytes' 2>/dev/null || echo 'not built')"
  echo
  echo "This script does not tear the container down: the adapter is read from the"
  echo "mounted directory, and removing the container is a separate, deliberate step."
} >> "$TMP"

mv "$TMP" "$OUT"
echo "written: $OUT"
