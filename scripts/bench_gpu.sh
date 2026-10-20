#!/usr/bin/env bash
# Stage-3 measured evidence: the SAME fixed prompt on the GPU-served model,
# run through the same probe as the CPU numbers, plus the CPU-vs-GPU table.
#
#   (a) inference  — the model's OpenAI-compatible API in isolation, reached
#                    from inside the compose network.  On the GPU stack this is
#                    the CUDA binary answering, still with no published port.
#   (b) gui        — the same prompt end-to-end through the published gradio
#                    HTTP API on http://localhost:7860 (browser -> gui ->
#                    inference), i.e. the GPU stack still behind the GUI.
#
# The request body is byte-for-byte the stage-2 one: temperature 0,
# max_tokens 160, chat_template_kwargs {"enable_thinking": false}.  Three runs
# per path.  Raw bodies and every response go to evidence/bench-gpu.txt; the
# CPU-vs-GPU table is appended from evidence/bench-cpu.txt and this file by
# scripts/bench_compare.py, so the compared numbers are read out of the
# committed artifact rather than retyped.
#
# Re-runnable; truncates its own artifact first so numbers never mix.
#
# Overrides:  RUNS=5  PROMPT="..."  GUI_URL=...
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=evidence/bench-gpu.txt
CPU_ARTIFACT=evidence/bench-cpu.txt
FILES=(-f compose.yaml -f compose.gpu.yaml)
RUNS="${RUNS:-3}"
# Consecutive executions of the whole measurement.  Two by default, because the
# GPU's completion-token count is not stable across executions at temperature 0;
# with one execution the artifact would be unable to show that.
EXECUTIONS="${EXECUTIONS:-2}"
GUI_URL="${GUI_URL:-http://localhost:7860}"
PROMPT="${PROMPT:-Write a Python function that returns the nth Fibonacci number. Include only the code and one short comment.}"

# The artifact is assembled in a temp file and moved into place at the very end,
# so the file a reader finds is always one complete benchmark: an artifact that
# is appended to run by run is both half-written if the script dies midway and
# silently mangled if two runs overlap.
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

# The probe shells out to `docker compose`; this is how it is pointed at the
# GPU stack (the override file keeps the service name `inference`).
export COMPOSE_FILES="compose.yaml compose.gpu.yaml"

# ---------------------------------------------------------------------------
# Preflight: the GPU stack has to be up and warm for these numbers to mean
# anything.  Start it if it is not, then wait for the healthcheck.
# ---------------------------------------------------------------------------
docker compose "${FILES[@]}" up -d --build >/dev/null
for i in $(seq 1 180); do
  state=$(docker inspect -f '{{.State.Health.Status}}' \
    "$(docker compose "${FILES[@]}" ps -q inference)" 2>/dev/null || echo starting)
  [ "$state" = "healthy" ] && break
  [ "$i" = 180 ] && { echo "gpu inference never became healthy" >&2; exit 1; }
  sleep 2
done

{
  echo "======================================================================"
  echo "bench-gpu.txt — fixed-prompt throughput on GPU (aiia-docker stage 3)"
  echo "======================================================================"
  echo "date          : $(date -Is)"
  echo "host          : $(uname -m), $(docker version --format '{{.Server.Version}}' | tr -d '\n') engine, $(docker compose version --short) compose"
  CPU_DESC=$(lscpu 2>/dev/null | grep -m1 -E 'Model name|Hardware' | cut -d: -f2- | sed 's/^ *//' || true)
  [ -n "${CPU_DESC:-}" ] || CPU_DESC=$(grep -m1 -iE 'model name|hardware' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//' || true)
  echo "cpu           : ${CPU_DESC:-unknown}  ($(nproc) threads)"
  echo "gpu           : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "gpu device    : $(docker compose "${FILES[@]}" exec -T inference nvidia-smi -L 2>/dev/null | head -1 || echo 'nvidia-smi unavailable')"
  echo "image         : $(docker inspect -f '{{.Config.Image}}' "$(docker compose "${FILES[@]}" ps -q inference)")"
  echo "compose files : compose.yaml + compose.gpu.yaml (override: build.target gpu, device reservation, offload env)"
  echo "runs per path : $RUNS"
  echo "prompt        : $PROMPT"
  echo
  echo "This is the same prompt and the same request body as evidence/bench-cpu.txt:"
  echo "  temperature 0, max_tokens 160, chat_template_kwargs {enable_thinking: false}"
  echo "Same host, same pinned model revision and sha256 — the only differences are"
  echo "the build target, the nvidia device reservation and the GPU offload settings."
  echo
  echo "Thinking-model note: the pinned Qwen3.5-2B GGUF is a reasoning model."
  echo "Both requests send {\"chat_template_kwargs\":{\"enable_thinking\":false}}"
  echo "plus a max_tokens cap; without them the reply lands in reasoning_content"
  echo "at thousands of tokens and the client times out."
  echo
  echo "Path notes:"
  echo "  inference — reached only from inside the compose network via"
  echo "              'docker compose -f compose.yaml -f compose.gpu.yaml exec -T"
  echo "              inference curl'; inference has no 'ports:' mapping, on the"
  echo "              GPU override either.  Probe pins temperature 0 and"
  echo "              max_tokens 160, so runs are deterministic and token counts"
  echo "              comparable across runs and across the two hosts."
  echo "  gui       — the published gradio HTTP API, unchanged by this stage."
  echo "              gui/app.py samples at its own temperature 0.7 / max_tokens"
  echo "              512, so run lengths differ; effective tokens/s is the"
  echo "              comparable metric.  gui tokens are counted by posting the"
  echo "              returned text to llama-server's own /tokenize."
  echo
  echo "Executions: $EXECUTIONS consecutive executions of $RUNS runs per path."
  echo "The GPU is deterministic WITHIN an execution but not across them: the same"
  echo "request at temperature 0 has been observed to emit one completion-token"
  echo "count on every run of one execution and a different one on every run of"
  echo "the next, so the completion-token count is treated as part of the"
  echo "measurement (quoted next to every number) rather than as a constant"
  echo "divided into elapsed time.  The stable quantity is effective tokens/s."
  echo "Every execution is kept in this artifact so the spread is visible instead"
  echo "of averaged away."
  echo
  echo "Comparison against $CPU_ARTIFACT is appended below by"
  echo "scripts/bench_compare.py, once per execution and once pooled."
  echo
  echo "======================================================================"
} > "$TMP"

for ex in $(seq 1 "$EXECUTIONS"); do
  EXEC_SUMMARY=$(mktemp)
  {
    echo
    echo "################################################################################"
    echo "### EXECUTION $ex of $EXECUTIONS"
    echo "################################################################################"
  } >> "$TMP"
  for mode in inference gui; do
    {
      echo
      echo "######################################################################"
      echo "# PATH: $mode — $RUNS runs (GPU, execution $ex)"
      echo "######################################################################"
    } >> "$TMP"
    for run in $(seq 1 "$RUNS"); do
      {
        echo
        echo "---------- $mode run $run/$RUNS (GPU) ----------"
        echo "started $run: $(date -Is)"
      } >> "$TMP"
      python3 scripts/bench_probe.py "$mode" "$PROMPT" 2>&1 | tee -a "$EXEC_SUMMARY" >> "$TMP"
      echo "  ...written"
    done
  done

  {
    echo
    echo "--- execution $ex summary ---"
  } >> "$TMP"
  python3 scripts/bench_probe.py --summarize < "$EXEC_SUMMARY" | tee -a "$TMP"

  {
    echo
    echo "--- execution $ex vs CPU ---"
  } >> "$TMP"
  python3 scripts/bench_compare.py "$CPU_ARTIFACT" "$EXEC_SUMMARY" \
    "$OUT (execution $ex only)" | tee -a "$TMP"
  rm -f "$EXEC_SUMMARY"
done

{
  echo
  echo "################################################################################"
  echo "### POOLED — all $EXECUTIONS executions, $RUNS runs per path each"
  echo "################################################################################"
} >> "$TMP"
grep -h '^__SUMMARY__ ' "$TMP" \
  | python3 scripts/bench_probe.py --summarize | tee -a "$TMP"

python3 scripts/bench_compare.py "$CPU_ARTIFACT" "$TMP" \
  "$OUT (all $EXECUTIONS executions pooled)" | tee -a "$TMP"

mv "$TMP" "$OUT"

echo
echo "raw output + per-execution summaries + CPU-vs-GPU tables in $OUT"
