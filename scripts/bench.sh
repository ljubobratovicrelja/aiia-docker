#!/usr/bin/env bash
# Stage-2 measured evidence: throughput and latency of a FIXED prompt, on CPU,
# down the two paths the assignment separates.
#
#   (a) inference  — the model's OpenAI-compatible API in isolation.  The port is
#                    `expose:`-only, so the call is made from inside the compose
#                    network with `docker compose exec -T inference curl ...`.
#   (b) gui        — the same prompt end-to-end through the published gradio HTTP
#                    API on http://localhost:7860 (browser -> gui -> inference).
#
# Three runs each.  Per run: elapsed seconds, completion tokens, effective
# tokens/s.  Plus the median across runs.  Raw output, including every response
# body, goes to evidence/bench-cpu.txt.
#
# Re-runnable; truncates its own artifact first so numbers never mix.
#
# Overrides:  RUNS=5  PROMPT="..."  GUI_URL=...
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=evidence/bench-cpu.txt
RUNS="${RUNS:-3}"
GUI_URL="${GUI_URL:-http://localhost:7860}"
PROMPT="${PROMPT:-Write a Python function that returns the nth Fibonacci number. Include only the code and one short comment.}"

# ---------------------------------------------------------------------------
# Preflight: the stack has to be up (and the model warm) for these numbers to
# mean anything.  Start it if it is not, then wait for the healthcheck.
# ---------------------------------------------------------------------------
docker compose up -d >/dev/null
for i in $(seq 1 120); do
  state=$(docker inspect -f '{{.State.Health.Status}}' "$(docker compose ps -q inference)" 2>/dev/null || echo starting)
  [ "$state" = "healthy" ] && break
  [ "$i" = 120 ] && { echo "inference never became healthy" >&2; exit 1; }
  sleep 2
done

{
  echo "======================================================================"
  echo "bench-cpu.txt — fixed-prompt throughput on CPU (aiia-docker stage 2)"
  echo "======================================================================"
  echo "date          : $(date -Is)"
  echo "host          : $(uname -m), $(docker version --format '{{.Server.Version}}' | tr -d '\n') engine, $(docker compose version --short) compose"
  CPU_DESC=$(lscpu 2>/dev/null | grep -m1 -E 'Model name|Hardware' | cut -d: -f2- | sed 's/^ *//' || true)
  [ -n "${CPU_DESC:-}" ] || CPU_DESC=$(grep -m1 -iE 'model name|hardware' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//' || true)
  echo "cpu           : ${CPU_DESC:-unknown}  ($(nproc) threads)"
  echo "runs per path : $RUNS"
  echo "prompt        : $PROMPT"
  echo
  echo "Thinking-model note: the pinned Qwen3.5-2B GGUF is a reasoning model."
  echo "Both requests send {\"chat_template_kwargs\":{\"enable_thinking\":false}}"
  echo "plus a max_tokens cap; without them the reply lands in reasoning_content"
  echo "at thousands of tokens and the client times out."
  echo
  echo "Path notes:"
  echo "  inference — reached only from inside the compose network via"
  echo "              'docker compose exec -T inference curl'; inference has no"
  echo "              'ports:' mapping at all.  Probe pins temperature 0 and"
  echo "              max_tokens 160, so runs are deterministic and token counts"
  echo "              comparable across runs."
  echo "  gui       — the published gradio HTTP API.  gui/app.py samples at its"
  echo "              own temperature 0.7 / max_tokens 512, so run lengths differ"
  echo "              slightly from the isolated path; that is precisely why the"
  echo "              comparable metric is effective tokens/s, not raw elapsed."
  echo "  gui tokens are counted by posting the returned text to llama-server's"
  echo "  own /tokenize, which was verified to equal usage.completion_tokens"
  echo "  exactly on the isolated path (gradio's stream carries no usage block)."
  echo
  echo "======================================================================"
} > "$OUT"

for mode in inference gui; do
  {
    echo
    echo "######################################################################"
    echo "# PATH: $mode — $RUNS runs"
    echo "######################################################################"
  } >> "$OUT"
  for run in $(seq 1 "$RUNS"); do
    {
      echo
      echo "---------- $mode run $run/$RUNS ----------"
      echo "started $run: $(date -Is)"
    } >> "$OUT"
    python3 scripts/bench_probe.py "$mode" "$PROMPT" >> "$OUT" 2>&1
    echo "  ...written"
  done
done

grep -h '^__SUMMARY__ ' "$OUT" \
  | python3 scripts/bench_probe.py --summarize | tee -a "$OUT"

echo
echo "raw output + summary in $OUT"
