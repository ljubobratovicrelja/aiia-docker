#!/usr/bin/env bash
# Re-runnable stage-1 evidence collector.
# Brings the stack up, waits for health, sends one fixed prompt through the
# GUI container's own HTTP API (gradio -> inference), and writes every
# artifact the report needs under evidence/.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=evidence
GUI_URL="${GUI_URL:-http://localhost:7860}"
PROMPT="Reply with exactly: aiia-docker stage 1 is alive."
mkdir -p "$OUT"

echo "== docker compose up -d --build ==" | tee "$OUT/up.log"
docker compose up -d --build 2>&1 | tee -a "$OUT/up.log"

echo "== waiting for inference to be healthy ==" | tee -a "$OUT/up.log"
for i in $(seq 1 120); do
  state=$(docker inspect -f '{{.State.Health.Status}}' \
    "$(docker compose ps -q inference)" 2>/dev/null || echo "starting")
  echo "  [$i] inference: $state" | tee -a "$OUT/up.log"
  [ "$state" = "healthy" ] && break
  sleep 2
done

echo "== waiting for the GUI to answer on $GUI_URL ==" | tee -a "$OUT/up.log"
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "$GUI_URL/" || echo "000")
  echo "  [$i] gui: HTTP $code" | tee -a "$OUT/up.log"
  [ "$code" = "200" ] && break
  sleep 2
done

echo "== docker compose ps ==" | tee "$OUT/compose-ps.txt"
docker compose ps 2>&1 | tee -a "$OUT/compose-ps.txt"

echo "== prompt through the GUI container's HTTP API =="
# gradio exposes the chat function as named endpoint /chat; this is the same
# path a browser takes: host -> gui (gradio) -> inference -> reply.
EVENT_ID=$(curl -s -X POST "$GUI_URL/gradio_api/call/v2/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"$PROMPT\"}" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['event_id'])")
{
  echo "PROMPT : $PROMPT"
  echo "GUI    : $GUI_URL"
  echo "EVENT  : $EVENT_ID"
  echo "--- raw SSE stream ---"
  curl -sN "$GUI_URL/gradio_api/call/chat/$EVENT_ID"
} 2>&1 | tee "$OUT/completion.txt"

echo "== inference PortBindings (must be empty / null) ==" | tee "$OUT/portbindings.txt"
docker inspect -f '{{json .NetworkSettings.Ports}}' \
  "$(docker compose ps -q inference)" 2>&1 | tee -a "$OUT/portbindings.txt"

echo "== docker stats --no-stream ==" | tee "$OUT/stats.txt"
docker stats --no-stream 2>&1 | tee -a "$OUT/stats.txt"

echo "evidence written to $OUT/"
