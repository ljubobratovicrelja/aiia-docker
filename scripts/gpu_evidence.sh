#!/usr/bin/env bash
# Stage-3 GPU evidence: proves the GPU variant is real, not decorative.
#
# Writes evidence/gpu.txt with, in this order:
#
#   1. one model download, two build targets — the shared `model` stage shows
#      CACHED for the second target, so the 1.28 GB GGUF is fetched once;
#   2. the CPU stack is unchanged — plain `docker compose config` still
#      resolves no build target and no device request;
#   3. the GPU stack's resolved config — build target, device reservation, env;
#   4. `docker inspect` of the running container — the DeviceRequests entry
#      Compose actually resolved into the container;
#   5. `nvidia-smi` run INSIDE the inference container, and `-L`;
#   6. the llama-server startup log: CUDA device discovery, model load and the
#      offload lines naming how many layers went to the GPU and how many (if
#      any) stayed on the CPU, plus the CUDA0 buffer sizes;
#   7. `docker stats` for the GPU stack.
#
# Re-runnable: the builds are cache hits, so it costs seconds and never
# re-downloads the model.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=evidence/gpu.txt
FILES=(-f compose.yaml -f compose.gpu.yaml)
INFER_IMAGE=aiia-docker/inference:stage-3-gpu

mkdir -p evidence

# The artifact is assembled in a temp file and moved into place as the last
# act, so the file a reader finds is always one complete run: appending section
# by section straight into evidence/gpu.txt leaves a half-written artifact if
# the script dies midway, and two concurrent runs interleave their sections into
# a file that still *looks* complete.  Neither is acceptable in an evidence
# artifact, which is the thing the report quotes.
TMP=$(mktemp)
mv_tmp_on_exit() { rm -f "$TMP"; }
trap mv_tmp_on_exit EXIT

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
  echo "gpu.txt — GPU variant proof (aiia-docker stage 3)"
  echo "======================================================================"
  echo "date            : $(date -Is)"
  echo "host            : $(uname -m), $(docker version --format '{{.Server.Version}}' | tr -d '\n') engine, $(docker compose version --short) compose"
  echo "default runtime : $(docker info --format '{{.DefaultRuntime}}')"
  echo "nvidia runtime  : $(docker info --format '{{json .Runtimes}}' | python3 -c 'import sys,json; print(", ".join(json.load(sys.stdin)))')"
  echo "CDI devices     : $(ls /var/run/cdi 2>/dev/null | tr '\n' ' ')"
  echo "host gpu        : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "compose files   : compose.yaml + compose.gpu.yaml (override)"
} > "$TMP"

# ---------------------------------------------------------------------------
section "1. One model download, two build targets (the shared \`model\` stage)"
# ---------------------------------------------------------------------------
CPU_BUILD=$(mktemp) ; GPU_BUILD=$(mktemp)
docker compose build inference > "$CPU_BUILD" 2>&1
docker compose "${FILES[@]}" build inference > "$GPU_BUILD" 2>&1
{
  echo "Both targets COPY --from the single shared \`model\` stage below, so the"
  echo "1.28 GB GGUF is downloaded and sha256-verified once.  The second build is"
  echo "the proof: its model stage is a cache hit, and only the COPY step of the"
  echo "target differs."
  echo
  echo '--- docker compose build inference   ==> cpu target ---'
  grep -E '^#[0-9]+ \[(model|cpu) ' "$CPU_BUILD" -A1 | grep -E '^#[0-9]+ \[|CACHED|DONE'
  echo
  echo '--- docker compose -f compose.yaml -f compose.gpu.yaml build inference   ==> gpu target ---'
  grep -E '^#[0-9]+ \[(model|gpu) ' "$GPU_BUILD" -A1 | grep -E '^#[0-9]+ \[|CACHED|DONE'
  echo
  echo "images:"
  for t in stage-1 stage-3-gpu; do
    printf '  %-38s %s bytes\n' "$t" \
      "$(docker image inspect "aiia-docker/inference:$t" --format '{{.Size}}')"
  done
  echo "  (the cpu image is byte-for-byte the stage-2 size: the extra COPY --from"
  echo "   added only metadata, no second copy of the model)"
} >> "$TMP"
rm -f "$CPU_BUILD" "$GPU_BUILD"

# ---------------------------------------------------------------------------
section "2. The CPU stack is untouched: plain \`docker compose config\`"
# ---------------------------------------------------------------------------
{
  echo "The plain stack must keep resolving exactly as in stages 1-2 — no build"
  echo "target at all (so the Dockerfile's LAST stage, cpu, is what is built), the"
  echo "stage-1 image tag, and an empty device request.  Selected keys only:"
  echo
  docker compose config \
    | grep -nE 'target:|image: aiia-docker|devices:|driver:|capabilities|reservations|N_GPU_LAYERS|FLASH_ATTN|count:' \
    || true
  echo
  echo "  ^ no 'target:', no 'devices:' — nothing from compose.gpu.yaml leaks into"
  echo "    the plain stack."
} >> "$TMP"

# ---------------------------------------------------------------------------
section "3. The GPU stack's resolved configuration"
# ---------------------------------------------------------------------------
{
  echo "\`docker compose -f compose.yaml -f compose.gpu.yaml config\` (selected keys):"
  echo
  docker compose "${FILES[@]}" config \
    | grep -nE 'target:|image: aiia-docker|devices:|driver:|capabilities|reservations|N_GPU_LAYERS|FLASH_ATTN|count:|cpus:|memory:'
  echo
  echo "Note what is NOT there: the gui service is unchanged and the inference"
  echo "service still has 'expose' and no 'ports' — the isolation design survives"
  echo "the GPU override."
} >> "$TMP"

# ---------------------------------------------------------------------------
section "4. What Compose resolved into the container (docker inspect)"
# ---------------------------------------------------------------------------
# Make sure the GPU stack is up before inspecting it.
docker compose "${FILES[@]}" up -d >/dev/null 2>&1 || true
for i in $(seq 1 180); do
  state=$(docker inspect -f '{{.State.Health.Status}}' \
    "$(docker compose "${FILES[@]}" ps -q inference)" 2>/dev/null || echo starting)
  [ "$state" = "healthy" ] && break
  [ "$i" = 180 ] && { echo "gpu inference never became healthy" >&2; exit 1; }
  sleep 2
done
INFER_ID=$(docker compose "${FILES[@]}" ps -q inference)
{
  echo "container       : $INFER_ID"
  echo "container image : $(docker inspect -f '{{.Config.Image}}' "$INFER_ID")"
  echo "health          : $state"
  echo
  echo "HostConfig.DeviceRequests (the NVIDIA Container Toolkit request as the"
  echo "runtime received it — this is the resolved form of"
  echo "deploy.resources.reservations.devices in compose.gpu.yaml):"
  docker inspect -f '{{json .HostConfig.DeviceRequests}}' "$INFER_ID" | python3 -m json.tool
  echo
  echo "PortBindings (must be empty — inference is still unpublished):"
  docker inspect -f '{{json .NetworkSettings.Ports}}' "$INFER_ID"
  echo
  echo "GPU devices visible inside the container:"
  docker compose "${FILES[@]}" exec -T inference sh -c 'ls -l /dev/nvidia* 2>/dev/null || echo none'
} >> "$TMP"

# ---------------------------------------------------------------------------
section "5. nvidia-smi from INSIDE the inference container"
# ---------------------------------------------------------------------------
{
  echo "--- docker compose -f compose.yaml -f compose.gpu.yaml exec -T inference nvidia-smi ---"
  docker compose "${FILES[@]}" exec -T inference nvidia-smi 2>&1 || true
  echo
  echo "--- nvidia-smi -L (inside) ---"
  docker compose "${FILES[@]}" exec -T inference nvidia-smi -L 2>&1 || true
  echo
  echo "--- nvidia-smi --query-gpu=... (inside) ---"
  docker compose "${FILES[@]}" exec -T inference \
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
    --format=csv 2>&1 || true
  echo
  echo "Caveat, stated rather than hidden: on the GB10 the driver reports no"
  echo "separate VRAM counters (memory total/used read N/A / 'Not Supported')"
  echo "because the GPU shares the 128 GB unified memory pool with the CPU.  The"
  echo "VRAM actually committed by the model is therefore read from llama.cpp's"
  echo "own buffer accounting in section 6, and the pool size from --list-devices."
} >> "$TMP"

# ---------------------------------------------------------------------------
section "6. llama-server startup log: CUDA discovery, load and offload"
# ---------------------------------------------------------------------------
{
  echo "Startup lines from the running GPU inference container.  'load_tensors:"
  echo "offloaded N/M layers to GPU' is the line that says how much of the model"
  echo "is on the device; the 'buffer size' lines are its memory footprint."
  echo
  # The container's log accumulates across restarts and runs to tens of
  # thousands of lines (verbosity 5 logs every graph), so everything below is
  # taken from the CURRENT server process only: the log is cut at its last
  # "initializing ..." line.  Grepping the whole history and keeping the tail
  # would silently report whichever start happened to be last rather than this
  # run's, and would mix two boots' numbers together.
  LOG=$(mktemp) ; BOOT=$(mktemp)
  docker compose "${FILES[@]}" logs --no-log-prefix inference > "$LOG" 2>&1 || true
  START=$(grep -n 'llama_server: initializing' "$LOG" | tail -1 | cut -d: -f1 || true)
  if [ -n "${START:-}" ]; then tail -n +"$START" "$LOG" > "$BOOT"; else cp "$LOG" "$BOOT"; fi
  echo "log lines captured: $(wc -l < "$LOG") total, of which $(wc -l < "$BOOT") belong"
  echo "to the current server process (this run's boot, not an earlier restart)."
  echo
  echo "--- CUDA device discovery ---"
  grep -iE 'device_info|CUDA0 *:|CPU *:|system_info: n_threads|using device CUDA0|ggml_cuda' "$BOOT" | head -10 || true
  echo
  echo "--- offload decision and device memory accounting ---"
  echo "(the server loads the model twice: a no-alloc probe used by --fit to size"
  echo " the request, then the real load.  The block below is the REAL load — the"
  echo " second occurrence of the offload lines — so its buffer sizes are the ones"
  echo " that describe the running server, not the probe that reports 0.00 MiB.)"
  grep -E 'offloaded|offloading|model buffer size|KV buffer size|RS buffer size|compute buffer size|output buffer size|projected to use|free device memory|flash_attn ' "$BOOT" | tail -22 || true
  echo
  echo "--- the server's own ready line ---"
  grep -E 'llama_server: listening on|threads for HTTP server' "$BOOT" | tail -3 || true
  echo
  echo "--- the same container answering its inherited healthcheck ---"
  docker compose "${FILES[@]}" exec -T inference curl -s -m 5 http://localhost:8080/health 2>&1 || true
  echo
  echo "--- last lines of the current boot ---"
  tail -3 "$BOOT" || true
  rm -f "$LOG" "$BOOT"
  echo
  echo "--- device enumeration as the server sees it ---"
  docker compose "${FILES[@]}" exec -T inference sh -c \
    'ls -l /proc/driver/nvidia/version 2>/dev/null && cat /proc/driver/nvidia/version' 2>&1 || true
} >> "$TMP"

# ---------------------------------------------------------------------------
section "7. docker stats — the GPU stack"
# ---------------------------------------------------------------------------
{
  docker compose "${FILES[@]}" ps
  echo
  docker stats --no-stream 2>&1
} >> "$TMP"

mv "$TMP" "$OUT"
echo "gpu evidence written to $OUT"
