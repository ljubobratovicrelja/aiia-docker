#!/usr/bin/env bash
# Stage-2 craft demonstration: prove the Docker layer cache actually does what
# the two Dockerfiles are ordered to achieve.
#
#   A. Rebuild with no changes            -> every build step CACHED.
#   B. Touch the GUI app code, rebuild    -> the naive expectation is that the
#      app layer re-runs and the pip layer above it stays CACHED.  B1 shows the
#      literal `touch` (nothing re-runs: buildkit keys COPY on the content
#      digest, not the mtime) and B2 the real content change (pip stays CACHED,
#      the app layer re-runs).  Both are shown because they differ.
#   C. Change a runtime setting in inference/Dockerfile (LLAMA_ARG_CTX_SIZE),
#      rebuild                             -> the ~1.3 GB model layer stays
#      CACHED, so no re-download happens.
#   D. Restore, rebuild                   -> back to the original image IDs,
#      working tree clean.
#
# It makes temporary edits, so it REQUIRES a clean tree to start and RESTORES it
# on exit (trap-based, so it restores even on failure).  The last thing it prints
# and the last thing in the artifact is the git status check.
#
# Re-runnable.  Writes evidence/build-cache.txt.  Exits non-zero if any
# expectation fails or the tree is left dirty.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=evidence/build-cache.txt
WATCHED=(gui/app.py inference/Dockerfile)
BACKUP=$(mktemp -d)
FAILED=0
LAST_CLASSIFIED=""

# ---------------------------------------------------------------------------
# classify: buildkit plain output -> one "CACHED/RE-RAN <instruction>" line per
# numbered step.  The step id appears on its instruction line and again on the
# result line, which is what makes this possible.
# ---------------------------------------------------------------------------
classify() {
  awk '
    # buildkit bookkeeping steps ("[internal] load …", "exporting to image",
    # provenance) are not Dockerfile instructions and only add noise here.
    /^#[0-9]+ \[/ {
      id = $1; sub(/^#/, "", id); $1 = ""; sub(/^ /, "", $0)
      cmd[id] = $0
    }
    /^#[0-9]+ CACHED/ {
      id = $1; sub(/^#/, "", id)
      if (id in cmd && cmd[id] !~ /\[internal\]|exporting|provenance/)
        printf "  CACHED  %s\n", cmd[id]
    }
    /^#[0-9]+ DONE/ {
      id = $1; sub(/^#/, "", id)
      if (id in cmd && cmd[id] !~ /\[internal\]|exporting|provenance/)
        printf "  RE-RAN  %s\n", cmd[id]
    }
  '
}

# build_and_report <label> -- <cmd...>
# Runs the build, captures the FULL --progress=plain log into the artifact, then
# appends the classified one-line-per-step view and stashes it for `expect`.
build_and_report() {
  local label="$1"; shift
  [ "${1:-}" = "--" ] && shift
  {
    echo
    echo "----------------------------------------------------------------------"
    echo "$label"
    echo "----------------------------------------------------------------------"
    echo "\$ $*"
  } >> "$OUT"

  local full rc=0
  full=$("$@" 2>&1) || rc=$?
  printf '%s\n' "$full" >> "$OUT"
  [ "$rc" -eq 0 ] || { echo "  !!! build exited $rc" >> "$OUT"; FAILED=1; }

  LAST_CLASSIFIED=$(printf '%s\n' "$full" | classify)
  {
    echo "  -- classified: one line per build step --"
    printf '%s\n' "$LAST_CLASSIFIED"
  } >> "$OUT"
}

# expect <CACHED|RE-RAN> <regex over the instruction text>
expect() {
  local want="$1" pattern="$2" hit
  hit=$(printf '%s\n' "$LAST_CLASSIFIED" | grep -E "^  ${want}[[:space:]]+.*${pattern}" | head -1 || true)
  if [ -n "$hit" ]; then
    echo "  >> VERIFIED: ${want} for step matching /${pattern}/" >> "$OUT"
    echo "     ${hit}" >> "$OUT"
  else
    echo "  >> NOT VERIFIED: no '${want}' line for a step matching /${pattern}/" >> "$OUT"
    FAILED=1
  fi
}

restore() {
  local rc=$?
  local f base
  for f in "${WATCHED[@]}"; do
    base=$(basename "$f")
    [ -f "$BACKUP/$base" ] && cp "$BACKUP/$base" "$f"
  done
  rm -rf "$BACKUP"
  return $rc
}

# ---------------------------------------------------------------------------
# Preflight: the files this script edits must match HEAD (otherwise the
# backup/restore would silently absorb someone else's uncommitted work), and
# the whole tree state is snapshotted so the exit check can compare like with
# like — including any untracked file that was already there.
# ---------------------------------------------------------------------------
for f in "${WATCHED[@]}"; do cp "$f" "$BACKUP/$(basename "$f")"; done
trap restore EXIT
trap 'echo "interrupted" >&2; exit 130' INT TERM

if ! git diff --quiet -- "${WATCHED[@]}"; then
  echo "refusing to run: uncommitted changes in ${WATCHED[*]}" >&2
  git diff --stat -- "${WATCHED[@]}" >&2
  exit 1
fi
# Status snapshots deliberately exclude this script's own artifact, which it
# rewrites by design; everything else must be untouched.
status_snapshot() { git status --short -- ":!$OUT"; }
STATUS_BEFORE=$(status_snapshot)

mkdir -p evidence
{
  echo "======================================================================"
  echo "build-cache.txt — Docker layer-cache demonstration (aiia-docker stage 2)"
  echo "======================================================================"
  echo "date            : $(date -Is)"
  echo "branch          : $(git rev-parse --abbrev-ref HEAD) ($(git rev-parse --short HEAD))"
  echo "engine / compose: $(docker version --format '{{.Server.Version}}') / $(docker compose version --short)"
  if [ -z "$STATUS_BEFORE" ]; then
    echo "working tree    : clean before the run"
  else
    echo "working tree    : these pre-existing entries, before the run:"
    printf '%s\n' "$STATUS_BEFORE" | sed 's/^/                  /'
  fi
  echo
  echo "Images before the run:"
  docker image ls --format '{{.Repository}}:{{.Tag}}  {{.ID}}  {{.Size}}' \
    | grep -E 'aiia-docker|llama.cpp' | sed 's/^/  /' || true
  echo
  echo "Read the classified view after each build:"
  echo "  CACHED = buildkit reused the layer from the local cache and did NOT"
  echo "           execute the instruction."
  echo "  RE-RAN = buildkit executed the instruction again."
  echo "For the model layer, CACHED is the difference between a 1.3 GB"
  echo "re-download and nothing at all."
  echo
  echo "NOTE on \`touch\`: buildkit keys a COPY step on the content digest of the"
  echo "copied files, not on their mtime.  A bare \`touch\` of app code therefore"
  echo "leaves the image fully cached (B1).  What the Dockerfile ordering actually"
  echo "buys is shown in B2, where the content changes: the pip layer survives,"
  echo "the app layer does not.  Both cases are recorded here rather than"
  echo "asserting the mtime story, which does not hold on this engine."
  echo
  echo "NOTE on re-running: the B2 marker line carries a unique timestamp and pid"
  echo "on every run.  That is deliberate — a fixed marker string would itself be a"
  echo "cache hit on the second run of this script and B2 would show CACHED for the"
  echo "app layer too, i.e. the demonstration would stop demonstrating.  This was"
  echo "found by re-running it."
  echo "======================================================================"
} > "$OUT"

# --- A: no changes at all -------------------------------------------------
{
  echo
  echo "======================================================================"
  echo "A. rebuild with NO changes — expect every step CACHED"
  echo "======================================================================"
} >> "$OUT"
build_and_report "A1. gui + inference, nothing changed" -- docker compose build --progress=plain
expect CACHED 'RUN pip install'
expect CACHED 'RUN mkdir -p /models'

# --- B1: literally `touch` ------------------------------------------------
{
  echo
  echo "======================================================================"
  echo "B1. touch gui/app.py and rebuild — the naive expectation is that the app"
  echo "    layer re-runs while the dependency layer above it stays cached"
  echo "======================================================================"
  echo "\$ touch gui/app.py"
} >> "$OUT"
touch gui/app.py
build_and_report "B1. build after a bare touch" -- docker compose build --progress=plain gui
expect CACHED 'RUN pip install'
expect CACHED 'COPY app.py'
{
  echo "  NOTE: everything stayed CACHED — mtime is not part of the COPY cache"
  echo "  key.  B2 changes the content instead, which is the case the Dockerfile"
  echo "  layer order is designed for."
} >> "$OUT"

# --- B2: a real content change -------------------------------------------
{
  echo
  echo "======================================================================"
  echo "B2. change the app code for real (append a comment) — this is the case"
  echo "    the Dockerfile layer order exists to handle"
  echo "======================================================================"
  # The marker is unique per run on purpose.  BuildKit caches a COPY step by
  # source content digest, so a fixed marker string would be a cache hit on the
  # second ever run of this script and B2 would show CACHED everywhere — the
  # demo would silently stop proving what it claims to prove.
  MARKER="# cache-demo marker $(date -Is) pid=$$ rand=$RANDOM"
  printf '\n%s\n' "$MARKER" >> gui/app.py
  echo "\$ printf '%s\\n' '$MARKER' >> gui/app.py"
  echo "\$ git diff --stat gui/app.py"
  git diff --stat gui/app.py
} >> "$OUT"
build_and_report "B2. build after the content change" -- docker compose build --progress=plain gui
expect CACHED 'RUN pip install'
expect RE-RAN 'COPY app.py'

{
  echo
  echo "  restoring gui/app.py from the backup"
} >> "$OUT"
cp "$BACKUP/app.py" gui/app.py

# --- C: a runtime setting change in the inference Dockerfile --------------
{
  echo
  echo "======================================================================"
  echo "C. change a RUNTIME SETTING in inference/Dockerfile:"
  echo "   LLAMA_ARG_CTX_SIZE=4096 -> 8192.  The ENV block sits below the model"
  echo "   layer on purpose, so the model layer must not be rebuilt."
  echo "======================================================================"
} >> "$OUT"
sed -i 's/LLAMA_ARG_CTX_SIZE=4096/LLAMA_ARG_CTX_SIZE=8192/' inference/Dockerfile
{
  echo "\$ sed -i 's/LLAMA_ARG_CTX_SIZE=4096/LLAMA_ARG_CTX_SIZE=8192/' inference/Dockerfile"
  echo "\$ git diff inference/Dockerfile"
  git diff inference/Dockerfile
} >> "$OUT"
build_and_report "C. build after the ENV change" -- docker compose build --progress=plain inference
expect CACHED 'RUN mkdir -p /models'

{
  echo
  echo "  restoring inference/Dockerfile from the backup"
} >> "$OUT"
cp "$BACKUP/Dockerfile" inference/Dockerfile

# --- D: restore + final rebuild ------------------------------------------
{
  echo
  echo "======================================================================"
  echo "D. tree restored; rebuild both services so the images are back to the"
  echo "   committed state.  Expect all CACHED — the restoration is itself a"
  echo "   cache hit, which is the point of content-addressed layers."
  echo "======================================================================"
} >> "$OUT"
build_and_report "D. final rebuild of both services after restoration" -- docker compose build --progress=plain
expect CACHED 'COPY app.py'
expect CACHED 'RUN mkdir -p /models'

# --- result --------------------------------------------------------------
{
  echo
  echo "======================================================================"
  echo "RESULT"
  echo "======================================================================"
  echo "images now (must match the IDs recorded at the top):"
  docker image ls --format '{{.Repository}}:{{.Tag}}  {{.ID}}  {{.Size}}' \
    | grep -E 'aiia-docker' | sed 's/^/  /' || true
  echo
  echo "\$ git status --short -- ':!$OUT'   (after the run)"
  status_snapshot
  echo
  echo "\$ git diff --stat -- gui/app.py inference/Dockerfile   (must be empty)"
  git diff --stat -- gui/app.py inference/Dockerfile
  echo "  (empty output above = the two edited files are byte-identical to HEAD)"
} >> "$OUT"

STATUS_AFTER=$(status_snapshot)
if [ "$STATUS_AFTER" != "$STATUS_BEFORE" ]; then
  {
    echo
    echo "!!! TREE STATE CHANGED — restoration FAILED"
    echo "before:"; printf '%s\n' "$STATUS_BEFORE"
    echo "after:";  printf '%s\n' "$STATUS_AFTER"
  } >> "$OUT"
  echo "FAILED: tree state changed (see $OUT)" >&2
  exit 1
fi
if [ -n "$STATUS_AFTER" ]; then
  {
    echo
    echo "tree state after == before (non-empty above only because of files"
    echo "that were already untracked before the run; no tracked file changed"
    echo "and git diff against HEAD for the two edited files is empty)."
  } >> "$OUT"
else
  {
    echo
    echo "TREE CLEAN: git status --short is empty, before and after."
  } >> "$OUT"
fi
if [ "$FAILED" -ne 0 ]; then
  echo "FAILED: an expectation was not verified (see $OUT)" >&2
  exit 1
fi
echo "OK — every expectation verified, tree restored clean; see $OUT"
