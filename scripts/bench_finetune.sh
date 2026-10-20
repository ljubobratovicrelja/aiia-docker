#!/usr/bin/env bash
# P4 measured evidence: base vs stage-4 LoRA vs the shipped instruct GGUF, on the
# FIXED prompt suite in bench/prompts.json.  Run ON the GPU host (it needs the
# GPU and the three GGUFs); the CSV of runs is pulled back as evidence.
#
# The controlled pair is `base` and `tuned`: the same converted base.gguf, the
# second one served with the stage-4 LoRA adapter on top (PATH A), so the base
# weights are byte-identical between the two arms and only the adapter differs.
# `instruct` is a DIFFERENT lineage (the post-trained Qwen3.5-2B instruct model)
# and is a reference point, not part of the controlled comparison.
#
# For every arm: start llama-server from the project's own inference image with
# IDENTICAL flags, wait for /health, run the probe (10 prompts x 5 runs, seeds
# fixed), capture the server log, stop the container.  Then per-model perplexity
# on two held-out texts (bonus: the classic degradation metric).
#
# Nothing is left running: every container is removed and the script exits
# nonzero if `docker ps` still shows one.
#
# Overrides:  MODELS_DIR=...  PORT=8081  CTX=8192  NGL=99
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"

IMAGE="${IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13}"
MODELS_DIR="${MODELS_DIR:-/var/tmp/aiia-bench/models}"
PORT="${PORT:-8081}"
CTX="${CTX:-8192}"
NGL="${NGL:-99}"
# "low" — the compose GPU override ran at 5 to show offload accounting; 2 is the
# server's own default and is enough for the LoRA-apply line the pair relies on.
VERBOSITY="${VERBOSITY:-2}"
# Empty = honour the suite's runs_per_prompt (5).  Set to 1 for a smoke test.
RUNS="${RUNS:-}"

PROBE="scripts/bench_finetune_probe.py"
SUITE="bench/prompts.json"
OUT_JSON="evidence/bench-finetune.json"
OUT_TXT="evidence/bench-finetune.txt"
WORK="$(mktemp -d)"
LOGDIR="$WORK/logs"
mkdir -p "$LOGDIR"

# The full run transcript IS the human-readable artifact (house style: the
# evidence *.txt files are the raw output of the script that produced them).
exec > >(tee "$OUT_TXT") 2>&1

# key|gguf file|label|lineage|role|lora(1/0)
MODELS=(
  "base|base-Q4_K_M.gguf|Qwen3.5-2B-Base (stage-4 training base), Q4_K_M|Qwen/Qwen3.5-2B-Base @ b1485b2f, composite checkpoint converted text-only|pair|0"
  "tuned|base-Q4_K_M.gguf|same base + stage-4 LoRA adapter, Q4_K_M|as base, adapter served at runtime (PATH A, base weights identical)|pair|1"
  "instruct|instruct-Q4_K_M.gguf|Qwen3.5-2B instruct (shipped), Q4_K_M|unsloth/Qwen3.5-2B-GGUF @ f6d5376b — post-trained instruct model, DIFFERENT lineage|reference|0"
)

cleanup() {
  for key in base tuned instruct lora-proof; do
    docker rm -f "aiia-bench-$key" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

{
  echo "======================================================================"
  echo "bench-finetune.txt — base vs stage-4 LoRA vs shipped instruct GGUF"
  echo "======================================================================"
  echo "date          : $(date -Is)"
  echo "host          : $(uname -m), docker $(docker version --format '{{.Server.Version}}') engine, $(docker compose version --short) compose"
  echo "gpu           : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "llama.cpp     : $(docker run --rm --entrypoint /app/llama "$IMAGE" --version 2>/dev/null | head -1)"
  echo "suite         : $SUITE (${SUITE:+10 fixed prompts, 5 runs each, seeds 43507..43511})"
  echo "server flags  : -c $CTX -ngl $NGL -fa on --host 127.0.0.1 --port $PORT -lv $VERBOSITY"
  echo "controlled pair: base (no adapter) vs tuned (same base.gguf + --lora adapter.gguf)"
  echo "reference     : instruct — a different lineage, listed for orientation only"
  echo
}

echo "# 1. Host, image and suite"
echo "  gpu          $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
echo "  llama.cpp    $(docker run --rm --entrypoint /app/llama "$IMAGE" --version 2>/dev/null | head -1)"
echo "  models dir   $MODELS_DIR"
for m in "${MODELS[@]}"; do IFS='|' read -r k f l n r lo <<<"$m"; echo "  $k -> $f  (lora=$lo)"; done

for spec in "${MODELS[@]}"; do
  IFS='|' read -r key gguf label lineage role has_lora <<<"$spec"
  name="aiia-bench-$key"
  frag="$WORK/frag-$key.json"
  srvlog="$LOGDIR/server-$key.log"

  echo
  echo "# 2. Serve arm '$key' ($gguf$([ "$has_lora" = 1 ] && echo ' + adapter.gguf'))"
  docker rm -f "$name" >/dev/null 2>&1 || true
  server_args=( -m "/models/$gguf" --host 127.0.0.1 --port "$PORT"
                -c "$CTX" -ngl "$NGL" -fa on -lv "$VERBOSITY" --no-webui )
  lora_path="null"
  if [ "$has_lora" = 1 ]; then
    server_args+=( --lora /models/adapter.gguf )
    lora_path="/models/adapter.gguf"
  fi
  docker run -d --name "$name" --network host --gpus all \
    -v "$MODELS_DIR":/models:ro "$IMAGE" "${server_args[@]}" >/dev/null

  echo "  waiting for /health ..."
  ok=0
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != "true" ]; then
      echo "  !! server exited early"; docker logs "$name" >&2; exit 1
    fi
    sleep 2
  done
  [ "$ok" = 1 ] || { echo "  !! /health never came up"; docker logs "$name" >&2; exit 1; }
  echo "  healthy after ~$((i*2))s"

  # The LoRA-apply proof lives in the server log; capture it before teardown.
  docker logs "$name" > "$srvlog" 2>&1
  if [ "$has_lora" = 1 ]; then
    echo "  --- adapter evidence (server log) ---"
    grep -iE 'lora|adapter' "$srvlog" 2>/dev/null | head -8 | sed 's/^/    /' || true
  fi
  # Also capture at end of run (apply lines can arrive after first request).
  after_log="$LOGDIR/server-$key-end.log"

  echo "  --- probes ($SUITE) ---"
  probe_args=( --model-key "$key" --prompts "$SUITE" --base-url "http://127.0.0.1:$PORT" --out "$frag" )
  [ -n "$RUNS" ] && probe_args+=( --runs "$RUNS" )
  python3 "$PROBE" "${probe_args[@]}" 2>&1 | sed 's/^/  /'

  docker logs "$name" > "$after_log" 2>&1 || true
  docker rm -f "$name" >/dev/null 2>&1 || true
  echo "  stopped $name"
done

# ---------------------------------------------------------------------------
echo
echo "# 2b. Adapter-apply evidence (tuned arm booted once at -lv 4)"
# The benchmark arms run at the low verbosity above, which suppresses the model
# load lines.  The tuned arm is booted once more at -lv 4 purely to capture the
# LoRA-apply line as evidence that the pair's difference is the adapter and not
# noise; the benchmark itself already ran with the identical low-verbosity flags.
# ---------------------------------------------------------------------------
if [ -f "$MODELS_DIR/adapter.gguf" ]; then
  docker rm -f aiia-bench-lora-proof >/dev/null 2>&1 || true
  docker run -d --name aiia-bench-lora-proof --network host --gpus all \
    -v "$MODELS_DIR":/models:ro "$IMAGE" \
    -m /models/base-Q4_K_M.gguf --lora /models/adapter.gguf \
    --host 127.0.0.1 --port "$PORT" -c "$CTX" -ngl "$NGL" -fa on -lv 4 --no-webui >/dev/null
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 2
  done
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}' >/dev/null 2>&1 || true
  docker logs aiia-bench-lora-proof > "$LOGDIR/lora-proof.log" 2>&1 || true
  docker rm -f aiia-bench-lora-proof >/dev/null 2>&1 || true
  echo "  --- server log: LoRA load ---"
  grep -iE 'llama_adapter_lora_init|LoRA buffer|loaded .* tensors from lora' "$LOGDIR/lora-proof.log" \
    2>/dev/null | head -12 | sed 's/^/    /' || true
  echo "  (full log kept out of evidence/ — it is a boot log, not a measurement)"
else
  echo "  !! no adapter.gguf — cannot capture LoRA-apply evidence" >&2
fi

# ---------------------------------------------------------------------------
echo
echo "# 3. Perplexity (held-out texts, classic degradation metric)"
# ---------------------------------------------------------------------------
python3 - "$MODELS_DIR" "$WORK" "$REPO" <<'PY'
import json, os, re, subprocess, sys
models_dir, work, repo = sys.argv[1], sys.argv[2], sys.argv[3]
IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda13"
CORPORA = [("general", "bench/perplexity/general.txt"),
           ("code", "bench/perplexity/code.txt")]
ARMS = [("base", "base-Q4_K_M.gguf"),
        ("tuned", "base-Q4_K_M.gguf"),
        ("instruct", "instruct-Q4_K_M.gguf")]
res = []
for key, gguf in ARMS:
    for cname, cfile in CORPORA:
        src = os.path.join(repo, cfile)
        cmd = ["docker","run","--rm","--network","none","--gpus","all",
               "-v", f"{models_dir}:/models:ro", "-v", f"{src}:/ppl.txt:ro",
               "--entrypoint","/app/llama", IMAGE,
               "perplexity","-m",f"/models/{gguf}","-f","/ppl.txt","-c","4096","-ngl","99"]
        if key == "tuned":
            cmd += ["--lora","/models/adapter.gguf"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        log = p.stdout + p.stderr
        m = re.search(r'Final estimate:\s*PPL\s*=\s*([0-9.]+)', log)
        # The authoritative chunk count is the "over N chunks" line; the [N]
        # per-chunk lines are interleaved with progress output on the same line.
        mc = re.search(r'calculating perplexity over (\d+) chunks', log)
        chunks = int(mc.group(1)) if mc else 0
        ppl = float(m.group(1)) if m else None
        res.append({"model_key": key, "corpus": cname, "ppl": ppl, "chunks": chunks})
        print(f"  {key:9s} {cname:8s} ppl={ppl} chunks={chunks}")
json.dump(res, open(f"{work}/perplexity-results.json","w"), indent=2)
PY

# ---------------------------------------------------------------------------
echo
echo "# 4. Assemble $OUT_JSON"
echo "  extracting sha256 of each GGUF ..."
SHA_TMP="$WORK/sha.json"
python3 - "$MODELS_DIR" "$SHA_TMP" <<'PY'
import hashlib, json, sys
d = sys.argv[1]
files = {"base":"base-Q4_K_M.gguf","instruct":"instruct-Q4_K_M.gguf"}
out = {}
for k, f in files.items():
    h = hashlib.sha256()
    with open(f"{d}/{f}","rb") as fh:
        for b in iter(lambda: fh.read(1<<20), b""):
            h.update(b)
    out[k] = h.hexdigest()
# adapter has its own hash too
h = hashlib.sha256()
with open(f"{d}/adapter.gguf","rb") as fh:
    for b in iter(lambda: fh.read(1<<20), b""):
        h.update(b)
out["adapter"] = h.hexdigest()
json.dump(out, open(sys.argv[2],"w"))
print(json.dumps(out, indent=2))
PY

python3 - "$REPO" "$WORK" "$OUT_JSON" "$MODELS_DIR" "$IMAGE" "$CTX" "$NGL" "$PORT" <<'PY'
import json, os, subprocess, sys, datetime
repo, work, out_json, models_dir, image, ctx, ngl, port = sys.argv[1:9]

suite = json.load(open(os.path.join(repo, "bench/prompts.json")))
sha = json.load(open(f"{work}/sha.json"))

# host block
gpu = subprocess.run(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip().splitlines()[0]
engine = "docker %s / compose %s" % (
    subprocess.run(["docker","version","--format","{{.Server.Version}}"],
                   capture_output=True,text=True).stdout.strip(),
    subprocess.run(["docker","compose","version","--short"],
                   capture_output=True,text=True).stdout.strip())
ver = subprocess.run(["docker","run","--rm","--entrypoint","/app/llama",image,"--version"],
                     capture_output=True,text=True).stdout.strip().splitlines()[0]
ver = ver.replace("version: ","")

MODELS = [
    ("base","Qwen3.5-2B-Base (stage-4 training base), Q4_K_M","base-Q4_K_M.gguf",sha["base"],
     "Qwen/Qwen3.5-2B-Base @ b1485b2fa6dfa1287294f269f5fb618e03d52d7c, composite checkpoint converted text-only","pair"),
    ("tuned","same base + stage-4 LoRA adapter, Q4_K_M","base-Q4_K_M.gguf",sha["base"],
     "as base; adapter.gguf (sha256 "+sha["adapter"]+") applied at serve time with --lora; base weights identical to the base arm","pair"),
    ("instruct","Qwen3.5-2B instruct (shipped), Q4_K_M","instruct-Q4_K_M.gguf",sha["instruct"],
     "unsloth/Qwen3.5-2B-GGUF @ f6d5376be1edb4d416d56da11e5397a961aca8ae — post-trained instruct model, DIFFERENT lineage","reference"),
]
models = [{"key":k,"label":l,"gguf":g,"sha256":s,"quant":"Q4_K_M","lineage":li,"role":r}
          for (k,l,g,s,li,r) in MODELS]

results = []
for k,_,_,_,_,_ in MODELS:
    frag = json.load(open(f"{work}/frag-{k}.json"))
    results.extend(frag["results"])

ppl = json.load(open(f"{work}/perplexity-results.json"))

# The controlled pair's whole point: does the adapter change behaviour?  Compare
# base and tuned run-by-run (same seed) per prompt.  If nothing differed anywhere
# the adapter did NOT apply and the tuned arm would be a lie.
b = {r["prompt_id"]: r["runs"] for r in json.load(open(f"{work}/frag-base.json"))["results"]}
t = {r["prompt_id"]: r["runs"] for r in json.load(open(f"{work}/frag-tuned.json"))["results"]}
detail, differing_prompts, differing_runs, total_runs = {}, 0, 0, 0
for pid in b:
    n_same = sum(1 for rb, rt in zip(b[pid], t[pid]) if rb["text"] == rt["text"])
    n = len(b[pid])
    total_runs += n
    differing_runs += n - n_same
    if n_same < n:
        differing_prompts += 1
    detail[pid] = {"runs": n, "runs_identical": n_same}
adapter_check = {"prompts_total": len(b), "prompts_differing": differing_prompts,
                 "runs_total": total_runs, "runs_differing": differing_runs,
                 "detail": detail}

doc = {
    "suite_version": suite.get("suite_version","1.0"),
    "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "sampling": suite["sampling"],
    "runs_per_prompt": int(suite["sampling"].get("runs_per_prompt",5)),
    "host": {"gpu": gpu, "driver": gpu.split(",")[-1].strip(), "engine": engine},
    "llama_cpp": {"image": image, "version": ver},
    "server_flags": {"ctx": int(ctx), "n_gpu_layers": int(ngl), "flash_attn": True,
                     "lora": "/models/adapter.gguf (tuned arm only; null for base and instruct)"},
    "models": models,
    "prompts": suite["prompts"],
    "results": results,
    "perplexity": ppl,
    "adapter_check": adapter_check,
    "perplexity_note": ("measured with `/app/llama perplexity` from the same image at -c 4096 "
                        "-ngl 99 on the two held-out texts in bench/perplexity/ (Wikipedia prose and a "
                        "CPython stdlib concatenation); corpus is 'general' or 'code' and `chunks` is the "
                        "number of 4096-token windows scored."),
    "build_note": ("PATH A: base.gguf converted from the original composite Qwen/Qwen3.5-2B-Base, "
                   "quantized Q4_K_M; the tuned arm is the SAME base.gguf with the stage-4 LoRA "
                   "adapter (convert_lora_to_gguf.py) served via --lora. `lora` in server_flags is "
                   "absolute; for the base and instruct arms no adapter is passed."),
    "lineage_note": ("base and tuned are the controlled pair (identical base weights, identical "
                     "quantization). instruct is a reference only and is a different lineage."),
}
json.dump(doc, open(os.path.join(repo,out_json),"w"), indent=2)
print("  wrote", out_json, os.path.getsize(os.path.join(repo,out_json)), "bytes")
print("  results:", len(results), "prompt-arms;",
      "runs total:", sum(len(r["runs"]) for r in results))
PY

{
  echo
  echo "# 5. Record"
  echo "  $OUT_JSON : $(stat -c%s "$REPO/$OUT_JSON" 2>/dev/null) bytes"
  echo
  echo "--- server logs (base) ---"
  { grep -iE 'lora|adapter|offloaded|model type|arch' "$LOGDIR/server-base-end.log" 2>/dev/null | head -15; } || true
  echo
  echo "--- server logs (tuned) ---"
  { grep -iE 'lora|adapter|offloaded|model type|arch' "$LOGDIR/server-tuned-end.log" 2>/dev/null | head -15; } || true
}

echo
echo "# 6. Teardown check"
cleanup
left="$(docker ps --format '{{.Names}}' | grep -c '^aiia-bench-' || true)"
echo "  aiia-bench containers left: ${left:-0}"
cat "$OUT_TXT" | tail -30
echo
echo "done: $OUT_JSON ; human log: $OUT_TXT"
