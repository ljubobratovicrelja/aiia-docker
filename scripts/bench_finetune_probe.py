#!/usr/bin/env python3
"""Run the fixed P4 prompt suite against one served model, one arm per call.

This is the *probe* half of the base-vs-fine-tune benchmark: it owns the request
contract and nothing else.  `bench_finetune.sh` owns the server lifecycle and
calls this once per model arm.

Why every sampling parameter is sent explicitly
-----------------------------------------------
llama.cpp's server defaults (temperature 0.80, top_p 0.95, top_k 40, min_p 0.05)
are NOT the sampling the suite asks for.  If any of them is left implicit the
three arms are compared under a decode the suite never specified, so each request
carries the full `sampling` block from `bench/prompts.json` plus the prompt's own
`max_tokens`.  Reproducibility comes from a fixed seed per run index
(`seed = seed_base + run`), not from greedy decoding — the suite is deliberately
sampled, so 5 runs per prompt show spread rather than one number.

Usage:
  bench_finetune_probe.py --model-key base \
      --prompts bench/prompts.json --base-url http://127.0.0.1:8081 \
      --out /tmp/frag-base.json [--runs 5]

Writes a JSON fragment {"model_key": ..., "results": [...]} where each result is
one prompt with its `runs` list.  A failed request is retried once; if it fails
again the run is recorded with `finish_reason` prefixed `error:` and an `error`
field, never silently dropped — a missing run would make the arms incomparable.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

CHAT_URL = "/v1/chat/completions"
# Order matters only for the log line; the body is built from the suite's block.
SAMPLING_KEYS = (
    "temperature", "top_p", "top_k", "min_p",
    "presence_penalty", "repeat_penalty", "chat_template_kwargs",
)


def build_body(prompt, sampling, seed):
    """Every decode parameter the suite fixes, made explicit, plus the seed."""
    body = {
        "messages": [{"role": "user", "content": prompt["prompt"]}],
        "max_tokens": prompt["max_tokens"],
        "seed": seed,
        "stream": False,
    }
    for k in SAMPLING_KEYS:
        if k in sampling:
            body[k] = sampling[k]
    return body


def post(url, body, timeout):
    """One POST; raises on transport/HTTP failure so the caller can retry."""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    return json.loads(raw), time.monotonic() - t0


def run_one(url, body, timeout):
    """POST with one retry; returns the fields the record needs."""
    last = None
    for attempt in (1, 2):
        try:
            data, dt = post(url, body, timeout)
            ch = data["choices"][0]
            usage = data.get("usage", {})
            text = ch["message"].get("content") or ""
            return {
                "text": text,
                "usage": {
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                },
                "latency_s": round(dt, 3),
                "finish_reason": ch.get("finish_reason") or "stop",
            }
        except Exception as e:  # noqa: BLE001 - recorded, not swallowed
            last = f"{type(e).__name__}: {e}"
            if isinstance(e, urllib.error.HTTPError) and e.code < 500:
                # A 4xx will not fix itself; surface it after the single retry.
                pass
            time.sleep(1.0)
    return {
        "text": "",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "latency_s": 0.0,
        "finish_reason": "error: request failed",
        "error": last,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--prompts", default="bench/prompts.json")
    ap.add_argument("--base-url", default="http://127.0.0.1:8081")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=0, help="override runs_per_prompt")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    with open(args.prompts) as f:
        suite = json.load(f)
    sampling = suite["sampling"]
    runs_per_prompt = args.runs or int(sampling.get("runs_per_prompt", 5))
    seed_base = int(sampling["seed_base"])
    url = args.base_url.rstrip("/") + CHAT_URL

    results = []
    for prompt in suite["prompts"]:
        runs = []
        for run in range(1, runs_per_prompt + 1):
            seed = seed_base + run
            body = build_body(prompt, sampling, seed)
            got = run_one(url, body, args.timeout)
            rec: dict = {"run": run, "seed": seed}
            rec.update(got)
            rec["params"] = body           # the exact decode sent, for the record
            runs.append(rec)
            print(
                "  %-18s run %d/%d seed %d  %5.2fs  %4d tok  %s"
                % (prompt["id"], run, runs_per_prompt, seed,
                   rec["latency_s"], rec["usage"]["completion_tokens"],
                   rec["finish_reason"]),
                flush=True,
            )
        results.append({"prompt_id": prompt["id"], "model_key": args.model_key,
                        "runs": runs})

    with open(args.out, "w") as f:
        json.dump({"model_key": args.model_key, "results": results}, f, indent=2)
    print("wrote %s (%d prompts x %d runs)" % (args.out, len(results), runs_per_prompt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
