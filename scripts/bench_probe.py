#!/usr/bin/env python3
"""One benchmark probe against the running aiia-docker stack.

Two paths, same fixed prompt:

  inference  the model's own OpenAI-compatible API, in isolation.  The port is
             `expose:`-only, so the request has to be made from inside the
             compose network: `docker compose exec -T inference curl ...`.
             Token count comes straight out of `usage.completion_tokens`.
  gui        the end-to-end path a browser takes: HTTP -> gui container
             (gradio) -> inference container -> back.  Gradio's response stream
             carries only the text, so the completion token count is obtained by
             posting that text to llama-server's own /tokenize endpoint inside
             the inference container.  Verified to agree exactly with
             `usage.completion_tokens` on the isolated path.

The prompt is a thinking-model trap unless handled: the pinned Qwen3.5-2B GGUF
fills `reasoning_content` for thousands of tokens and never writes `content`
unless `chat_template_kwargs.enable_thinking` is false.  Both the request body
here and `gui/app.py` send it, plus a `max_tokens` cap.

Usage:
  bench_probe.py <inference|gui>       # one run, raw response + summary line
  bench_probe.py --summarize           # read summary lines on stdin, print table
"""

import json
import statistics
import subprocess
import sys
import time
import urllib.request

REPO = __file__.rsplit("/scripts/", 1)[0]
GUI_URL = "http://localhost:7860"
INFER_URL = "http://localhost:8080/v1/chat/completions"
TOKENIZE_URL = "http://localhost:8080/tokenize"
MAX_TOKENS = 160
PROMPT = (
    "Write a Python function that returns the nth Fibonacci number. "
    "Include only the code and one short comment."
)
SEP = "__HTTP_META__"


def compose(*args, **kw):
    """Run `docker compose ...` from the repo root."""
    return subprocess.run(
        ["docker", "compose", *args], cwd=REPO, capture_output=True, text=True, **kw
    )


def in_container_curl(url, body=None, extra=()):
    """curl inside the inference container; returns stdout."""
    cmd = ["exec", "-T", "inference", "curl", "-s", *extra, url]
    if body is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", body]
    r = compose(*cmd)
    if r.returncode != 0:
        raise RuntimeError(f"docker compose exec failed: {r.stderr.strip()}")
    return r.stdout


def body_for(prompt):
    return json.dumps(
        {
            "model": "aiia-local",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )


def count_tokens(text):
    """Exact token count of a string, via the serving container's tokenizer."""
    out = in_container_curl(TOKENIZE_URL, json.dumps({"content": text}))
    return len(json.loads(out)["tokens"])


def probe_inference(prompt):
    """Model API in isolation.  Elapsed time is curl's own time_total."""
    raw = in_container_curl(
        INFER_URL,
        body_for(prompt),
        extra=("-w", f"\\n{SEP} %{{http_code}} %{{time_total}}"),
    )
    payload, _, meta = raw.rpartition(SEP)
    status, elapsed = meta.split()
    if status != "200":
        raise RuntimeError(f"inference HTTP {status}")
    data = json.loads(payload)
    return {
        "raw": json.dumps(data, indent=2),
        "elapsed_s": float(elapsed),
        "completion_tokens": int(data["usage"]["completion_tokens"]),
        "text": data["choices"][0]["message"].get("content") or "",
    }


def probe_gui(prompt):
    """End-to-end through the published gradio port.  Elapsed is host wall clock
    from the POST that starts the job to the last byte of the result stream."""
    t0 = time.monotonic()
    req = urllib.request.Request(
        f"{GUI_URL}/gradio_api/call/v2/chat",
        data=json.dumps({"message": prompt}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        event_id = json.load(r)["event_id"]

    with urllib.request.urlopen(
        f"{GUI_URL}/gradio_api/call/chat/{event_id}", timeout=600
    ) as r:
        stream = r.read().decode()
    elapsed = time.monotonic() - t0

    # SSE: one or more `event: <name>` / `data: <json>` pairs; the result is the
    # `complete` event, whose data is a JSON array [value, ...].
    result = None
    for block in stream.split("\n\n"):
        lines = dict(
            line.split(": ", 1) for line in block.strip().splitlines() if ": " in line
        )
        if lines.get("event") == "error":
            raise RuntimeError(f"gradio error event: {lines.get('data')}")
        if lines.get("event") == "complete":
            result = json.loads(lines["data"])[0]
    if result is None:
        raise RuntimeError(f"no complete event in stream: {stream[:400]}")
    return {
        "raw": stream,
        "elapsed_s": elapsed,
        "completion_tokens": count_tokens(result),
        "text": result,
    }


def summarize(stream):
    prefix = "__SUMMARY__ "
    rows = []
    for line in stream:
        line = line.strip()
        if line.startswith(prefix):
            line = line[len(prefix):]
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    if not rows:
        print("no summary lines on stdin", file=sys.stderr)
        return 1
    out = []
    out.append("")
    out.append("=" * 78)
    out.append("SUMMARY — fixed prompt, %d inference + %d gui runs"
               % (len([r for r in rows if r["mode"] == "inference"]),
                  len([r for r in rows if r["mode"] == "gui"])))
    out.append("=" * 78)
    out.append(
        "%-11s %-5s %10s %10s %12s" % ("path", "run", "elapsed_s", "comp_tok", "tokens/s")
    )
    med = {}
    for mode in ("inference", "gui"):
        mrows = [r for r in rows if r["mode"] == mode]
        for i, r in enumerate(mrows, 1):
            out.append(
                "%-11s %-5d %10.3f %10d %12.2f"
                % (mode, i, r["elapsed_s"], r["completion_tokens"], r["tokens_per_s"])
            )
        if mrows:
            med[mode] = {
                "elapsed_s": statistics.median(r["elapsed_s"] for r in mrows),
                "completion_tokens": statistics.median(r["completion_tokens"] for r in mrows),
                "tokens_per_s": statistics.median(r["tokens_per_s"] for r in mrows),
                "pooled_tokens_s": sum(r["completion_tokens"] for r in mrows)
                / sum(r["elapsed_s"] for r in mrows),
            }
    out.append("-" * 78)
    out.append("median across runs (tokens/s = median of the per-run effective rates;")
    out.append("pooled = total completion tokens / total elapsed):")
    for mode, m in med.items():
        out.append(
            "%-11s %10.3f %10d %12.2f   pooled %.2f tok/s"
            % (mode, m["elapsed_s"], m["completion_tokens"], m["tokens_per_s"], m["pooled_tokens_s"])
        )
    if "inference" in med and "gui" in med:
        out.append("")
        out.append(
            "gui - inference median elapsed: %+.3f s.  Not a like-for-like latency"
            % (med["gui"]["elapsed_s"] - med["inference"]["elapsed_s"])
        )
        out.append(
            "difference: this probe pins temperature 0 / max_tokens 160 on the"
        )
        out.append(
            "isolated call, while gui/app.py samples at its own temperature 0.7 /"
        )
        out.append(
            "max_tokens 512 — so the gui run generates more tokens.  Effective"
        )
        out.append(
            "tokens/s is the comparable number; the elapsed delta is mostly extra"
        )
        out.append("generation, not gradio overhead.")
    print("\n".join(out))


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__, file=sys.stderr)
        return 2
    if args[0] == "--summarize":
        return summarize(sys.stdin)
    mode = args[0]
    if mode not in ("inference", "gui"):
        print(f"unknown mode {mode}", file=sys.stderr)
        return 2
    prompt = args[1] if len(args) > 1 else PROMPT
    probe = probe_inference if mode == "inference" else probe_gui
    res = probe(prompt)
    print(f"--- raw response ({mode}) ---")
    print(res["raw"])
    print("--- completion text ---")
    print(res["text"])
    print("--- measured ---")
    tps = res["completion_tokens"] / res["elapsed_s"]
    print(
        "%-11s elapsed %.3f s | completion_tokens %d | %.2f tokens/s"
        % (mode, res["elapsed_s"], res["completion_tokens"], tps)
    )
    print(
        "__SUMMARY__ "
        + json.dumps(
            {
                "mode": mode,
                "elapsed_s": round(res["elapsed_s"], 3),
                "completion_tokens": res["completion_tokens"],
                "tokens_per_s": round(tps, 2),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
