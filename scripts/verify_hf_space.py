#!/usr/bin/env python3
"""End-to-end check of the P5 Space: does the *public* API answer with the fine-tuned model?

No local model, no local GPU, no local container: this talks to the Space's own Gradio
API over HTTPS the way any third party would, and records what came back.  It also
reports the Space's runtime stage and the hardware it is actually on, because "the Space
exists" and "the Space is serving on ZeroGPU" are different claims.

Usage:
    python3 scripts/verify_hf_space.py                 # uses the default Space + prompt
    python3 scripts/verify_hf_space.py --prompt "..." --max-new-tokens 200

Optional: export HF_TOKEN to have the script also read the Space's runtime metadata
(stage, hardware, quota-relevant details); the inference call itself needs no token.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_SPACE = "ljubobratovicrelja/qwen3.5-2b-magicoder-demo"
DEFAULT_PROMPT = ("Write a Python function that returns the sum of the squares of the "
                  "even numbers in a list.")


def get_json(url, token=None, timeout=60):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def post_json(url, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def find_chat_endpoint(domain):
    """Ask the Space which endpoints it exposes, and pick the one that answers a prompt.

    Gradio 6 serves the app config at `/config` (the `/gradio_api` prefix applies to the
    call endpoints, not to the config), and it publishes the *signature* of every named
    endpoint at `/gradio_api/info` — which is what makes this script robust to the app's
    parameter list changing.
    """
    info = get_json(f"https://{domain}/gradio_api/info")
    endpoints = info.get("named_endpoints", {})
    # Prefer the function we defined; ChatInterface also exposes internal helpers.
    for candidate in ("/generate", "/chat", "/_submit_fn"):
        if candidate in endpoints:
            return candidate, endpoints[candidate], sorted(endpoints)
    for name, spec in endpoints.items():
        if not name.startswith("/_"):
            return name, spec, sorted(endpoints)
    raise RuntimeError(f"no callable endpoint found; endpoints: {sorted(endpoints)}")


def build_data(spec, prompt, max_new_tokens):
    """Fill the endpoint's parameters by name, so the call matches the real signature."""
    data = []
    for param in spec.get("parameters", []):
        name = (param.get("parameter_name") or "").lower()
        if name in ("message", "prompt", "text", "input"):
            data.append(prompt)
        elif "token" in name:
            data.append(max_new_tokens)
        elif name in ("history", "chat_history"):
            data.append([])
        else:
            data.append(None)
    return data


def call_endpoint(domain, api_name, data):
    """POST then read the SSE stream; returns the final payload."""
    started = time.monotonic()
    event = post_json(f"https://{domain}/gradio_api/call/{api_name}", {"data": data})
    event_id = event.get("event_id")
    if not event_id:
        raise RuntimeError(f"no event_id in {event}")
    result, last = None, None
    req = urllib.request.Request(f"https://{domain}/gradio_api/call/{api_name}/{event_id}")
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event:"):
                last = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                payload = line.split(":", 1)[1].strip()
                if payload and payload != "null":
                    result = payload
            elif last == "complete":
                break
    return result, time.monotonic() - started


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=DEFAULT_SPACE)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    domain = args.space.replace("/", "-").replace(".", "-").replace("_", "-") + ".hf.space"

    print(f"# space        : {args.space}")
    print(f"# url          : https://huggingface.co/spaces/{args.space}")
    print(f"# domain       : https://{domain}")

    if token:
        try:
            rt = get_json(f"https://huggingface.co/api/spaces/{args.space}/runtime", token)
            hw = rt.get("hardware") or {}
            print(f"# runtime stage: {rt.get('stage')}")
            print(f"# hardware     : current={hw.get('current')} requested={hw.get('requested')}")
            for d in rt.get("domains", []):
                print(f"# domain stage : {d.get('domain')} -> {d.get('stage')}")
        except Exception as exc:  # noqa: BLE001
            print(f"# runtime      : (could not read: {exc})")
    else:
        print("# runtime      : (no HF_TOKEN set — skipping the metadata read)")

    api_name, spec, all_names = find_chat_endpoint(domain)
    api_path = api_name.lstrip("/")
    params = [(p.get("parameter_name"), (p.get("python_type") or {}).get("type"))
              for p in spec.get("parameters", [])]
    print(f"# endpoints    : {len(all_names)} named; using '{api_path}'")
    print(f"# signature    : {params}")

    data = build_data(spec, args.prompt, args.max_new_tokens)
    shown = [d if not isinstance(d, str) or len(d) < 44 else d[:41] + "..." for d in data]
    print(f"# call data    : {shown}")
    print(f"\n# prompt       : {args.prompt}")
    print(f"# max_new_tokens: {args.max_new_tokens}")
    print("# calling the public API ...")
    try:
        payload, latency = call_endpoint(domain, api_path, data)
    except urllib.error.HTTPError as exc:
        print(f"# FAILED: HTTP {exc.code} {exc.read().decode()[:200]}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"# FAILED: {type(exc).__name__}: {exc}")
        return 1

    print(f"# latency      : {latency:.1f} s (including any queue wait and cold start)")
    print("# raw payload  :", (payload or "")[:400])
    try:
        data = json.loads(payload or "null")
        text = data[0] if isinstance(data, list) and data else str(data)
        print("\n--- answer ---")
        print(text[:800])
    except Exception:  # noqa: BLE001
        print("\n--- answer (unparsed) ---")
        print((payload or "")[:800])
    return 0


if __name__ == "__main__":
    sys.exit(main())
