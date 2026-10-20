#!/usr/bin/env python3
"""Judging instrument for the P4 benchmark: mechanical checks across all 150 runs.

Read-only. Prints the numbers quoted in docs/BENCHMARK_REPORT.md; changes nothing.

    python3 scripts/analyze_bench_finetune.py

Two checks are deliberately stricter than a naive keyword/regex scan, because the
naive versions were wrong:

  * list-item counting counts *markup or plain-line* items, so a model that
    returns a bare line-per-item list (no "1." / "-") is not scored as "0 items";
  * the format-1 check requires the parsed value to be a JSON *object*, so a
    quoted JSON *string* (or a fence, or trailing prose) fails, as it should.

Neither check is semantic grading — see the caveats printed at the end.
"""
import json
import re
import statistics
import collections
import sys
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "evidence/bench-finetune.json")
SUITE = os.path.join(REPO, "bench/prompts.json")

d = json.load(open(DATA))
src = json.load(open(SUITE))
prompts = {p["id"]: p for p in src["prompts"]}
by = collections.defaultdict(dict)           # (prompt, arm) -> list of runs
for r in d["results"]:
    by[(r["prompt_id"], r["model_key"])] = r["runs"]

ARMS = ["base", "tuned", "instruct"]
FMT_KEYS = {"city", "country", "population_estimate"}


def list_items(text):
    """Count list items, accepting both markup ('1.' / '-' / '*') and plain lines."""
    lines = [l for l in text.splitlines() if l.strip()]
    marked = [l for l in lines if re.match(r"^\s*(\d+[.)]|[-*\u2022])\s+\S", l)]
    if marked:
        return len(marked), "marked"
    return len(lines), "plain"


def json_object_ok(text):
    """True only if the whole reply is a single JSON object with exactly the wanted keys."""
    body = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I).strip()
    try:
        o = json.loads(body)
    except Exception:
        return False, "not-json"
    if not isinstance(o, dict):
        return False, "json-" + type(o).__name__     # catches a JSON string/array/number
    return set(o) == FMT_KEYS, ("ok" if set(o) == FMT_KEYS else "keys=" + ",".join(sorted(o)) or "{}")


def bullet_count(text):
    """Count list items, taking the larger of the ordered and unordered styles so a
    reply that wraps numbered items under a bold header bullet is not overcounted."""
    ordered = len([l for l in text.splitlines() if re.match(r"^\s*\d+[.)]\s+\S", l)])
    unordered = len([l for l in text.splitlines() if re.match(r"^\s*[-*\u2022]\s+\S", l)])
    return max(ordered, unordered)


def hit_1805(text):
    t = text.replace(" ", "")
    return "18:05" in t or "1805" in t


print("=" * 100)
print("P4 BENCHMARK — MECHANICAL ANALYSIS OF evidence/bench-finetune.json")
print(f"suite {d['suite_version']}  generated {d['generated_at']}  runs/prompt {d['runs_per_prompt']}"
      f"  seeds {src['sampling']['seed_base']+1}..{src['sampling']['seed_base']+d['runs_per_prompt']}")
print("=" * 100)

print("\n--- AGGREGATE PER ARM (all 10 prompts x 5 runs = 50 runs each) ---")
print(f"{'arm':9} {'runs':>4} {'tok total':>10} {'tok med':>8} {'tok min':>7} {'tok max':>7}"
      f" {'capped':>7} {'lat med s':>10} {'empty':>5}")
for arm in ARMS:
    runs = [r for k, v in by.items() if k[1] == arm for r in v]
    toks = [r["usage"]["completion_tokens"] for r in runs]
    lat = [r["latency_s"] for r in runs]
    capped = sum(1 for r in runs if r["finish_reason"] == "length")
    empty = sum(1 for r in runs if not r["text"].strip())
    print(f"{arm:9} {len(runs):>4} {sum(toks):>10} {statistics.median(toks):>8.0f}"
          f" {min(toks):>7} {max(toks):>7} {capped:>5}/{len(runs)} {statistics.median(lat):>10.2f} {empty:>5}")

print("\n--- PER PROMPT, PER ARM (tokens / cap / latency + machine checks) ---")
for pid in [p["id"] for p in src["prompts"]]:
    p = prompts[pid]
    print(f"\n{pid}  [{p['category']}]  max_tokens={p['max_tokens']}"
          + (f"  check: {p['check']}" if p.get("check") else ""))
    for arm in ARMS:
        runs = by[(pid, arm)]
        toks = [r["usage"]["completion_tokens"] for r in runs]
        capped = sum(1 for r in runs if r["finish_reason"] == "length")
        print(f"  {arm:9} tok med {statistics.median(toks):6.0f} (min {min(toks):3d} max {max(toks):3d})"
              f"  capped {capped}/5  latency med {statistics.median([r['latency_s'] for r in runs]):5.2f}s")
        for i, r in enumerate(runs, 1):
            t = r["text"]
            if pid == "general-trivial-2":
                n, kind = list_items(t)
                print(f"      run {i}: {n} items ({kind})   want 5 -> {'PASS' if n == 5 else 'FAIL'}")
            elif pid == "general-context-1":
                print(f"      run {i}: {bullet_count(t)} bullet lines   want 5 -> "
                      f"{'PASS' if bullet_count(t) == 5 else 'FAIL'}")
            elif pid == "reasoning-1":
                print(f"      run {i}: {'says 18:05 -> PASS' if hit_1805(t) else 'no 18:05 -> FAIL'}"
                      + ("   (hit token cap)" if r["finish_reason"] == "length" else ""))
            elif pid == "format-1":
                ok, why = json_object_ok(t)
                print(f"      run {i}: {'PASS' if ok else 'FAIL'} ({why})   raw: {t.strip()[:46]!r}")
            elif pid == "code-context-1":
                low = t.lower()
                flags = [k for k in ("popitem(last=true)", "last=false", "last=true",
                                     "evict", "most recently used", "move_to_end", "wrong end",
                                     "insertion") if k in low]
                print(f"      run {i}: eviction/ordering keywords: {','.join(flags) or 'none'}"
                      + ("   (hit token cap)" if r["finish_reason"] == "length" else ""))

print("\n--- FORMAT-1 STRICT PASS RATE (single JSON object, exactly the three keys) ---")
for arm in ARMS:
    runs = by[("format-1", arm)]
    n = sum(1 for r in runs if json_object_ok(r["text"])[0])
    print(f"  {arm:9} {n}/5 pass")

print("\n--- GENERAL-TRIVIAL-2 ITEM COUNT (want exactly 5) ---")
for arm in ARMS:
    runs = by[("general-trivial-2", arm)]
    counts = [list_items(r["text"])[0] for r in runs]
    print(f"  {arm:9} {sum(1 for c in counts if c == 5)}/5 exact-five   (counts: {counts})")

print("\n--- GENERAL-CONTEXT-1 BULLET COUNT (want exactly 5) ---")
for arm in ARMS:
    runs = by[("general-context-1", arm)]
    counts = [bullet_count(r["text"]) for r in runs]
    print(f"  {arm:9} {sum(1 for c in counts if c == 5)}/5 exact-five   (counts: {counts})")

print("\n--- REASONING-1 (true answer 18:05; 14:20 + 3h45m) ---")
for arm in ARMS:
    runs = by[("reasoning-1", arm)]
    ok = sum(1 for r in runs if hit_1805(r["text"]))
    cap = sum(1 for r in runs if r["finish_reason"] == "length")
    print(f"  {arm:9} {ok}/5 contain 18:05   {cap}/5 hit the token cap")

print("\n--- PERPLEXITY (llama perplexity, -c 4096; `chunks` = number of 4096-token windows) ---")
for row in d["perplexity"]:
    print(f"  {row['model_key']:9} {row['corpus']:8} ppl {row['ppl']:.4f}  ({row['chunks']} chunks)")

print("\n--- NOTES COPIED FROM THE EVIDENCE FILE ---")
print("  build:  ", d["build_note"])
print("  lineage:", d["lineage_note"])
print("  adapter check:", json.dumps(d["adapter_check"]["runs_differing"]) + "/"
      + str(d["adapter_check"]["runs_total"]), "base-vs-tuned runs differ (all prompts differ)")

print("""
CAVEATS for anything read off the block above
  * temperature 1.0 + presence_penalty 2.0 makes single samples noisy; five runs per
    prompt is a small sample, so only a pattern repeated across runs is worth reporting.
  * the item/bullet/JSON checks are structural, not semantic: they cannot tell whether
    the code actually runs or the bug is the *right* bug.
  * 'capped' means the reply ran into max_tokens (finish_reason == length) and was cut
    off mid-sentence; that is a truncation, not a wrong answer.
  * perplexity corpora are two 4096-token chunks each — small enough that the numbers
    are indicative at best.
""")
