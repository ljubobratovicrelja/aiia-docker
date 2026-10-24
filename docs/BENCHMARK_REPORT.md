# P4 benchmark — did the LoRA fine-tune help, and did it break anything?

This is the reading of `evidence/bench-finetune.json` against the fixed suite in
`bench/prompts.json`. It answers one question: **did the stage-4 LoRA adapter improve the coding
behaviour it was trained for, and did it degrade anything else?** The method was fixed before the
run, the numbers below come from a runnable command, and the raw generations
are quoted rather than summarised wherever the reading depends on them.

## What was measured, in one paragraph

Three model arms, one prompt suite, five runs each. The **controlled pair** is `base`
(`Qwen3.5-2B-Base`, pinned revision, Q4_K_M) and `tuned` (the *same* GGUF bytes served with
`--lora adapter.gguf`, i.e. the stage-4 adapter) — base weights byte-identical, the adapter the only
difference. `instruct` (`unsloth/Qwen3.5-2B-GGUF`, the shipped post-trained model) is a **labelled
reference of a different lineage**, never part of the pair. The suite is 10 fixed prompts × 5 runs ×
3 arms = **150 generations**. Sampling follows the Qwen3.5 card's non-thinking recommendation
(`temperature 1.0, top_p 1.0, top_k 20, min_p 0.0, presence_penalty 2.0`), so the five runs of one
prompt are **five samples, not five repeats** — a finding has to hold across runs, and each claim
below says how many of the five carry it. Reproducibility comes from a fixed seed per run index
(`43507…43511`), not from greedy decoding.

The harness was verified before judging: the adapter genuinely loads (server log
`llama_adapter_lora_init`, `adapter.lora.alpha f32 = 32.000000`, `loaded 300 tensors from lora file`),
an independent 1-run re-run reproduced run 1 **byte-identically on 30/30** prompt-arm pairs, and
base-versus-tuned texts differ on **50/50** runs — the pair is real and the harness is deterministic
given the seeds. Those are stated as facts, not re-derived here.

## Headline verdict

**The adapter did not demonstrably improve coding correctness, and on the one hard coding task that
has a verifiable right answer it did not help at all.** Both the base and the tuned model fail to
find the real bug in the `code-context-1` review case; the tuned arm is not better than the base at
it (arguably no arm is), and on `code-context-2` neither produces a clean, working validator.

What the adapter *did* change, consistently and measurably, is **style and length**: tuned responses
are roughly **45 % shorter** than base (6,594 vs 11,908 completion tokens over 50 runs) and almost
never run into the token cap (0/50 vs base 7/50). That is a real, repeatable effect — but shorter is
not the same as better, and several prompts show the terseness crossing into *failure*: dropped
docstrings, dropped examples, and a new habit of describing a reply in the third person instead of
writing it.

Against the base, the tuned arm shows **one clear win** (`reasoning-1`: 2/5 correct vs 0/5, and
0/5 versus 4/5 truncations), **one clear regression** (`general-context-2`: 4/5 replies are
third-person meta-descriptions instead of the requested email), and **mostly noise** on the rest.
The honest one-line verdict: *the adapter mostly changed style, not correctness; it helped on
short arithmetic and hurt on instruction-following, and it did not solve the coding task it was
trained for.*

The instruct arm is included only as a reference and is visibly worse in places (5/5 truncations on
the reasoning canary, a function that ignores its argument on `code-trivial-1`) — which is a useful
reminder that raw brevity and coherence trade off, but it is a different lineage and is **not**
evidence about the adapter.

## The coding prompts

### `code-trivial-1` — a short Python function (trained-for task)

The function body is easy and **all three arms get the body right in almost every run** — base 5/5,
tuned 5/5, instruct 3/5 produce a generator/loop that computes the right sum. The differences are in
the wrapping, and they are the whole story:

- **Base** writes long docstrings and usage examples and then *mangles the arithmetic in the
  examples*. Run 1: `# Output: 26 (because only 4 is even, square is 16 = 26 + 1 + 9)`. Run 3 runs
  the example aloud and talks itself into a corner (`… 16+64=80. Wait! Only two evens? …`).
- **Tuned** returns the same correct body but strips the wrapper. Run 4 is the whole reply:
  ```python
  def sum_even_squares(nums):
      return sum(x * x for x in nums if x % 2 == 0)
  # Usage example
  nums = [1, 3, 4, 8, 5, 9]
  print(sum_even_squares(nums))  # Output: 16 (…)
  ```
  No docstring, and the printed `16` is wrong (4²+8²=80). The prompt asked for *a docstring and one
  usage example*; run 4 supplies neither a docstring nor a correct example.
- **Instruct** is the cautionary one. Run 1's headline function **ignores its argument entirely**:
  ```python
  for i in range(0, 200):      # Check up to 199 …
      if i % 2 == 0:
          running_total += (i * i)
  ```
  and all five runs hit the 320-token cap. Brevity is not the issue here; correctness is.

So on the trained-for task: **no correctness regression, a mild instruction-following regression**
(tuned drops the requested docstring/example in 1/5, and the examples it keeps are wrong — a weakness
shared with base). The interesting signal is that base's *extra* tokens buy verbose prose, not
correct arithmetic.

### `code-trivial-2` — SQL (does the adapter overfit to Python?)

Mixed, ~a wash, lots of noise. Base happens to nail the idiomatic query in 2–3 of 5:
```sql
SELECT c.name, SUM(o.amount) AS total
FROM customers c JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name
ORDER BY total DESC
LIMIT 3;
```
Tuned's clean runs are just as good and much shorter — run 1 is five lines with no prose — but its
bad runs are worse: run 4 references a non-existent table `FROM order` and a `GROUP BY c.id` with no
`c` in scope; run 3 groups by `orders.id` instead of the customer. Its two clean runs (1 and 2)
correctly join `customers` and group the sum. Base's bad runs are a `GROUP BY 1` on `orders` alone
(`orders` has no `name`, so it does not run — run 3) and a semicolon before `LIMIT` (run 4). Instruct
is similar (one run emits `LIMIT 3.5` and a `GROUP_CONCAT` that does not aggregate).

Verdict: **no evidence the adapter overfit to Python.** No arm reliably clears the SQL prompt, which
is a property of a 2B model, not of the adapter.

### `code-context-1` — the code review (the interesting one)

This is the prompt that matters most, and it is where the honest answer is *nobody passed*. The
module is meant to be an LRU cache. **The real bug** is that `get` and the update branch of `put`
call `self._data.move_to_end(key, last=False)` (move the used key to the **front** = most-recently-used),
but a *new* key is appended to the **back**, and eviction pops the **back** with
`self._data.popitem(last=True)`. The two operations disagree about which end is "recent", so the
eviction removes the **most recently inserted/used** key and lets the oldest untouched key survive.
Verified directly:

```text
capacity 2: insert a, b, c        ->  evicts b, keeps a   (should evict a)
capacity 3: put a,b,c; get(a); put d ->  evicts c, keeps a,b   (should evict b)
```
The fix is one line — in `put`'s new-key branch, also move the freshly inserted key to the front
(`self._data.move_to_end(key, last=False)` after `self._data[key] = value`) — after which the trace
is correct (evicts `a`, then `b`). The suite's `check` ("LRU evicts the most-recently-used key") is
right in spirit; precisely, it evicts the most-recently-*inserted* key.

No arm gets there, and the failure mode is **confident hallucination**, which is worth quoting
because it is the single most alarming thing in the record:

- **base run 1** invents a bug in an attribute that does not exist: *"Bug: The constructor does not
  initialize `m` to `0`."* There is no `m`.
- **base run 2** is self-assured nonsense: *"Here `total` is a variable name that already exists in
  the scope … Python will resolve it with the next highest identifier, which in this case is usually
  `"__name__"`."* That is not how Python works, and the code is fine there.
- **base run 4** identifies a "bug" in `get` and "fixes" it by flipping the very direction that is
  part of the real problem.
- **tuned run 5** presents a "corrected version" that is **byte-for-byte the original code** — it
  changed nothing and claimed a fix.
- **tuned runs 3, 4, 1** variously blame hit/miss counters, a "collision", or a "double deletion";
  none of those mechanisms exists.

The only run that faintly touches the truth is **instruct run 3**, which points at the right line —
*"`popitem(last=True)` … removes a recently added key to free space"* — before wandering off into
`pop()`/`popleft()` and "recency levels". So: base **0/5**, tuned **0/5**, instruct **1/5 partial**.
The keyword scanner in the analysis script flags eviction words in most runs, which is exactly why a
keyword hit is not a pass — the words are there, the diagnosis is not.

This is the decisive negative result: the kind of bug the fine-tune was meant to make the model
better at is not solved by either arm of the pair, and the tuned arm is not distinguishable from base
on it.

### `code-context-2` — implement a JSON-schema validator (no third-party libraries)

Also poor across the board, and again roughly a wash. The task is hard for a 2B model. The common
failures:

- **base run 1** ignores "do not use any third-party library" and does `import jsonschema`, then
  calls it with nonsense arguments. Base run 2 has a genuine Python syntax error in a string
  join. Base run 4 calls an undefined `get_tags`/`records`.
- **tuned** is shorter and no more correct. Run 1 references an undefined `RECORD_VALIDATION_FIELDS`
  and calls `getattr(type(value), 'validate', …)()`; run 3 contains a syntax error; run 5 is a
  two-line stub. Only run 2 is a plausible-looking validator, and its `sensor_id` regex is escaped
  wrong (`\d\{4\}` matches literally).
- **instruct** produces the two cleanest-looking attempts (runs 3 and 5) and three incoherent ones;
  run 5 is arguably the most nearly-correct validator in the whole file.

If anything, the *reference* model is marginally more likely to emit something coherent here, but
with 5 runs and this much brokenness the honest call is **insufficient evidence of any adapter
effect** on this prompt. The machine check ("enforces the enum and the required list") is only
partially met anywhere, and a keyword count cannot separate a validator that *runs* from one that
merely *mentions* `enum`.

## The general / non-coding prompts (degradation canaries)

### `general-trivial-1` — explain why the sky is blue (3 sentences)

All arms are mediocre and all three-sentence instructions are loosely honoured. No arm is clearly
worse. Worth noting the tuned arm's most wrong run, which is fluent but self-contradictory — *"all
colors scatter equally. Blue light has a shorter wavelength and gets scattered more than red light"* —
and a base run that leaks a chat-template token (`… reddening … </think>`). This prompt shows noise,
not a signal.

### `general-trivial-2` — list five ways to reduce water use

The declared check is "exactly 5 items", and **every arm passes 5/5** once the counter is fixed (the
orchestrator's original regex scored 0 for a plain line-per-item list, which all three arms produced
in their first run — that was a *tool* bug, not a model one; tune run 1 and base run 1 are plain
unmarked lines). No regression here.

### `general-context-1` — summarise ~600 words in five bullets + state the claim

Instruction-following degrades with tuning here, gently. Counting the bullet lines:

| arm | runs with exactly 5 bullets |
|---|---|
| base | **4/5** (run 4 gives 4) |
| tuned | **3/5** (run 2 gives a paragraph, run 3 gives 4) |
| instruct | **1/5** |

Tuned run 2 drops the list entirely and answers in one flowing paragraph; base keeps the structure
more often. Base's main-claim sentences are also more specific (*"… immutability is often a source of
hidden inefficiency in production containers …"*) than tuned's, which tend to trail off into
generalities (*"understanding immutability and caching … is crucial for managing them efficiently"*).
Direction of travel is negative for tuned, but it is 5 runs and the margin is one run — treat it as a
hint, not a verdict.

### `general-context-2` — draft an email reply (the clearest regression)

This is the strongest, most quotable adapter regression. The prompt asks for *a reply from Ana*. The
tuned arm answers by **describing** a reply instead of writing one, in **4 of 5 runs**:

- run 3: *"Ana replies to Mark by agreeing to the deadline and reminding him about the template
  usage. She also asks for the missing figures …"*
- run 4: *"Ana replies to Mark stating her availability by 6 pm Wednesday …"*
- run 5: *"Ana responds to Mark's email … She keeps the response concise, staying within the word
  limit …"*
- run 1 is the same failure with the roles swapped (*"Replying to Ana: It seems like a good plan …"*).

Only run 2 is an actual draft. Base produces a real reply in most runs, e.g. run 2:
*"**Ana replies:** Thursday 18:00 works for me. To avoid any concerns later—Finance closes
Thursday—and to ensure consistency I need the missing cost breakdown by region and headcount
movement."* — it agrees to the deadline and (roughly) asks for the missing figure, as asked. Instruct
mostly writes real replies too, landing the requirement cleanly in run 3 (*"Thursday 18:00 works
perfectly … could you provide the latest **headcount figures**?"*).

A model that meta-describes the requested artifact is a genuine instruction-following failure, and
it appears at a rate (4/5) that survives the small sample. **This is the clearest "the adapter
degraded something non-coding."**

## The two canaries with verifiable answers

### `reasoning-1` — 14:20 + 3 h 45 m (true answer **18:05**)

The one place the tuned arm clearly beats base:

| arm | correct (says 18:05) | hit the 400-token cap |
|---|---|---|
| base | **0/5** | 4/5 |
| tuned | **2/5** | **0/5** |
| instruct | 0/5 | 5/5 |

Tuned is the only arm that reliably *stops* and the only arm that ever lands the right answer — but
it is still wrong in 3 of 5, and wrong in ways that show the arithmetic is not actually reliable:

- tuned run 1 converts to minutes correctly (`860 + 225 = 1085`) and then divides wrong: *"1085 ÷ 60
  = 18 with a remainder of 7 … the train arrives at 18:07."*
- tuned run 3: *"the arrival time is 15:05."*
- tuned run 5: *"1085 minutes … Therefore, the arrival time is 6:05 AM the next day."*
- base, when it does commit, is confidently wrong: run 3 computes `6:20 PM` then prints **19:20**.

So the win is real *as far as it goes* — tuned goes from never-correct-but-rambling to
sometimes-correct-and-terse — but "sometimes right on a single-digit-minute addition" is a low bar
and the other three runs show the same arithmetic fragility as base. This is not evidence of a
general reasoning gain; it is evidence that the adapter learned to **stop and answer**, which is a
useful but different thing.

### `format-1` — reply with only a JSON object (strict)

Strict pass = the whole reply is a single JSON **object** with exactly `city`, `country`,
`population_estimate` (a quoted JSON *string*, a fence, or trailing prose all fail):

| arm | strict pass |
|---|---|
| base | 3/5 |
| tuned | 4/5 |
| instruct | 4/5 |

Base run 1 emits `{}` and base run 5 emits the object three times wrapped in template artifacts
(`…}  {"city":…}  '{"city":…}'`). The tuned arm has no adversarial run here except **run 1**, which
returns a quoted JSON *string* — the exact failure the orchestrator warned the naive checker would
miss:

```text
"{\"city\":\"Lisbon\",\"country\":\"Portugal\",\"population_estimate\":507416}"
```

That is one run in five, and tuned's other four are clean, so the honest read is **format adherence
is broadly kept, with one string-collapse sample**. Instruct's single failure is the same `{}` base
produced. Nothing here is a strong regression, but it is a reminder that "the tuned arm is shorter"
also means "fewer tokens in which to recover from a bad start".

## The quantitative table

All numbers from `python3 scripts/analyze_bench_finetune.py` over the 150 generations.

| arm | runs | completion tokens (total) | median | min | max | runs hitting `max_tokens` | median latency |
|---|---|---|---|---|---|---|---|
| base | 50 | 11,908 | 185 | 2 | 700 | 7/50 | 0.63 s |
| tuned | 50 | **6,594** | 88 | 24 | 612 | **0/50** | **0.43 s** |
| instruct | 50 | 14,330 | 240 | 2 | 700 | 18/50 | 0.83 s |

Split by prompt type (median completion tokens):

| arm | coding prompts (4) | general prompts (6) |
|---|---|---|
| base | 322 | 106 |
| tuned | **120** | **76** |
| instruct | 451 | 90 |

Held-out **perplexity** (`llama perplexity`, `-c 4096`, two corpora), lower is better:

| arm | general prose | code | note |
|---|---|---|---|
| base | **9.0791** | **3.0116** | |
| tuned | 9.1941 | 3.0758 | +1.3 % general, +2.1 % code vs base |
| instruct | 10.1246 | 3.2293 | different lineage |

**How to read these numbers, and where they are proxies.** Completion-token count is a proxy for
verbosity, not for quality — and it is a *confounded* proxy here: `max_tokens` is per-prompt and
identical across arms, so a shorter reply can mean "more decisive" or "cut short early". The
"capped" column is clean (it is `finish_reason == "length"`, a hard truncation signal), but everything
else about length is descriptive. Latency is derived from completion tokens at near-constant
throughput and so mostly restates length; it is not an independent measurement. The perplexity
numbers are the cleanest quantitative signal in the file, but they rest on **two 4096-token chunks
per corpus** — far too little to separate a 1–2 % shift from chunk choice. The tuned arm's perplexity
is worse than base on *both* corpora, which is the expected direction for fine-tuning on a narrow
distribution, but read it as a hint, not a measurement.

## What improved, what degraded, what is noise

**Improved (holds across runs):**
- **Verbosity discipline.** 6,594 vs 11,908 tokens, and 0/50 truncations vs 7/50. Tuned answers in
  one pass where base rambles. `reasoning-1` is where this pays off: 2/5 correct vs 0/5.

**Degraded (holds across runs):**
- **`general-context-2` instruction-following**: 4/5 replies are third-person descriptions of the
  email instead of the email. Base does not do this.
- **`general-context-1` structure**: exactly-5-bullets drops from base 4/5 to tuned 3/5 (small),
  and tuned's main-claim sentences are vaguer.
- **Perplexity**: worse than base on both corpora (+1.3 % / +2.1 %) — the expected cost of a narrow
  fine-tune.

**Unchanged / noise (do not claim either way):**
- **Coding correctness.** `code-trivial-1` body correct in ~all runs of both arms; `code-context-1`
  unsolved by both (0/5 vs 0/5); `code-context-2` broken in both; `code-trivial-2` a wash.
- **`format-1`** and **`general-trivial-2`**: within noise; both arms essentially satisfy the check.
- **`general-trivial-1`**: noise.

The single most important line in this table is the `code-context-1` row: the adapter was trained to
write and review code, and on the one code-review prompt with a single correct answer, **the tuned
model is indistinguishable from the base — both are wrong, both are confident about it.** Everything
the adapter demonstrably bought is stylistic.

## Limits of this evidence (read before quoting any of it)

- **Five runs is a small sample, and sampling is noisy by design.** At `temperature 1.0` with
  `presence_penalty 2.0`, a single run is not evidence about the model, only about one draw. Every
  "x/5" above is a thin margin; treat 1-run differences (e.g. base 3/5 vs tuned 4/5 on `format-1`) as
  noise, and only the 4/5-vs-1/5 patterns as signal.
- **Structural checks are not semantic grading.** Counting bullets, items, JSON keys and the string
  `18:05` is mechanical; it cannot tell whether returned code *runs*, whether a validator *enforces*
  what it names, or whether a diagnosis is the *right* bug. The `code-context-1` keyword flags in the
  script do not distinguish insight from reciting "evict".
- **`instruct` is a different lineage.** It is the shipped post-trained model, not a third point on a
  controlled axis. Where it looks better (validator attempts) or worse (reasoning truncations) that
  is a statement about *base-vs-post-training*, not about the adapter.
- **Perplexity corpora are two chunks each.** The 1–2 % gaps are within reach of chunk selection.
- **The `reasoning-1` "win" is small and narrow.** 2/5 on a single-digit addition is a low bar, and
  the same prompt shows tuned failing arithmetic in the other three runs.
- **This measures 2B-scale behaviour, not the LoRA method.** A negative result on a 2B base, a
  formulaic 3,000-record corpus, and five samples per prompt is a statement about *this* adapter on
  *this* suite, not about LoRA fine-tuning in general.

**What would change the conclusion.** A larger sample (≥ 20 runs/prompt to put error bars on the
x/5 counts); an *executed* check — actually run the returned code (`ast.parse` + a tiny unit test per
code prompt, and a `pytest`-style pass on the validator) instead of reading it, which is the
`ast.parse`/pass@1 metric that is still open and that this suite deliberately did not
run; a temperature-0 or many-sample-majority read on the canaries; and a HumanEval slice to measure
coding ability on held-out problems rather than ten hand-picked prompts. If a *executed* pass rate on
the two code-context prompts came out base ≈ tuned, as the qualitative read here predicts, that
would confirm "the adapter changed style, not correctness."

## Housekeeping

This report is the qualitative complement to the base-vs-adapter comparison, not the two executed
metrics: `ast.parse` adherence and HumanEval pass@1 are still not measured, and this suite did not
execute returned code. That remains the open item.

## Reproduce

```bash
python3 scripts/analyze_bench_finetune.py
```

Reads `evidence/bench-finetune.json` and `bench/prompts.json`, prints the aggregate and per-prompt
tables, the strict `format-1` pass rate, the item/bullet/reasoning checks, and the perplexity rows.
Nothing is written. The two orchestrator-side flaws are fixed in that script: list items are counted
whether or not they carry `1.`/`-` markup, and the `format-1` check requires a JSON **object** (a
quoted JSON string now fails, as it should).
