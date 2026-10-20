# JOURNAL — aiia-docker (MUIA 43506, Assignment 2)

Work log for this repository: **decisions, findings, problems hit, and what they cost**.
Append-only, newest block at the bottom, one `## YYYY-MM-DD` block per session.

## What does NOT belong here

- **Anything the git log already says.** No "added compose.yaml", no "ran the container", no
  per-commit narration. If `git log --oneline` or a diff answers it, it is noise here.
- Restatements of the plan (`docs/PROJECT_PLAN.md`) or of the brief (`docs/assignment/`).
- Progress roll-ups. This is a log of *what was learned*, not a status board.

## What DOES belong here

- **Problems and surprises** — especially the ones that took real time or forced a design
  change, e.g. anything that fights LoRA on the target architecture, a dataset that is not what
  the card promised, a Docker build that breaks on aarch64, a parameter that has no effect.
- **Decisions with their reason**, when the reason is not visible in the code (why CPU-first,
  why the model is baked in rather than mounted, why an adapter is served instead of merged).
- **Measured numbers** worth remembering later (latency, RSS against a limit, adapter size,
  loss / eval deltas) — a number with no home in code or config.
- **Environment quirks** that will bite the next person: arm64 wheel availability, what a base
  image ships with, what fails silently.

## 2026-10-20
**Activity:** Repository created empty by Relja; skeleton and project plan written before any
development, as the review checkpoint.
**Findings:** `conversion/__init__.py` in the local llama.cpp checkout already maps
`Qwen3_5ForCausalLM` onto its qwen conversion path, so a Qwen3.5 GGUF is reachable with the
existing tooling — but whether LoRA *adapters* work on a hybrid Mamba/attention model is an open
question, not an assumption; it is being verified before the training phase is planned in
detail. Recorded in `docs/PROJECT_PLAN.md` §6 as risk R1.
**Findings (risk R1 — LoRA on the hybrid arch):** `Qwen/Qwen3.5-2B` is a composite VLM checkpoint
(`Qwen3_5ForConditionalGeneration`, vision tensors included); the SFT base is
`Qwen/Qwen3.5-2B-Base` loaded text-only as `Qwen3_5ForCausalLM`, and the two layouts use different
weight prefixes (`model.language_model.layers.*` vs `model.layers.*`) — an adapter must be trained
and loaded against the same one. `all-linear` is the wrong instrument here: it would reach into the
vision tower and it silently skips names that do not match, so the target list is explicit. QLoRA is
discouraged on this family; on the GB10, neither `causal_conv1d` nor `fla` has a build, so GDN runs
on the slow PyTorch reference (acceptable at this scale).
**Findings (risk R5 — HF Spaces: free route confirmed, first reading wrong):** the first pass read
"Gradio and Docker Spaces run on compute and require a paid plan" as a blanket cost and concluded the
bullet needed PRO. Wrong: the same sentence continues *"Free personal accounts in good standing can
still host up to 2 Gradio Spaces running on ZeroGPU"*. ZeroGPU is a shared RTX PRO 6000 Blackwell
via the `@spaces.GPU` decorator, PyTorch/transformers-oriented, 48 GB in the default slice, with a
**5 min/day GPU quota** for free accounts. So the spec's clause — publish the model to Spaces and
serve inference over an API — is reachable at zero cost; only a *Docker* Space needs PRO. The Space
is a second deployment of the same artefacts (transformers there, llama.cpp locally), not the same
server.
**Findings (risk R6 — Compose keys, verified on the engine):** `deploy.resources.limits` **does**
take effect under plain `docker compose up` (Engine 29.2.1 / Compose v5.0.2), so the service-level
`mem_limit`/`cpus` shorthand is unnecessary and declaring both invites drift. `expose:` with no
`ports:` was confirmed to leave `PortBindings` empty — that is the isolation requirement made real.
**Findings (risk R3 — CUDA torch on arm64, and a wrong claim of my own):** the plan had carried
"PyPI/CUDA-index aarch64 wheels stop at torch 2.0.1" as a fact. It is false. It came from grepping
the index and reading the **tail of an unsorted list** — lexicographically `2.9.1` sorts after
`2.14.1`, so the newest wheels sat earlier in the output than I looked. Sorted numerically,
`download.pytorch.org/whl/cu130` ships aarch64 wheels through **2.14.1+cu130** (cp310–cp315), and
PyPI's default index is current too. The wheels are CUDA builds: the aarch64 wheel carries
`libtorch_cuda.so` (~461 MB) and pins `cuda-toolkit==13.0.3`. Lesson worth keeping: when the claim
decides a build decision, sort the list properly or query the version metadata — never eyeball an
index listing.
**Findings (risk R4 — corpus):** Magicoder-OSS-Instruct-75K is the pick (MIT, 75,197 rows, ~194 MiB,
`problem`/`solution`); CodeAlpaca-20k (7.7 MiB) is the light fallback but its `output` is not always
code. Measured evidence will be two numbers: `ast.parse` adherence and pass@1 on a 50-problem
HumanEval slice (~5–8 min on a 2B Q4).
**Findings (risk R2 — llama.cpp LoRA path):** inference and runtime adapter loading both work for
`qwen35`, and every Qwen3.5 projection — GDN included — is routed through `build_lora_mm`. The trap
is on the conversion side: `convert_lora_to_gguf.py` raises `NotImplementedError` in the V-head
reorder branch for any adapter targeting `linear_attn.out_proj` (upstream #21125, fixed by #28324,
merged 2026-09-30, absent from the local checkout at HEAD 2026-08-27). Base weights are stored in a
V-head-reordered layout different from HF's, so a mis-reordered adapter multiplies the wrong
weights rather than failing loudly. Non-matmul SSM params can never be LoRA targets.
**Findings (stage 1 build):** two real traps, both cost build cycles and both now in this course's
ops notes. (a) The llama.cpp server image only resolves its own `libllama-server-impl.so` while the
working directory is `/app` — any other `WORKDIR` kills it with "error while loading shared
libraries". Confirmed by hand: works at `/app`, fails at `/models`. The Dockerfile therefore keeps
the inherited `WORKDIR` and addresses the model by absolute path, rather than exporting
`LD_LIBRARY_PATH=/app` to paper over it. (b) The pinned Qwen3.5-2B GGUF is a **thinking** model: the
server fills `reasoning_content` and can burn thousands of tokens before `content` is ever written
(observed 2355 tokens at ~7.7 t/s, client timeout, empty answer). `chat_template_kwargs:
{enable_thinking: false}` fixes it — the reply then lands in ~1.5 s. Worth knowing for the HF Space,
which will hit the same behaviour.
**Notes (stage 1 measured):** inference settles at ~1.4–1.8 GiB RSS against its 4 GiB limit on CPU;
the gradio container runs comfortably in 512 MiB and survives `read_only: true` with `HOME=/tmp` and
a `/tmp` tmpfs — that combination is not obvious and was worth testing rather than assuming.
**Duties / To-do:** Green light given. Working convention from here: one stage at a time, one PR per stage, merge before the next stage starts.
**Finding (stage 2 — `touch` does not invalidate a `COPY` layer):** the cache demonstration everyone
expects — `touch` the app code, rebuild, and watch the app layer re-run while the dependency layer
above it stays cached — **does not happen** on this engine. BuildKit keys a `COPY` step on the
**content digest** of the copied files, not on their mtime, so after `touch gui/app.py` the rebuild
reports `CACHED` for every step including `COPY app.py`; nothing re-runs at all. The demonstration
therefore shows both cases, and the property the Dockerfile layer order actually buys is shown with a
real content edit: pip layer `CACHED`, `COPY app.py` re-run. Worth keeping because the brief itself
recommends "take the Dockerfile order into account to make proper use of the cache" — anyone testing
that advice with `touch` would conclude, wrongly, that the ordering does nothing. Cost: one build
cycle, and it changed what the demo had to contain.
**Finding (stage 2 — a cache demo can stop demonstrating itself):** the first version of
`scripts/cache_demo.sh` appended a fixed marker line to `gui/app.py` to force the app layer to
rebuild. It worked — and then it stopped working on the second run: because a `COPY` step is keyed on
source content, the marker layer from run 1 was already in the build cache, so run 2 reported `CACHED`
for the app layer and the script's own assertion caught the difference between "the demo proved it"
and "the demo looked like it proved it". The marker now carries a timestamp and pid. General lesson
worth keeping: **a demonstration that writes a file it also caches against is not idempotent until
its input is unique** — and the assertion is what made the failure visible instead of a silently
degraded artifact.
**Finding (stage 2 — Engine 29 changed what `docker images` reports):** the table now prints
`DISK USAGE` and `CONTENT SIZE` columns and **`CONTENT SIZE` reads `0B`**, which is a trap if the
report quotes that column as "the image size". Authoritative byte counts come from
`docker image inspect --format '{{.Size}}'`. Also new: image size and the sum of layer sizes differ
by under 1 MB here, so layer arithmetic is a sound way to say where the bytes are.
**Finding (stage 2 — gradio's result stream carries no usage block):** the `POST
/gradio_api/call/v2/chat` + SSE result returns only the text, so completion tokens cannot be read off
the end-to-end path the way they can off the isolated API (`usage.completion_tokens`). Counting them
by posting the returned text to llama-server's own `/tokenize` gives exactly the same number as the
usage block when checked against the isolated call — that is the method `scripts/bench_probe.py`
uses, and it is the only way to state effective tokens/s for the GUI path without guessing.
**Notes (stage 2 — measured, CPU-only, Cortex-X925, 20 threads, shared host):** fixed 38-token prompt
at **9.08 tokens/s** median through the model's own API inside the compose network (median 4.183 s),
and **8.84 tokens/s** end-to-end through gradio at 67–89 tokens per run. The GUI's elapsed time is
longer than the isolated call's, but that is **not** transport overhead — `gui/app.py` samples at
temperature 0.7 / max_tokens 512 while the probe pins 0 / 160, so it simply generates more tokens;
tokens/s is the number the two paths can be compared on, which is why the evidence reports both the
per-run rate and the median instead of only latency. Decision recorded: the GUI's sampling parameters
were **not** changed to make the benchmark prettier.
**Notes (stage 2 — image arithmetic):** `inference` is 2 117 416 629 B, of which the pinned GGUF layer
is **1.28 GB (60.5 % of all layer bytes)** and the upstream apt layer 706 MB; `gui` is 461 208 892 B
with the `pip install` layer at 317 MB (68.6 %). Keeping `--no-cache-dir` is worth **84.1 MB
(+18.2 %)** — measured by building the alternative, not assumed: without the flag the pip layer grows
317 MB → 401 MB (545.29 MB image). That is the one size claim the report can make with numbers behind
it.
**Notes:** The plan carries a **pinned-sources table** (model card, corpora, llama.cpp docs, Compose
spec) and the rule that documents in this repo cite those and only those — a fact re-derived by
search can silently become a different fact, and the report has to be reproducible.
The brief's "train a model for a few epochs" is the only item that requires real ML work;
everything else in the grading list is Dockerfile and Compose craft. Timing: 4 days to the 25 Oct
00:00 cut-off, and the report is capped at 6 pages *including* code.
