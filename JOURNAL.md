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
and **8.84 tokens/s** end-to-end through gradio at 67–89 tokens per run. Re-run during review gave
8.88 / 8.49 tok/s on the same scripts — a few percent of drift between executions, so the committed
artifact is the reference and no single figure should be quoted as exact. The GUI's elapsed time is
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

**Finding (stage 3 — a Compose profile cannot re-parameterise an existing service):** the plan said
the GPU variant would be a `profiles: ["gpu"]` entry. That is wrong for this case and the note in
§6.1 R6 is corrected: a profile decides whether a service is *started*; it cannot change the build
target, the image tag or add a device reservation to a service that `compose.yaml` already defines
(the merge that would be needed does not exist). The mechanism that does work is a second file —
`compose.gpu.yaml`, applied with `-f compose.yaml -f compose.gpu.yaml`, whose keys merge over the
first. Profiles stay in the plan for P4, where the training stage is a *new* service and profiles
are exactly right.

**Finding (stage 3 — a target-less Docker build builds the LAST stage, and it bit us):** the
Dockerfile was first written as `model` → `cpu` → `gpu`. With no `--target`, BuildKit builds the
final stage, so plain `docker compose build` — the default path that must keep behaving as in
stages 1–2 — quietly produced the **4.4 GB CUDA image** while still tagging it `:stage-1`. Nothing
failed; the mistake was visible only by comparing image sizes (5.68 GB vs the expected 2.12 GB).
The fix is to make `cpu` the last stage and say so in the file. General lesson: **"X is the default
build" is a property of stage order, not of a comment or a name** — verify it by size or digest,
never by reading the Dockerfile.

**Finding (stage 3 — the CUDA image's server logs at debug verbosity, and loads the model twice):**
`server-cuda13` (build 11749) starts with `verbosity = 5`; one startup plus twelve benchmark requests
produced **60 014 log lines**. Worse for evidence work, it loads the model **twice**: first a
`no_alloc` probe that `--fit` uses to size the request, then the real load. The probe's block prints
`CUDA0 model buffer size = 0.00 MiB` and `offloaded 25/25 layers to GPU` — i.e. if you grep the log
and take the *first* match, you conclude the model is fully offloaded and yet resident nowhere.
The real load's block is the second one: 1211.05 MiB in `CUDA0`. The capture step in
`scripts/gpu_evidence.sh` therefore cuts the log at the last `initializing ...` line and greps only
the current boot — the first version grepped the whole history, which reports whichever restart
happened to be last and mixes two boots' buffer sizes into one block.

**Finding (stage 3 — the GB10 reports no VRAM counters, so "how much VRAM" needs two sources):**
`nvidia-smi` on this host prints `Not Supported` for `Memory-Usage` and `[N/A]` for
`memory.total/used`, because the GPU shares the 128 GB unified pool with the CPU — the usual
"check VRAM before and after" method simply does not exist here. The per-process column does work,
and that is the number used: `/app/llama-server` → **1589 MiB**. llama.cpp's own accounting agrees
in shape: `CUDA0` model 1211.05 MiB + KV 48.00 + recurrent 77.06 + compute 56.02 ≈ 1392 MiB, which
matches its own `projected to use 1392 MiB of device memory`. Measured, not assumed. Note also that
this committed memory does **not** show up in `docker stats`: the container reports ~1 GiB of RSS
against its 4 GiB limit while holding 1.6 GB on the device, because the unified-memory allocation is
the driver's, not the cgroup's — so `docker stats` is not a superset of GPU usage on this machine.

**Finding (stage 3 — at temperature 0 the GPU is not reproducible even against itself):** the
isolated probe pins temperature 0 and max_tokens 160, so it was expected to be deterministic *and
identical* across backends. It is deterministic **within** an execution but not across them. The CPU
stack emitted 38 completion tokens on every run of every execution. The GPU emitted **59** on all
three runs of one execution, then **77** on all three runs of the next three executions — same
prompt, same request body, same container, same image, no configuration change in between. Greedy
decoding is only greedy if the arithmetic is bit-identical, and CPU vs CUDA (with flash-attention on
the GPU side) is not; a near-tie flips, the trajectory diverges, and the completion stops at a
different place. **This changed the shape of the benchmark:** the completion-token count is part of
the measurement, not a constant to divide into elapsed time, so `scripts/bench_gpu.sh` now runs the
whole measurement **twice** (`EXECUTIONS=2`) and keeps both executions in the artifact, every
comparison is made on **effective tokens/s** with the token count printed beside it, and
`bench_compare.py` refuses to print a speedup factor at all when the counts disagree. Measured on
the committed pair: 138.77 tok/s (execution 1) and 133.93 tok/s (execution 2) isolated, i.e. 15.63×
and 15.08× over the CPU's 8.88; 15.12× and 15.69× through the GUI. What looks stable is the **rate**
(~134–139 tok/s isolated, within ~4 % between executions) and the **ratio** (~15.1–15.6×), not the
answer. A raw wall-clock speedup would have been flattering (7.47× isolated, 20.39× GUI) and wrong.
"Same prompt, same answer on both backends" is not a property this stack has; "same prompt, roughly
the same rate" is — and that is the only claim the numbers support.

**Decision (P4 host — train on the RTX 3090 Ti machine, not the Spark):** the LoRA we planned (D7:
bf16, explicit `target_modules`, no quantization) fits 24 GB comfortably — a 2B in bf16 is ~5 GB of
weights plus activations and optimizer state — and that machine measured ~2.4× the GB10 on this very
model (320.66 vs 134.40 tok/s). The tie-breaker is **disk, not speed**: the Spark's `/` is 97 % full
(32 GB free) while the other machine has 311 GB free, and the training base image alone
(`nvcr.io/nvidia/pytorch:26.09-py3`, multi-arch — both amd64 and arm64 exist) is ~12 GB compressed
and ~25 GB unpacked. Consequence for the pipeline: training runs off the serving host, the adapter
comes back (~30 MB artifact), gets converted to GGUF and served on the stage-3 stack, so base vs
adapter is measured on the same benchmark. **Fallback if the run grows past 24 GB** (full-parameter
SFT rather than LoRA): the Spark's 128 GB unified memory is the reason to keep the GB10 in the
picture at all, and disk would have to be freed first. Unverified and worth one timed spike before a
long run: whether the GDN (Mamba) layers get their fast conv kernels on either host — `causal-conv1d`
is source-only on both architectures, Triton ships wheels for both.

**Notes (stage 3 — measured, GB10):** the 1.28 GB GGUF is downloaded once by the shared `model`
stage; the second target's build is a pure cache hit (`model 2/2 RUN … CACHED`, `COPY --from`
0.0–0.5 s). Splitting the download into its own stage cost **nothing** in image size: the CPU image
is byte-for-byte its stage-2 size, 2 117 416 629 B — the extra `COPY --from` layer carries only
metadata. The GPU image is 5 684 548 299 B, of which the CUDA 13 base is 4.40 GB. Engine detail:
`count: all` in `deploy.resources.reservations.devices` resolves to `DeviceRequests[0].Count = -1`,
not to a count — expected once seen, but it is what the `docker inspect` output shows and it looks
like a bug until you know.

**Finding (stage 3 — a wrong backtick escape wrote wrong text into an evidence artifact):** the
first draft of `gpu_evidence.sh` had ``echo "--- plain\` docker compose build …\` (cpu target) ---"``
inside a redirection block. Doubled backslashes do **not** protect a backtick in a double-quoted
string, so bash ran the inner text as a command substitution: the artifact printed
`--- plain\ (cpu target) ---` and two `no such service: \` errors went to the build log. It was
visible only by reading the generated artifact, not the script. General lesson: when a script
*writes* an artifact, read the artifact — a syntax mistake in a generator is silent in a way it
never is in a program.

**Findings (P4 — the corpus is not the shape the plan described, and the base model is already good
at it):** the Magicoder repo carries **no parquet at all**. At the pinned revision (unchanged, and
current) it holds a single `data-oss_instruct-decontaminated.jsonl` of **203 201 169 B** — the size
matched the plan's "~194 MiB", the format did not, and a loader written against the plan would have
404'd. The trainer streams that file out of the HF cache line by line, which also makes a
`--scan-limit` possible without decoding the whole corpus. More important: the base model's loss on
this corpus is **~0.46 at step 1**, and the reason is measurable — on 200 records the sequence is
208/524/1024 tokens (min/median/max), of which only **57/234/612** are *supervised* (the prompt is
masked), and the solutions are template-heavy (`#include <iostream>`, `cout`, repeated function
signatures copied from the problem statement). So the loss curve will move in a narrow band, and a
full run that "improves" 0.46 → 0.42 is a normal result, not a broken one. **The consequence is a
decision: the loss curve cannot be the evidence for this bullet** — §6.1 R4's two metrics
(`ast.parse` adherence, a HumanEval pass@1 slice) measured base-vs-adapter are, and the curve is a
figure showing the schedule behaved.

**Findings (P4 — the composite→text-only load is a silent trap, and this checkpoint is exactly the
case it bites):** `Qwen/Qwen3.5-2B-Base` ships the composite layout — 632 tensors: 320
`model.language_model.*`, 297 `model.visual.*`, 15 `mtp.*` — while `Qwen3_5ForCausalLM` (transformers
5.20.2) builds a `model.layers.*` module tree. `Qwen3_5ForCausalLM.from_pretrained(repo)` therefore
matches **zero** weights: it emits a warning, initialises the text tower at random and trains that.
The trainer re-keys the state dict itself and treats a non-empty `missing`/`unexpected` set as a hard
error instead of a warning. Verified on the smoke run: **320 tensors loaded, 312 dropped** (`mtp` +
`model.visual`), `missing: []`, `unexpected: []`, **1 881 825 088** text parameters. One legitimate
absence to keep in mind when reading that check: `lm_head.weight` is not in the checkpoint because
the embeddings are tied — it is the one name allowed to be missing, and the code says so.

**Findings (P4 — training image and environment, measured):** the training image is
**38 248 955 946 B (38.25 GB)**, of which the `nvcr.io/nvidia/pytorch:26.09-py3` base is
**38 065 164 029 B (38.07 GB)** — the whole training stack (transformers, peft, datasets, matplotlib)
adds **184 MB**. The plan's "~25 GB unpacked" estimate for the base was **13 GB low**, which matters
only because it is the number that decided against the disk-poor serving host; the decision stands,
the margin does not. Two more environment facts worth carrying: torch inside that image is
**2.14.0a0+b2c75dd062.nv26.9.68203377**, an alpha build, which is a second reason not to layer a
`pip install torch` on top of it; and the GDN reference fallbacks are now *measured* rather than
predicted — the log reports `causal_conv1d_fn` and `chunk_gated_delta_rule` both falling back to
their pure-PyTorch implementations because neither `causal_conv1d` nor `flash-linear-attention` is
installed. `USE_HUB_KERNELS=0` is pinned in the image so that stays true instead of transformers
resolving a hub kernel at run time. And peft writes the adapter in **fp32**: 62 561 688 B for r=16
across 10 target-module names (**15 630 336** trainable parameters, 0.82 % of 1.897 B) — not the
~30 MB the plan predicted, which is the number that matters for what the adapter costs to ship.

**Findings (P4 — the full run, measured):** 374 optimizer steps (2 epochs × 187) in **1801 s** on the
3090 Ti — 4.76 s/step, 2071.6 padded tokens/s median, flat for the whole run. Loss **0.4070 → 0.3359**,
per-step minimum **0.1763 at step 293**, and the interval means carry the shape: ~0.41 over steps 1–25,
a plateau at ~0.33 to step 175, a second drop to ~0.26 by step 300, then flat ~0.27 for the last 75
steps while the cosine LR decays to zero. Adapter 62 561 688 B, trainable 15 630 336 (0.82 %). Device
peak: the sampler covering the second half saw **13 249 MiB**, while the smoke run — identical
configuration, sampled from step 1 — showed **18 063 MiB** in its device table and 17 905 MiB in its own
sampler. That asymmetry is the finding worth keeping: a
sampler started late reports a **lower bound**, and the honest peak for the configuration is the number
from the run that was sampled from the first step (18 GB of 24 GB, ~6 GB headroom).

**Findings (P4 — the adapter is written as root into a bind mount):** `save_pretrained` runs as the
container's root user, so `adapter_model.safetensors` lands on the host as `root:root`, mode `600` —
`ls` shows the file sitting there while a plain `scp` of the directory fails with "Permission denied".
Retrieval is `docker cp <container>:/out/adapter/. <dir>` (which runs as root), or a matching `user:` on
the service. Worth knowing before the adapter is needed: the run is 30 minutes and the file is the
deliverable.

**Findings (P4 benchmark — the adapter changed style, not correctness):** 150 generations (10 fixed
prompts × 5 sampled runs × 3 arms; base and tuned serve the *same* GGUF, the adapter is the only
difference). Verdict: no demonstrated coding gain. On the code-review prompt neither arm of the pair
finds the real bug — the cache evicts the most-recently-*inserted* key, because only `get` moves a key
to the front, so `popitem(last=True)` never evicts the LRU — and both arms hallucinate confidently about
it; one tuned run's "corrected version" is the original code unchanged. What is real and repeatable:
tuned answers are ~45 % shorter (6 594 vs 11 908 completion tokens over 50 runs) and never hit the token
cap (0/50 vs base 7/50). That terseness is a genuine win on `reasoning-1` (2/5 correct vs 0/5) and a
genuine regression on the email prompt, where 4/5 tuned replies are third-person *descriptions* of the
requested email rather than the email. Held-out perplexity moves the wrong way, slightly: 9.0791 → 9.1941
(general prose) and 3.0116 → 3.0758 (code). Two things worth keeping from the base side: at these
settings the *base* model leaks chat-template artefacts mid-answer (`...}`) and contradicts itself
arithmetically (computes 6:20 PM, prints 19:20); and simple clock arithmetic is unreliable in every arm —
tuned's best runs reach 18:05, one reaches it and then relabels it "6:05 AM the next day".

**Findings (P4 benchmark — what the method had to be, twice):** the shipped GGUF is the **instruct**
lineage while the adapter was trained on the **Base** checkpoint, so the only honest pair is base vs
base+adapter at identical quantization; measuring the adapter against the production model would have
silently reported "base vs post-trained" as "the adapter's effect". And the first-run canaries are
seed-driven, not arm-driven: at seed 43507 all three arms returned `{}` for the strict-JSON prompt, which
is a sampling artefact that a single-run benchmark would have published as a model difference.

**Findings (P5 — the HF free tier no longer covers Spaces, and the docs still say it does):** the
assignment's last bullet is satisfiable without paying on paper and not in practice. Creating a Gradio
Space through the API returns **402** on the default `cpu-basic` ("hosting Gradio and Docker Spaces on
free cpu-basic requires a PRO subscription") and **402** again with `hardware: "zero-a10g"`
("You must be subscribed to PRO to host Spaces with ZeroGPU. If you recently created your account,
please wait 30 days or request a community grant"), while the Spaces docs still promise *"free personal
accounts in good standing can still host up to 2 Gradio Spaces running on ZeroGPU"*. The UI agrees with
the API, not the docs: Gradio and Docker are both badged **Paid**, and only **Static** is free — and
Static cannot execute a 3.8 GB model (Gradio-Lite runs Python in the visitor's browser via Pyodide;
Transformers.js would need an ONNX export of a hybrid Gated-DeltaNet architecture that does not exist).
With PRO the same API call succeeds immediately, so the paywall was the whole blocker. Recorded because
it is a platform change that an assignment written earlier cannot have anticipated.

**Findings (P5 — four things that cost a deploy cycle each, worth not rediscovering):** (a)
`gr.ChatInterface(examples=[...])` must be a list of **lists** when `additional_inputs` is used; a flat
list raises at construction and the Space dies in RUNTIME_ERROR with the traceback only in the run log.
(b) Gradio 6 serves the app config at `/config`, **not** `/gradio_api/config` — the `/gradio_api` prefix
applies to the *call* endpoints, and the endpoint signatures live at `/gradio_api/info`, which is the
robust way to discover how to call a Space. (c) The model must be moved to `cuda` at module level: CUDA
is emulated outside `@spaces.GPU`, so the 3.76 GB load costs no GPU quota, and the run log shows ZeroGPU
packing the tensors before the app starts. (d) The served app has to reproduce the benchmark's sampler
exactly, `presence_penalty=2.0` included — the same weights answer a Fibonacci prompt in 116 tokens with
the penalty and 220 (capped, with a docstring) without it, so a demo that drops the parameter shows a
different model than the one that was measured.

**Findings (P5 — the merged model needs none of the training-side machinery):** the published
checkpoint loads with plain `AutoModelForCausalLM.from_pretrained` — `Qwen3_5ForCausalLM`, bf16, 320/320
tensors, no `missing`/`unexpected` keys, and `peft` never imported — which is exactly why route (b) was
chosen over publishing the adapter and asking every consumer to reproduce the composite→text-only
re-key. The merge is deterministic (a repeat run produced a byte-identical safetensors) and the Hub's
LFS oid for the uploaded file equals the locally computed sha256.

**Findings (P4 — an evidence generator can fail in the direction of looking fine):** two bugs in the
smoke record were found only by re-running it and reading the artifact it produced. (a) The VRAM
sampler wrote through a single redirected stdout, which is block-buffered, so a run short enough to
finish before the buffer filled wrote **zero** samples — and the section then printed a device table
captured while the GPU was idle, in a file whose purpose is to prove the GPU was used. (b) The "wait
until the first loss line appears" check keyed on the loss log, which survives from the previous run,
so on a re-run it returned immediately and the same idle table was captured. Fixes: append per line
(which is unbuffered without `stdbuf`) and truncate the loss log before launching. Same family as the
backtick bug above — an evidence script is only as honest as its worst re-run, and the tell is a
section that is present but empty.

**Findings (P4 — an inherited `EXPOSE` makes the trainer's isolation look weaker than it is):** the
NGC base image `EXPOSE`s 6006 and 8888 itself, so `docker compose ps` prints `6006/tcp, 8888/tcp` in
the PORTS column of a service that declares no `ports:` and no `expose:` at all. Nothing is published
— Compose publishes only what `ports:` maps, and there is no mapping here — but the column reads like
one, which is the kind of detail that turns a correct claim ("the trainer has no interface") into a
contested one during a review. Worth one sentence in the artifact rather than a silent confusion.
