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
**Duties / To-do:** Awaiting Relja's green light on this plan before any development.
**Notes:** The plan carries a **pinned-sources table** (model card, corpora, llama.cpp docs, Compose
spec) and the rule that documents in this repo cite those and only those — a fact re-derived by
search can silently become a different fact, and the report has to be reproducible.
The brief's "train a model for a few epochs" is the only item that requires real ML work;
everything else in the grading list is Dockerfile and Compose craft. Timing: 4 days to the 25 Oct
00:00 cut-off, and the report is capped at 6 pages *including* code.
