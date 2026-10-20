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
**Duties / To-do:** Awaiting Relja's green light on this plan before any development.
**Notes:** The plan carries a **pinned-sources table** (model card, corpora, llama.cpp docs, Compose
spec) and the rule that documents in this repo cite those and only those — a fact re-derived by
search can silently become a different fact, and the report has to be reproducible.
The brief's "train a model for a few epochs" is the only item that requires real ML work;
everything else in the grading list is Dockerfile and Compose craft. Timing: 4 days to the 25 Oct
00:00 cut-off, and the report is capped at 6 pages *including* code.
