# Project plan — Assignment 2 (Docker) · MUIA 43506

**Repository:** `ljubobratovicrelja/aiia-docker` · **Authors:** Relja Ljubobratović, Alex Marcondes
**Deadline:** 2026-10-25 00:00 (Moodle, *Submission Practical Assignment Docker*)
**Weight:** 25% of the practical grade

Status: **pre-development checkpoint.** Skeleton and plan only; no code yet.

---

## 1. Objective

Assemble a Docker container able to run a pre-trained Deep Learning model for inference, with the
inference call wrapped in a web server, and a *separate* container running only a graphical
interface that requests it — the two isolated with Docker Compose.

We do this with a **small language model served by `llama-server`**, fine-tuned for a few epochs
into a "coder" adapter, so that the same repo covers the base assignment and the optional training
clause without a second stack.

## 2. Where each graded item is satisfied

| Graded item (brief) | How it is met | Artifact |
|---|---|---|
| Inference in a container on a pre-trained model | `llama-server` serving a Qwen3.5-2B GGUF | `inference/` |
| GUI used for the inference, isolated in its own container | gradio app, only published port | `gui/`, `compose.yaml` |
| Inference wrapped in a web server | `llama-server` (OpenAI-compatible HTTP), plus a thin FastAPI layer if a custom route is needed | `inference/` |
| Dockerfile quality: instructions, existing images, cache layers | `FROM` the official llama.cpp image; app layers ordered above the model layer; cache-hit demo in evidence | `inference/Dockerfile` |
| Detail of run parameters | memory/cpu limits, shm size, read-only rootfs, non-root, healthcheck gating, GPU reservation | `compose.yaml` |
| Compose structure for GUI → local API | `gui` depends on `inference` being healthy; inference port not published | `compose.yaml` |
| **Optional:** train a model for a few epochs | LoRA fine-tune of the served model on code-instruction data | `training/` |
| **Max grade:** NVIDIA Container Toolkit + training/inference on GPU | `server-cuda13` build target + device reservation (GPU already available on the host) | `compose.yaml` (gpu profile) |
| **Max grade:** Hugging Face Spaces + online API inference | Space publishing the same small model | `spaces/` or `gui/` |
| **Max grade:** other improvements | cache-layer demonstration, image size work, healthcheck-driven startup, model pinning + sha256 | evidence + report |

## 3. Architecture

```
        browser
           |
    [ gui ]  gradio only, the single published port (7860)
           |  http://inference:8080/v1
    [ inference ]  llama-server + the GGUF, internal-only (8080 not published)
           |
      models/  (named volume: GGUF base + LoRA adapter)

    [ training ]  (off the serving path: dataset prep + peft LoRA, GPU)
```

- **Two containers is the graded minimum and stays the floor.** A third service is only added if
  it earns its place (the `training` container is a separate stage, not part of the runtime path).
- Inference is reachable **only** from the compose network — that is what makes the isolation
  real rather than nominal.
- The model is **baked into the image** at a pinned HF revision with a `sha256` check, and also
  exposed as a volume so a base-model swap can be demonstrated without a rebuild.

## 4. Decisions already taken

| # | Decision | Why |
|---|---|---|
| D1 | Runtime is **llama.cpp / GGUF** (`llama-server`), not transformers+torch, not vLLM, not HF TGI | Single binary, CPU-first, OpenAI-compatible server comes free; TGI's own docs call CPU "not the intended platform" |
| D2 | `FROM ghcr.io/ggml-org/llama.cpp:server` | The brief rewards *leveraging existing images*; a hand-rolled build earns nothing extra. Verified by inspection, see Appendix A |
| D3 | Model **baked in**, pinned revision + sha256 | "Runs on almost any machine" should mean self-contained; also gives something concrete to say about image size and cache layers |
| D4 | Inference port **not published** | The GUI must be the only entry point — this is the isolation requirement |
| D5 | `depends_on: service_healthy` | The base image ships its own `/health` healthcheck, so gating the GUI start costs nothing |
| D6 | Base model **Qwen3.5-2B** (dense, Apache-2.0) | General small model; the point of the training half is to *make* it a coder, so it must not already be one |
| D7 | Fine-tune by **LoRA**, served as an adapter | No re-quantisation, ~30 MB artifact, one extra flag at serve time; keeps the training stage off the runtime path |

## 5. Considered and rejected

- **YOLO / ultralytics + COCO128** for the training bullet: cheaper evidence (mAP before/after in
  ~1 minute on GPU, official public Dockerfile to modify). Rejected because the LLM route is the
  one that keeps a single coherent stack and a portfolio-worthy demo; the cost is that the
  training evidence is weaker and has to be framed honestly (see R1).
- **Pointing the stack at the remote HF Inference API**: moves inference *outside* the container
  and weakens the core criterion. HF Spaces is a *publishing* target, not a runtime.
- **`llama_index`**: an orchestration layer, not an engine. It earns its place only if a RAG index
  is added; a ~20-line FastAPI wrapper does the same job otherwise.

## 6. Phases and exit criteria

Each phase ends with something executable that proves it, not with "it should work".

| Phase | Deliverable | Exit criterion (the proof) |
|---|---|---|
| **P0** ✅ | Skeleton, plan, brief embedded | This document committed and pushed |
| **P1** | CPU two-container baseline: `inference/` + `gui/`, `compose.yaml` | `docker compose up` → a prompt typed in the browser returns a completion, and `docker compose ps` shows inference with no published port |
| **P2** | Dockerfile / Compose craft pass | cache-layer demo (touch app code → rebuild → model layer `CACHED`), non-root `USER`, read-only rootfs where it holds, limits in effect (`docker inspect` shows them) |
| **P3** | GPU profile (`server-cuda13` + device reservation) | `docker compose --profile gpu up` serves, and tok/s CPU vs GPU measured on the same prompt |
| **P4** | LoRA training stage | adapter file produced, loss curve recorded, base vs adapter outputs on a fixed prompt set captured |
| **P5** | HF Space | public Space URL answering a request |
| **P6** | Report (≤6 pp) | **not started — awaiting Relja's call** |

## 7. Risks and open questions

- **R1 — LoRA on a hybrid Mamba/attention model.** Qwen3.5 dense models are not plain
  transformers (GDN linear-attention layers alongside full attention). Two unknowns: which modules
  `peft` must target, and whether `convert_lora_to_gguf.py` handles the arch. Under verification
  in parallel with this plan. **Fallback:** merge the adapter in HF and convert the merged model,
  or serve the base model and keep the training as a documented standalone stage with its own
  evidence.
- **R2 — Honest evidence from a 2B.** A few epochs of LoRA on a 2B changes *behaviour* (code-block
  shape, docstrings, adherence) long before it changes *capability*. The report must show both:
  adherence measured objectively (`ast.parse` on outputs) and a small pass@1 delta reported
  as it is, not as an improvement.
- **R3 — aarch64 wheels.** Anything torch-based runs on the DGX Spark's GB10, where
  `pytorch.org/whl/cu130` has no current aarch64 wheels — the PyPI aarch64 build or a CUDA base
  image is the route.
- **R4 — 6-page budget.** Every optional item buys marks and costs pages. Order of work is
  deliberately: graded core first, optional items after.
- **R5 — Qwen3.5 in llama.cpp.** The conversion package maps `Qwen3_5*` already, but a converted
  GGUF that loads and generates must be demonstrated, not assumed.

## 8. Evidence to collect (the report is built from this, not the reverse)

- `docker compose ps` / `up` output showing the healthcheck gating the GUI start;
- training loss curve, adapter size, base-vs-adapter outputs on a fixed prompt set;
- `docker stats` RSS against the memory limit;
- image size (`docker images`) and a build-cache hit demonstration;
- tokens/s and end-to-end latency, CPU vs GPU.

## 9. Open decisions (defaults applied unless overruled)

| # | Question | Default taken |
|---|---|---|
| O1 | Qwen3.5-2B vs a Qwen3-1.7B-class model | **Qwen3.5-2B** |
| O2 | Corpus slice | Magicoder-OSS-Instruct-75K, trimmed; CodeAlpaca-20k if a smaller run is wanted |
| O3 | HF Space | **yes** (second max-grade bullet) |
| O4 | Adapter served, or training kept as a separate stage | serve it if the GGUF adapter conversion works; otherwise separate stage |
| O5 | GPU: compose profile or the only target | profile, so the CPU-first portability claim survives |

---

## Appendix A — base image, verified by inspection (2026-10-08, aarch64 host)

`ghcr.io/ggml-org/llama.cpp:server` — 836 MB unpacked, of which ~706 MB is one apt layer
(`libgomp1 curl ffmpeg`).

- entrypoint `/app/llama-server`, `WORKDIR /app`, `ENV LLAMA_ARG_HOST=0.0.0.0`
- ships a `HEALTHCHECK` on `http://localhost:8080/health`
- contains `curl` and `python3` (no `wget`) — a model can be downloaded inside the image
- runtime-dispatched ARM CPU backends, plus `libmtmd` (multimodal)
- build `b11459` / rev `f498f864…`, built 2026-10-07 — pin a digest for reproducibility
- note: the org moved `ggerganov` → `ggml-org`; old `ghcr.io/ggerganov/...` paths are stale

Host GPU: Docker's default runtime is already `nvidia` and the CDI devices are registered, so
GPU-in-container needs no extra plumbing on this machine.
