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

**This project is about the Docker setup, not about the code.** The deliverable that gets graded is
the container image, the Compose specification and the run parameters; the model and the small
amount of application code exist only to give the containers something real to do. So: keep
application code minimal, prefer configuration over code wherever a choice exists, and treat the
YAML as the primary artifact of the assignment — it should be exemplary on its own terms (service
separation, healthcheck gating, resource limits, GPU reservation, profiles, volumes).

We serve a **small language model with `llama-server`**, fine-tuned for a few epochs into a "coder"
adapter, so that one stack covers both the base assignment and the optional training clause.

## 2. Where each graded item is satisfied

| Graded item (brief) | How it is met | Artifact |
|---|---|---|
| Inference in a container on a pre-trained model | `llama-server` serving a Qwen3.5-2B GGUF | `inference/` |
| GUI used for the inference, isolated in its own container | gradio app, the only published port | `gui/`, `compose.yaml` |
| Inference wrapped in a web server | `llama-server` (OpenAI-compatible HTTP), plus a thin FastAPI layer if a custom route is needed | `inference/` |
| Dockerfile quality: instructions, existing images, cache layers | `FROM` the official llama.cpp image; app layers ordered above the model layer; cache-hit demo in evidence | `inference/Dockerfile` |
| Detail of run parameters | memory/cpu limits, shm size, read-only rootfs, non-root, healthcheck gating, GPU reservation | `compose.yaml` |
| Compose structure for GUI → local API | `gui` starts only once `inference` is healthy; inference port not published | `compose.yaml` |
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

- **Two containers is the graded minimum and stays the floor.** A third service is only added if it
  earns its place; `training` is a separate stage, never on the runtime path.
- Inference is reachable **only** from the compose network — that is what makes the isolation real
  rather than nominal.
- The model is **baked into the image** at a pinned revision with a `sha256` check, and also
  exposed as a volume so a base-model swap can be demonstrated without a rebuild.

## 4. Decisions taken

| # | Decision | Why |
|---|---|---|
| D1 | Runtime is **llama.cpp / GGUF** (`llama-server`) | Single binary, CPU-first, and the OpenAI-compatible web server — which the brief requires — comes with it, so the "wrap the inference in a web server" step needs no extra code |
| D2 | `FROM ghcr.io/ggml-org/llama.cpp:server` | The brief rewards *leveraging existing images*; a hand-rolled build earns nothing extra. Verified by inspection, see Appendix A |
| D3 | Model **baked in**, pinned revision + sha256 | "Runs on almost any machine" should mean self-contained; also gives something concrete to say about image size and cache layers |
| D4 | Inference port **not published** | The GUI must be the only entry point — this is the isolation requirement |
| D5 | `depends_on: service_healthy` | The base image ships its own `/health` healthcheck, so gating the GUI start costs nothing |
| D6 | Base model **`Qwen/Qwen3.5-2B-Base`**, loaded text-only as `Qwen3_5ForCausalLM` | A general small model that must not already be a coder, and the `-Base` checkpoint because we are doing SFT; text-only so the vision tower stays out of the graph (§6.1 R1) |
| D7 | Fine-tune by **LoRA with an explicit `target_modules` list**, served as an adapter **if the converter fix is in** — otherwise merge in HF and convert the merged model | No re-quantisation, ~30 MB artifact, one extra flag at serve time; the fallback exists because the in-tree converter cannot handle `linear_attn.out_proj` (§6.1 R2) |
| D8 | Application code stays minimal; the Compose spec is the artifact | See §1 — the assignment grades the container setup |

## 5. Phases and exit criteria

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

## 6. Risks and open questions

Each item below is being answered by an investigation, and this section is rewritten with the
answer (and its evidence) before any development starts. Nothing marked *pending* is an assumption
we will build on.

| # | Question to answer | Why it matters | Status |
|---|---|---|---|
| R1 | Can `peft` LoRA a Qwen3.5-2B (hybrid Mamba/attention, GDN layers)? Which module names must `target_modules` cover? | Decides whether the "train a few epochs" clause is a small job or a research project | **answered — §6.1** |
| R2 | Does llama.cpp serve a Qwen3.5 GGUF, and can it load a LoRA adapter for that architecture? Can `convert_lora_to_gguf.py` convert a peft LoRA of it? | Decides whether the adapter is served (D7 as written) or the training stays a separate documented stage | **answered — §6.1** |
| R3 | What is the exact CUDA-PyTorch install route on aarch64 / GB10 (index, version, wheel availability), and is a CUDA base image needed for the `training` container? | Sizes the training environment work and pins the training container's `FROM` | pending |
| R4 | Dataset facts (exact repo, revision, row count, license, on-disk size) for the corpus candidates, and which objective metric fits the 6-page budget | Removes "which dataset / how do we show it worked" from the critical path | pending |
| R5 | Can a Hugging Face Space host this model, with which SDK and hardware tier, at what cost — and what are the exact publish steps? | The second max-grade bullet; if it is not free in practice, we drop it deliberately rather than discover it late | pending |
| R6 | Which of the run parameters we intend to use are actually supported by the Docker/Compose version installed here, and what is the correct syntax for each? | The YAML is a graded artifact; it must be valid on this engine, not aspirational | pending |

**Constraint (not an investigation):** the report is capped at **6 pages including the code**, so the
number of optional items we attempt is bounded by the page budget, not by ambition.

### 6.1 Answers

**R1 — LoRA on Qwen3.5-2B: yes, with an explicit target list.**

- `Qwen/Qwen3.5-2B` is a **composite VLM checkpoint** (`Qwen3_5ForConditionalGeneration`; `text_config`
  + `vision_config`, and the vision tensors ship in the same repo). There is no official text-only 2B.
- The text base for this job is **`Qwen/Qwen3.5-2B-Base` loaded text-only as `Qwen3_5ForCausalLM`
  with `Qwen3_5TextConfig`** — the `-Base` checkpoint because we are doing SFT, and the text class
  because we do not want the vision tower in the training graph. This matches the arch mapping in
  llama.cpp's own conversion package (`Qwen3_5ForCausalLM`).
- **Prefix caveat:** composite weights are named `model.language_model.layers.N.*`; text-only
  re-saves are `model.layers.N.*`. Train and load the adapter against **one** layout — mixing them
  is a known hard failure elsewhere.
- Module names, read from the hub weight index and the modelling source: attention layers carry
  `q_proj / k_proj / v_proj / o_proj` (+ `q_norm`, `k_norm`; `q_proj` is double-width because
  `attn_output_gate` is on), the 18 GDN layers carry `linear_attn.in_proj_qkv`, `in_proj_z`,
  `in_proj_b`, `in_proj_a`, `out_proj`, and the non-linear `conv1d`, `A_log`, `dt_bias`, `norm`.
- **Use an explicit `target_modules` list, never `all-linear`.** `all-linear` would also attach
  adapters to the **vision tower** on a composite checkpoint, still cannot adapt the non-linear SSM
  params, and *silently skips* names that do not match — a typo yields a frozen model with no
  warning. Baseline set: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` plus
  `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, `linear_attn.out_proj` (see R2 for why
  `out_proj` may have to be dropped). Verify with `print_trainable_parameters()`.
- Further constraints found: **QLoRA / 4-bit is explicitly discouraged on Qwen3.5** (abnormal
  quantisation damage); on GB10 (SM121) neither `causal_conv1d` nor `fla` ship a build, so the GDN
  path falls back to the slow PyTorch reference (acceptable at our scale); TRL's `sample_packing`
  is silently ignored for this family; and `peft` issue #2878 reports `all-linear` behaving
  differently on aarch64 than on x86 — another reason to avoid it.

**R2 — llama.cpp: inference yes; adapter loading yes; the converter is broken in this checkout.**

- The GGUF arch string is **`qwen35`** (not `qwen3_5`), classified **hybrid** rather than recurrent,
  with graphs in `src/models/qwen35.cpp`.
- LoRA is applied inside the llama graph (`build_lora_mm`) and qwen35 routes **every** projection
  through it, the GDN ones included — so **runtime adapter loading works**. The adapter's
  `general.architecture` must equal the base model's, and every tensor is name- and
  shape-checked against the base tensor manifest.
- **However:** base Qwen3.5 linear-attention weights are stored in a V-head-reordered layout that
  differs from the HF checkpoint, so the adapter must be reordered the same way or LoRA multiplies
  the wrong weights. The converter does this — when it works.
- **`convert_lora_to_gguf.py` crashes in our checkout for any adapter that targets
  `linear_attn.out_proj`**: `LoraTorchTensor.reshape` raises `NotImplementedError` inside the
  V-head reorder branch. Upstream issue **#21125**, fixed by **PR #28324** (merged 2026-09-30,
  `b016f461`) — **not in our tree**, whose HEAD is 2026-08-27.
- Therefore three options, in order of preference: **(a)** update the local llama.cpp checkout past
  the fix (and check what build the serving image carries); **(b)** drop `linear_attn.out_proj`
  from `target_modules` — the row-axis GDN projections convert fine; **(c)** bypass the adapter
  path and merge in HF, then convert the full model.
- Non-matmul SSM parameters (`conv1d`, `A_log`, `dt_bias`, norms) **cannot** be LoRA targets at all.
- Separate serving risk worth carrying: upstream issue **#26916**, Qwen3.5 hybrid failing to load
  with `blk.N.attn_norm.weight` not found (MTP-layer loading), workaround `--no-mtp`.

## 7. Evidence to collect (the report is built from this, not the reverse)

- `docker compose ps` / `up` output showing the healthcheck gating the GUI start;
- training loss curve, adapter size, base-vs-adapter outputs on a fixed prompt set;
- `docker stats` RSS against the memory limit;
- image size (`docker images`) and a build-cache hit demonstration;
- tokens/s and end-to-end latency, CPU vs GPU.

## 8. Open decisions (defaults applied unless overruled)

| # | Question | Default taken |
|---|---|---|
| O1 | Qwen3.5-2B vs a Qwen3-1.7B-class model | **`Qwen3.5-2B-Base`, text-only** (§6.1 R1) |
| O2 | Corpus slice | decided by R4 |
| O3 | HF Space | **yes** (second max-grade bullet), contingent on R5 |
| O4 | Adapter served, or training kept as a separate stage | **adapter served** if the llama.cpp converter fix is applied; else merged model (§6.1 R2) |
| O5 | GPU: compose profile or the only target | profile, so the CPU-first portability claim survives |

## 9. Pinned sources

Use these; do not substitute alternates found by searching, so that every document in this repo
points at the same thing.

| What | Where |
|---|---|
| Brief (PDF + extracted text) | `docs/assignment/` (mirrors `muia/43506…/notes/theory/AIIA_DockerAssignment.pdf`) |
| llama.cpp repository | https://github.com/ggml-org/llama.cpp |
| llama.cpp Docker images and tags | https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md |
| llama.cpp server (flags, `--lora`) | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md |
| Model card — Qwen3.5-2B (composite), revision `15852e8c16360a2fea060d615a32b45270f8a8fc` | https://huggingface.co/Qwen/Qwen3.5-2B |
| Text base for the SFT — Qwen3.5-2B-Base | https://huggingface.co/Qwen/Qwen3.5-2B-Base |
| Corpus — Magicoder-OSS-Instruct-75K | https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K |
| Corpus — Magicoder-Evol-Instruct-110K | https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K |
| Corpus — CodeAlpaca-20k | https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k |
| Corpus — OpenCodeInstruct | https://huggingface.co/datasets/nvidia/OpenCodeInstruct |
| peft / LoRA | https://huggingface.co/docs/peft |
| Transformers Qwen3.5 model doc (text-only load path) | https://huggingface.co/docs/transformers/en/model_doc/qwen3_5 |
| Axolotl Qwen3.5 fine-tune guide (LoRA target modules for GDN) | https://docs.axolotl.ai/docs/models/qwen3.5.html |
| llama.cpp LoRA converter fix for linear-attn `out_proj` — PR #28324 (issue #21125) | https://github.com/ggml-org/llama.cpp/pull/28324 |
| HF Spaces | https://huggingface.co/docs/hub/spaces |
| Compose specification | https://docs.docker.com/reference/compose-file/ |
| GPU access in Compose | https://docs.docker.com/reference/compose-file/services/#gpus |
| Base image (CPU) | `ghcr.io/ggml-org/llama.cpp:server` |
| Base image (CUDA 13) | `ghcr.io/ggml-org/llama.cpp:server-cuda13` |

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
