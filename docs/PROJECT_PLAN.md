# Project plan — Assignment 2 (Docker) · MUIA 43506

**Repository:** `ljubobratovicrelja/aiia-docker` · **Author:** Relja Ljubobratović · **individual work**
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
| **Max grade:** NVIDIA Container Toolkit + training/inference on GPU | `server-cuda13` build target behind a Compose **override file** (a profile cannot re-parameterise an existing service), one shared model stage so the GGUF is fetched once, device reservation + offload env | `compose.gpu.yaml` + `inference/Dockerfile` |
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
| D9 | `training/` builds on **`nvcr.io/nvidia/pytorch:26.09-py3`**, bf16, `trl.SFTTrainer`, explicit `target_modules` | CUDA torch is a wheel install on arm64, so the base image is the only real choice; bf16 because a 2B against 128 GB of unified memory needs no quantization (§6.1 R3) |

## 5. Phases and exit criteria

Each phase ends with something executable that proves it, not with "it should work".

| Phase | Deliverable | Exit criterion (the proof) |
|---|---|---|
| **P0** ✅ | Skeleton, plan, brief embedded | This document committed and pushed |
| **P1** ✅ | CPU two-container baseline: `inference/` + `gui/`, `compose.yaml` | Verified: a prompt through the **GUI container's HTTP API** returns a completion generated inside the inference container; `PortBindings` on inference is `null` while the GUI is published on 7860; limits/shm/read-only/user applied to **both** services |
| **P2** ✅ | Craft demonstration and measured CPU evidence | Verified: every criterion now has an artifact plus a regeneration command in `docs/EVIDENCE.md` (rows 1–10, 14). Cache ordering proved by building three ways: nothing changed → **every step `CACHED`**; app code edited → the pip layer `CACHED` and only `COPY app.py` re-runs; `LLAMA_ARG_CTX_SIZE` changed → the **1.28 GB model layer `CACHED`**, nothing re-downloaded (`evidence/build-cache.txt`). Image sizes and the full layer breakdown measured (`evidence/image-sizes.txt`), with one size decision actually taken rather than asserted — `pip install --no-cache-dir` kept after building the alternative and measuring it: **461.21 MB vs 545.29 MB (+84.1 MB, +18.2 %)** (`evidence/size-experiment.txt`). Fixed prompt measured on CPU through **both** paths: **8.88 tok/s median isolated** (4.280 s, 38 tokens) and **8.49 tok/s through the GUI** (`evidence/bench-cpu.txt`) — the numbers rows 1–2 and the stage-3 baseline quote |
| **P3** ✅ | GPU build target + Compose override file, and CPU-vs-GPU numbers | Verified: `inference/Dockerfile` builds a `cpu` target (the default — it is the file's last stage) and a `gpu` target (`server-cuda13`) out of **one shared model stage**, so switching targets never re-downloads the 1.28 GB GGUF (the second build's model layer is `CACHED`); `compose.gpu.yaml` is an **override file** — not a profile: a profile can add a service but cannot re-parameterise an existing one — setting `build.target: gpu`, a distinct image tag, `deploy.resources.reservations.devices` and the offload env; the running container holds a real device request (`HostConfig.DeviceRequests`) and offloads **25/25 layers to GPU** (`CUDA0 model buffer size = 1211.05 MiB`, 1589 MiB per `nvidia-smi` **inside** the container); same fixed prompt, same request body: **134.40 tok/s GPU vs 8.88 CPU** (15.14× effective tokens/s; pooled 135.48 vs 8.74, and 15.63× then 15.08× in the two executions kept in the artifact); plain `docker compose up -d --build` still builds and runs the CPU target unchanged |
| **P4** ✅ | LoRA training stage | Verified and **merged** (`db76964`, PR #4). Adapter produced (62 561 688 B), loss curve recorded (`evidence/loss-curve.png`, three panels), **full run** — 374 optimizer steps, exit 0, no OOM, loss 0.4070 → 0.3359, 4.76 s/step, ~30 min, ~18 GB of 24 GB peak (`evidence/train-run.txt`, [`TRAINING_REPORT.md`](TRAINING_REPORT.md)). **Base vs adapter measured** — 150 generations (10 fixed prompts × 5 sampled runs × 3 arms), adapter against its own base at identical quantization, read in [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md): the adapter changed style far more than correctness (~45 % shorter, 0/50 vs 7/50 truncations), one win (`reasoning-1` 2/5 vs 0/5), one regression (`general-context-2`: 4/5 third-person meta-descriptions instead of the requested email), no demonstrated coding gain. **Remaining (not part of P4's own criteria):** plan §6.1 R4's `ast.parse` adherence and HumanEval pass@1 slice. **Host: the RTX 3090 Ti machine, not the Spark** — 24 GB is ample for the bf16 LoRA, it is ~2.4× the GB10 on this model, and the Spark's `/` has only 32 GB free against 311 GB there (the training base image alone is 38.25 GB unpacked). Adapter returns to the Spark for GGUF conversion and serving on the stage-3 stack |
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
| R3 | What is the exact CUDA-PyTorch install route on aarch64 / GB10 (index, version, wheel availability), and is a CUDA base image needed for the `training` container? | Sizes the training environment work and pins the training container's `FROM` | **answered — §6.1** |
| R4 | Dataset facts (exact repo, revision, row count, license, on-disk size) for the corpus candidates, and which objective metric fits the 6-page budget | Removes "which dataset / how do we show it worked" from the critical path | **answered — §6.1** |
| R5 | Can a Hugging Face Space host this model, with which SDK and hardware tier, at what cost — and what are the exact publish steps? | The second max-grade bullet; if it is not free in practice, we drop it deliberately rather than discover it late | **answered — §6.1** |
| R6 | Which of the run parameters we intend to use are actually supported by the Docker/Compose version installed here, and what is the correct syntax for each? | The YAML is a graded artifact; it must be valid on this engine, not aspirational | **answered — §6.1** |

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

**R3 — CUDA PyTorch on aarch64/GB10: a wheel install, not a build.**

- A claim carried earlier in this plan — that aarch64 CUDA wheels stop at torch 2.0.1 — is **wrong**.
  It came from reading the *tail* of an unsorted index listing. Re-checked with a numeric sort over
  the simple indexes: `download.pytorch.org/whl/cu130` ships aarch64 wheels up to **2.14.1+cu130**
  (cp310–cp315), and PyPI's default index up to **2.14.1**.
- Those aarch64 wheels are **CUDA-enabled, not CPU-only**: the wheel's central directory carries
  `torch/lib/libtorch_cuda.so` (~461 MB uncompressed) and its metadata pins `cuda-toolkit==13.0.3`,
  `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`.
- `training/` base image: **`nvcr.io/nvidia/pytorch:26.09-py3`** (arm64 digest confirmed with
  `docker manifest inspect`; CUDA 13.4.1, torch 2.14.0a0, ~12.3 GB compressed). Leaner alternative:
  `nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04` (~2.1 GB) plus `pip install torch==2.14.1`.
  No compilation either way.
- Stack on top: `transformers`, `peft`, `trl`, `accelerate`, `datasets`, `sentencepiece`, `protobuf`.
  **Trainer: `trl.SFTTrainer` in bf16, no quantization** — a 2B in bf16 is ~4 GB against 128 GB of
  unified memory, so 8-bit/4-bit buys nothing, and QLoRA is discouraged on this family anyway
  (§6.1 R1). Two caveats: the released `peft` has no `qwen3_5` mapping, so `target_modules` must be
  listed explicitly; and the checkpoint is a hybrid VLM with an MTP head, so pass the processor via
  `processing_class` — or fall back to a hand-rolled peft + `Trainer` loop if it fights.
- Environment cost: ~5–25 min image pull plus ~2–4 min of pip. No build step.
- Not verified: Blackwell sm_120/121 coverage in the wheel's fatbin, the NGC image's exact Python
  version, and `SFTTrainer` actually running on this hybrid checkpoint — nothing was installed on
  the host, so that verification belongs to P4.

**R4 — corpus and metric: decided.**

- **Corpus: `ise-uiuc/Magicoder-OSS-Instruct-75K`**, revision `5f839b1f368a76b161028bb9edff055db34022b2`,
  MIT, 75,197 train rows, ~194 MiB, columns `problem` / `solution` (plus `lang`, `seed`).
  `load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train[:3000]")`.
- Fallback: `sahil2801/CodeAlpaca-20k` (cc-by-4.0, 20,022 rows, ~7.7 MiB, one JSON file,
  `instruction` / `input` / `output`) — trivial to load, but some `output` values are prose rather
  than code, so it supports a weaker claim. Not chosen: `Magicoder-Evol-Instruct-110K`
  (apache-2.0, 111,183 rows, 243 MiB) and `nvidia/OpenCodeInstruct` (cc-by-4.0, 6.4 GiB, 5M rows
  claimed) — both waste bandwidth for a few-thousand-sample run.
- **Two metrics, both cheap and both reported for base and adapter:**
  1. *Instruction adherence* — the share of generated answers whose extracted code block passes
     `ast.parse`. No execution, no sandbox, and it is the number that actually moves.
  2. *pass@1 on a 50-problem slice of HumanEval* (`openai/openai_humaneval`, `split="test"`,
     164 problems) — greedy, `max_tokens≈256`, run the problem's own test harness in a subprocess
     with a timeout. ~5–8 minutes for 50 problems on a 2B Q4; the full set is 20–30 minutes.
- Carried uncertainty: OpenCodeInstruct's 5M row count is README-sourced (the datasets server has
  indexed 1.4M); the runtime figures are projections, to be measured on the real GPU.

**R5 — HF Space: the free route exists, and it is a Gradio Space on ZeroGPU.**

- Re-read of the spec: the body says *"Optionally you can … publish the model in Hugging Face Spaces
  so that inference can be done through an API"*, and the max-grade list repeats it as *"Upload to
  Hugging face spaces and use of online API inference"*. Nothing in it requires a paid tier, and
  nothing requires the Space to run our container.
- Free personal accounts **can host up to 2 Gradio Spaces on ZeroGPU** — "Free personal accounts in
  good standing can still host up to 2 Gradio Spaces running on ZeroGPU". ZeroGPU is a shared RTX
  PRO 6000 Blackwell handed out through the `@spaces.GPU` decorator, explicitly built for
  PyTorch/`transformers` apps; a 2B fits easily in the 48 GB `large` slice.
- What needs PRO (~$9/month) is a **Docker** Space, and compute-backed Spaces generally *outside*
  that ZeroGPU exception. That is the expensive route, not the only one — and not the one the spec
  asks for.
- Plan: publish the **fine-tuned model** (base + adapter, or merged) to the Hub — free — and serve it
  from a small **Gradio Space on ZeroGPU**, which gets an API for free (`gradio_client` / the Space's
  own HTTP endpoints). The Space is a second deployment of the *same artefacts*: `transformers` there,
  llama.cpp locally.
- Quota to respect: **5 minutes of GPU time per day** on a free account (2 minutes unauthenticated),
  medium queue priority; CPU-tier Spaces sleep after 48 h idle and wake on the next visit. Ample for
  a demo, not for a benchmark run.
- Judgement: **do it** — it is an evening's work, the app itself is small, and it is the only
  max-grade bullet left that costs nothing but time.

**R5 refresh — re-read against the live docs (2026-10-21), because the free tier narrowed since the
answer above was written.** Two of these change the plan; one corrects a claim made in conversation.

- **Creating a Gradio Space on `cpu-basic` is now paid too**, not just Docker: the docs say *"Static
  Spaces are free for everyone. Gradio and Docker Spaces run on compute and require a paid plan to
  create: PRO for personal accounts, Team or Enterprise for organizations. Free personal accounts in
  good standing can still host up to 2 Gradio Spaces running on ZeroGPU."* So ZeroGPU is not merely
  the cheap route — it is the **only free compute route**, and the plan above still holds.
- **"Good standing" has a definition to check before building**: *verified email and account older
  than 30 days*. Worth confirming on the account we publish under, because the failure would appear
  only at creation time.
- **Quota is charged on effective duration, times a size multiplier** — `large` (half RTX Pro 6000
  Blackwell, 48 GB) is 1×, `xlarge` (96 GB) is 2×, so a 45 s task on `xlarge` costs 90 s of quota. The
  free tier gets **5 minutes/day** (unauthenticated 2, PRO 40 + $1/10 min beyond). This corrects the
  earlier note that the *declared* duration drives the charge: the declared duration sets the maximum
  runtime per call and improves queue priority, and the charge follows actual use. A 2B at ~5 s per
  answer is therefore tens of requests/day on `large`, which is the number to state in the report.
- **ZeroGPU is Gradio-SDK only**, PyTorch 2.8+ (3.12.12 / 3.10.13 available), and models must be moved
  to `cuda` at module level — CUDA is emulated outside `@spaces.GPU`, so the load itself is free and
  lazy-loading inside the decorator is explicitly discouraged. `torch.compile` is unsupported.
- **Hardware selection is a Space setting**, not something the model card declares; the docs' own step
  list starts with *"Make sure the ZeroGPU hardware is selected in your Space settings."* Creating the
  repo and pushing the app are API operations; the hardware selection may need the UI, which is the one
  step that could require a human click.

**R5 correction — measured on the live account, 2026-10-21: the free Space route is closed.** The
paragraphs above were right about the docs and wrong about the account. What actually happened:

- `POST /api/repos/create` with `sdk: gradio` → **402**: *"hosting Gradio and Docker Spaces on free
  cpu-basic requires a PRO subscription"*.
- The same call with `hardware: "zero-a10g"` (confirmed as the valid ZeroGPU id by the API's own enum of
  hardware options) → **402**: *"You must be subscribed to PRO to host Spaces with ZeroGPU. If you
  recently created your account, please wait 30 days or request a community grant."*
- The Hub's Space-creation UI agrees and is blunter: **Gradio and Docker are both badged "Paid"**; only
  **Static** is free. Its templates (Gradio-Lite, Transformers.js, React, Svelte, Vue, Blank) all run
  client-side or not at all.

So the documented "up to 2 Gradio Spaces on ZeroGPU for free accounts in good standing" is not granted
for this account. What is left: **Static** Spaces, which can host a front end but cannot execute a
3.8 GB model — Gradio-Lite runs Python in the visitor's browser through Pyodide/WASM (no GPU, no server
compute), and Transformers.js would need an ONNX export plus transformers.js support for the hybrid
Gated-DeltaNet architecture, which does not exist. Serving *our* model from a Space therefore needs
**PRO ($9/month → Gradio on ZeroGPU, 40 GPU-min/day; or Docker on cpu-basic)** or an Inference Endpoint
(paid per hour). The model itself is already published and needs none of this:
`ljubobratovicrelja/qwen3.5-2b-magicoder-lora`.

Consequence for the max-grade bullet *"Upload to Hugging Face spaces and use of online API inference"*:
the **upload** half is satisfied and verifiable today; the **serving** half depends on a paid tier that
did not exist when the assignment was written. Decision recorded in O3.

**R6 — the Compose keys that actually take effect on this engine.**

Verified by running each against Docker Engine 29.2.1 / Compose v5.0.2 (non-swarm `docker compose up`):

| Intent | Key to use |
|---|---|
| CPU / memory limits | `deploy.resources.limits.cpus` + `.memory` — canonical and effective here; the service-level `cpus`/`mem_limit` shorthand also works, so declare it in one place only |
| Shared memory | `shm_size: "128m"` |
| Read-only root + writable scratch | `read_only: true` with `tmpfs: ["/tmp:size=64m,mode=1777"]` |
| Non-root | `user: "65534:65534"` |
| Healthcheck gating | `healthcheck.test: ["CMD", …]` and `depends_on: {inference: {condition: service_healthy}}` |
| GPU | `deploy.resources.reservations.devices: [{driver: nvidia, count: all, capabilities: [gpu]}]` — canonical; `gpus: all` is equivalent but needs Compose ≥ 2.30 |
| GPU variant of an *existing* service | **`-f compose.yaml -f compose.gpu.yaml`** (an override file). Correction to an earlier line of this plan: `profiles: ["gpu"]` does **not** work for this — a profile controls whether a service is *started*, and cannot re-parameterise a service that `compose.yaml` already defines. Profiles remain the right tool for adding a *new* service (P4's `training`) |
| Model storage | top-level `volumes:` plus `services.<name>.volumes: ["models:/models"]` |
| Internal-only port | `expose: ["8080"]` **and no `ports:`** — verified as empty `PortBindings` |

Not verified: a differential cgroup test of `deploy` limits against the shorthand.

**Closed by P3:** a container does get working CUDA. The official `server-cuda13` image resolves
the GB10 for real (`CUDA0: NVIDIA GB10`), the device reservation arrives as
`HostConfig.DeviceRequests = [{Driver: nvidia, Count: -1, Capabilities: [[gpu]]}]`, the server
offloads 25/25 layers, and `nvidia-smi` run *inside* the container lists `/app/llama-server` as a
GPU process. Two engine-level details found on the way: `count: all` resolves to `Count: -1`
(not a count), and the GB10's unified memory makes `nvidia-smi` report `N/A` for total/used VRAM
while the per-process column still works — that column, plus llama.cpp's own buffer accounting, is
where a VRAM number has to come from on this machine.


### 6.2 P4 implementation — the LoRA stage, and the smoke run that proves it

Built 2026-10-20 (`stage-4-lora`). The stage is `training/` (image + trainer + entrypoint), a
`training` service in `compose.yaml` under a profile, `scripts/train_lora.sh` (launcher),
`scripts/plot_loss.py` (curve + live monitor) and `scripts/train_smoke.sh` (the record). Measured
smoke run: `evidence/train-smoke.txt`.

| Decision | Why | Verified by |
|---|---|---|
| Trains on the **second machine** (RTX 3090 Ti 24 GB), never on the serving host | §6.1 R3 + the P4 row: the serving host has 32 GB of free disk against the training image's measured 38.25 GB, and the 3090 Ti is the faster device on this model | the smoke record's host/GUI lines and `df` |
| `training/Dockerfile` on **`nvcr.io/nvidia/pytorch:26.09-py3`** (plan D9) | CUDA torch preinstalled, multi-arch, no wheel roulette per architecture (§6.1 R3). Measured cost, correcting the plan's estimate: **38.25 GB unpacked, of which the base image is 38.07 GB**, i.e. the whole training stack (transformers/peft/datasets/matplotlib) adds **184 MB**. The plan said "~25 GB" — low by 13 GB, and it is still affordable on a host with 225 GB free | `evidence/train-smoke.txt` §2 |
| The service is added under **`profiles: ["train"]`**, not as an override, not in the default stack | A profile is the mechanism for *adding* a service (R6); the stage-1 property — plain `docker compose up -d` starts inference + gui and nothing else — has to survive a new optional stage | `docker compose config --services` → `inference gui`; `docker compose up -d --dry-run` creates only those two containers (smoke §1) |
| No `ports:`, no `expose:`; `restart: "no"` | A trainer needs no interface, only its mounted output dir. And a restart policy on a long job turns a crash into a loop that restarts from step 0 and **overwrites the loss log** on each pass | `compose.yaml`; `docker compose ps training` shows no published port |
| Weights and corpus in a **mounted host directory**, never baked | The smoke run's 4.5 GB download has to be the full run's download | smoke §9: `/var/tmp/aiia-hf-cache` = 4.5 G with both snapshots symlinked |
| **No `trl.SFTTrainer`** — a plain `peft` + torch loop | The plan itself named SFTTrainer as the risk on this checkpoint ("fall back to a hand-rolled loop if it fights"), `sample_packing` is ignored for this family, and the required per-step JSONL loss log is a training-loop concern | `training/train_lora.py`; the smoke's loss log |
| The `-Base` checkpoint is loaded **text-only by explicit re-keying**, not by `from_pretrained` | The repo is the composite layout (`model.language_model.*` + `model.visual.*` + `mtp.*`); `Qwen3_5ForCausalLM` expects `model.layers.*`. A plain `from_pretrained` matches **zero** weights, warns, and trains a randomly initialised model — the exact failure R1 warned about. Non-empty `missing`/`unexpected` is a hard error here | smoke §7 `run.json`: **320 tensors loaded, 312 dropped** (`mtp`, `model.visual`), `missing: []`, `unexpected: []`, **1 881 825 088** text parameters |
| LoRA `r=16 / alpha=32 / dropout 0.05`, **explicit** `target_modules` incl. the GDN projections | D7/R1: `all-linear` would reach into the vision tower and silently skip unmatched names | `adapter_config.json` in smoke §7; **15 630 336** trainable = 0.82 % of 1.897 B — and the count is *asserted* in code, because peft's typo mode is a silently frozen model |
| Loss on **answer tokens only**, prompt masked with `-100`; prompt rendered with the tokenizer's own chat template | The `problem` field is long (454 tokens on the first record); unmasked, most of the gradient goes into reproducing problem phrasings. The chat template is used because the GGUF converted from this adapter carries the same one — training on any other format trains for a prompt the server never sends | smoke §8 log line `chat_template=True`; measured on 200 records: sequence tokens 208/524/1024 (min/median/max), **supervised tokens 57/234/612** |
| Effective batch **2 × 8 = 16 sequences** of ≤ 1024 tokens (~8.4 k padded tokens/step) | A conventional SFT budget for a 2B, and it fits: peak device use measured at 18 063 MiB of 24 564 | smoke §4 + `run.json` (`peak_vram_gib: 10.539` allocated by torch) |
| `USE_HUB_KERNELS=0` | The GDN layer has an optional hub-kernel path; on a non-SM121 device it resolves to nothing, and on any device it adds a network dependency and a trust boundary to a run whose point is reproducibility | container log: `causal_conv1d_fn` and `chunk_gated_delta_rule` both report the reference PyTorch fallback |
| Fixed seed, but **bitwise reproducibility not claimed** | Same reason as the stage-3 finding: at bf16 a near-tie flips and the trajectory diverges | JOURNAL 2026-10-20 (stage 3) |

**Smoke result (2026-10-20, RTX 3090 Ti, 10 optimizer steps on 24 records drawn from a 200-record
scan, one execution, container torn down afterwards):** exit 0, loss **0.4567 → 0.3944**, 4.3–5.3 s
per optimizer step, ~2.0–2.2 k padded tokens/s, peak **18 063 MiB** in the device table and
**17 905 MiB** in the sampler (both with `python3` holding 17 414 MiB), adapter **62 561 688 B**
(60 MB).

**Full-run projection and the reason the loss curve is not the evidence.** 3000 records at 16 per
step is 187 steps per epoch, so 2 epochs ≈ 375 optimizer steps ≈ **29 min at 4.7 s/step**, plus ~2 min
already-cached load — call it **35 min**. Two caveats a reader should have in advance: (a) the base
model's loss on this corpus is already **~0.46 at step 1**, because the median answer is only 234
code tokens and the corpus is formulaic — the curve will move in a narrow band and its *shape*, not
its size, is what a figure can honestly show; (b) the adapter came out at **62.5 MB, not the ~30 MB**
the plan predicted, because peft stores LoRA parameters in fp32 regardless of the bf16 base. The
discriminating evidence for this bullet is therefore §6.1 R4's two metrics (`ast.parse` adherence and
pass@1 on a HumanEval slice), measured base vs adapter — not the loss delta.

### 6.3 P4 benchmark — base vs adapter, and how it is measured

The design below was fixed **before** the measurement ran, so the method could not be chosen after
seeing the result. Suite: `bench/prompts.json`. Harness: `scripts/bench_finetune.sh` +
`scripts/bench_finetune_probe.py`. Record: `evidence/bench-finetune.json`. Reading:
`docs/BENCHMARK_REPORT.md`.

| Decision | Why |
|---|---|
| Sampling follows the **model card**, not temperature 0 | The card's recommendation for **non-thinking mode, text tasks** — the mode this stack serves — is `temperature=1.0, top_p=1.00, top_k=20, min_p=0.0, presence_penalty=2.0, repetition_penalty=1.0`. Greedy decoding is also not reproducible here: stages 2–3 measured the same request giving different completions on different backends at `temperature 0`, so it buys determinism it does not actually have. Reproducibility comes from a **fixed seed per run** (`seed_base + run`) instead. |
| Every sampling parameter is sent **explicitly** | llama.cpp's own defaults differ from the card's (temp 0.80, top_k 40, top_p 0.95, min_p 0.05). Leaving any of them implicit would silently benchmark a configuration nobody recommended. |
| The controlled pair is **base vs base + adapter**, both from the pinned revision, both **Q4_K_M** | The shipped production GGUF is the **instruct** lineage (`unsloth/Qwen3.5-2B-GGUF`), while the adapter was trained on `Qwen3.5-2B-Base`. Comparing the adapter against the instruct model would conflate "what the adapter did" with "base vs post-trained". The instruct model is measured as a labelled **reference point** only. |
| 10 fixed prompts × 5 runs × 3 models, coding **and** non-coding, trivial **and** larger-context | Degradation from code-only SFT shows up as lost format adherence, repetition, language mixing and worse instruction following — none of which a coding-only suite would reveal. Two prompts are canaries with verifiable answers (arithmetic; JSON-only output). |
| `max_tokens` per prompt, far below the card's 32k suggestion | A fixed generation budget is what makes the comparison fair; 32k outputs would also blow the 8192 context and the wall clock. |
| Held-out **perplexity** on general prose and on code | A quantitative complement to the qualitative reading: the classic way to see a general-ability regression. |



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
| O2 | Corpus slice | **Magicoder-OSS-Instruct-75K, `train[:3000]`** (§6.1 R4); CodeAlpaca-20k if a lighter run is wanted |
| O3 | HF Space | **model published; the Space is blocked by the Hub's paywall** — the creation UI badges Gradio and Docker as *Paid* (only Static is free) and both API creation paths return **402**, for `cpu-basic` and for `hardware: zero-a10g` alike, so the free route R5 described is gone for this account (§6.1 R5 correction). **Done:** `ljubobratovicrelja/qwen3.5-2b-magicoder-lora` is public and verified by LFS oid. **Ready:** the Gradio app is written and committed (`space/`), and deploys in minutes on PRO ($9/mo → ZeroGPU, 40 GPU-min/day) or an HF community grant. Static Spaces cannot execute a 3.8 GB model (Gradio-Lite is browser-side Pyodide; Transformers.js would need an ONNX export for the hybrid Gated-DeltaNet architecture, which does not exist) | awaiting Relja's call |
| O4 | Adapter served, or training kept as a separate stage | **adapter served** if the llama.cpp converter fix is applied; else merged model (§6.1 R2) |
| O5 | GPU: how the variant is selected | **resolved by P3**: an override file (`compose.gpu.yaml`) rather than a profile — the CPU path stays the default build and the CPU-first portability claim survives, while the GPU variant still reuses the same service, model layer and GUI |

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
| HF Spaces overview (hardware, limits, plan requirement) | https://huggingface.co/docs/hub/spaces-overview |
| HF ZeroGPU (shared GPU, `@spaces.GPU`, daily quotas) | https://huggingface.co/docs/hub/spaces-zerogpu |
| HF Spaces — Docker SDK and `app_port` | https://huggingface.co/docs/hub/spaces-sdks-docker |
| HF Spaces — Gradio SDK and ZeroGPU | https://huggingface.co/docs/hub/spaces-sdks-gradio |
| HF pricing (tiers, PRO, GPU rates) | https://huggingface.co/pricing |
| HF datasets-server API (rows, splits, sizes) | https://datasets-server.huggingface.co/ |
| NGC PyTorch container (training base image) | https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch |
| NGC PyTorch release notes (arm64 / CUDA / torch versions) | https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/ |
| PyTorch aarch64 CUDA wheels | https://download.pytorch.org/whl/cu130/torch/ |
| TRL SFTTrainer | https://huggingface.co/docs/trl |
| Eval set — HumanEval | https://huggingface.co/datasets/openai/openai_humaneval |
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
