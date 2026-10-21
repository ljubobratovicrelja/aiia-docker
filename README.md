# aiia-docker — Assignment 2 (Docker), MUIA 43506

> Part of the MUIA master's (Universidad de Alicante, 2026-27),
> course **43506 — Arquitecturas e Infraestructuras para IA** (resp. David Ortiz Pérez).


Deployment of a Deep Learning architecture for inference inside Docker containers, with the
graphical interface isolated from the inference in separate Compose services.

**Deadline:** 2026-10-25 00:00 (Moodle, *Submission Practical Assignment Docker* dropbox).
**Weight:** 25% of the practical grade of 43506.

## What the assignment asks for

- one container running a pre-trained DL model for inference, with the inference call wrapped in
  a web server (Flask / FastAPI);
- a second container running *only* a very simple GUI (gradio) that requests the first;
- the two isolated via Docker Compose;
- a report, max 6 pages excluding cover/TOC, including the code and the commands used.

Optional (and the max-grade clauses): train the model for a few epochs, publish it to Hugging
Face Spaces for API inference, NVIDIA Container Toolkit + training/inference on GPU, or any other
improvement to the Dockerfile / compose / run parameters.

The full brief is in [`docs/assignment/`](docs/assignment/) (PDF + extracted markdown).

## Layout

```
docs/            assignment brief (pdf + md), project plan, architecture decisions
inference/       container 1: model + web server wrapping the inference call
gui/             container 2: gradio interface, talks to container 1 over the internal network
training/        the "few epochs" stage (dataset prep + fine-tune), outside the serving path
evidence/        run logs, measurements and figures the report is built from
```

## Run

```bash
docker compose up -d --build
docker compose ps
```

GUI: <http://localhost:7860> — the only published port. The inference API is reachable
only from inside the compose network (`http://inference:8080/v1`).

```bash
./scripts/evidence.sh    # bring up, health-check, send a fixed prompt, collect evidence
docker compose down
```

## GPU variant (NVIDIA Container Toolkit)

`inference/Dockerfile` builds **two targets out of one shared model stage**: the 1.28 GB pinned
GGUF is downloaded and sha256-verified once by the `model` stage, and both targets `COPY --from`
it, so switching between them never re-downloads the model (the second build's model layer reports
`CACHED`). The CPU target is the file's **last** stage, which is what makes it what a target-less
build gets — so the command above keeps working exactly as before.

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build   # GPU (CUDA 13)
docker compose -f compose.yaml -f compose.gpu.yaml down
```

`compose.gpu.yaml` is an **override file**, not a profile: a profile can add a service, but it
cannot re-parameterise one that already exists, and here the existing `inference` service needs a
different build target (`gpu`), a different image tag, the device reservation
(`deploy.resources.reservations.devices: [{driver: nvidia, count: all, capabilities: [gpu]}]`) and
the offload settings (`LLAMA_ARG_N_GPU_LAYERS=99`, `LLAMA_ARG_FLASH_ATTN=on` — names read off the
image's own `--help`). The GUI, the isolation and the model are unchanged; the override only
changes which binary serves them.

On this host (one GB10) it is not decorative: the server loads with `offloaded 25/25 layers to
GPU`, `CUDA0 model buffer size = 1211.05 MiB`, `nvidia-smi` **inside** the container lists
`/app/llama-server` holding 1589 MiB, and the same fixed prompt runs at **134.40 tokens/s against
8.88 on CPU** — 15.14× effective tokens/s. Full logs and the CPU-vs-GPU tables: `evidence/gpu.txt`
and `evidence/bench-gpu.txt`. The same branch and the same two Compose files also run unchanged on a
second machine with an **RTX 3090 Ti** — 320.66 tok/s against 10.00 on that host's own CPU (32.07×) —
see §2.1 of [`docs/EVIDENCE.md`](docs/EVIDENCE.md).

## Training (optional bullet, off the serving path)

`training/` fine-tunes the served base model by LoRA SFT on `ise-uiuc/Magicoder-OSS-Instruct-75K`
(plan §6.2). It is a **third, optional Compose service**: it is gated by a profile, so

```bash
docker compose up -d          # still starts inference + gui, and nothing else
docker compose config --services          # -> inference, gui
docker compose --profile train config --services   # -> inference, gui, training
```

Training runs on the machine that has the GPU, not on the host that serves. Two things the launcher
expects from that machine are environmental rather than project choices, and both are parameters
rather than assumptions: a writable Docker configuration directory (`TRAIN_DOCKER_HOME`, needed where
`$HOME` is not writable for the Docker CLI) and a non-interactive shell, so a script is sent over
stdin instead of as a quoted one-liner.

```bash
# --- on the training host, from a copy of this repo ------------------------
export TRAIN_DOCKER_HOME=/var/tmp/train-home   # any writable dir, where $HOME is not writable

./scripts/train_lora.sh up        # build if needed, then start DETACHED
./scripts/train_lora.sh tail      # follow the JSONL loss log (live monitor #1)
docker compose --profile train logs -f training      # container log (monitor #2)
./scripts/train_lora.sh status    # container state + last logged step
./scripts/train_lora.sh down      # remove the container; the image stays
```

The full run is the launcher with no knobs set; a smoke run is the same command with them:

```bash
MAX_STEPS=10 TRAIN_SIZE=24 SCAN_LIMIT=200 ./scripts/train_lora.sh up    # smoke-sized
```

Three properties are deliberate and worth knowing before changing anything here:

- **The run configuration has one home** — the argparse defaults in `training/train_lora.py`.
  `training/run.sh` and the Compose file only *override* the handful of knobs a smoke run needs, so a
  smoke run and the full run cannot drift apart. `run.json` lands next to the adapter and records the
  effective values.
- **The weights and the corpus are mounted, never baked** (`/hf-cache`, `/out` on the host). The
  first run downloads ~4.7 GB once; every later run reuses it. `TRAIN_HF_CACHE` and `TRAIN_OUT`
  choose the directories.
- **The loss log is a first-class artifact**, not a copy of stdout: one flushed JSON object per step
  (`step, loss, lr, tokens_s, elapsed_s`), which makes `train_lora.sh tail` a live monitor over a
  bind mount and the same file the loss curve afterwards. On this host (matplotlib is already here,
  and the file is plain JSON):

```bash
python3 scripts/plot_loss.py --log training/out/loss.jsonl      # summary + evidence/loss-curve.png
```

The smoke test — 10 steps, one execution, teardown included — and every number quoted for this
stage live in [`evidence/train-smoke.txt`](evidence/train-smoke.txt), regenerated by
`bash scripts/train_smoke.sh` on the training host.

The **full run** (2 epochs, 374 steps, ~30 min) is recorded the same way, except that its script
documents a run that already happened instead of driving one:

```bash
TRAIN_RUN_DIR=/var/tmp/aiia-fullrun bash scripts/train_record.sh    # -> evidence/train-run.txt
```

Its result and how to read the loss curve are in [`docs/TRAINING_REPORT.md`](docs/TRAINING_REPORT.md).
The adapter is **not** committed (62.5 MB binary) and is written by the container's root user, so pull
it out with `docker cp` rather than a plain copy:

```bash
docker cp aiia-docker-training-1:/out/adapter/. ./training/out/adapter/
```

## Evidence

The graded artifact is the Docker/Compose configuration, so the measurements that back it up live in
[`evidence/`](evidence/) and are regenerated by the scripts in [`scripts/`](scripts/). Every graded
criterion of the brief is mapped to the artifact that proves it, and to the command that regenerates
it, in **[`docs/EVIDENCE.md`](docs/EVIDENCE.md)**.

```bash
./scripts/cache_demo.sh          # layer-cache proof      -> evidence/build-cache.txt
./scripts/bench.sh               # CPU tokens/s + latency -> evidence/bench-cpu.txt
./scripts/bench_gpu.sh           # GPU tokens/s + CPU-vs-GPU table -> evidence/bench-gpu.txt
./scripts/gpu_evidence.sh        # device request, nvidia-smi inside, CUDA init + offload, stats
./scripts/image-sizes.sh         # image sizes + layers   -> evidence/image-sizes.txt
./scripts/size-experiment.sh     # demonstrated size win  -> evidence/size-experiment.txt
./scripts/evidence.sh            # stack health, isolation, one end-to-end completion
./scripts/train_smoke.sh         # training smoke run -> evidence/train-smoke.txt  (on the GPU host)
./scripts/train_record.sh        # full training run  -> evidence/train-run.txt    (on the GPU host)
./scripts/bench_finetune.sh      # base vs adapter, 150 generations -> evidence/bench-finetune.json
python3 scripts/analyze_bench_finetune.py   # the tables the benchmark report quotes
```

Headline numbers (fixed prompt, 3 runs each): **8.88 tokens/s** median on CPU through the model's own
API inside the compose network and **8.49 tokens/s** end-to-end through the published gradio port;
on the GPU variant the same prompt runs at **134.40** (isolated) and **132.62** (through the GUI)
tokens/s. `inference` 2.12 GB CPU / 5.68 GB GPU, `gui` 461 MB, with the 1.28 GB model layer staying
`CACHED` when only a runtime setting changes. The GPU answer is *not* byte-identical to the CPU one,
and not even to its own previous one: at `temperature 0` the CPU stack emitted 38 completion tokens
on every run, while the GPU emitted 77 on all six committed runs after an earlier execution had
emitted 59 on all three of its runs. `scripts/bench_gpu.sh` therefore runs the measurement **twice**
and keeps both executions, and every comparison is made on effective tokens/s with the token counts
quoted beside it — see the caveats in `docs/EVIDENCE.md`.

## Status

Stage 5 (P5, *"Upload to Hugging Face spaces and use of online API inference"*) is **live**:

- model: [`ljubobratovicrelja/qwen3.5-2b-magicoder-lora`](https://huggingface.co/ljubobratovicrelja/qwen3.5-2b-magicoder-lora) — the merged, standalone text-only checkpoint (bf16, 3.76 GB), public
- demo: [`ljubobratovicrelja/qwen3.5-2b-magicoder-demo`](https://huggingface.co/spaces/ljubobratovicrelja/qwen3.5-2b-magicoder-demo) → https://ljubobratovicrelja-qwen3-5-2b-magicoder-demo.hf.space — a Gradio Space on ZeroGPU, serving the chat UI *and* its own API (`gradio_client`, `/gradio_api/call/generate`); `evidence/hf-space.txt` is a call to it from outside, with the runtime and hardware read back

Serving is the one part of this project that needs a paid tier: HF's free tier refuses compute Spaces on `cpu-basic` and ZeroGPU alike (402), so the Space runs on a one-month PRO subscription. Everything else here is free and public.

Stage 4 (P4, the optional "train a model for a few epochs" bullet) adds `training/` — a LoRA SFT of
the served base model, built and run **only** under the `train` profile and only on the GPU host, so
the two-container default stack is untouched — plus `scripts/train_lora.sh`, `scripts/plot_loss.py`,
`scripts/train_smoke.sh` and `evidence/train-smoke.txt`. The stage is implemented, smoke-proven and
**fully run**: 374 optimizer steps (2 epochs) in 1801 s on the RTX 3090 Ti, exit 0, no OOM, loss
0.4070 → 0.3359, adapter 62 561 688 B — [`evidence/train-run.txt`](evidence/train-run.txt),
[`evidence/loss-curve.png`](evidence/loss-curve.png) and
[`docs/TRAINING_REPORT.md`](docs/TRAINING_REPORT.md). The base-vs-adapter comparison has since been
measured — 150 generations, the adapter served against its own base at identical quantization — and
reads as a **negative result on correctness with a large style change**; the reading is in
[`docs/BENCHMARK_REPORT.md`](docs/BENCHMARK_REPORT.md). Still unmeasured: plan §6.1 R4's
`ast.parse`/pass@1 metrics.

Stage 3 (GPU variant and CPU-vs-GPU profiling) adds the `cpu`/`gpu` build targets over one shared
model stage, [`compose.gpu.yaml`](compose.gpu.yaml), `scripts/bench_gpu.sh`,
`scripts/gpu_evidence.sh` and the artifacts `evidence/gpu.txt` + `evidence/bench-gpu.txt`. The CPU
path is untouched and still the default.

Stage 2 (craft demonstration and measured CPU evidence) adds the evidence scripts and artifacts above
and [`docs/EVIDENCE.md`](docs/EVIDENCE.md); it changes no application behaviour.

Stage 1 (CPU two-container baseline) is implemented and running: llama-server serves a pinned
Qwen3.5-2B GGUF in one container, a gradio UI in a second, the two wired by `compose.yaml` with the
inference port reachable only inside the compose network. See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)
for the staged plan; work log and findings in [`JOURNAL.md`](JOURNAL.md).
