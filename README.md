# aiia-docker — Assignment 2 (Docker), MUIA 43506

> Part of the MUIA master's (Universidad de Alicante, 2026-27),
> course **43506 — Arquitecturas e Infraestructuras para IA** (resp. David Ortiz Pérez).


Deployment of a Deep Learning architecture for inference inside Docker containers, with the
graphical interface isolated from the inference in separate Compose services.


## What the assignment asks for

- one container running a pre-trained DL model for inference, with the inference call wrapped in
  a web server;
- a second container running *only* a very simple GUI (gradio) that requests the first;
- the two isolated via Docker Compose;
- a report, max 6 pages excluding cover/TOC, including the code and the commands used.

The official brief is distributed through Moodle and is not redistributed here.

What this repository adds on top: a GPU build target behind a Compose override file, a LoRA
fine-tune of the served model in `training/`, and the merged checkpoint published on Hugging Face
with a Gradio Space serving it.

## Layout

```
docs/            the two measurement reports
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
see [`evidence/gpu-3090ti/`](evidence/gpu-3090ti/).

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
[`evidence/`](evidence/) and are regenerated by the scripts in [`scripts/`](scripts/). The list below gives the
script that regenerates each artifact, and the report quotes them by path.

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
quoted beside it — see the caveats recorded beside the artifacts in `evidence/`.
