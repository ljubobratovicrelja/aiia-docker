# Evidence — where each graded criterion is proved

Every row below maps a criterion from the brief
([`docs/assignment/AIIA_DockerAssignment.md`](assignment/AIIA_DockerAssignment.md)) to the artifact
that demonstrates it and the exact command that regenerates that artifact. Nothing in the table is
asserted without an artifact behind it; the numbers are cited where they are relevant.

Host: DGX Spark, aarch64, Docker Engine 29.2.1, Compose v5.0.2, NVIDIA runtime default,
one GB10 GPU. Rows 1–10 and 14 are the CPU baseline (stage 2); row 12 is the GPU variant
(stage 3), which runs the *same* pinned model on the *same* host through a Compose override
file — so the two sets of numbers are directly comparable.

## 1. The criteria

| # | Criterion (brief wording) | Where it is met | Artifact that proves it | Command that regenerates it |
|---|---|---|---|---|
| 1 | *"a Docker container capable of running a Deep Learning architecture to perform inference on a pre-trained model"* | `inference/` — `ghcr.io/ggml-org/llama.cpp:server` serving a **pinned** Qwen3.5-2B Q4_K_M GGUF (repo `unsloth/Qwen3.5-2B-GGUF`, revision `f6d5376…`, sha256 verified at build) | `evidence/bench-cpu.txt` — fixed prompt, 3 runs, **8.88 tokens/s median** (isolated) and **8.49 tokens/s** (through the GUI); raw response bodies included | `./scripts/bench.sh` |
| 2 | *"Performing inference (locally and through GUI) within a Docker container"* | one path via the model's own HTTP API, one path browser → `gui` → `inference` | `evidence/bench-cpu.txt` (both paths, medians + per-run numbers), `evidence/completion.txt` | `./scripts/bench.sh`; `./scripts/evidence.sh` |
| 3 | *"A graphic interface must be used for doing the inference … isolated from the inference using different containers with Docker Compose"* | `gui/` (gradio) is the **only** published port; `inference` has `expose:` and no `ports:` | `evidence/compose-ps.txt` (`0.0.0.0:7860->7860/tcp` on gui, `8080/tcp` on inference) | `docker compose ps` |
| 4 | *"it will be necessary to encapsulate the inference function in a web server (flask, FastAPI)"* | the llama.cpp server image **already is** an OpenAI-compatible HTTP server, so the inference call is wrapped in a web server without extra code (plan D1) | raw `POST /v1/chat/completions` response in `evidence/bench-cpu.txt`; `evidence/compose-ps.txt` shows the inherited healthcheck passing | `docker compose exec -T inference curl -s http://localhost:8080/v1/models`; `docker compose exec -T inference curl -s http://localhost:8080/health` |
| 5 | Dockerfile quality — *"Many architectures … have public Dockerfiles available"* / *"leveraging existing images"* | `inference/Dockerfile`: `FROM ghcr.io/ggml-org/llama.cpp:server` (837 MB base, 24 layers) and only **two** own instructions — no hand-rolled rebuild | `evidence/image-sizes.txt` (layer breakdown: 706 MB of the base is one upstream apt layer) | `./scripts/image-sizes.sh` |
| 6 | Dockerfile quality — *"The order of definition of the Dockerfile should be taken into account to make proper use of the cache"* | model layer above the `ENV` runtime settings; `requirements.txt` above `COPY app.py` | `evidence/build-cache.txt` — **A**: nothing changed → every step `CACHED`; **B2**: app code edited → pip layer `CACHED`, only `COPY app.py` re-runs; **C**: `LLAMA_ARG_CTX_SIZE` changed → the 1.28 GB model layer `CACHED`, nothing re-downloaded | `./scripts/cache_demo.sh` |
| 7 | Dockerfile quality — *"optimize the space used by the images"* | one measured size decision: `pip install --no-cache-dir` kept, after building the alternative and measuring it | `evidence/size-experiment.txt` — **461.21 MB with the flag vs 545.29 MB without (+84.1 MB, +18.2 %)**; the pip layer grows 317 MB → 401 MB. The smaller variant is the committed one | `./scripts/size-experiment.sh` |
| 8 | *"Detail of the parameters used to run the container"* (GPU, RAM limit, shared memory, "etc.") | `compose.yaml`: `deploy.resources.limits.cpus/memory` (4 CPU / 4 GiB inference, 1 CPU / 512 MiB gui), `shm_size: 128m`, `read_only: true`, `tmpfs /tmp`, `user: 65534:65534`, `restart: unless-stopped`, `depends_on: service_healthy` | `evidence/compose-config.txt` (resolved spec), `evidence/stats.txt` (RSS against the limits, ~1.4–1.8 GiB), `evidence/up.log` (healthcheck gating the GUI start) | `docker compose config > evidence/compose-config.txt`; `docker stats --no-stream`; `./scripts/evidence.sh` |
| 9 | *"It is mandatory that there is a well-defined logical separation for the various containers. One will only be used for inference and the other will only serve as a graphical interface"* | two services, one job each; `gui` gets `API_URL=http://inference:8080/v1` and nothing else | `evidence/compose-config.txt`, `evidence/compose-ps.txt`, `evidence/portbindings.txt` | `docker compose config > evidence/compose-config.txt`; `docker compose ps`; `docker inspect -f '{{json .NetworkSettings.Ports}}' "$(docker compose ps -q inference)"` |
| 10 | *"Docker Compose structure to perform graphical inference using a local API"* | `depends_on: {inference: {condition: service_healthy}}`, inference reachable only on the compose network | `evidence/up.log` (gui starts only after inference is healthy), `evidence/portbindings.txt` (empty `PortBindings` = the isolation is real, not nominal) | `./scripts/evidence.sh` |
| 11 | Optional: *"you can train a model for a few epochs"* | planned: LoRA SFT of `Qwen/Qwen3.5-2B-Base` on Magicoder-OSS-Instruct-75K, off the serving path | **not implemented in this stage** — design and risks in [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §6.1 R1/R2/R4; `training/` is a placeholder | n/a (stage P4) |
| 12 | Max grade: *"NVIDIA container toolkit + training and inference on GPU"* | `inference/Dockerfile` now has **two build targets over one model download**: `cpu` (the default — it is the file's last stage) and `gpu` (`ghcr.io/ggml-org/llama.cpp:server-cuda13`), both `COPY --from` the shared `model` stage. `compose.gpu.yaml` is an **override file** (a profile can add a service, not re-parameterise one) setting `build.target: gpu`, image tag `stage-3-gpu`, `deploy.resources.reservations.devices: [{driver: nvidia, count: all, capabilities: [gpu]}]` and the offload env read off the image's own `--help`: `LLAMA_ARG_N_GPU_LAYERS=99`, `LLAMA_ARG_FLASH_ATTN=on` | `evidence/gpu.txt` — `nvidia-smi` **inside** the inference container (the container itself, via the toolkit), showing `/app/llama-server` holding **1589 MiB** on the GB10, the `HostConfig.DeviceRequests` the runtime actually received, and `load_tensors: offloaded 25/25 layers to GPU` with `CUDA0 model buffer size = 1211.05 MiB`. `evidence/bench-gpu.txt` — two consecutive executions of the same fixed prompt, **134.40 tokens/s on GPU vs 8.88 on CPU** (15.14× effective tokens/s; pooled 135.48 vs 8.74; the GUI path 132.62 vs 8.49, and the same comparison read 15.63×/15.08× in execution 1 and 15.12×/15.69× in execution 2) | `docker compose -f compose.yaml -f compose.gpu.yaml up -d --build`; `./scripts/gpu_evidence.sh`; `./scripts/bench_gpu.sh` |
| 13 | Max grade: *"Upload to Hugging Face spaces and use of online API inference"* | planned: Gradio Space on ZeroGPU serving the fine-tuned model | **not implemented in this stage**; the free route is confirmed in PROJECT_PLAN §6.1 R5 | n/a (stage P5) |
| 14 | Max grade: *"Any other improvements to the Dockerfile, Docker Compose YAML file or parameters to run the container"* | non-root + read-only rootfs + tmpfs (both services), model pinned by revision **and** sha256-verified at build, healthcheck-driven startup, inference port unpublished, resource limits, and the cache/size work itself | `evidence/build-cache.txt`, `evidence/image-sizes.txt`, `evidence/size-experiment.txt`, `evidence/portbindings.txt`, `evidence/stats.txt` | `./scripts/cache_demo.sh`; `./scripts/image-sizes.sh`; `./scripts/size-experiment.sh` |

Rows 11 and 13 are listed with their real status rather than omitted: they are later stages of
the same plan, and this stage claims nothing about them.

## 2. Numbers at a glance

Run-to-run variance on this shared host is a few percent (two consecutive executions of `bench.sh`
gave 9.08 / 8.88 tok/s isolated; the two GPU executions committed in `bench-gpu.txt` gave 138.77 and
133.93 tok/s isolated — the same few-percent band on either device); the numbers below are from the
**committed** `evidence/bench-cpu.txt` and `evidence/bench-gpu.txt`, which the table quotes, and the
artifacts are authoritative.

| Measurement | Value | Artifact |
|---|---|---|
| `inference` image | 2 117 416 629 B (2.12 GB) | `image-sizes.txt` |
| — its largest layer | 1.28 GB, the GGUF download (60.5 % of all layer bytes) | `image-sizes.txt` |
| `gui` image | 461 208 892 B (461 MB) | `image-sizes.txt` |
| — its largest layer | 317 MB, the `pip install` layer (68.6 %) | `image-sizes.txt` |
| base image | `ghcr.io/ggml-org/llama.cpp:server`, 836 580 789 B (837 MB) | `image-sizes.txt` |
| GUI image without `--no-cache-dir` | 545 290 931 B (545.29 MB) — **+84.1 MB** | `size-experiment.txt` |
| inference API, isolated (median of 3) | 4.280 s, 38 tokens, **8.88 tok/s** (pooled 8.74) | `bench-cpu.txt` |
| same prompt via the GUI (median of 3) | 8.839 s, 75 tokens, **8.49 tok/s** (pooled 8.64) | `bench-cpu.txt` |
| **same two paths on the GPU variant** (2 executions × 3 runs, pooled) | isolated 0.573 s, 77 tokens, **134.40 tok/s** (pooled 135.48); GUI 0.433 s, 57 tokens, **132.62 tok/s** (pooled 134.27) | `bench-gpu.txt` |
| **GPU vs CPU, same fixed prompt, isolated path** | **15.14×** the tokens/s — 134.40 vs 8.88 — i.e. **7.4 ms/token vs 112.6 ms/token**; pooled 15.50× | `bench-cpu.txt` + `bench-gpu.txt` |
| **the same comparison, execution by execution** | execution 1: 138.77 tok/s = **15.63×**; execution 2: 133.93 tok/s = **15.08×** (GUI: 15.12× then 15.69×) — the speedup holds to within ~4 % across executions | `bench-gpu.txt` |
| **the raw wall-clock ratio, shown only to be rejected** | 7.47× isolated / 20.39× through the GUI — inflated, because the GPU runs emitted more tokens; the rate is the honest number | `bench-gpu.txt` |
| layers offloaded to the device | **25/25 to GPU**, 0 left on CPU | `gpu.txt` |
| memory committed on the GPU, read from inside the container | **1589 MiB** held by `/app/llama-server`; llama.cpp's own accounting: 1211.05 MiB model + 48.00 KV + 77.06 recurrent + 56.02 compute = 1392 MiB on `CUDA0` | `gpu.txt` |
| `inference` GPU image | 5 684 548 299 B (5.68 GB) — the CUDA 13 base is 4.40 GB of it; the CPU image stayed byte-for-byte its stage-2 size (2 117 416 629 B) | `gpu.txt` |
| model downloads per stage-3 build of both targets | **one** — the shared `model` stage is `CACHED` for the second target | `gpu.txt` §1 |
| rebuild with no changes | every step `CACHED`, 0 instructions executed | `build-cache.txt` A |
| edit app code | pip layer `CACHED`, `COPY app.py` re-runs | `build-cache.txt` B2 |
| change `LLAMA_ARG_CTX_SIZE` | 1.28 GB model layer `CACHED` — no re-download | `build-cache.txt` C |
| gui → inference overhead | not separable from noise: both paths land in the same 8.0–9.1 tok/s band across runs, so any gradio hop cost sits below the run-to-run variance of a multi-second generation on a shared host | `bench-cpu.txt` |

### 2.1 The same stack on a second GPU

The identical committed branch, the identical `compose.yaml` + `compose.gpu.yaml` and the identical
pinned model (revision and sha256 unchanged) were also run on a second machine — **RTX 3090 Ti,
Ryzen 9 7950X3D, x86_64** — with nothing changed but the host. Each machine is compared against its
**own** CPU baseline, so neither comparison mixes hosts:

- DGX Spark / GB10 — CPU 8.88 → GPU **134.40 tok/s** isolated (**15.14×**); GUI 8.49 → 132.62.
- Second machine / RTX 3090 Ti — CPU 10.00 → GPU **320.66 tok/s** isolated (**32.07×**); GUI 9.79 → 310.00.
- The two CPU baselines are within ~13 % of each other, so on this workload the difference is the
  GPU: the 3090 Ti serves the 2B Q4 model at roughly **2.4×** the GB10's rate.
- The mechanism is identical on both hosts — `offloaded 25/25 layers to GPU`, `CUDA0 model buffer
  size = 1211.05 MiB` (~1392 MiB projected), device request
  `[{driver: nvidia, count: all, capabilities: [gpu]}]`. One instrumentation difference is worth
  knowing: the GB10 withholds VRAM counters (`N/A`, unified memory) where the 3090 Ti reports
  **2191 MiB / 24564 MiB** from inside the container.

Artifacts: `evidence/gpu-3090ti/` — a condensed `gpu.txt` (device-use proof only) plus
`bench-cpu.txt` and `bench-gpu.txt`, produced by the same scripts on that host.

## 3. Reproduce everything

```bash
git checkout stage-3-gpu
docker compose up -d --build        # CPU stack (the default build target)
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build   # GPU stack instead

./scripts/evidence.sh               # stage-1 set: up.log, compose-ps, completion, portbindings, stats
./scripts/cache_demo.sh             # layer-cache proof  -> evidence/build-cache.txt
./scripts/bench.sh                  # CPU throughput     -> evidence/bench-cpu.txt
./scripts/bench_gpu.sh              # GPU throughput + CPU-vs-GPU table -> evidence/bench-gpu.txt
./scripts/gpu_evidence.sh           # GPU proof set      -> evidence/gpu.txt
./scripts/image-sizes.sh            # sizes + layers     -> evidence/image-sizes.txt
./scripts/size-experiment.sh        # demonstrated size win -> evidence/size-experiment.txt
docker compose config > evidence/compose-config.txt
```

`cache_demo.sh` makes temporary edits to `gui/app.py` and `inference/Dockerfile`; it restores both
from a backup on exit (via `trap`, so it also restores after a failure) and its last act is to
compare `git status --short` against the snapshot taken before the run. `bench.sh` and
`image-sizes.sh` are read-only apart from their own artifact.

## 4. Caveats a reader should know

- **The two benchmark paths sample differently.** The isolated call pins `temperature 0` and
  `max_tokens 160` for comparability across runs; `gui/app.py` uses its own `temperature 0.7` /
  `max_tokens 512`, so GUI runs generate a different number of tokens. That is why *effective
  tokens/s* is the comparable metric, and why the elapsed delta is not a pure transport overhead.
- **`touch` does not invalidate a `COPY` layer on this engine.** BuildKit keys `COPY` on the content
  digest, not the mtime, so `touch gui/app.py` rebuilds nothing (`build-cache.txt` B1). The cache
  property the Dockerfile was ordered for is shown with a real content edit (B2), which is what
  actually happens when the GUI code changes.
- **The cache demo re-runs safely because its marker is unique.** `cache_demo.sh` appends a marker
  line carrying a timestamp and pid to `gui/app.py` for section B2. A fixed marker string would
  itself be a cache hit on the second run, so the app layer would report `CACHED` and the
  demonstration would quietly stop demonstrating; this was found by re-running it, and the script now
  generates the marker fresh each time.
- **The model is a thinking model.** Every request in this repository sends
  `chat_template_kwargs: {enable_thinking: false}` plus a `max_tokens` cap. Without it the reply is
  written to `reasoning_content` for thousands of tokens and `content` stays empty.
- **`docker images` on Engine 29 prints `DISK USAGE` / `CONTENT SIZE`, and the `CONTENT SIZE` column
  reads `0B` here.** The per-image byte counts quoted above come from
  `docker image inspect --format '{{.Size}}'`, not from that column.
- Single-digit tokens/s on CPU is expected: this is a shared CPU host serving a 2B Q4 model
  without a GPU.
- **The two backends do not produce identical text at `temperature 0` — and the GPU is not even
  reproducible against itself.** On the isolated path the CPU stack emitted 38 completion tokens on
  every run of every execution; the GPU emitted **77** on all six committed runs (two executions),
  but earlier executions of the *same* script produced **59** on all three of their runs — same
  prompt, same request body, same server, same image. It is stable within an execution and not
  across them. Greedy decoding is only "greedy" if the arithmetic is identical, and the CPU and CUDA
  kernels (with flash-attention on the GPU side) are not bit-for-bit equivalent, so a near-tie flips
  and the trajectory diverges from there. The same effect appears between two *CPU* backends: the
  x86 baseline in §2.1 emitted 59 tokens where this host's ARM CPU emitted 38 — so the
  divergence is a property of the arithmetic, not of the GPU.
  **Consequence: the completion-token count is part of the measurement, not a constant.** The
  comparison is made on *effective tokens/s* with the token count quoted beside every number, and
  the artifact keeps *two* executions so the spread is visible rather than averaged away. What is
  stable is the rate: 15.63× and 15.08× on the isolated path in the two committed executions, 15.12×
  and 15.69× through the GUI. Raw wall-clock is never the headline — for the committed pair it would
  have read 7.47× (isolated) and 20.39× (GUI), inflated because the GPU run did more work per request.
  "Same prompt, same answer on both backends" is *not* a property this stack has; "same prompt,
  roughly the same rate" is.
- **On the GB10 the driver exposes no separate VRAM counter.** `nvidia-smi`'s memory columns read
  `Not Supported` / `[N/A]` because the GPU shares the 128 GB unified pool with the CPU. The
  per-process column still works, and that is where the 1589 MiB for `llama-server` comes from;
  llama.cpp's own `CUDA0 ... buffer size` accounting gives the breakdown.
- **The CUDA image's server logs at debug verbosity by default.** One startup plus twelve benchmark
  requests produced 60 014 log lines, so `evidence/gpu.txt` greps the log instead of dumping it —
  and greps the *current server process* only, cut at its last `initializing ...` line, so a restart
  cannot mix two boots' numbers into the block that is quoted. It also loads the model **twice** at
  startup — a `no_alloc` probe that `--fit` uses to size the request (reporting `0.00 MiB` buffers
  while still claiming `offloaded 25/25 layers to GPU`) and then the real load. The numbers quoted
  are the second one: 1211.05 MiB in `CUDA0`, and it is the first one a careless grep would catch.
- **The CPU image's size did not move.** Splitting the download into a shared stage and copying it
  into each target added one layer that carries no bytes: 2 117 416 629 B before and after.
