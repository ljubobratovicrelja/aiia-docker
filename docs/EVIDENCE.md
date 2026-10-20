# Evidence — where each graded criterion is proved

Every row below maps a criterion from the brief
([`docs/assignment/AIIA_DockerAssignment.md`](assignment/AIIA_DockerAssignment.md)) to the artifact
that demonstrates it and the exact command that regenerates that artifact. Nothing in the table is
asserted without an artifact behind it; the numbers are cited where they are relevant.

Host: DGX Spark, aarch64, Docker Engine 29.2.1, Compose v5.0.2, NVIDIA runtime default.
Everything here is CPU-only — GPU is a later stage (see the last three rows).

## 1. The criteria

| # | Criterion (brief wording) | Where it is met | Artifact that proves it | Command that regenerates it |
|---|---|---|---|---|
| 1 | *"a Docker container capable of running a Deep Learning architecture to perform inference on a pre-trained model"* | `inference/` — `ghcr.io/ggml-org/llama.cpp:server` serving a **pinned** Qwen3.5-2B Q4_K_M GGUF (repo `unsloth/Qwen3.5-2B-GGUF`, revision `f6d5376…`, sha256 verified at build) | `evidence/bench-cpu.txt` — fixed prompt, 3 runs, **9.08 tokens/s median** (isolated) and **8.84 tokens/s** (through the GUI); raw response bodies included | `./scripts/bench.sh` |
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
| 12 | Max grade: *"NVIDIA container toolkit + training and inference on GPU"* | planned: `server-cuda13` build target + `deploy.resources.reservations.devices` under a Compose profile | **not implemented in this stage** — this stage's numbers are deliberately CPU-only, and the host's NVIDIA runtime is already the default (PROJECT_PLAN Appendix A) | n/a (stage P3) |
| 13 | Max grade: *"Upload to Hugging Face spaces and use of online API inference"* | planned: Gradio Space on ZeroGPU serving the fine-tuned model | **not implemented in this stage**; the free route is confirmed in PROJECT_PLAN §6.1 R5 | n/a (stage P5) |
| 14 | Max grade: *"Any other improvements to the Dockerfile, Docker Compose YAML file or parameters to run the container"* | non-root + read-only rootfs + tmpfs (both services), model pinned by revision **and** sha256-verified at build, healthcheck-driven startup, inference port unpublished, resource limits, and the cache/size work itself | `evidence/build-cache.txt`, `evidence/image-sizes.txt`, `evidence/size-experiment.txt`, `evidence/portbindings.txt`, `evidence/stats.txt` | `./scripts/cache_demo.sh`; `./scripts/image-sizes.sh`; `./scripts/size-experiment.sh` |

Rows 11–13 are listed with their real status rather than omitted: they are later stages of the same
plan, and this stage claims nothing about them.

## 2. Numbers at a glance

| Measurement | Value | Artifact |
|---|---|---|
| `inference` image | 2 117 416 629 B (2.12 GB) | `image-sizes.txt` |
| — its largest layer | 1.28 GB, the GGUF download (60.5 % of all layer bytes) | `image-sizes.txt` |
| `gui` image | 461 208 892 B (461 MB) | `image-sizes.txt` |
| — its largest layer | 317 MB, the `pip install` layer (68.6 %) | `image-sizes.txt` |
| base image | `ghcr.io/ggml-org/llama.cpp:server`, 836 580 789 B (837 MB) | `image-sizes.txt` |
| GUI image without `--no-cache-dir` | 545 290 931 B (545.29 MB) — **+84.1 MB** | `size-experiment.txt` |
| inference API, isolated (median of 3) | 4.183 s, 38 tokens, **9.08 tok/s** (pooled 9.05) | `bench-cpu.txt` |
| same prompt via the GUI (median of 3) | 7.467 s, 67 tokens, **8.84 tok/s** (pooled 8.82) | `bench-cpu.txt` |
| rebuild with no changes | every step `CACHED`, 0 instructions executed | `build-cache.txt` A |
| edit app code | pip layer `CACHED`, `COPY app.py` re-runs | `build-cache.txt` B2 |
| change `LLAMA_ARG_CTX_SIZE` | 1.28 GB model layer `CACHED` — no re-download | `build-cache.txt` C |
| gui → inference overhead | ≈ 0.1 s (a GUI run of the same token count lands within ~0.1 s of the API run) | `bench-cpu.txt` |

## 3. Reproduce everything

```bash
git checkout stage-2-evidence
docker compose up -d --build        # stack up; inference healthy, gui published on 7860

./scripts/evidence.sh               # stage-1 set: up.log, compose-ps, completion, portbindings, stats
./scripts/cache_demo.sh             # layer-cache proof  -> evidence/build-cache.txt
./scripts/bench.sh                  # CPU throughput     -> evidence/bench-cpu.txt
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
- Single-digit tokens/s is expected: this is a shared CPU host serving a 2B Q4 model without a GPU.
