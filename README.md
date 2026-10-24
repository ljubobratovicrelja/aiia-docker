# aiia-docker

Docker Compose deployment of a quantized LLM behind a Gradio interface. `llama-server` (llama.cpp)
serves `Qwen3.5-2B` (Q4_K_M) in one container; a Gradio app in a second container is the only thing exposed
to the host. A GPU build target, a LoRA fine-tune of the served model and a published Hugging Face
Space sit alongside it.

Assignment 2 of **43506 — Arquitecturas e Infraestructuras para IA**, MUIA, Universidad de Alicante
(2026-27).

## Run

```bash
docker compose up -d --build
```

The interface is at <http://localhost:7860>, the only published port. The model API is reachable at
`http://inference:8080/v1` from inside the Compose network only.

```bash
./scripts/evidence.sh    # health check, one end-to-end completion, artifacts
docker compose down
```

## GPU

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
```

`compose.gpu.yaml` is an override file: it switches `inference` to the `gpu` target (CUDA 13), adds
the device reservation and the offload settings. Both targets build from one shared model stage, so
switching does not re-download the 1.28 GB GGUF.

The fixed prompt runs at 8.9 tokens/s on the GB10's CPU and 9.5 on the 3090 Ti's; on the GPU, 134.4
(GB10) and 281.5 (RTX 3090 Ti), median over 3 runs per route — 6 on the GPU. Logs and full tables in
[`evidence/`](evidence/); the 3090 Ti run is [`evidence/mint-3090ti-2026-10-24/`](evidence/mint-3090ti-2026-10-24/),
and `README.txt` there maps each artifact to the number it supports. Images are 2.13 GB (cpu target),
5.37 GB (gpu target) and 442 MB (gui) of files — *not* the 3.75 / 9.02 / 0.64 GB that
`docker image inspect .Size` reports on that engine; `image-sizes-reconciliation.txt` shows why.

## Training

`training/` fine-tunes the served base model by LoRA on `ise-uiuc/Magicoder-OSS-Instruct-75K`, as a
Compose service behind the `train` profile.

```bash
export TRAIN_DOCKER_HOME=/var/tmp/train-home   # where $HOME is not writable for the Docker CLI
./scripts/train_lora.sh up      # build if needed, start detached
./scripts/train_lora.sh tail    # follow the loss log
./scripts/train_lora.sh down
```

A smoke run is the same command with `MAX_STEPS=10 TRAIN_SIZE=24 SCAN_LIMIT=200`.

The full run is 374 optimizer steps (2 epochs, about 30 minutes on the 3090 Ti), loss 0.4070 to
0.3359. Reading: [`docs/TRAINING_REPORT.md`](docs/TRAINING_REPORT.md); the base-vs-adapter
comparison: [`docs/BENCHMARK_REPORT.md`](docs/BENCHMARK_REPORT.md).

## Model and demo

- merged checkpoint: [`ljubobratovicrelja/qwen3.5-2b-magicoder-lora`](https://huggingface.co/ljubobratovicrelja/qwen3.5-2b-magicoder-lora)
- Space: [`ljubobratovicrelja/qwen3.5-2b-magicoder-demo`](https://huggingface.co/spaces/ljubobratovicrelja/qwen3.5-2b-magicoder-demo)

## Layout

```
inference/   llama-server image, cpu and gpu build targets
gui/         Gradio front end
training/    LoRA fine-tune, behind a Compose profile
space/       the Gradio app published as a Hugging Face Space
scripts/     benchmarks, evidence collection, training launcher
evidence/    logs and measurements
docs/        training and benchmark reports
bench/       prompt suite and perplexity corpora
```

## Evidence

Every artifact in `evidence/` is produced by a script in `scripts/`: `bench.sh` and `bench_gpu.sh`
(throughput), `cache_demo.sh` (layer cache), `image-sizes.sh` and `size-experiment.sh` (image size),
`gpu_evidence.sh` (device and offload), `train_smoke.sh` and `train_record.sh` (training),
`bench_finetune.sh` and `analyze_bench_finetune.py` (base vs adapter).

Two host-specific directories hold re-measurements taken on the machine the report describes, and say so
in their own `README.txt`: `evidence/gpu-3090ti/` (the first 3090 Ti run) and
`evidence/mint-3090ti-2026-10-24/` (the re-measurement, including four artifacts that have no script
because they are one-off observations — the host's own `nvidia-smi`/`docker version` output, RSS against
the resolved cgroup limits, device memory sampled during generation, and the resolved port bindings). Each
of those four prints the pipeline that produced it in its header.

