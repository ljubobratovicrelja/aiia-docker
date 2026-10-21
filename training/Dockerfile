# syntax=docker/dockerfile:1
#
# Stage-4 (P4) image: LoRA SFT of the served Qwen3.5-2B base.
#
# This image is NOT on the serving path.  Nothing in the inference/gui stack
# references it; it is built and run only under the `train` Compose profile, on
# a machine with a GPU.  Keeping the trainer out of the served images is the
# point of the arrangement: the serving image stays 2.12 GB (CPU) / 5.68 GB
# (GPU) and does not grow a torch dependency to satisfy an optional bullet.
#
# ---------------------------------------------------------------------
# Base image: `nvcr.io/nvidia/pytorch:26.09-py3`, as the project plan decided
# (D9/R3).  Multi-arch (amd64 + arm64), CUDA 13, torch preinstalled.
#
# Rejected leaner alternative: `nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04`
# (~2.1 GB) + `pip install torch`.  It is smaller and would work, but the
# plan's whole reason for the NGC image was to stop fighting CUDA-torch
# availability per architecture (R3) — building that fight back in to save disk
# on a host with 285 GB free buys nothing, and it puts a `pip install` of a
# ~1 GB CUDA wheel in the critical path of every rebuild.
#
# The cost is real and is recorded rather than hidden: ~25 GB unpacked, a long
# first pull.  It is a one-time cost per host and the weights themselves are
# mounted, not baked (see below).
# ---------------------------------------------------------------------
FROM nvcr.io/nvidia/pytorch:26.09-py3

# USE_HUB_KERNELS=0 — transformers >= 5.16 will fetch optional "kernel" packages
# from the HF hub for layers like Qwen3_5GatedDeltaNet when a GPU-specific
# mapping exists.  The GDN mapping in the pinned transformers version is
# registered for SM121 (GB10) and gfx1151 only, so on the 3090 Ti used here it
# would resolve to nothing — but on any host it adds a network dependency and a
# trust boundary to a training run whose whole point is reproducibility.  Pinned
# off: the pure-PyTorch reference path of the same layer runs instead.
#
# HF_HOME points at a mounted host directory (see compose.yaml).  The 4.5 GB
# base checkpoint and the 194 MiB corpus are NOT baked into the image: the smoke
# run's download has to be reused by the full run, and a rebuild must never
# re-download them.
ENV USE_HUB_KERNELS=0 \
    HF_HOME=/hf-cache \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Cheap and stable at the top, app code at the bottom: the pip layer only
# changes when requirements.txt does, so editing train_lora.py costs a rebuild
# of seconds, not a re-resolve of the whole environment.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY train_lora.py /app/train_lora.py
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

# run.sh turns the handful of environment knobs a smoke run needs into flags and
# leaves every hyperparameter at the committed default inside train_lora.py —
# one place holds the run configuration, so the smoke and the full run cannot
# drift apart.
ENTRYPOINT ["/app/run.sh"]
