# syntax=docker/dockerfile:1
#
# Stage 1 — inference container, two build targets over one model download.
#
# There is ONE model stage and it is shared: the 1.28 GB pinned GGUF is
# downloaded and sha256-verified exactly once, in `model`, and both targets
# `COPY --from=model`.  Building the second target therefore finds the model
# stage in the layer cache and pays nothing for it (build logs in
# evidence/gpu.txt).
#
#   target cpu  — ghcr.io/ggml-org/llama.cpp:server          (DEFAULT build)
#   target gpu  — ghcr.io/ggml-org/llama.cpp:server-cuda13   (opt-in)
#
# **The CPU stage is deliberately the LAST stage in this file.**  A build with
# no `--target` builds the final stage, so "cpu is the default" is a property of
# the file's stage order and not of a comment: plain
# `docker compose up -d --build` (compose.yaml alone, stages 1-2 behaviour) and
# a bare `docker build .` both produce the CPU image, and only
# `docker compose -f compose.yaml -f compose.gpu.yaml up -d` asks for `gpu`.
# Getting this backwards is a silent failure — the CPU service quietly runs the
# 4.4 GB CUDA image — so the order is load-bearing.
#
# Layer order inside both targets is deliberate: everything cheap and stable
# sits at the top, the inherited image below it, and only the settings that
# change often (the LLAMA_ARG_* runtime config) sit at the bottom — so a config
# tweak never triggers a re-download of the model.
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Shared stage: fetch the pinned model.  Nothing else happens here, so this one
# download layer is what both targets below reuse.
# ---------------------------------------------------------------------
FROM ghcr.io/ggml-org/llama.cpp:server AS model

# --- Pinned model identity (build args) ------------------------------
ARG MODEL_REPO=unsloth/Qwen3.5-2B-GGUF
ARG MODEL_REVISION=f6d5376be1edb4d416d56da11e5397a961aca8ae
ARG MODEL_FILE=Qwen3.5-2B-Q4_K_M.gguf
ARG MODEL_SHA256=aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223

# Download once, verify the digest, keep nothing else in the layer.
# The CPU server image is deliberately the fetch stage for BOTH targets: it
# ships `curl` and `sha256sum`, and using one identical stage is what lets
# BuildKit serve the same cached layer to the cpu and the gpu build.
# The model is addressed by absolute path on purpose: the base image's
# llama-server resolves its own libllama-server-impl.so relative to the working
# directory, which is why the inherited WORKDIR (/app) must not be changed.
# Verified: running the image's binary with -w /models fails with
# "error while loading shared libraries: libllama-server-impl.so".
RUN mkdir -p /models \
 && curl -fL --retry 3 -o "/models/${MODEL_FILE}" \
        "https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/${MODEL_FILE}" \
 && echo "${MODEL_SHA256}  /models/${MODEL_FILE}" | sha256sum -c -

# ---------------------------------------------------------------------
# Target: GPU.  The CUDA 13 SBSA build of the same server, over the same model
# layer.  The only differences from the CPU target are the base image and the
# offload settings; the model, the port, the alias and the inherited healthcheck
# are identical, so nothing that goes through the GUI can tell which target is
# running.
# ---------------------------------------------------------------------
FROM ghcr.io/ggml-org/llama.cpp:server-cuda13 AS gpu

COPY --from=model /models /models

# GPU offload settings — names taken from the image's own `--help`
# (`docker run --rm ghcr.io/ggml-org/llama.cpp:server-cuda13 --help`):
#   LLAMA_ARG_N_GPU_LAYERS  (-ngl / --gpu-layers), 99 = all layers
#   LLAMA_ARG_FLASH_ATTN    (-fa / --flash-attn)
ENV LLAMA_ARG_MODEL=/models/Qwen3.5-2B-Q4_K_M.gguf \
    LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080 \
    LLAMA_ARG_CTX_SIZE=4096 \
    LLAMA_ARG_ALIAS=aiia-local \
    LLAMA_ARG_N_GPU_LAYERS=99 \
    LLAMA_ARG_FLASH_ATTN=on

# ---------------------------------------------------------------------
# Target: CPU — and the LAST stage, so it is what a target-less build gets.
# Identical to the stage-1/2 image except that the model now arrives as a
# `COPY --from` of the shared stage instead of an inline download.
# ---------------------------------------------------------------------
FROM ghcr.io/ggml-org/llama.cpp:server AS cpu

COPY --from=model /models /models

# --- Runtime configuration (changes most often, stays at the bottom) --
ENV LLAMA_ARG_MODEL=/models/Qwen3.5-2B-Q4_K_M.gguf \
    LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080 \
    LLAMA_ARG_CTX_SIZE=4096 \
    LLAMA_ARG_ALIAS=aiia-local

# The server image already ships HEALTHCHECK curl -f http://localhost:8080/health
# and ENTRYPOINT ["/app/llama-server"], so neither is restated here.