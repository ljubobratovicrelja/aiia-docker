# syntax=docker/dockerfile:1
#
# Stage 1 — inference container.
# Serves a pinned GGUF with llama-server's OpenAI-compatible HTTP API.
#
# Layer order is deliberate: everything cheap and stable sits at the top,
# the multi-GB model layer sits next, and only settings that change often
# (the LLAMA_ARG_* runtime config) sit below it — so a config tweak never
# re-downloads the model.
# ---------------------------------------------------------------------

FROM ghcr.io/ggml-org/llama.cpp:server

# --- Pinned model identity (build args) ------------------------------
ARG MODEL_REPO=unsloth/Qwen3.5-2B-GGUF
ARG MODEL_REVISION=f6d5376be1edb4d416d56da11e5397a961aca8ae
ARG MODEL_FILE=Qwen3.5-2B-Q4_K_M.gguf
ARG MODEL_SHA256=aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223

# --- Model layer (the heavy one) -------------------------------------
# Download once, verify the digest, and keep only the model in the layer.
WORKDIR /models
RUN curl -fL --retry 3 -o "${MODEL_FILE}" \
        "https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/${MODEL_FILE}" \
 && echo "${MODEL_SHA256}  ${MODEL_FILE}" | sha256sum -c -

# --- Runtime configuration (changes most often, stays at the bottom) --
# LD_LIBRARY_PATH is part of this block on purpose: the base image's
# llama-server only resolves its own libllama-server-impl.so while the CWD is
# /app, and this Dockerfile sets WORKDIR /models. Without this it dies with
# "error while loading shared libraries". Verified fix.
ENV LD_LIBRARY_PATH=/app \
    LLAMA_ARG_MODEL=/models/${MODEL_FILE} \
    LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080 \
    LLAMA_ARG_CTX_SIZE=4096 \
    LLAMA_ARG_ALIAS=aiia-local

# The base image already ships HEALTHCHECK curl -f http://localhost:8080/health
# and ENTRYPOINT ["/app/llama-server"], so neither is restated here.
