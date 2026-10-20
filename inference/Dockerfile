# syntax=docker/dockerfile:1
#
# Stage 1 — inference container.
# Serves a pinned GGUF with llama-server's OpenAI-compatible HTTP API.
#
# Layer order is deliberate: everything cheap and stable sits at the top, the
# multi-GB model layer sits next, and only the settings that change often (the
# LLAMA_ARG_* runtime config) sit below it — so a config tweak never triggers a
# re-download of the model.
# ---------------------------------------------------------------------

FROM ghcr.io/ggml-org/llama.cpp:server

# --- Pinned model identity (build args) ------------------------------
ARG MODEL_REPO=unsloth/Qwen3.5-2B-GGUF
ARG MODEL_REVISION=f6d5376be1edb4d416d56da11e5397a961aca8ae
ARG MODEL_FILE=Qwen3.5-2B-Q4_K_M.gguf
ARG MODEL_SHA256=aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223

# --- Model layer (the heavy one) -------------------------------------
# Download once, verify the digest, keep nothing else in the layer.
# The model is addressed by absolute path on purpose: the base image's
# llama-server resolves its own libllama-server-impl.so relative to the working
# directory, which is why the inherited WORKDIR (/app) must not be changed.
# Verified: running the image's binary with -w /models fails with
# "error while loading shared libraries: libllama-server-impl.so".
RUN mkdir -p /models \
 && curl -fL --retry 3 -o "/models/${MODEL_FILE}" \
        "https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/${MODEL_FILE}" \
 && echo "${MODEL_SHA256}  /models/${MODEL_FILE}" | sha256sum -c -

# --- Runtime configuration (changes most often, stays at the bottom) --
ENV LLAMA_ARG_MODEL=/models/${MODEL_FILE} \
    LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080 \
    LLAMA_ARG_CTX_SIZE=4096 \
    LLAMA_ARG_ALIAS=aiia-local

# The base image already ships HEALTHCHECK curl -f http://localhost:8080/health
# and ENTRYPOINT ["/app/llama-server"], so neither is restated here.
