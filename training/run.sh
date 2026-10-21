#!/usr/bin/env bash
# Entrypoint of the training image.
#
# It exists so that the run configuration lives in exactly ONE place
# (the argparse defaults in train_lora.py, which are the committed
# hyperparameters) while a smoke run can still shrink the schedule without
# rebuilding the image or editing the Compose file.  Only the variables below
# are looked at; anything unset leaves the committed default in place.
#
# Every test is an `if`, never `[ ... ] && ...`: the latter returns non-zero
# when the variable is unset, which under `set -e` ends the container before it
# starts — a failure that looks like a crash in the trainer.
set -euo pipefail

ARGS=(
  --output-dir "${OUT_DIR:-/out/adapter}"
  --loss-log   "${LOSS_LOG:-/out/loss.jsonl}"
)

add_if_set() {                # add_if_set <ENV_NAME> <FLAG>
  local value="${!1:-}"
  if [ -n "$value" ]; then
    ARGS+=("$2" "$value")
  fi
}

add_if_set MAX_STEPS    --max-steps
add_if_set TRAIN_SIZE   --train-size
add_if_set SCAN_LIMIT   --scan-limit
add_if_set EPOCHS       --epochs
add_if_set BATCH_SIZE   --batch-size
add_if_set GRAD_ACCUM   --grad-accum
add_if_set MAX_SEQ_LEN  --max-seq-len
add_if_set LR           --lr
add_if_set SEED         --seed
add_if_set TARGET_MODULES --target-modules

echo "# run.sh: python3 /app/train_lora.py ${ARGS[*]} $*"
exec python3 /app/train_lora.py "${ARGS[@]}" "$@"
