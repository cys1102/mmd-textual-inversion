#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-CompVis/stable-diffusion-v1-4}"
DATASET="${DATASET:-pets}"
NUM_TRIAL="${NUM_TRIAL:-3}"
DIST_MATCH="${DIST_MATCH:-0.005}"
EXAMPLES_PER_CLASS="${EXAMPLES_PER_CLASS:-4 8 16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-125}"
LEARNING_RATE="${LEARNING_RATE:-2e-3}"
SNR_GAMMA="${SNR_GAMMA:-5}"

accelerate launch ti_mmd.py \
  --pretrained_model_name_or_path "${MODEL_NAME}" \
  --dataset "${DATASET}" \
  --examples-per-class ${EXAMPLES_PER_CLASS} \
  --num-trials "${NUM_TRIAL}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --initializer_token "${DATASET}" \
  --dist_match "${DIST_MATCH}" \
  --snr_gamma "${SNR_GAMMA}"
