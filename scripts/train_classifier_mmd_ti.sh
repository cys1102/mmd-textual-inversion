#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-CompVis/stable-diffusion-v1-4}"
DATASET="${DATASET:-pets}"
NUM_TRIAL="${NUM_TRIAL:-3}"
DIST_MATCH="${DIST_MATCH:-0.005}"
EXAMPLES_PER_CLASS="${EXAMPLES_PER_CLASS:-8}"
NUM_SYNTHETIC="${NUM_SYNTHETIC:-5}"
SYNTHETIC_PROBABILITY="${SYNTHETIC_PROBABILITY:-0.5}"

python train_classifier.py \
  --logdir "${DATASET}-baselines/textual-inversion-mmd${DIST_MATCH}" \
  --synthetic-dir "aug/textual-inversion-mmd${DIST_MATCH}/{dataset}-{seed}-{examples_per_class}" \
  --model-path "${MODEL_NAME}" \
  --dataset "${DATASET}" \
  --prompt "a photo of a {name}" \
  --aug textual-inversion \
  --guidance-scale 7.5 \
  --strength 0.5 \
  --mask 0 \
  --inverted 0 \
  --probs 1 \
  --compose parallel \
  --num-synthetic "${NUM_SYNTHETIC}" \
  --synthetic-probability "${SYNTHETIC_PROBABILITY}" \
  --num-trials "${NUM_TRIAL}" \
  --examples-per-class ${EXAMPLES_PER_CLASS} \
  --dist_match "${DIST_MATCH}"
