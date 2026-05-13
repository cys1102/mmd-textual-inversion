#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-pets}"
NUM_TRIAL="${NUM_TRIAL:-3}"
EXAMPLES_PER_CLASS="${EXAMPLES_PER_CLASS:-8}"

python train_classifier.py \
  --logdir "${DATASET}-baselines/real-only" \
  --dataset "${DATASET}" \
  --num-trials "${NUM_TRIAL}" \
  --examples-per-class ${EXAMPLES_PER_CLASS}
