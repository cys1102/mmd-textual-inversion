#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-pets}"
NUM_TRIAL="${NUM_TRIAL:-3}"
DIST_MATCH="${DIST_MATCH:-0.005}"
EXAMPLES_PER_CLASS="${EXAMPLES_PER_CLASS:-4 8 16}"

python aggregate_embeddings.py \
  --dataset "${DATASET}" \
  --num-trials "${NUM_TRIAL}" \
  --examples-per-class ${EXAMPLES_PER_CLASS} \
  --dist_match "${DIST_MATCH}"
