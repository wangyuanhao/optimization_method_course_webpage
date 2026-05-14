#!/usr/bin/env bash
set -euo pipefail

# Change these for a quick test.
MAX_PASSES=${MAX_PASSES:-10}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_INTERVAL=${EVAL_INTERVAL:-0.25}
OUTPUT_DIR=${OUTPUT_DIR:-results}

DATASETS=(mnist cifar10)
ALGORITHMS=(sgd spider sarah storm page)

for dataset in "${DATASETS[@]}"; do
  for alg in "${ALGORITHMS[@]}"; do
    python -m src.train \
      --dataset "$dataset" \
      --algorithm "$alg" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --max-passes "$MAX_PASSES" \
      --batch-size "$BATCH_SIZE" \
      --eval-interval-passes "$EVAL_INTERVAL" \
      --output-dir "$OUTPUT_DIR"
  done
done

python -m src.plot --results-dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/figure13_level1_reproduction.pdf"
