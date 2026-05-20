#!/usr/bin/env bash
set -euo pipefail
OUT=exp/task_002_baseline_repro/20260520_104300

python scripts/task_002_baseline_repro.py \
  --output-root "$OUT" \
  --models FCN U-Net \
  --seeds 42 123 2024 \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 2

python scripts/task_002_baseline_repro.py \
  --output-root "$OUT" \
  --skip-smoke \
  --resume \
  --models 'FCN + FA-DCG' 'U-Net + FA-DCG' \
  --seeds 2024 \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 2
