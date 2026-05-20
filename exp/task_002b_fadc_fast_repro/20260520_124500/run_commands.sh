#!/usr/bin/env bash
set -euo pipefail
OUT=exp/task_002b_fadc_fast_repro/20260520_124500
python scripts/task_002b_fadc_fast_repro.py --output-root "$OUT" --models 'U-Net + Fast FA-DCG' 'FCN + Fast FA-DCG' --seeds 2024 --epochs 50 --batch-size 8 --num-workers 2
python scripts/task_002b_fadc_fast_repro.py --output-root "$OUT" --skip-benchmark --resume --models 'U-Net + Fast FA-DCG' 'FCN + Fast FA-DCG' --seeds 42 123 --epochs 50 --batch-size 8 --num-workers 2
