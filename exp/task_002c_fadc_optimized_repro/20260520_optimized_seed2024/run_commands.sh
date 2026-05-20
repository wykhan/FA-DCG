#!/usr/bin/env bash
set -euo pipefail
python scripts/task_002c_fadc_optimized_repro.py --output-root exp/task_002c_fadc_optimized_repro/20260520_optimized_seed2024 --models 'FCN + Optimized FA-DCG' 'U-Net + Optimized FA-DCG' --seeds 2024 --epochs 50 --batch-size 8 --num-workers 2
