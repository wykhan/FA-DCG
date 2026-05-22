#!/usr/bin/env bash
set -euo pipefail
python scripts/task_002c_fadc_optimized_repro.py --output-root exp/task_002c_fadc_optimized_multiseed/20260522_v1_1_seeds42_123 --models 'FCN + Optimized FA-DCG' 'U-Net + Optimized FA-DCG' --seeds 42 123 --epochs 50 --batch-size 8 --num-workers 2
