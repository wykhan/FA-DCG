#!/usr/bin/env bash
set -euo pipefail
python scripts/task_004c_fadcg_v2c_repro.py --output-root exp/task_004c_fadcg_v2c_repro/20260521_fadcg_v2c_seed2024 --models 'FCN + FA-DCG V2c' 'U-Net + FA-DCG V2c' --seeds 2024 --epochs 50 --batch-size 8 --num-workers 2
