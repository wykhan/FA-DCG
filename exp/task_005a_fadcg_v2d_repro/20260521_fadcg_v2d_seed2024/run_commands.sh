#!/usr/bin/env bash
set -euo pipefail
python scripts/task_005a_fadcg_v2d_repro.py --output-root exp/task_005a_fadcg_v2d_repro/20260521_fadcg_v2d_seed2024 --models 'FCN + FA-DCG V2d' 'U-Net + FA-DCG V2d' --seeds 2024 --epochs 50 --batch-size 8 --num-workers 2
