#!/usr/bin/env bash
set -euo pipefail
python scripts/task_004b_fadcg_v2b_repro.py --output-root exp/task_004b_fadcg_v2b_repro/20260521_fadcg_v2b_seed2024 --models 'FCN + FA-DCG V2b' 'U-Net + FA-DCG V2b' --seeds 2024 --epochs 50 --batch-size 8 --num-workers 2
