#!/usr/bin/env bash
set -euo pipefail
python scripts/task_004a_fadcg_v2a_repro.py --output-root exp/task_004a_fadcg_v2a_repro/20260521_fadcg_v2a_seed2024 --models 'FCN + FA-DCG V2a' 'U-Net + FA-DCG V2a' --seeds 2024 --epochs 50 --batch-size 8 --num-workers 2
