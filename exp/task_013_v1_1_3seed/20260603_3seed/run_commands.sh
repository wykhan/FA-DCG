#!/usr/bin/env bash
set -euo pipefail
python scripts/task_013_v1_1_3seed.py --output-root exp/task_013_v1_1_3seed/20260603_3seed --data-root /home/superws/dataset/HISEA1_flooding_dataset --seeds 42 123 2026 --epochs 50 --batch-size 8 --num-workers 2 --resume
