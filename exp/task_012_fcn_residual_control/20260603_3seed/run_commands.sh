#!/usr/bin/env bash
set -euo pipefail
python scripts/task_012_fcn_residual_control.py --output-root exp/task_012_fcn_residual_control/20260603_3seed --data-root /home/superws/dataset/HISEA1_flooding_dataset --seeds 42 123 2026 2027 2028 2029 --epochs 50 --batch-size 8 --num-workers 2 --resume
