#!/usr/bin/env bash
set -euo pipefail
python scripts/task_007_v3_fcrg_repro.py --output-root exp/task_007_v3_fcrg_repro/20260522_seed2024 --architectures fcn unet --variants v1_1 v2d_nomsc v3_fcrg_a v3_fcrg_b --seeds 2024 --epochs 50 --batch-size 8 --num-workers 2
