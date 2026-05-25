#!/usr/bin/env bash
set -euo pipefail
python scripts/task_007_v3_fcrg_repro.py --output-root exp/task_007_v3_fcrg_repro/20260522_v3a_nomsc_seeds42_123 --architectures fcn unet --variants v2d_nomsc v3_fcrg_a --seeds 42 123 --epochs 50 --batch-size 8 --num-workers 2
