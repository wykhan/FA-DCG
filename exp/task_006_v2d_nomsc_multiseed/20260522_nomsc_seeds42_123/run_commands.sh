#!/usr/bin/env bash
set -euo pipefail
python scripts/task_006_v2d_ablation_repro.py --output-root exp/task_006_v2d_nomsc_multiseed/20260522_nomsc_seeds42_123 --architectures fcn unet --variants no_msc --seeds 42 123 --epochs 50 --batch-size 8 --num-workers 2
