#!/usr/bin/env bash
set -euo pipefail
python scripts/task_006_v2d_ablation_repro.py --output-root exp/task_006_v2d_ablation_repro/20260521_seed2024 --architectures fcn unet --variants full no_channel no_boundary no_speckle no_boundary_no_speckle no_all_gates no_msc no_local_var --seeds 2024 --epochs 50 --batch-size 8 --num-workers 2
