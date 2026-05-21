#!/usr/bin/env bash
set -euo pipefail
python scripts/task_005c_placement_repro.py --output-root exp/task_005c_placement_repro/20260521_v1_1_v2d_placement_seed2024 --models 'FCN + V1.1 after conv3' 'FCN + V1.1 after conv4' 'FCN + V2d after conv3' 'FCN + V2d after conv4' 'U-Net + V1.1 after encoder2' 'U-Net + V1.1 after encoder3' 'U-Net + V2d after encoder2' 'U-Net + V2d after encoder3' --seeds 2024 --epochs 50 --batch-size 8 --num-workers 2 --alpha-init 0.25
