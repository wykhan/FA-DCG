#!/usr/bin/env bash
set -euo pipefail
python scripts/task_003_fadc_aligned_repro.py --output-root exp/task_003_fadc_aligned_repro/20260520_fadc_aligned_seed2024 --models 'FCN + FADC' 'U-Net + FADC' --seeds 2024 --epochs 50 --batch-size 8 --num-workers 2
