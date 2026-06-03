# Task 012 FCN Residual-Control Report

- git branch: `codex/fadcg-v1.1-baseline`
- git commit hash: `3241630b77b5d6306e68441c41d8138cb41129ef`
- dirty-worktree status:

```text
?? PROMPTS/task_012.md
?? SAR_FEM1/models/improved/fcn_residual_only.py
?? exp/task_012_fcn_residual_control/
?? scripts/task_012_fcn_residual_control.py
```

- dataset path: `/home/superws/dataset/HISEA1_flooding_dataset`
- split sizes: train=1404, val=468, test=468
- output directory: `/home/superws/2026_Projects/FA_DCG/exp/task_012_fcn_residual_control/20260603_3seed`
- PyTorch: `2.0.1+cu118`; CUDA: `11.8`

## Model Configs

- `FCN + Residual-only`: two depthwise residual blocks at the same 512-channel and 4096-channel stages as FCN V2d; formula `out = x + alpha * z`, alpha initialized to 0.5; no gates or descriptors.
- `FCN + V2d-noBoundary-noLocalVar`: `FCNWithFADCGV2dAblation(variant="no_boundary_no_local_var")`; config `{'channel': True, 'boundary': False, 'speckle': True, 'msc': False, 'local_var': False}`; `freq_init=random`.

## Protocol

- input size: 256 x 256; batch size: 8; epochs: 50
- optimizer: Adam, lr=0.0001, cosine eta_min=1e-06; loss=BCEWithLogitsLoss
- seeds: 42, 123, 2026, 2027, 2028, 2029
- train is used for training, val for checkpoint selection, and test for final evaluation only
- augmentation: random horizontal flip, random vertical flip, random rotation within +/-10 degrees
- metrics: sigmoid threshold 0.5; mIoU averages foreground/background IoU; Dice is foreground Dice
- complexity: THOP MACs when available; latency uses batch size 1, eval mode, no data loading

## Completed Runs

- completed rows: 12
- complexity rows: 3

## Failed Or Skipped Runs

- None.

## Raw Per-Seed Results

| method | seed | val mIoU | test mIoU | test Dice | best epoch |
|---|---:|---:|---:|---:|---:|
| FCN + Residual-only | 42 | 0.812359 | 0.810101 | 0.824341 | 44 |
| FCN + Residual-only | 123 | 0.813624 | 0.820251 | 0.835830 | 44 |
| FCN + Residual-only | 2026 | 0.823259 | 0.818909 | 0.833238 | 41 |
| FCN + Residual-only | 2027 | 0.813297 | 0.814864 | 0.829978 | 40 |
| FCN + Residual-only | 2028 | 0.817498 | 0.816800 | 0.830969 | 35 |
| FCN + Residual-only | 2029 | 0.814069 | 0.816129 | 0.830410 | 39 |
| FCN + V2d-noBoundary-noLocalVar | 42 | 0.817795 | 0.819768 | 0.834294 | 44 |
| FCN + V2d-noBoundary-noLocalVar | 123 | 0.822926 | 0.822367 | 0.837575 | 39 |
| FCN + V2d-noBoundary-noLocalVar | 2026 | 0.813003 | 0.817876 | 0.833110 | 41 |
| FCN + V2d-noBoundary-noLocalVar | 2027 | 0.819315 | 0.824693 | 0.840543 | 39 |
| FCN + V2d-noBoundary-noLocalVar | 2028 | 0.820075 | 0.822568 | 0.838063 | 31 |
| FCN + V2d-noBoundary-noLocalVar | 2029 | 0.819691 | 0.822278 | 0.837709 | 39 |

## Summary

| method | seeds | test mIoU mean/std | test Dice mean/std | params | MACs | latency ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCN + Residual-only | 6 | 0.816176 +/- 0.003552 | 0.830794 +/- 0.003842 | 136402371 | 20174340096 | 1.596377 | 626.418478 |
| FCN + V2d-noBoundary-noLocalVar | 6 | 0.821592 +/- 0.002399 | 0.836882 +/- 0.002716 | 144973893 | 20186698240 | 2.113892 | 473.061148 |

## Paired Seed Delta

| seed | residual mIoU | V2d mIoU | delta mIoU | residual Dice | V2d Dice | delta Dice |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.810101 | 0.819768 | 0.009667 | 0.824341 | 0.834294 | 0.009953 |
| 123 | 0.820251 | 0.822367 | 0.002116 | 0.835830 | 0.837575 | 0.001745 |
| 2026 | 0.818909 | 0.817876 | -0.001033 | 0.833238 | 0.833110 | -0.000128 |
| 2027 | 0.814864 | 0.824693 | 0.009829 | 0.829978 | 0.840543 | 0.010565 |
| 2028 | 0.816800 | 0.822568 | 0.005768 | 0.830969 | 0.838063 | 0.007093 |
| 2029 | 0.816129 | 0.822278 | 0.006149 | 0.830410 | 0.837709 | 0.007299 |

- mean paired delta mIoU: `0.005416`; std: `0.004259`
- mean paired delta Dice: `0.006088`; std: `0.004358`

## Decision

- Decision rule result: positive evidence for V2d beyond residual learning on FCN.
- Positive paired seeds: 5/6.
- No LaTeX paper edits were made.
