# Task 013 FA-DCG V1.1 Three-Seed Report

- git branch: `codex/fadcg-v1.1-baseline`
- git commit hash: `61f31b50d3e60f7a5feb250e099a4249bb73fab7`
- dirty-worktree status:

```text
?? PROMPTS/task_013.md
?? exp/task_013_v1_1_3seed/
?? scripts/task_013_v1_1_3seed.py
```

- dataset path: `/home/superws/dataset/HISEA1_flooding_dataset`
- split sizes: train=1404, val=468, test=468
- output directory: `/home/superws/2026_Projects/FA_DCG/exp/task_013_v1_1_3seed/20260603_3seed`
- PyTorch: `2.0.1+cu118`; CUDA: `11.8`

## Model Configs

- `FCN + FA-DCG V1.1`: `FCNWithOptimizedFADCG`, two `VectorizedLightFADC` blocks on the original deep FCN path.
- `U-Net + FA-DCG V1.1`: `UNetWithOptimizedFADCG`, one `VectorizedLightFADC` block at the U-Net bottleneck.
- These are fresh three-seed train/val/test V1.1 results, not reused validation-only rows.

## Protocol

- input size: 256 x 256; batch size: 8; epochs: 50
- optimizer: Adam, lr=0.0001, cosine eta_min=1e-06; loss=BCEWithLogitsLoss
- seeds: 42, 123, 2026
- train is used for training, val for checkpoint selection, and test for final evaluation only
- augmentation: random horizontal flip, random vertical flip, random rotation within +/-10 degrees
- metrics: sigmoid threshold 0.5; mIoU averages foreground/background IoU; Dice is foreground Dice
- complexity: THOP MACs when available; latency uses batch size 1, eval mode, no data loading

## Completed Runs

- completed rows: 6
- complexity rows: 4

## Failed Or Skipped Runs

- None.

## Raw Per-Seed Results

| method | seed | val mIoU | test mIoU | test Dice | best epoch |
|---|---:|---:|---:|---:|---:|
| FCN + FA-DCG V1.1 | 42 | 0.817509 | 0.817411 | 0.831165 | 44 |
| FCN + FA-DCG V1.1 | 123 | 0.819196 | 0.826753 | 0.842768 | 43 |
| FCN + FA-DCG V1.1 | 2026 | 0.816979 | 0.821146 | 0.836399 | 35 |
| U-Net + FA-DCG V1.1 | 42 | 0.852341 | 0.877796 | 0.893829 | 38 |
| U-Net + FA-DCG V1.1 | 123 | 0.846115 | 0.876444 | 0.892397 | 30 |
| U-Net + FA-DCG V1.1 | 2026 | 0.856473 | 0.879836 | 0.895006 | 31 |

## Summary

| method | seeds | test mIoU mean/std | test Dice mean/std | params | MACs | latency ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCN + FA-DCG V1.1 | 3 | 0.821770 +/- 0.004702 | 0.836778 +/- 0.005811 | 144932419 | 20183159296 | 1.846534 | 541.555099 |
| U-Net + FA-DCG V1.1 | 3 | 0.878025 +/- 0.001707 | 0.893744 +/- 0.001307 | 31578178 | 54663054336 | 2.773101 | 360.607167 |

## Notes

- No LaTeX paper edits were made.
