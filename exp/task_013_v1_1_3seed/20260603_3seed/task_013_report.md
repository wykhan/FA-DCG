# Task 013 FA-DCG V1.1 6-seed Report

- git branch: `codex/fadcg-v1.1-baseline`
- git commit hash: `f6c665f13ad1aedd9c9b4f66ee4de7a32221021e`
- dirty-worktree status:

```text
M exp/task_013_v1_1_3seed/20260603_3seed/complexity_raw.csv
 M exp/task_013_v1_1_3seed/20260603_3seed/diagnostics_raw.csv
 M exp/task_013_v1_1_3seed/20260603_3seed/environment.txt
 M exp/task_013_v1_1_3seed/20260603_3seed/raw_metrics_per_seed.csv
 M exp/task_013_v1_1_3seed/20260603_3seed/run_commands.sh
 M exp/task_013_v1_1_3seed/20260603_3seed/summary_mean_std.csv
 M scripts/task_013_v1_1_3seed.py
?? exp/task_013_v1_1_3seed/20260603_3seed/logs/fcn_fadcg_v1_1_seed2027.log
?? exp/task_013_v1_1_3seed/20260603_3seed/logs/fcn_fadcg_v1_1_seed2028.log
?? exp/task_013_v1_1_3seed/20260603_3seed/logs/fcn_fadcg_v1_1_seed2029.log
?? exp/task_013_v1_1_3seed/20260603_3seed/logs/unet_fadcg_v1_1_seed2027.log
?? exp/task_013_v1_1_3seed/20260603_3seed/logs/unet_fadcg_v1_1_seed2028.log
?? exp/task_013_v1_1_3seed/20260603_3seed/logs/unet_fadcg_v1_1_seed2029.log
```

- dataset path: `/home/superws/dataset/HISEA1_flooding_dataset`
- split sizes: train=1404, val=468, test=468
- output directory: `/home/superws/2026_Projects/FA_DCG/exp/task_013_v1_1_3seed/20260603_3seed`
- PyTorch: `2.0.1+cu118`; CUDA: `11.8`

## Model Configs

- `FCN + FA-DCG V1.1`: `FCNWithOptimizedFADCG`, two `VectorizedLightFADC` blocks on the original deep FCN path.
- `U-Net + FA-DCG V1.1`: `UNetWithOptimizedFADCG`, one `VectorizedLightFADC` block at the U-Net bottleneck.
- These are fresh 6-seed train/val/test V1.1 results, not reused validation-only rows.

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
- complexity rows: 4

## Failed Or Skipped Runs

- None.

## Raw Per-Seed Results

| method | seed | val mIoU | test mIoU | test Dice | best epoch |
|---|---:|---:|---:|---:|---:|
| FCN + FA-DCG V1.1 | 42 | 0.817509 | 0.817411 | 0.831165 | 44 |
| FCN + FA-DCG V1.1 | 123 | 0.819196 | 0.826753 | 0.842768 | 43 |
| FCN + FA-DCG V1.1 | 2026 | 0.816979 | 0.821146 | 0.836399 | 35 |
| FCN + FA-DCG V1.1 | 2027 | 0.819213 | 0.823914 | 0.839171 | 40 |
| FCN + FA-DCG V1.1 | 2028 | 0.822531 | 0.825949 | 0.840147 | 49 |
| FCN + FA-DCG V1.1 | 2029 | 0.816463 | 0.820464 | 0.834846 | 39 |
| U-Net + FA-DCG V1.1 | 42 | 0.852341 | 0.877796 | 0.893829 | 38 |
| U-Net + FA-DCG V1.1 | 123 | 0.846115 | 0.876444 | 0.892397 | 30 |
| U-Net + FA-DCG V1.1 | 2026 | 0.856473 | 0.879836 | 0.895006 | 31 |
| U-Net + FA-DCG V1.1 | 2027 | 0.850540 | 0.877485 | 0.893202 | 29 |
| U-Net + FA-DCG V1.1 | 2028 | 0.852235 | 0.871877 | 0.888628 | 30 |
| U-Net + FA-DCG V1.1 | 2029 | 0.851524 | 0.878032 | 0.894573 | 46 |

## Summary

| method | seeds | test mIoU mean/std | test Dice mean/std | params | MACs | latency ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCN + FA-DCG V1.1 | 6 | 0.822606 +/- 0.003572 | 0.837416 +/- 0.004145 | 144932419 | 20183159296 | 1.371750 | 728.995998 |
| U-Net + FA-DCG V1.1 | 6 | 0.876912 +/- 0.002701 | 0.892939 +/- 0.002310 | 31578178 | 54663054336 | 2.693438 | 371.272662 |

## Notes

- No LaTeX paper edits were made.
