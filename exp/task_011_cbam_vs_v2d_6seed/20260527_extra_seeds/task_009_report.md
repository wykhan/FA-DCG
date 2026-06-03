# Task 009 FADC-SRFG Hybrid Report

- git branch: `codex/fadcg-v1.1-baseline`
- git commit hash: `490e20fbe3ca68a33346caf02181c1424146a77b`
- dataset path: `/home/superws/dataset/HISEA1_flooding_dataset`
- split sizes: train=1404, val=468, test=468
- output directory: `/home/superws/2026_Projects/FA_DCG/exp/task_011_cbam_vs_v2d_6seed/20260527_extra_seeds`
- PyTorch: `2.0.1+cu118`; CUDA: `11.8`

## Model List

- Reused baselines: U-Net, U-Net + CBAM, U-Net + FADC-aligned, U-Net + SRFG from task_008 when available.
- U-Net + CBAM (cbam, priority baseline): confirmation baseline: U-Net bottleneck CBAM under the task_009 protocol
- U-Net + V2d-noBoundary-noLocalVar (v2d_no_boundary_no_local_var, priority B): V2d ablation without boundary, MSC, or local variance; keeps channel and speckle gates

## Protocol

- data: train for training, val for checkpoint selection, test for final metrics
- input size: 256 x 256; batch size: 8; epochs: 50
- optimizer: Adam, lr=0.0001, cosine eta_min=1e-06; loss=BCEWithLogitsLoss
- seeds: 2027, 2028, 2029
- augmentation: random horizontal flip, random vertical flip, random rotation within +/-10 degrees
- metrics: sigmoid threshold 0.5; mIoU averages foreground/background IoU; Dice is foreground Dice
- complexity: THOP MACs when available; latency uses batch size 1, eval mode, no data loading

## Completed Runs

- completed or reused rows: 10
- complexity rows: 0

## Failed Or Skipped Runs

- None.

## Summary

| method | variant | test mIoU | test Dice | delta vs CBAM | params | MACs | latency ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | baseline | 0.856289 | 0.873615 | -0.020406 | NA | NA | NA | NA |
| U-Net + CBAM | cbam | 0.876695 | 0.892750 | 0.000000 | NA | NA | NA | NA |
| U-Net + FADC-aligned | fadc_aligned | 0.878748 | 0.894914 | 0.002053 | NA | NA | NA | NA |
| U-Net + SRFG | srfg_v2d_nomsc | 0.875874 | 0.892199 | -0.000821 | NA | NA | NA | NA |
| U-Net + V2d-noBoundary-noLocalVar | v2d_no_boundary_no_local_var | 0.878784 | 0.895108 | 0.002089 | NA | NA | NA | NA |

## Decision Notes

- Best seed-screening test mIoU: U-Net + V2d-noBoundary-noLocalVar (v2d_no_boundary_no_local_var) = 0.878784.
- Delta vs U-Net + CBAM: 0.002089 mIoU.
- Decision rule: improvement is positive but below 0.003 mIoU; treat as inconclusive until multi-seed confirmation.
- Factual raw data only; no LaTeX paper edits were made.
