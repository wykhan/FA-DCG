# Task 009 FADC-SRFG Hybrid Report

- git branch: `codex/fadcg-v1.1-baseline`
- git commit hash: `d8feb311d67183a18e392501b9c6f5eafe78449c`
- dataset path: `/home/superws/dataset/HISEA1_flooding_dataset`
- split sizes: train=1404, val=468, test=468
- output directory: `/home/superws/2026_Projects/FA_DCG/exp/task_009_fadc_srfg_hybrid/20260526_081709`
- PyTorch: `2.0.1+cu118`; CUDA: `11.8`

## Model List

- Reused baselines: U-Net, U-Net + CBAM, U-Net + FADC-aligned, U-Net + SRFG from task_008 when available.
- U-Net + FADC-aligned (fadc_aligned, priority baseline): confirmation baseline: local PyTorch FADC-aligned block under the task_009 protocol

## Protocol

- data: train for training, val for checkpoint selection, test for final metrics
- input size: 256 x 256; batch size: 8; epochs: 50
- optimizer: Adam, lr=0.0001, cosine eta_min=1e-06; loss=BCEWithLogitsLoss
- seeds: 42, 123
- augmentation: random horizontal flip, random vertical flip, random rotation within +/-10 degrees
- metrics: sigmoid threshold 0.5; mIoU averages foreground/background IoU; Dice is foreground Dice
- complexity: THOP MACs when available; latency uses batch size 1, eval mode, no data loading

## Completed Runs

- completed or reused rows: 21
- complexity rows: 13

## Failed Or Skipped Runs

- None.

## Summary

| method | variant | test mIoU | test Dice | delta vs CBAM | params | MACs | latency ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | baseline | 0.856289 | 0.873615 | -0.019278 | 31042369 | 54662266880 | 2.834703 | 352.770639 |
| U-Net + CBAM | cbam | 0.875567 | 0.891556 | 0.000000 | 31173539 | 54662817280 | 3.105889 | 321.968987 |
| U-Net + FADC-CBAM-SpatialOnly | fadc_cbam_spatial_only | 0.873406 | 0.889196 | -0.002161 | 31483886 | 54682806784 | 4.597029 | 217.531793 |
| U-Net + FADC-SARSpatial-noVar | fadc_sar_spatial_no_var | 0.877584 | 0.893429 | 0.002017 | 31483935 | 54683343616 | 4.800261 | 208.322009 |
| U-Net + FADC-SARSpatial-withVar | fadc_sar_spatial_with_var | 0.875967 | 0.891018 | 0.000400 | 31483984 | 54683356160 | 4.449567 | 224.740948 |
| U-Net + FADC-SR-branch | fadc_sr_branch | 0.873023 | 0.887726 | -0.002544 | 31494027 | 54685665280 | 3.820457 | 261.748763 |
| U-Net + FADC-SR-branch + SARSpatial-noVar | fadc_sr_branch_sar_spatial_no_var | 0.878156 | 0.894004 | 0.002589 | 31494176 | 54686227200 | 4.148882 | 241.028783 |
| U-Net + FADC-SR-high | fadc_sr_high | 0.874392 | 0.889974 | -0.001175 | 31494027 | 54685665280 | 3.788785 | 263.936891 |
| U-Net + FADC-SR-high + SARSpatial-noVar | fadc_sr_high_sar_spatial_no_var | 0.873964 | 0.890681 | -0.001603 | 31494176 | 54686227200 | 4.102940 | 243.727643 |
| U-Net + FADC-aligned | fadc_aligned | 0.870252 | 0.886702 | -0.005315 | 31483786 | 54682781696 | 3.586797 | 278.800264 |
| U-Net + SRFG | srfg_v2d_nomsc | 0.875874 | 0.892199 | 0.000307 | 31606851 | 54670918656 | 2.892448 | 345.727891 |
| U-Net + V2d-noBoundary-noLocalVar | v2d_no_boundary_no_local_var | 0.876026 | 0.891790 | 0.000459 | 31587395 | 54666200064 | 2.839628 | 352.158845 |
| U-Net + V2d-noLocalVar | v2d_no_local_var | 0.878237 | 0.893934 | 0.002670 | 31616067 | 54673277952 | 2.848233 | 351.094874 |

## Decision Notes

- Best seed-screening test mIoU: U-Net + V2d-noLocalVar (v2d_no_local_var) = 0.878237.
- Delta vs U-Net + CBAM: 0.002670 mIoU.
- Decision rule: improvement is positive but below 0.003 mIoU; treat as inconclusive until multi-seed confirmation.
- Factual raw data only; no LaTeX paper edits were made.
