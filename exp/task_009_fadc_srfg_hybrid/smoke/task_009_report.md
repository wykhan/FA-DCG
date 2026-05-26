# Task 009 FADC-SRFG Hybrid Report

- git branch: `codex/fadcg-v1.1-baseline`
- git commit hash: `d8feb311d67183a18e392501b9c6f5eafe78449c`
- dataset path: `/home/superws/dataset/HISEA1_flooding_dataset`
- split sizes: train=1404, val=468, test=468
- output directory: `/home/superws/2026_Projects/FA_DCG/exp/task_009_fadc_srfg_hybrid/smoke`
- PyTorch: `2.0.1+cu118`; CUDA: `11.8`

## Model List

- Reused baselines: U-Net, U-Net + CBAM, U-Net + FADC-aligned, U-Net + SRFG from task_008 when available.
- U-Net + V2d-noLocalVar (v2d_no_local_var, priority A): historical V2d ablation without local variance descriptor
- U-Net + FADC-SR-high (fadc_sr_high, priority A): FADC-aligned main operator; SRFG reliability modulates high-frequency selected bands
- U-Net + FADC-SARSpatial-noVar (fadc_sar_spatial_no_var, priority A): FADC-aligned plus SAR-guided spatial mask without local variance
- U-Net + FADC-SR-high + SARSpatial-noVar (fadc_sr_high_sar_spatial_no_var, priority A): combined FADC high-band reliability and SAR-guided spatial mask
- U-Net + FADC-CBAM-SpatialOnly (fadc_cbam_spatial_only, priority A): control: FADC-aligned plus generic CBAM-style spatial-only mask
- U-Net + V2d-noBoundary-noLocalVar (v2d_no_boundary_no_local_var, priority B): V2d ablation without boundary, MSC, or local variance; keeps channel and speckle gates
- U-Net + FADC-SR-branch (fadc_sr_branch, priority B): FADC-aligned main operator; SRFG reliability modulates dilation branch responses
- U-Net + FADC-SARSpatial-withVar (fadc_sar_spatial_with_var, priority B): FADC-aligned plus SAR-guided spatial mask with local variance descriptor
- U-Net + FADC-SR-branch + SARSpatial-noVar (fadc_sr_branch_sar_spatial_no_var, priority B): combined branch reliability and SAR-guided spatial mask

## Protocol

- data: train for training, val for checkpoint selection, test for final metrics
- input size: 256 x 256; batch size: 2; epochs: 1
- optimizer: Adam, lr=0.0001, cosine eta_min=1e-06; loss=BCEWithLogitsLoss
- seeds: 2026
- augmentation: random horizontal flip, random vertical flip, random rotation within +/-10 degrees
- metrics: sigmoid threshold 0.5; mIoU averages foreground/background IoU; Dice is foreground Dice
- complexity: THOP MACs when available; latency uses batch size 1, eval mode, no data loading

## Completed Runs

- completed or reused rows: 13
- complexity rows: 13

## Failed Or Skipped Runs

- None.

## Summary

| method | variant | test mIoU | test Dice | delta vs CBAM | params | MACs | latency ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | baseline | 0.856289 | 0.873615 | -0.022930 | 31042369 | 54662266880 | 2.972820 | 336.380946 |
| U-Net + CBAM | cbam | 0.879219 | 0.894610 | 0.000000 | 31173539 | 54662817280 | 3.242636 | 308.391076 |
| U-Net + FADC-CBAM-SpatialOnly | fadc_cbam_spatial_only | 0.210856 | 0.205571 | -0.668363 | 31483886 | 54682806784 | 4.781593 | 209.135303 |
| U-Net + FADC-SARSpatial-noVar | fadc_sar_spatial_no_var | 0.211794 | 0.205892 | -0.667425 | 31483935 | 54683343616 | 4.845860 | 206.361719 |
| U-Net + FADC-SARSpatial-withVar | fadc_sar_spatial_with_var | 0.213395 | 0.206468 | -0.665824 | 31483984 | 54683356160 | 5.003156 | 199.873839 |
| U-Net + FADC-SR-branch | fadc_sr_branch | 0.214330 | 0.206834 | -0.664889 | 31494027 | 54685665280 | 4.810338 | 207.885600 |
| U-Net + FADC-SR-branch + SARSpatial-noVar | fadc_sr_branch_sar_spatial_no_var | 0.211798 | 0.205908 | -0.667421 | 31494176 | 54686227200 | 5.328940 | 187.654581 |
| U-Net + FADC-SR-high | fadc_sr_high | 0.214037 | 0.206698 | -0.665182 | 31494027 | 54685665280 | 4.795535 | 208.527307 |
| U-Net + FADC-SR-high + SARSpatial-noVar | fadc_sr_high_sar_spatial_no_var | 0.212866 | 0.206292 | -0.666353 | 31494176 | 54686227200 | 4.943861 | 202.271058 |
| U-Net + FADC-aligned | fadc_aligned | 0.878748 | 0.894914 | -0.000471 | 31483786 | 54682781696 | 3.739252 | 267.433200 |
| U-Net + SRFG | srfg_v2d_nomsc | 0.875874 | 0.892199 | -0.003345 | 31606851 | 54670918656 | 3.045658 | 328.336221 |
| U-Net + V2d-noBoundary-noLocalVar | v2d_no_boundary_no_local_var | 0.210220 | 0.205323 | -0.668999 | 31587395 | 54666200064 | 2.974656 | 336.173269 |
| U-Net + V2d-noLocalVar | v2d_no_local_var | 0.214290 | 0.206803 | -0.664929 | 31616067 | 54673277952 | 2.988937 | 334.567103 |

## Decision Notes

- Best seed-screening test mIoU: U-Net + CBAM (cbam) = 0.879219.
- Delta vs U-Net + CBAM: 0.000000 mIoU.
- Decision rule: no hybrid beats CBAM in seed-2026 screening.
- Factual raw data only; no LaTeX paper edits were made.
