# Task 008 Raw Report

- git branch: `codex/fadcg-v1.1-baseline`
- git commit hash: `d8feb311d67183a18e392501b9c6f5eafe78449c`
- dataset path: `/home/superws/dataset/HISEA1_flooding_dataset`
- output directory: `/home/superws/2026_Projects/FA_DCG/exp/task_008_main_module_table/20260525_181115`
- PyTorch: `2.0.1+cu118`; CUDA: `11.8`; device: `cuda`; GPU: `NVIDIA GeForce RTX 4090`

## Model List

- FCN (baseline)
- FCN + SE (SE)
- FCN + CBAM (CBAM)
- FCN + FADC-aligned (FADC-aligned)
- FCN + SRFG (SRFG)
- U-Net (baseline)
- U-Net + SE (SE)
- U-Net + CBAM (CBAM)
- U-Net + FADC-aligned (FADC-aligned)
- U-Net + SRFG (SRFG)

## Training Protocol

- input size: 256 x 256
- input channels: 1
- batch size: 8
- epochs: 50
- optimizer: Adam, initial lr=0.0001, cosine annealing eta_min=1e-06
- loss: BCEWithLogitsLoss
- augmentation: random horizontal flip, random vertical flip, random rotation within +/-10 degrees
- seeds: 2026

## Insertion Positions

- FCN SE/CBAM/FADC-aligned: inserted after `conv5`, before `fc6`.
- FCN SRFG: internal V2d-noMSC implementation mapped to `SRFG`; replaces FCN classifier gates as implemented in `FCNWithFADCGV2dAblation(variant="no_msc")`.
- U-Net SE/CBAM/FADC-aligned/SRFG: inserted after bottleneck, before decoder.
- FADC-aligned is the local PyTorch implementation inspired by FADC, not the official implementation.

## Metric Definitions

- Existing repository binary metrics: sigmoid threshold 0.5; foreground IoU and background IoU averaged as mIoU; Dice computed on foreground mask.
- Best epoch is selected by maximum mIoU on the validation split; final mIoU/Dice in CSV summaries are measured on the test split using that checkpoint.

## FLOPs And FPS Protocol

- FLOPs tool: THOP. Reported values are MACs; one multiply-add is counted as one MAC.
- input tensor for complexity and latency: `[1, 1, 256, 256]`.
- latency: eval mode, no_grad, warmup=50, timed_iters=300, no data loading or preprocessing time.

## Completed Runs

- completed accuracy runs: 10/10
- summary rows: 10/10
- complexity rows: 10/10

## Failed Or Skipped Runs

- None.

## Summary

| backbone | method | seeds | test mIoU mean | test Dice mean | Params | MACs | latency ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FCN | FCN | 1 | 0.649654 | 0.623303 | 134271937 | 20518027264 | 2.656144 | 376.485674 |
| FCN | FCN + SE | 1 | 0.674237 | 0.658948 | 134305249 | 20518093312 | 2.704110 | 369.807470 |
| FCN | FCN + CBAM | 1 | 0.681313 | 0.670457 | 134304803 | 20518132352 | 2.748988 | 363.770242 |
| FCN | FCN + FADC-aligned | 1 | 0.674524 | 0.664087 | 134410730 | 20520633344 | 3.553430 | 281.418274 |
| FCN | FCN + SRFG | 1 | 0.823353 | 0.839845 | 145061445 | 20192006656 | 1.946957 | 513.622137 |
| U-Net | U-Net | 1 | 0.856289 | 0.873615 | 31042369 | 54662266880 | 2.763470 | 361.863915 |
| U-Net | U-Net + SE | 1 | 0.874927 | 0.891253 | 31174529 | 54662661120 | 2.791419 | 358.240689 |
| U-Net | U-Net + CBAM | 1 | 0.879219 | 0.894610 | 31173539 | 54662817280 | 2.851861 | 350.648230 |
| U-Net | U-Net + FADC-aligned | 1 | 0.878748 | 0.894914 | 31483786 | 54682781696 | 3.758301 | 266.077704 |
| U-Net | U-Net + SRFG | 1 | 0.875874 | 0.892199 | 31606851 | 54670918656 | 2.895684 | 345.341546 |

## Key Observations

- Raw data only; no paper table formatting was changed.
- SRFG in this report is V2d-noMSC.
