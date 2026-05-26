# Task 008 Raw Report

- git branch: `codex/fadcg-v1.1-baseline`
- git commit hash: `d8feb311d67183a18e392501b9c6f5eafe78449c`
- dataset path: `/home/superws/2026_Projects/FA_DCG/SAR_FEM1/data/flood_dataset`
- output directory: `/home/superws/2026_Projects/FA_DCG/exp/task_008_main_module_table/smoke_check`
- PyTorch: `2.0.1+cu118`; CUDA: `11.8`; device: `cuda`; GPU: `NVIDIA GeForce RTX 4090`

## Model List

- FCN (baseline)
- U-Net + SRFG (SRFG)

## Training Protocol

- input size: 256 x 256
- input channels: 1
- batch size: 8
- epochs: 1
- optimizer: Adam, initial lr=0.0001, cosine annealing eta_min=1e-06
- loss: BCEWithLogitsLoss
- augmentation: random horizontal flip, random vertical flip, random rotation within +/-10 degrees
- seeds: 42

## Insertion Positions

- FCN SE/CBAM/FADC-aligned: inserted after `conv5`, before `fc6`.
- FCN SRFG: internal V2d-noMSC implementation mapped to `SRFG`; replaces FCN classifier gates as implemented in `FCNWithFADCGV2dAblation(variant="no_msc")`.
- U-Net SE/CBAM/FADC-aligned/SRFG: inserted after bottleneck, before decoder.
- FADC-aligned is the local PyTorch implementation inspired by FADC, not the official implementation.

## Metric Definitions

- Existing repository binary metrics: sigmoid threshold 0.5; foreground IoU and background IoU averaged as mIoU; Dice computed on foreground mask.
- Best epoch is selected by maximum validation mIoU on the existing validation split.

## FLOPs And FPS Protocol

- FLOPs tool: THOP. Reported values are MACs; one multiply-add is counted as one MAC.
- input tensor for complexity and latency: `[1, 1, 256, 256]`.
- latency: eval mode, no_grad, warmup=1, timed_iters=2, no data loading or preprocessing time.

## Completed Runs

- completed accuracy runs: 2/2
- summary rows: 2/10
- complexity rows: 2/10

## Failed Or Skipped Runs

- None.

## Summary

| backbone | method | seeds | mIoU mean | Dice mean | Params | MACs | latency ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FCN | FCN | 1 | 0.325548 | 0.000000 | 134271937 | 20518027264 | 3.011379 | 332.073779 |
| U-Net | U-Net + SRFG | 1 | 0.174452 | 0.517314 | 31606851 | 54670918656 | 2.951522 | 338.808248 |

## Key Observations

- Raw data only; no paper table formatting was changed.
- SRFG in this report is V2d-noMSC.
