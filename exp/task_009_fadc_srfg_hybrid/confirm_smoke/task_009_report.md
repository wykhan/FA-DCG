# Task 009 FADC-SRFG Hybrid Report

- git branch: `codex/fadcg-v1.1-baseline`
- git commit hash: `d8feb311d67183a18e392501b9c6f5eafe78449c`
- dataset path: `/home/superws/dataset/HISEA1_flooding_dataset`
- split sizes: train=1404, val=468, test=468
- output directory: `/home/superws/2026_Projects/FA_DCG/exp/task_009_fadc_srfg_hybrid/confirm_smoke`
- PyTorch: `2.0.1+cu118`; CUDA: `11.8`

## Model List

- Reused baselines: U-Net, U-Net + CBAM, U-Net + FADC-aligned, U-Net + SRFG from task_008 when available.
- U-Net + CBAM (cbam, priority baseline): confirmation baseline: U-Net bottleneck CBAM under the task_009 protocol
- U-Net + FADC-aligned (fadc_aligned, priority baseline): confirmation baseline: local PyTorch FADC-aligned block under the task_009 protocol

## Protocol

- data: train for training, val for checkpoint selection, test for final metrics
- input size: 256 x 256; batch size: 2; epochs: 1
- optimizer: Adam, lr=0.0001, cosine eta_min=1e-06; loss=BCEWithLogitsLoss
- seeds: 7
- augmentation: random horizontal flip, random vertical flip, random rotation within +/-10 degrees
- metrics: sigmoid threshold 0.5; mIoU averages foreground/background IoU; Dice is foreground Dice
- complexity: THOP MACs when available; latency uses batch size 1, eval mode, no data loading

## Completed Runs

- completed or reused rows: 6
- complexity rows: 4

## Failed Or Skipped Runs

- None.

## Summary

| method | variant | test mIoU | test Dice | delta vs CBAM | params | MACs | latency ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | baseline | 0.856289 | 0.873615 | 0.395478 | 31042369 | 54662266880 | 3.186097 | 313.863638 |
| U-Net + CBAM | cbam | 0.460811 | 0.525480 | 0.000000 | 31173539 | 54662817280 | 3.271202 | 305.698027 |
| U-Net + FADC-aligned | fadc_aligned | 0.460575 | 0.525632 | -0.000236 | 31483786 | 54682781696 | 4.598140 | 217.479219 |
| U-Net + SRFG | srfg_v2d_nomsc | 0.875874 | 0.892199 | 0.415063 | 31606851 | 54670918656 | 3.656659 | 273.473718 |

## Decision Notes

- Best seed-screening test mIoU: U-Net + SRFG (srfg_v2d_nomsc) = 0.875874.
- Delta vs U-Net + CBAM: 0.415063 mIoU.
- Decision rule: run multi-seed confirmation for this candidate.
- Factual raw data only; no LaTeX paper edits were made.
