# task_002c Optimized FA-DCG Report

## Implementation

- Added `VectorizedLightFADC`, a behavior-compatible FA-DCG-v1 block.
- Preserved original FA-DCG insertion points for both FCN and U-Net.
- Replaced per-channel Python convolution loops with a single depthwise grouped convolution.
- Moved FCN `up_dim` registration from `forward` to `__init__`.
- Did not reuse task_002b FastFADCG branch-fusion structure.

## Accuracy vs Old FA-DCG

| Method | Old mIoU | Optimized mIoU | Delta | Old Dice | Optimized Dice | Delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FCN + Optimized FA-DCG | 0.844849 | 0.844089 | -0.000760 | 0.907507 | 0.906666 | -0.000841 | faster_and_comparable |
| U-Net + Optimized FA-DCG | 0.896663 | 0.891287 | -0.005376 | 0.941023 | 0.937523 | -0.003501 | faster_and_comparable |

## Speed Benchmark

| Method | Implementation | Full-batch latency ms | Params M |
| --- | --- | ---: | ---: |

## Run

- Seeds: `42, 123`.
- Epochs: `50`.
- Batch size: `8`.
- Evaluation uses the local validation split.
