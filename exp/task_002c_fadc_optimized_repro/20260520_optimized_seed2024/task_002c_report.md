# task_002c FA-DCG V1.1 Report

## Implementation

- FA-DCG V1.0 denotes the original `LightFADC` implementation.
- FA-DCG V1.1 denotes the accepted optimized baseline implemented by `VectorizedLightFADC`.
- Added `VectorizedLightFADC`, a behavior-compatible FA-DCG V1.1 block.
- Preserved original FA-DCG V1.0 insertion points for both FCN and U-Net.
- Replaced per-channel Python convolution loops with a single depthwise grouped convolution.
- Moved FCN `up_dim` registration from `forward` to `__init__`.
- Did not reuse task_002b FastFADCG branch-fusion structure.

## Accuracy vs Old FA-DCG

| Method | Old mIoU | Optimized mIoU | Delta | Old Dice | Optimized Dice | Delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FCN + Optimized FA-DCG | 0.844849 | 0.842804 | -0.002045 | 0.907507 | 0.905853 | -0.001654 | faster_and_comparable |
| U-Net + Optimized FA-DCG | 0.896663 | 0.892461 | -0.004202 | 0.941023 | 0.938098 | -0.002925 | faster_and_comparable |

## Speed Benchmark

| Method | Implementation | Full-batch latency ms | Params M |
| --- | --- | ---: | ---: |
| FCN + old FA-DCG | old | 214.826 | 144.9324 |
| FCN + Optimized FA-DCG | optimized | 8.317 | 144.9324 |
| U-Net + old FA-DCG | old | 65.931 | 31.5782 |
| U-Net + Optimized FA-DCG | optimized | 20.350 | 31.5782 |

## Run

- Seeds: `2024`.
- Epochs: `50`.
- Batch size: `8`.
- Evaluation uses the local validation split.
