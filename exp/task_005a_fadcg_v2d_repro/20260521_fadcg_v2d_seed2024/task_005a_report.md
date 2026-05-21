# task_005a FA-DCG V2d Report

## Implementation Summary

- task_005a: Speckle-Robust Frequency Gate.
- V2d keeps V2a's conservative depthwise residual enhancement and channel gate.
- V2d separates useful boundary frequency from speckle-like local variance cues.
- The boundary gate starts as identity and the suppression path starts weak, so the model starts close to V2a/V1.1.

## Relation to V1.1 and FADC

- Relation to V1.1: conservative residual extension with SAR-specific frequency filtering.
- FADC idea borrowed: frequency-aware modulation.
- SAR adaptation: useful high-frequency boundaries are separated from unstable local variance/speckle cues.
- Lightweight design: grouped descriptor heads; no dynamic dilation or heavy adaptive kernels.

## Speed Benchmark

| Method | Implementation | Full-batch latency ms | Params M |
| --- | --- | ---: | ---: |
| FCN + FA-DCG V1.1 | v1.1 | 8.381 | 144.9324 |
| FCN + FA-DCG V2d | v2d | 8.806 | 145.1444 |
| U-Net + FA-DCG V1.1 | v1.1 | 20.338 | 31.5782 |
| U-Net + FA-DCG V2d | v2d | 20.714 | 31.6253 |

## Seed 2024 Results

| Method | V1.1 mIoU | V2d mIoU | Delta | V1.1 Dice | V2d Dice | Delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FCN + FA-DCG V2d | 0.842804 | 0.845975 | 0.003171 | 0.905853 | 0.908181 | 0.002328 | v2d_better |
| U-Net + FA-DCG V2d | 0.892461 | 0.893472 | 0.001011 | 0.938098 | 0.938617 | 0.000519 | v2d_better |

## Comparison with task_004 Variants

| Method | V2a mIoU | V2b mIoU | V2c mIoU | V2d mIoU | Best |
| --- | ---: | ---: | ---: | ---: | --- |
| FCN | 0.847235 | 0.842661 | 0.846577 | 0.845975 | V2a |
| U-Net | 0.888433 | 0.892354 | 0.891023 | 0.893472 | V2d |

V2d does not exceed V2a on FCN, but it remains above V1.1. On U-Net, V2d is the best result among V2a/V2b/V2c/V2d and is the first frequency-aware variant to exceed V1.1.

## Diagnostics

| Method | Module | boundary gate | speckle gate | suppression ratio | beta |
| --- | --- | ---: | ---: | ---: | ---: |
| FCN + FA-DCG V2d | module0 | 0.993897 | 0.018564 | 0.998121 | 0.101224 |
| FCN + FA-DCG V2d | module1 | 1.000252 | 0.017964 | 0.998200 | 0.100181 |
| U-Net + FA-DCG V2d | module0 | 1.004199 | 0.018123 | 0.998206 | 0.099001 |

The speckle suppression branch remains weak and conservative after training. This indicates that the improvement is not caused by heavy high-frequency suppression; instead, V2d mostly keeps V2a-like enhancement while adding a small SAR-specific noise-aware correction.

## Recommendation for task_005b

- Use V2d as the current U-Net-preferred FA-DCG V2.x candidate.
- Keep V2a as the current FCN-preferred candidate because it still has the best FCN mIoU.
- For task_005b, test boundary-aware auxiliary supervision on V2d for U-Net and V2a/V2d for FCN.

## Run Settings

- Seeds: `2024`.
- Epochs: `50`.
- Batch size: `8`.
- Evaluation uses the local validation split.
