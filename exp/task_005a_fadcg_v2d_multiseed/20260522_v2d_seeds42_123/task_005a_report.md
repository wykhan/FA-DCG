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

## Seed 2024 Results

| Method | V1.1 mIoU | V2d mIoU | Delta | V1.1 Dice | V2d Dice | Delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FCN + FA-DCG V2d | 0.842804 | 0.841177 | -0.001627 | 0.905853 | 0.904497 | -0.001356 | v2d_comparable |
| U-Net + FA-DCG V2d | 0.892461 | 0.890540 | -0.001921 | 0.938098 | 0.937050 | -0.001048 | v2d_comparable |

## Recommendation for task_005b

- If V2d improves, use it as the structural base before trying boundary-aware auxiliary supervision.
- If V2d does not improve, move to task_005b because the model may need stronger boundary supervision rather than another feature gate.

## Run Settings

- Seeds: `42, 123`.
- Epochs: `50`.
- Batch size: `8`.
- Evaluation uses the local validation split.
