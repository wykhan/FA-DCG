# task_004a FA-DCG V2a Report

## Implementation Summary

- Candidate A: SAR Frequency-Gated V1.1.
- V2a keeps V1.1's depthwise residual enhancement and channel gate.
- V2a adds a lightweight SAR frequency gate based on local high-frequency residual and multi-scale consistency.
- The frequency gate is initialized as identity, so the model starts close to V1.1.

## Relation to V1.1 and FADC

- Relation to V1.1: minimal conservative extension.
- FADC idea borrowed: frequency-aware modulation.
- SAR adaptation: local residual and multi-scale consistency descriptors target speckle/boundary cues.
- Lightweight design: grouped descriptor head with about 18C extra weights per V2a block.

## Speed Benchmark

| Method | Implementation | Full-batch latency ms | Params M |
| --- | --- | ---: | ---: |
| FCN + FA-DCG V1.1 | v1.1 | 8.326 | 144.9324 |
| FCN + FA-DCG V2a | v2a | 8.585 | 145.0154 |
| U-Net + FA-DCG V1.1 | v1.1 | 20.290 | 31.5782 |
| U-Net + FA-DCG V2a | v2a | 20.494 | 31.5966 |

## Seed 2024 Results

| Method | V1.1 mIoU | V2a mIoU | Delta | V1.1 Dice | V2a Dice | Delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FCN + FA-DCG V2a | 0.842804 | 0.847235 | 0.004431 | 0.905853 | 0.909229 | 0.003376 | v2a_better |
| U-Net + FA-DCG V2a | 0.892461 | 0.888433 | -0.004028 | 0.938098 | 0.935235 | -0.002863 | v2a_comparable |

## Recommendation for task_004b

- If V2a improves or matches V1.1, use it as the stable base for frequency-biased dilation in 004b.
- If V2a underperforms, inspect diagnostics before adding dilation dynamics.

## Run Settings

- Seeds: `2024`.
- Epochs: `50`.
- Batch size: `8`.
- Evaluation uses the local validation split.
