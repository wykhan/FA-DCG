# task_004b FA-DCG V2b Report

## Implementation Summary

- Candidate B: Frequency-Biased Dilation Preference.
- V2b keeps V2a's SAR frequency gate and V1.1-style residual channel-gated enhancement.
- V2b adds three shared-kernel dilation branches with rates [1, 2, 3].
- The branch preference is initialized toward dilation=1, so the model starts close to V1.1/V2a.

## Relation to V1.1 and FADC

- Relation to V1.1: conservative residual extension with learnable receptive-field allocation.
- FADC idea borrowed: frequency-aware dilation preference.
- SAR adaptation: high-frequency cues preserve small structures and boundaries; low-frequency cues can allocate more large-context response.
- Lightweight design: branches share one depthwise kernel; the extra head is grouped by channel.

## Speed Benchmark

| Method | Implementation | Full-batch latency ms | Params M |
| --- | --- | ---: | ---: |
| FCN + FA-DCG V1.1 | v1.1 | 8.354 | 144.9324 |
| FCN + FA-DCG V2b | v2b | 9.125 | 145.2780 |
| U-Net + FA-DCG V1.1 | v1.1 | 20.358 | 31.5782 |
| U-Net + FA-DCG V2b | v2b | 20.989 | 31.6550 |

## Seed 2024 Results

| Method | V1.1 mIoU | V2b mIoU | Delta | V1.1 Dice | V2b Dice | Delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FCN + FA-DCG V2b | 0.842804 | 0.842661 | -0.000143 | 0.905853 | 0.906207 | 0.000354 | v2b_comparable |
| U-Net + FA-DCG V2b | 0.892461 | 0.892354 | -0.000107 | 0.938098 | 0.937855 | -0.000243 | v2b_comparable |

## Diagnostics

The learned branch weights remain conservative and do not collapse to uniform weights.

| Method | Module | d1 weight | d2 weight | d3 weight | large dilation |
| --- | --- | ---: | ---: | ---: | ---: |
| FCN + FA-DCG V2b | module0 | 0.863734 | 0.120128 | 0.016137 | 0.136266 |
| FCN + FA-DCG V2b | module1 | 0.866813 | 0.117310 | 0.015876 | 0.133187 |
| U-Net + FA-DCG V2b | module0 | 0.856803 | 0.126120 | 0.017077 | 0.143197 |

Frequency gates also stay close to identity:

| Method | Module | frequency gate mean | frequency gate std |
| --- | --- | ---: | ---: |
| FCN + FA-DCG V2b | module0 | 0.994787 | 0.014843 |
| FCN + FA-DCG V2b | module1 | 0.999974 | 0.005977 |
| U-Net + FA-DCG V2b | module0 | 1.003547 | 0.015353 |

## Recommendation for task_004c

- If V2b improves U-Net without FCN collapse, consider refining dilation bias in 004c.
- If V2b underperforms V2a, keep V2a as the safer narrative baseline and move 004c toward speckle suppression or U-Net skip calibration.

## Run Settings

- Seeds: `2024`.
- Epochs: `50`.
- Batch size: `8`.
- Evaluation uses the local validation split.
