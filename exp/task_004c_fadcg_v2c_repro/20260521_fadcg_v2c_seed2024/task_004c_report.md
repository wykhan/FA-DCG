# task_004c FA-DCG V2c Report

## Implementation Summary

- Selected candidate refinement from task_004a/task_004b.
- FCN V2c keeps V2a because FCN benefited most from conservative frequency gating.
- U-Net V2c keeps V2b at the bottleneck and adds one V2a skip-level frequency calibration block.
- The skip calibration block is initialized weakly with alpha=0.25.

## Relation to V1.1 and FADC

- Relation to V1.1: conservative residual extension, selected per backbone.
- FADC idea borrowed: frequency-aware modulation and receptive-field preference.
- SAR adaptation: bottleneck context handles homogeneous flood interiors; skip calibration targets boundaries and narrow water bodies.
- Lightweight design: only one extra skip block is added for U-Net, and FCN remains V2a-sized.

## Speed Benchmark

| Method | Implementation | Full-batch latency ms | Params M |
| --- | --- | ---: | ---: |
| FCN + FA-DCG V1.1 | v1.1 | 8.376 | 144.9324 |
| FCN + FA-DCG V2c | v2c | 8.524 | 145.0154 |
| U-Net + FA-DCG V1.1 | v1.1 | 20.372 | 31.5782 |
| U-Net + FA-DCG V2c | v2c | 21.453 | 31.8010 |

## Seed 2024 Results

| Method | V1.1 mIoU | V2c mIoU | Delta | V1.1 Dice | V2c Dice | Delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FCN + FA-DCG V2c | 0.842804 | 0.846577 | 0.003773 | 0.905853 | 0.908685 | 0.002832 | v2c_better |
| U-Net + FA-DCG V2c | 0.892461 | 0.891023 | -0.001438 | 0.938098 | 0.936816 | -0.001282 | v2c_comparable |

## Comparison with 004a and 004b

| Method | V2a mIoU | V2b mIoU | V2c mIoU | Best |
| --- | ---: | ---: | ---: | --- |
| FCN | 0.847235 | 0.842661 | 0.846577 | V2a |
| U-Net | 0.888433 | 0.892354 | 0.891023 | V2b |

V2c keeps the FCN improvement direction but does not exceed V2a. For U-Net, skip-level frequency calibration improves over V2a but remains lower than V2b.

## Diagnostics

| Method | Module | frequency gate mean | alpha | d1 | d2 | d3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FCN + FA-DCG V2c | module0 | 0.994739 | 0.487892 | NA | NA | NA |
| FCN + FA-DCG V2c | module1 | 0.999964 | 0.496775 | NA | NA | NA |
| U-Net + FA-DCG V2c bottleneck | module0 | 1.002845 | 0.526389 | 0.858151 | 0.124935 | 0.016914 |
| U-Net + FA-DCG V2c skip | module1 | 1.010555 | 0.285436 | NA | NA | NA |

The bottleneck dilation branch remains conservative and does not collapse to uniform weights. The skip block learns a slightly stronger alpha than its 0.25 initialization, but the final U-Net result suggests the extra skip calibration is not better than V2b alone.

## Recommendation

- Do not promote the skip-calibrated U-Net V2c over V2b.
- For task_004's final finding, use V2a as the FCN-preferred variant and V2b as the U-Net-preferred variant.
- If a single named FA-DCG V2.0 is required, the practical choice is the backbone-aware selected configuration: FCN uses V2a, U-Net uses V2b.

## Run Settings

- Seeds: `2024`.
- Epochs: `50`.
- Batch size: `8`.
- Evaluation uses the local validation split.
