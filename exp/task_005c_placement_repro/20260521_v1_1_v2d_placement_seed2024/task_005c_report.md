# task_005c Placement Ablation Report

## Implementation Summary

- This experiment tests the best insertion position for FA-DCG V1.1 and V2d.
- It keeps the block definitions fixed and changes only where one block is inserted.
- FCN tests after conv3 and after conv4; U-Net tests after encoder stage 2 and stage 3.
- Existing deep/bottleneck results are used as original-placement references.

## Paper Story

FA-DCG is framed as a plug-and-play feature extraction module. Therefore, insertion position is a core design variable: early/mid placement may enhance water-boundary and narrow-channel cues before repeated downsampling, while too-early placement may amplify SAR speckle and low-level texture noise.

## Seed 2024 Results

| Method | mIoU | Dice | Accepted Original mIoU | Delta | Local Original mIoU | Delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FCN + V1.1 after conv3 | 0.653409 | 0.756980 | 0.842804 | -0.189395 | 0.844849 | -0.191440 | under_accepted_original |
| FCN + V1.1 after conv4 | 0.651189 | 0.756623 | 0.842804 | -0.191615 | 0.844849 | -0.193660 | under_accepted_original |
| FCN + V2d after conv3 | 0.660667 | 0.765494 | 0.845975 | -0.185308 | 0.845975 | -0.185308 | under_accepted_original |
| FCN + V2d after conv4 | 0.653649 | 0.757387 | 0.845975 | -0.192326 | 0.845975 | -0.192326 | under_accepted_original |
| U-Net + V1.1 after encoder2 | 0.885331 | 0.933787 | 0.892461 | -0.007130 | 0.896663 | -0.011332 | under_accepted_original |
| U-Net + V1.1 after encoder3 | 0.888685 | 0.936085 | 0.892461 | -0.003776 | 0.896663 | -0.007978 | comparable_to_accepted_original |
| U-Net + V2d after encoder2 | 0.880935 | 0.931215 | 0.893472 | -0.012537 | 0.893472 | -0.012537 | under_accepted_original |
| U-Net + V2d after encoder3 | 0.888610 | 0.936217 | 0.893472 | -0.004862 | 0.893472 | -0.004862 | comparable_to_accepted_original |

## Best Placement Summary

| Architecture | Block | Best Placement | Best mIoU | Delta vs Accepted Original | Recommendation |
| --- | --- | --- | ---: | ---: | --- |
| fcn | v1.1 | after_conv3 | 0.653409 | -0.189395 | keep_original_or_retest |
| fcn | v2d | after_conv3 | 0.660667 | -0.185308 | keep_original_or_retest |
| unet | v1.1 | after_encoder3 | 0.888685 | -0.003776 | keep_original_or_retest |
| unet | v2d | after_encoder3 | 0.888610 | -0.004862 | keep_original_or_retest |

## Interpretation

- FCN does not support replacing the original deep FA-DCG path with a single early/mid block. Both conv3 and conv4 placements collapse toward plain-FCN-level performance, even though the models remain numerically stable.
- U-Net is more tolerant of early/mid FA-DCG insertion. Encoder stage 3 is consistently better than encoder stage 2 for both V1.1 and V2d.
- However, U-Net encoder stage 3 remains slightly below the accepted original bottleneck/deep references. The result supports keeping the conservative bottleneck placement as the main paper setting.
- V2d does not reverse the placement conclusion. Its speckle-robust gate improves FCN after-conv3 over V1.1 after-conv3, but the gap to original deep V2d remains too large.

## Recommendation

- Do not replace the main FA-DCG V1.1/V2d configuration with early/mid single-block placement.
- Use task_005c as a placement ablation showing that plug-and-play does not mean every insertion point is equally effective.
- For the paper story, frame the current deep/bottleneck placement as the stable choice for this small SAR dataset, while U-Net encoder stage 3 can be reported as a near-comparable but not superior alternative.

## Speed Benchmark

| Method | Params M | Full-batch latency ms | GPU MB |
| --- | ---: | ---: | ---: |
| FCN + V1.1 after conv3 | 134.3076 | 10.489 | 912.8 |
| FCN + V1.1 after conv4 | 134.4088 | 10.422 | 912.3 |
| FCN + V2d after conv3 | 134.3194 | 10.775 | 911.9 |
| FCN + V2d after conv4 | 134.4323 | 10.556 | 912.4 |
| U-Net + V1.1 after encoder2 | 31.0520 | 22.101 | 1002.3 |
| U-Net + V1.1 after encoder3 | 31.0780 | 21.642 | 1001.6 |
| U-Net + V2d after encoder2 | 31.0579 | 27.898 | 1145.7 |
| U-Net + V2d after encoder3 | 31.0898 | 24.178 | 1001.7 |

## Run Settings

- Seeds: `2024`.
- Epochs: `50`.
- Batch size: `8`.
- Alpha init: `0.25`.
- Evaluation uses the local validation split.
