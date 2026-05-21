# task_003 FADC-Aligned Report

## Feasibility Result

- Implemented a local PyTorch FADC-aligned module.
- No MMCV/MMSeg dependency is used.
- Official AdaDR is approximated with spatial dilation-branch mixing rather than deformable convolution.
- SAR-specific FA-DCG V2.0 design is deferred to task_004.

## Implemented Components

- FreqSelect: high/low band feature reweighting.
- AdaDR: spatially adaptive dilation branch mixing over dilation rates 1, 2, 3, and 4.
- AdaKern: low/high depthwise kernel decomposition with lightweight attention gates.

## Deviations from Official Code

- Does not use `mmcv.ops.ModulatedDeformConv2d` or learned deformable offsets.
- Uses branch mixing as a local differentiable approximation of adaptive dilation.
- Integrated as a residual feature enhancement block in FCN and U-Net.

## Integration Positions

- FCN + FADC: after `conv3`, C=256.
- U-Net + FADC: after encoder stage 3, C=256.

## Speed Benchmark

| Method | Implementation | Full-batch latency ms | Params M |
| --- | --- | ---: | ---: |
| FCN + FA-DCG V1.1 | v1.1 | 8.397 | 144.9324 |
| FCN + FADC | fadc_aligned | 10.673 | 134.3209 |
| U-Net + FA-DCG V1.1 | v1.1 | 20.454 | 31.5782 |
| U-Net + FADC | fadc_aligned | 23.828 | 31.0913 |

## Seed 2024 Results

| Method | V1.1 mIoU | FADC mIoU | Delta | V1.1 Dice | FADC Dice | Delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FCN + FADC | 0.842804 | 0.663577 | -0.179227 | 0.905853 | 0.771850 | -0.134003 | underperforming |
| U-Net + FADC | 0.892461 | 0.887575 | -0.004886 | 0.938098 | 0.934839 | -0.003259 | aligned_and_comparable |

## Recommendation

- Keep FA-DCG V1.1 as the practical baseline for FCN because FCN + FADC underperforms strongly.
- U-Net + FADC is close enough to V1.1 to justify later comparison, but it should not replace V1.1 from this single-seed result.
- Use task_004 to design the SAR-specific lightweight FA-DCG V2.0 rather than treating this official-aligned FADC as the final SAR module.

## Run Settings

- Seeds: `2024`.
- Epochs: `50`.
- Batch size: `8`.
- Evaluation uses the local validation split.
