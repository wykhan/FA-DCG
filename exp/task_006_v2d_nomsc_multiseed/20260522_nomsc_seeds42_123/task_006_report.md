# task_006 V2d Minimal Sufficient Ablation Report

## Goal

Search for the minimal sufficient FA-DCG V2d by removing gates and descriptors from the full V2d block.

## Results

| Architecture | Variant | Params M | mIoU | Delta mIoU | Dice | Delta Dice | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fcn | no_msc | 145.0614 | 0.841316 | 0.000000 | 0.905402 | 0.000000 | equivalent_to_full |
| unet | no_msc | 31.6069 | 0.890907 | 0.000000 | 0.937191 | 0.000000 | equivalent_to_full |

## Decision Notes

- `equivalent_to_full` means abs(delta mIoU) < 0.001 from a single seed.
- `slightly_worse_but_simpler` means the variant loses less than 0.003 mIoU and reduces parameters.
- A simplified variant should only replace full V2d after checking both FCN and U-Net behavior.

## Run Settings

- Architectures: `fcn, unet`.
- Variants: `no_msc`.
- Seeds: `42, 123`.
- Epochs: `50`.
- Batch size: `8`.
- Evaluation uses the local validation split.
