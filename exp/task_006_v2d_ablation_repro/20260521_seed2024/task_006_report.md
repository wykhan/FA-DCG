# task_006 V2d Minimal Sufficient Ablation Report

## Goal

Search for the minimal sufficient FA-DCG V2d by removing gates and descriptors from the full V2d block.

## Results

| Architecture | Variant | Params M | mIoU | Delta mIoU | Dice | Delta Dice | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fcn | full | 145.1444 | 0.845375 | 0.000000 | 0.907630 | 0.000000 | full_reference |
| fcn | no_channel | 136.6189 | 0.836355 | -0.009020 | 0.901098 | -0.006532 | important_drop |
| fcn | no_boundary | 145.0568 | 0.843355 | -0.002020 | 0.906143 | -0.001487 | slightly_worse_but_simpler |
| fcn | no_speckle | 145.0154 | 0.843500 | -0.001875 | 0.906690 | -0.000940 | slightly_worse_but_simpler |
| fcn | no_boundary_no_speckle | 144.9278 | 0.844877 | -0.000498 | 0.907139 | -0.000491 | equivalent_to_full |
| fcn | no_all_gates | 136.4024 | 0.840829 | -0.004546 | 0.904586 | -0.003044 | important_drop |
| fcn | no_msc | 145.0614 | 0.848363 | 0.002988 | 0.910157 | 0.002527 | better_than_full |
| fcn | no_local_var | 145.1029 | 0.845585 | 0.000210 | 0.907911 | 0.000281 | equivalent_to_full |
| unet | full | 31.6253 | 0.893472 | 0.000000 | 0.938617 | 0.000000 | full_reference |
| unet | no_channel | 31.0997 | 0.889718 | -0.003754 | 0.936340 | -0.002277 | important_drop |
| unet | no_boundary | 31.6058 | 0.894634 | 0.001162 | 0.939136 | 0.000519 | better_than_full |
| unet | no_speckle | 31.5966 | 0.888433 | -0.005039 | 0.935235 | -0.003382 | important_drop |
| unet | no_boundary_no_speckle | 31.5772 | 0.892461 | -0.001011 | 0.938098 | -0.000519 | slightly_worse_but_simpler |
| unet | no_all_gates | 31.0516 | 0.890295 | -0.003177 | 0.936652 | -0.001965 | important_drop |
| unet | no_msc | 31.6069 | 0.895189 | 0.001717 | 0.939710 | 0.001093 | better_than_full |
| unet | no_local_var | 31.6161 | 0.897975 | 0.004503 | 0.941652 | 0.003035 | better_than_full |

## Decision Notes

- `equivalent_to_full` means abs(delta mIoU) < 0.001 from a single seed.
- `slightly_worse_but_simpler` means the variant loses less than 0.003 mIoU and reduces parameters.
- A simplified variant should only replace full V2d after checking both FCN and U-Net behavior.

## Run Settings

- Architectures: `fcn, unet`.
- Variants: `full, no_channel, no_boundary, no_speckle, no_boundary_no_speckle, no_all_gates, no_msc, no_local_var`.
- Seeds: `2024`.
- Epochs: `50`.
- Batch size: `8`.
- Evaluation uses the local validation split.
