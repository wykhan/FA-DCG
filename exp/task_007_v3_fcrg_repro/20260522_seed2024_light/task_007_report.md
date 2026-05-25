# task_007 V3 FCRG Seed-2024 Report

## Executive Summary

This run evaluates frequency-conditioned response gates under the validation protocol. It is not a final independent-test evaluation.

## Rationale

task_007 follows the GPT proposal: avoid making B/S more complex and instead transform the generic residual response gate into a SAR frequency-conditioned response gate.

## Evaluated Structures

- `v1_1`: residual enhancement plus original channel response gate.
- `v2d_nomsc`: current V2d-noMSC reference.
- `v3_fcrg_a`: direct frequency-conditioned response gate.
- `v3_fcrg_b`: V1.1 response gate with residual SAR-frequency modulation.

## Main Results

| Architecture | Variant | Params M | mIoU | Delta vs V1.1 | Delta vs V2d-noMSC | Dice | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fcn | v1_1 | 144.9324 | 0.840983 | 0.000000 | -0.008205 | 0.904855 | worse_than_nomsc |
| fcn | v2d_nomsc | 145.0614 | 0.849188 | 0.008205 | 0.000000 | 0.910095 | v2d_nomsc_reference |
| fcn | v3_fcrg_a | 137.5212 | 0.845847 | 0.004864 | -0.003341 | 0.908071 | worse_than_nomsc |
| fcn | v3_fcrg_b | 145.7681 | 0.845709 | 0.004726 | -0.003479 | 0.907424 | worse_than_nomsc |
| unet | v1_1 | 31.5782 | 0.892461 | 0.000000 | -0.002728 | 0.938098 | worse_than_nomsc |
| unet | v2d_nomsc | 31.6069 | 0.895189 | 0.002728 | 0.000000 | 0.939710 | v2d_nomsc_reference |
| unet | v3_fcrg_a | 31.3148 | 0.897218 | 0.004757 | 0.002029 | 0.941095 | better_than_nomsc |
| unet | v3_fcrg_b | 31.7749 | 0.895481 | 0.003020 | 0.000292 | 0.939956 | comparable_to_nomsc |

## Run Settings

- Architectures: `fcn, unet`.
- Variants: `v1_1, v2d_nomsc, v3_fcrg_a, v3_fcrg_b`.
- Seeds: `2024`.
- Epochs: `50`.
- Batch size: `8`.
- Evaluation uses the local validation split.

## Recommendation Rule

Use `v3_fcrg_b` as the final FA-DCG variant only if it matches or exceeds V2d-noMSC. Use `v3_fcrg_a` only if it clearly exceeds `v3_fcrg_b` and remains stable.

## Interpretation

The lightweight corrected implementation changes the first implementation's cost profile substantially. The frequency-conditioning bottleneck is capped, so the V3 variants remain close to the V1.1/V2d-noMSC cost range instead of adding a large fully connected response branch.

On FCN, neither V3-FCRG variant matches V2d-noMSC:

- `v3_fcrg_a`: mIoU 0.845847, delta vs V2d-noMSC -0.003341.
- `v3_fcrg_b`: mIoU 0.845709, delta vs V2d-noMSC -0.003479.

However, both FCN V3 variants improve over V1.1, and `v3_fcrg_a` uses fewer parameters than V1.1:

- FCN V1.1: 144.9324M params, mIoU 0.840983.
- FCN V3-FCRG-A: 137.5212M params, mIoU 0.845847.

On U-Net, V3-FCRG-A is the best seed-2024 result:

- U-Net V2d-noMSC: mIoU 0.895189, Dice 0.939710.
- U-Net V3-FCRG-A: mIoU 0.897218, Dice 0.941095.
- Delta: +0.002029 mIoU and +0.001385 Dice.

U-Net V3-FCRG-B is comparable to V2d-noMSC:

- U-Net V3-FCRG-B: mIoU 0.895481, delta vs V2d-noMSC +0.000292.

The diagnostics show that V3-FCRG-B learns non-zero frequency modulation:

- FCN module 0 lambda: -0.012072.
- FCN module 1 lambda: -0.001664.
- U-Net module 0 lambda: 0.012850.

Thus the frequency branch is not purely ignored after training. V3-FCRG-A also produces non-collapsed response gates, especially on U-Net, where the gate mean is 0.5405 and std is 0.3139.

## Recommendation

Do not replace V2d-noMSC with V3-FCRG as a universal FCN+U-Net final method yet. The seed-2024 evidence is mixed:

- V3-FCRG-A is promising for U-Net and deserves multi-seed confirmation there.
- V3-FCRG-B is safer and comparable on U-Net, but it does not beat V2d-noMSC on FCN.
- FCN still prefers V2d-noMSC in this experiment.

The next best step is:

```text
Run multi-seed confirmation for U-Net V3-FCRG-A, U-Net V2d-noMSC, and U-Net V1.1.
For FCN, keep V2d-noMSC as the reference unless a lighter FCN-specific FCRG variant is designed.
```
