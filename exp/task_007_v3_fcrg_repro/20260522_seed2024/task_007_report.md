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
| fcn | v1_1 | 144.9324 | 0.843120 | 0.000000 | -0.004325 | 0.906027 | worse_than_nomsc |
| fcn | v2d_nomsc | 145.0614 | 0.847445 | 0.004325 | 0.000000 | 0.908951 | v2d_nomsc_reference |
| fcn | v3_fcrg_a | 153.4475 | 0.836096 | -0.007024 | -0.011349 | 0.901944 | worse_than_nomsc |
| fcn | v3_fcrg_b | 157.7131 | 0.845357 | 0.002237 | -0.002088 | 0.908126 | worse_than_nomsc |
| unet | v1_1 | 31.5782 | 0.892461 | 0.000000 | -0.002728 | 0.938098 | worse_than_nomsc |
| unet | v2d_nomsc | 31.6069 | 0.895189 | 0.002728 | 0.000000 | 0.939710 | v2d_nomsc_reference |
| unet | v3_fcrg_a | 32.1014 | 0.888919 | -0.003542 | -0.006270 | 0.936162 | worse_than_nomsc |
| unet | v3_fcrg_b | 32.3649 | 0.893119 | 0.000658 | -0.002070 | 0.938102 | worse_than_nomsc |

## Run Settings

- Architectures: `fcn, unet`.
- Variants: `v1_1, v2d_nomsc, v3_fcrg_a, v3_fcrg_b`.
- Seeds: `2024`.
- Epochs: `50`.
- Batch size: `8`.
- Evaluation uses the local validation split.

## Recommendation Rule

Use `v3_fcrg_b` as the final FA-DCG variant only if it matches or exceeds V2d-noMSC. Use `v3_fcrg_a` only if it clearly exceeds `v3_fcrg_b` and remains stable.
