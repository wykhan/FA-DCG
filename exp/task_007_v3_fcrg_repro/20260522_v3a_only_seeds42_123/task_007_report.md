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
| fcn | v3_fcrg_a | 137.5212 | 0.836892 | -0.005912 | -0.011471 | 0.901653 | worse_than_nomsc |
| unet | v3_fcrg_a | 31.3148 | 0.885474 | -0.006987 | -0.009715 | 0.933752 | worse_than_nomsc |

## Run Settings

- Architectures: `fcn, unet`.
- Variants: `v3_fcrg_a`.
- Seeds: `42, 123`.
- Epochs: `50`.
- Batch size: `8`.
- Evaluation uses the local validation split.

## Recommendation Rule

Use `v3_fcrg_b` as the final FA-DCG variant only if it matches or exceeds V2d-noMSC. Use `v3_fcrg_a` only if it clearly exceeds `v3_fcrg_b` and remains stable.
