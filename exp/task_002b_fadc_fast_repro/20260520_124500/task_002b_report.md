# task_002b Fast FA-DCG-v1 Report

## Executive Summary

- Implemented vectorized Fast FA-DCG-v1 without deleting the old LightFADC implementations.
- Replaced old U-Net bottleneck LightFADC with FastFADCG.
- Replaced the old FCN 512/4096-channel FA-DCG path with a single FastFADCG after conv5 at 512 channels.
- Conditional Stage B was completed: both optimized models were run for seeds 42, 123, and 2024.
- Evaluation remains the current validation protocol; this is not independent test-set reproduction.

## Why the Old Implementation Was Inefficient

- Old LightFADC looped over channels in Python and converted learned dilation preferences through `.item()` into integer dilations.
- Old FCN added FA-DCG-like processing at a 4096-channel stage and lazily created `up_dim` inside `forward`.

## New Implementation Details

- FastFADCG uses four vectorized depthwise branches with dilation rates 1, 2, 3, and 4.
- Channel-wise branch weights are differentiable: `softmax(theta - beta * abs(d_cont - d_k))`.
- The channel gate remains a lightweight GAP + 1x1 MLP + sigmoid gate.

## Implementation Audit

| Issue | Old implementation | New implementation | Fixed? |
| --- | --- | --- | --- |
| per-channel Python loop | yes | no | yes |
| `.item()` integer dilation | yes | no | yes |
| 4096-channel FA-DCG in FCN | yes | no | yes |
| module created in forward | yes | no | yes |
| differentiable branch fusion | no | yes | yes |

## Speed Benchmark Results

| Model | Old time(ms) | New time(ms) | Speedup | Old Params(M) | New Params(M) | Notes |
| ----- | -----------: | -----------: | ------: | ------------: | ------------: | ----- |
| FCN + Fast FA-DCG | 217.030 | 9.972 | 21.76x | 144.9324 | 134.3268 | same full-batch benchmark |
| U-Net + Fast FA-DCG | 67.032 | 20.512 | 3.27x | 31.5782 | 31.2175 | same full-batch benchmark |

## Training Results

| Model | Old mIoU | New mIoU | Delta mIoU | Old Dice | New Dice | Delta Dice | Status |
| ----- | -------: | -------: | ---------: | -------: | -------: | ---------: | ------ |
| FCN + Fast FA-DCG | 0.844849 | 0.653098 | -0.191751 | 0.907507 | 0.761202 | -0.146305 | faster_but_worse |
| U-Net + Fast FA-DCG | 0.896663 | 0.881403 | -0.015260 | 0.941023 | 0.931304 | -0.009719 | faster_but_worse |

## Comparison With task_002 Old FA-DCG Results

- Old-reference values are reused from task_002 rather than rerun.
- FLOPs are recorded as `NA`; no FLOPs dependency was introduced.

## Recommendation for task_003

- Use FastFADCG as the engineering baseline for task_003 module work because it fixes the implementation defects and is much faster in the controlled benchmark.
- Do not treat the corrected FCN integration as a drop-in accuracy replacement for the old FCN + FA-DCG result; its validation mIoU dropped substantially.
- U-Net + Fast FA-DCG is the better candidate for continued development: it is much faster than the old U-Net FA-DCG benchmark and only moderately below the task_002 old-reference mIoU.
- Keep the old implementation as historical reference.
- Continue to avoid independent-test claims until a test split and event metadata are available.
