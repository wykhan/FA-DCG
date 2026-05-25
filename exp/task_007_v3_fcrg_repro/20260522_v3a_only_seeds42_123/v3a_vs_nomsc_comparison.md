# V3-FCRG-A vs V2d-noMSC Multi-Seed Comparison

## Sources

V3-FCRG-A seeds 42/123:

```text
exp/task_007_v3_fcrg_repro/20260522_v3a_only_seeds42_123/metrics_per_seed.csv
```

V3-FCRG-A seed 2024 and V2d-noMSC seed 2024:

```text
exp/task_007_v3_fcrg_repro/20260522_seed2024_light/metrics_per_seed.csv
```

V2d-noMSC seeds 42/123:

```text
exp/task_006_v2d_nomsc_multiseed/20260522_nomsc_seeds42_123/metrics_per_seed.csv
```

## Per-Seed Results

| Architecture | Seed | V2d-noMSC mIoU | V3-FCRG-A mIoU | Delta | V2d-noMSC Dice | V3-FCRG-A Dice | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FCN | 42 | 0.835672 | 0.834479 | -0.001193 | 0.901421 | 0.899999 | -0.001422 |
| FCN | 123 | 0.846960 | 0.839305 | -0.007655 | 0.909384 | 0.903307 | -0.006077 |
| FCN | 2024 | 0.849188 | 0.845847 | -0.003341 | 0.910095 | 0.908071 | -0.002024 |
| U-Net | 42 | 0.888316 | 0.881377 | -0.006939 | 0.935594 | 0.931411 | -0.004183 |
| U-Net | 123 | 0.893497 | 0.889572 | -0.003925 | 0.938788 | 0.936093 | -0.002695 |
| U-Net | 2024 | 0.895189 | 0.897218 | +0.002029 | 0.939710 | 0.941095 | +0.001385 |

## Three-Seed Summary

| Architecture | Variant | mIoU Mean | mIoU Std | Dice Mean | Dice Std | Params M |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FCN | V2d-noMSC | 0.843940 | 0.007247 | 0.906967 | 0.004811 | 145.0614 |
| FCN | V3-FCRG-A | 0.839877 | 0.005705 | 0.903792 | 0.004051 | 137.5212 |
| U-Net | V2d-noMSC | 0.892334 | 0.003580 | 0.938031 | 0.002149 | 31.6069 |
| U-Net | V3-FCRG-A | 0.889389 | 0.007922 | 0.936200 | 0.004849 | 31.3148 |

## Interpretation

V3-FCRG-A does not beat V2d-noMSC under three-seed confirmation.

For FCN:

```text
V3-FCRG-A mean mIoU is 0.004063 lower than V2d-noMSC.
```

For U-Net:

```text
V3-FCRG-A mean mIoU is 0.002945 lower than V2d-noMSC.
```

The seed-2024 U-Net improvement is not stable across seeds 42 and 123. V3-FCRG-A remains parameter-efficient, especially on FCN, but the accuracy tradeoff is not favorable enough to replace V2d-noMSC as the final method.

## Recommendation

Do not select V3-FCRG-A as the final paper method.

Keep **V2d-noMSC** as the stronger final candidate for now. If the paper story still needs a unified response-gate framing, it is better to revise the explanation around V2d-noMSC rather than replace it with V3-FCRG-A.

