# 2026-05-22 task_006 V2d-noMSC 多 seed 补充实验

## 实验目标

在 task_006 单 seed 消融确定 `V2d-noMSC` 是最稳定的最小充分候选后，补充 seeds 42 和 123，形成 `42/123/2024` 三 seed 结果。

本次补充只跑最终候选：

```text
FCN + V2d-noMSC
U-Net + V2d-noMSC
```

## 实验命令

```text
python scripts/task_006_v2d_ablation_repro.py \
  --output-root exp/task_006_v2d_nomsc_multiseed/20260522_nomsc_seeds42_123 \
  --architectures fcn unet \
  --variants no_msc \
  --seeds 42 123 \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 2
```

seed2024 结果来自 task_006 主实验：

```text
exp/task_006_v2d_ablation_repro/20260521_seed2024/metrics_per_seed.csv
```

## Per-Seed Results

| Architecture | Seed | mIoU | Dice | Params M | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| FCN | 42 | 0.835672 | 0.901421 | 145.0614 | 42 |
| FCN | 123 | 0.846960 | 0.909384 | 145.0614 | 43 |
| FCN | 2024 | 0.848363 | 0.910157 | 145.0614 | 41 |
| U-Net | 42 | 0.888316 | 0.935594 | 31.6069 | 38 |
| U-Net | 123 | 0.893497 | 0.938788 | 31.6069 | 45 |
| U-Net | 2024 | 0.895189 | 0.939710 | 31.6069 | 33 |

## Three-Seed Summary

| Architecture | mIoU mean | mIoU std | Dice mean | Dice std | Params M |
| --- | ---: | ---: | ---: | ---: | ---: |
| FCN + V2d-noMSC | 0.843665 | 0.006958 | 0.906987 | 0.004836 | 145.0614 |
| U-Net + V2d-noMSC | 0.892334 | 0.003581 | 0.938031 | 0.002160 | 31.6069 |

## Interpretation

V2d-noMSC 的三 seed 结果显示：

- FCN 上 seed42 明显低于 seed123/2024，但三 seed 均值仍达到 `0.843665`。
- U-Net 上三 seed 更稳定，mIoU std 为 `0.003581`。
- seed2024 单次结果中 V2d-noMSC 同时超过 full V2d；多 seed 后需要避免只用 seed2024 的提升幅度做强结论。

更稳妥的论文表述应是：

```text
The task_006 ablation identifies multi-scale consistency as a redundant
descriptor in the current validation setting. The resulting V2d-noMSC variant
retains competitive and stable performance across three seeds, while using a
simpler frequency-selection design.
```

中文表述：

```text
消融实验表明，跨尺度一致性描述符在当前 SAR 洪水分割设置中并非必要。删除该描述符后的 V2d-noMSC 在三 seed 复现实验中保持稳定性能，并具有更简洁的频率选择结构。
```

## Artifacts

```text
exp/task_006_v2d_nomsc_multiseed/20260522_nomsc_seeds42_123/metrics_per_seed.csv
exp/task_006_v2d_nomsc_multiseed/20260522_nomsc_seeds42_123/metrics_summary.csv
exp/task_006_v2d_nomsc_multiseed/20260522_nomsc_seeds42_123/task_006_report.md
```
