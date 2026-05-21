# 2026-05-21 task_004c FA-DCG V2c 实验记录

## 实验目标

task_004c 用于收敛 task_004a 和 task_004b 的实验发现，验证一个“按骨干选择结构”的 FA-DCG V2.0 候选。

设计假设：

- FCN 对动态 dilation 更敏感，因此保留 task_004a 中表现最好的 V2a 保守频率门控。
- U-Net 能更好吸收动态感受野，因此保留 task_004b 中表现更好的 V2b bottleneck。
- 额外验证 Candidate D 思路：在 U-Net 最高层 skip 特征上加入一个低强度 V2a 频率校准块，观察是否能进一步改善边界和小水体。

## 实现要点

- `FCNWithFADCGV2c`：结构等同 FCN + V2a。
- `UNetWithFADCGV2c`：bottleneck 使用 V2b，最高层 skip 使用 V2a 频率校准。
- skip 频率校准的 `alpha_init = 0.25`，限制早期扰动。
- 诊断同时采集 V2a 与 V2b 模块。

## 验证

单元测试：

```text
python scripts/task_004c_test_fadcg_v2c.py
task_004c FA-DCG V2c tests passed
```

完整实验：

```text
python scripts/task_004c_fadcg_v2c_repro.py \
  --output-root exp/task_004c_fadcg_v2c_repro/20260521_fadcg_v2c_seed2024 \
  --models 'FCN + FA-DCG V2c' 'U-Net + FA-DCG V2c' \
  --seeds 2024 \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 2
```

## 结果

| Method | V1.1 mIoU | V2c mIoU | Delta | V1.1 Dice | V2c Dice | Delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FCN + FA-DCG V2c | 0.842804 | 0.846577 | +0.003773 | 0.905853 | 0.908685 | +0.002832 | v2c_better |
| U-Net + FA-DCG V2c | 0.892461 | 0.891023 | -0.001438 | 0.938098 | 0.936816 | -0.001282 | v2c_comparable |

速度与参数：

| Method | V1.1 latency ms | V2c latency ms | Params M |
| --- | ---: | ---: | ---: |
| FCN | 8.376 | 8.524 | 145.0154 |
| U-Net | 20.372 | 21.453 | 31.8010 |

## 与 004a/004b 对比

| Method | V2a mIoU | V2b mIoU | V2c mIoU | Best |
| --- | ---: | ---: | ---: | --- |
| FCN | 0.847235 | 0.842661 | 0.846577 | V2a |
| U-Net | 0.888433 | 0.892354 | 0.891023 | V2b |

## 诊断观察

U-Net V2c 的 bottleneck V2b 分支权重：

| Module | d1 | d2 | d3 | large dilation |
| --- | ---: | ---: | ---: | ---: |
| bottleneck | 0.858151 | 0.124935 | 0.016914 | 0.141849 |

skip 校准模块的 `alpha` 从 0.25 增长到 0.285436，说明模型确实使用了该模块，但最终指标没有超过 V2b。

## 结论

task_004c 的结论是：skip 前移校准不是当前最优方向。最稳妥的 task_004 结论应是 backbone-aware selection：

- FCN 推荐 V2a：保守 SAR 频率门控，获得最佳 FCN 指标。
- U-Net 推荐 V2b：频率偏置动态 dilation，获得最佳 U-Net 指标并接近 V1.1。

如果论文中需要一个统一的 FA-DCG V2.0 名称，可以将 V2.0 定义为“面向骨干的频率感知 FA-DCG 框架”，其中 FCN 实例使用 V2a，U-Net 实例使用 V2b。不建议把 skip-calibrated V2c 作为最终主结果。
