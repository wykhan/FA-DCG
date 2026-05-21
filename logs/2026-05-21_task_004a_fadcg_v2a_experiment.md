# 2026-05-21 task_004a FA-DCG V2a 实验记录

## 实验目标

task_004a 验证 Candidate A：SAR Frequency-Gated V1.1。

该版本以 FA-DCG V1.1 为稳定基线，保留深度可分离残差增强、通道门控和残差缩放结构，只加入一个轻量 SAR 频率门控。频率描述符由局部高频残差和多尺度一致性组成，用于调制残差增强强度，而不是替换 V1.1 的主路径。

## 实现要点

- 新增 `SARFrequencyGate`：使用 `avg3`、`avg7` 构造 `local_high = abs(x - avg3(x))` 和 `multi_scale_consistency = abs(avg3(x) - avg7(x))`。
- 新增 `FADCGV2a`：输出为 `x + alpha * channel_gate(z) * frequency_gate(x) * z`。
- 频率门控以 identity 方式初始化，初始输出接近 1.0，训练起点接近 V1.1。
- FCN 保持 V1.1 的 `fc6_light` 与 `fc7_light` 插入位置。
- U-Net 保持 V1.1 的 bottleneck 插入位置。
- 诊断采集改为真实验证 batch 的前向输入，不再使用零输入占位。

## 验证

单元测试：

```text
python scripts/task_004a_test_fadcg_v2a.py
task_004a FA-DCG V2a tests passed
```

完整实验：

```text
python scripts/task_004a_fadcg_v2a_repro.py \
  --output-root exp/task_004a_fadcg_v2a_repro/20260521_fadcg_v2a_seed2024 \
  --models 'FCN + FA-DCG V2a' 'U-Net + FA-DCG V2a' \
  --seeds 2024 \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 2
```

## 结果

| Method | V1.1 mIoU | V2a mIoU | Delta | V1.1 Dice | V2a Dice | Delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FCN + FA-DCG V2a | 0.842804 | 0.847235 | +0.004431 | 0.905853 | 0.909229 | +0.003376 | v2a_better |
| U-Net + FA-DCG V2a | 0.892461 | 0.888433 | -0.004028 | 0.938098 | 0.935235 | -0.002863 | v2a_comparable |

速度与参数：

| Method | V1.1 latency ms | V2a latency ms | Params M |
| --- | ---: | ---: | ---: |
| FCN | 8.326 | 8.585 | 145.0154 |
| U-Net | 20.290 | 20.494 | 31.5966 |

参数增量约为 0.06%，速度开销约为 FCN +3.1%、U-Net +1.0%。

## 诊断观察

真实验证激活上的频率门控均值接近 1：

- FCN module0 `frequency_gate_mean = 0.9948`，module1 `frequency_gate_mean = 1.0000`。
- U-Net module0 `frequency_gate_mean = 1.0044`。

这说明 V2a 没有激进改变 V1.1 的残差路径，而是做了保守的频率调制。FCN 获得小幅提升；U-Net 略低于 V1.1，但差距在 0.005 mIoU 以内，属于可比结果。

## 结论

task_004a 证明了“在 V1.1 上加入轻量 SAR 频率门控”是可行方向：它几乎不增加参数和延迟，并且在 FCN 上带来小幅提升。由于 U-Net 未超过 V1.1，004b 不宜直接扩大结构复杂度，应优先围绕频率门控的位置、强度和动态感受野偏置做受控消融。
