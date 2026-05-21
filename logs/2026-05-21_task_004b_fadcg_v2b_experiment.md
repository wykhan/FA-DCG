# 2026-05-21 task_004b FA-DCG V2b 实验记录

## 实验目标

task_004b 验证 Candidate B：Frequency-Biased Dilation Preference。

该版本在 V2a 的 SAR 频率门控基础上，加入轻量动态感受野分支。核心问题是：频率线索是否能帮助模型在小感受野边界增强和大感受野区域上下文之间做更合理的分配。

## 实现要点

- 新增 `FADCGV2b`。
- 保留 V2a 的 `SARFrequencyGate`。
- 新增 `FrequencyDilationBias`，从 `[local_high, multi_scale_consistency]` 频率描述符生成 dilation 分支权重。
- 分支 dilation 为 `[1, 2, 3]`。
- 三个 dilation 分支共享同一个 depthwise kernel，避免明显增加参数。
- 分支 logits 初始化为 `[2, 0, -2]`，使初始权重强烈偏向 dilation=1，避免 task_003 中动态感受野过强导致 FCN 退化。

## 验证

单元测试：

```text
python scripts/task_004b_test_fadcg_v2b.py
task_004b FA-DCG V2b tests passed
```

完整实验：

```text
python scripts/task_004b_fadcg_v2b_repro.py \
  --output-root exp/task_004b_fadcg_v2b_repro/20260521_fadcg_v2b_seed2024 \
  --models 'FCN + FA-DCG V2b' 'U-Net + FA-DCG V2b' \
  --seeds 2024 \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 2
```

## 结果

| Method | V1.1 mIoU | V2b mIoU | Delta | V1.1 Dice | V2b Dice | Delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FCN + FA-DCG V2b | 0.842804 | 0.842661 | -0.000143 | 0.905853 | 0.906207 | +0.000354 | v2b_comparable |
| U-Net + FA-DCG V2b | 0.892461 | 0.892354 | -0.000107 | 0.938098 | 0.937855 | -0.000243 | v2b_comparable |

速度与参数：

| Method | V1.1 latency ms | V2b latency ms | Params M |
| --- | ---: | ---: | ---: |
| FCN | 8.354 | 9.125 | 145.2780 |
| U-Net | 20.358 | 20.989 | 31.6550 |

参数增量仍较小：FCN 约 +0.3456M，U-Net 约 +0.0768M。速度开销约为 FCN +9.2%、U-Net +3.1%。

## 诊断观察

真实验证激活上的 dilation 分支权重：

| Method | Module | d1 | d2 | d3 | large dilation |
| --- | --- | ---: | ---: | ---: | ---: |
| FCN | module0 | 0.863734 | 0.120128 | 0.016137 | 0.136266 |
| FCN | module1 | 0.866813 | 0.117310 | 0.015876 | 0.133187 |
| U-Net | module0 | 0.856803 | 0.126120 | 0.017077 | 0.143197 |

分支权重没有退化为均匀分布，也没有让大 dilation 失控。整体仍以 dilation=1 为主，dilation=2/3 提供小比例上下文补充。

## 与 V2a 的关系

V2a 结果：

| Method | V2a mIoU | V2a Dice |
| --- | ---: | ---: |
| FCN + FA-DCG V2a | 0.847235 | 0.909229 |
| U-Net + FA-DCG V2a | 0.888433 | 0.935235 |

V2b 相比 V2a：

- FCN 从 0.847235 降到 0.842661，说明 FCN 对 dilation 分支仍然敏感，V2a 更适合 FCN。
- U-Net 从 0.888433 提升到 0.892354，几乎追平 V1.1，说明动态感受野更适合 U-Net 的 bottleneck + skip 结构。

## 结论

task_004b 证明频率偏置动态感受野是可行但需要克制的方向。它没有造成 task_003 式 FCN 坍塌，并且让 U-Net 明显优于 V2a、接近 V1.1。后续 task_004c 更适合围绕 U-Net 结构做精修，例如将频率校准前移到 skip/中层特征，或在 V2a/V2b 基础上加入轻量 speckle suppression。
