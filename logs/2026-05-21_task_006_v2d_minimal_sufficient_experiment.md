# 2026-05-21 task_006 V2d 最小充分结构实验记录

## 实验目标

task_006 目标是寻找最小充分 V2d：在不引入新模块、不引入 V1.1/V2a 论文叙事的前提下，通过删减 V2d 内部组件判断 full V2d 是否过度设计。

Full V2d 形式为：

```text
y = x + alpha * channel_gate * boundary_gate * (1 - beta * speckle_gate) * z
```

本轮测试组件包括：

- channel gate
- boundary gate
- speckle gate
- boundary + speckle 双频率门控
- all gates
- multi-scale consistency descriptor
- local variance descriptor

## 实现与验证

新增文件：

```text
SAR_FEM1/models/improved/fadcg_v2d_ablation.py
SAR_FEM1/models/improved/fcn_fadcg_v2d_ablation.py
SAR_FEM1/models/improved/unet_fadcg_v2d_ablation.py
scripts/task_006_test_v2d_ablation.py
scripts/task_006_v2d_ablation_repro.py
```

单元测试：

```text
python scripts/task_006_test_v2d_ablation.py
task_006 FA-DCG V2d ablation tests passed
```

完整实验命令：

```text
python scripts/task_006_v2d_ablation_repro.py \
  --output-root exp/task_006_v2d_ablation_repro/20260521_seed2024 \
  --architectures fcn unet \
  --variants full no_channel no_boundary no_speckle \
             no_boundary_no_speckle no_all_gates no_msc no_local_var \
  --seeds 2024 \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 2
```

## 结果

| Architecture | Variant | Params M | mIoU | Delta vs Full | Dice | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| FCN | full | 145.1444 | 0.845375 | 0.000000 | 0.907630 | reference |
| FCN | no_channel | 136.6189 | 0.836355 | -0.009020 | 0.901098 | important_drop |
| FCN | no_boundary | 145.0568 | 0.843355 | -0.002020 | 0.906143 | slight_drop |
| FCN | no_speckle | 145.0154 | 0.843500 | -0.001875 | 0.906690 | slight_drop |
| FCN | no_boundary_no_speckle | 144.9278 | 0.844877 | -0.000498 | 0.907139 | equivalent |
| FCN | no_all_gates | 136.4024 | 0.840829 | -0.004546 | 0.904586 | important_drop |
| FCN | no_msc | 145.0614 | 0.848363 | +0.002988 | 0.910157 | better |
| FCN | no_local_var | 145.1029 | 0.845585 | +0.000210 | 0.907911 | equivalent |
| U-Net | full | 31.6253 | 0.893472 | 0.000000 | 0.938617 | reference |
| U-Net | no_channel | 31.0997 | 0.889718 | -0.003754 | 0.936340 | important_drop |
| U-Net | no_boundary | 31.6058 | 0.894634 | +0.001162 | 0.939136 | better |
| U-Net | no_speckle | 31.5966 | 0.888433 | -0.005039 | 0.935235 | important_drop |
| U-Net | no_boundary_no_speckle | 31.5772 | 0.892461 | -0.001011 | 0.938098 | slight_drop |
| U-Net | no_all_gates | 31.0516 | 0.890295 | -0.003177 | 0.936652 | important_drop |
| U-Net | no_msc | 31.6069 | 0.895189 | +0.001717 | 0.939710 | better |
| U-Net | no_local_var | 31.6161 | 0.897975 | +0.004503 | 0.941652 | better |

## 主要观察

### 1. Channel gate 是必要组件

去掉 channel gate 后两个骨干都明显下降：

```text
FCN:   -0.009020 mIoU
U-Net: -0.003754 mIoU
```

这说明 V2d 的稳定收益并不只是来自 depthwise residual enhancement，也不是频率门控单独完成的。通道响应选择仍然是核心组件。

### 2. 只保留 depthwise residual 不够

`no_all_gates` 明显低于 full：

```text
FCN:   -0.004546 mIoU
U-Net: -0.003177 mIoU
```

说明 V2d 不是单纯靠 depthwise residual enhancement 起作用。

### 3. Multi-scale consistency 可能是过度设计

`no_msc` 在两个骨干上都优于 full：

```text
FCN:   +0.002988 mIoU
U-Net: +0.001717 mIoU
```

这是本轮最稳定的简化方向。当前证据表明，`multi_scale_consistency = abs(avg3 - avg7)` 并非必要，反而可能给小样本训练引入额外扰动。

### 4. Boundary gate 的必要性较弱

U-Net 去掉 boundary gate 后反而提升：

```text
U-Net no_boundary: +0.001162 mIoU
```

FCN 去掉 boundary gate 会轻微下降：

```text
FCN no_boundary: -0.002020 mIoU
```

因此 boundary gate 不是稳定必需项。它可能对 FCN 有轻微帮助，但对 U-Net 可能冗余。

### 5. Speckle gate 对 U-Net 很重要

U-Net 去掉 speckle gate 后明显下降：

```text
U-Net no_speckle: -0.005039 mIoU
```

FCN 也有轻微下降：

```text
FCN no_speckle: -0.001875 mIoU
```

这支持 speckle-aware correction 的论文叙事，但要注意：该 correction 不一定需要 multi-scale consistency。

### 6. Local variance 的结论不完全一致

U-Net 去掉 local variance 反而大幅提升：

```text
U-Net no_local_var: +0.004503 mIoU
```

FCN 去掉 local variance 基本等价：

```text
FCN no_local_var: +0.000210 mIoU
```

这说明 full V2d 中 local variance 也可能不是稳定必要项。若后续选择最终模型，`no_local_var` 值得作为候选，但它在 FCN 上不如 `no_msc`。

## 当前结论

Full V2d 存在过度设计。最稳定的最小充分候选是：

```text
V2d-noMSC
```

理由：

- FCN 和 U-Net 都超过 full V2d。
- 参数量略低于 full。
- 保留 channel gate、boundary gate、speckle gate 和 local variance。
- 只删除 multi-scale consistency，结构解释仍然完整。
- 两个骨干的方向一致，适合论文统一主模型。

备选候选：

```text
V2d-noLocalVar
```

理由：

- U-Net 提升最大。
- FCN 与 full 基本等价。

但它会削弱“local variance 建模 speckle-like 高频”的 SAR-specific 叙事。因此除非后续多 seed 继续支持它，否则不建议优先作为最终论文主模型。

## 下一步建议

建议将 `V2d-noMSC` 作为最小充分 V2d 候选，进行多 seed 确认：

```text
FCN + V2d-noMSC, seeds 42/123/2024
U-Net + V2d-noMSC, seeds 42/123/2024
```

同时保留 full V2d 和 `no_local_var` 作为对照：

- full V2d：证明简化版优于原设计。
- no_local_var：验证 U-Net 上的高提升是否稳定，避免单 seed 偶然性。

论文目前不应直接写 full V2d 为最终版本。更稳妥的方向是：

```text
通过 task_006 消融，发现 multi-scale consistency 不是必要项，因此采用更简洁的 V2d-noMSC 作为最终 FA-DCG V2d。
```

## 产物

```text
exp/task_006_v2d_ablation_repro/20260521_seed2024/metrics_per_seed.csv
exp/task_006_v2d_ablation_repro/20260521_seed2024/metrics_summary.csv
exp/task_006_v2d_ablation_repro/20260521_seed2024/task_006_report.md
exp/task_006_v2d_ablation_repro/20260521_seed2024/figures/fadcg_v2d_ablation_diagnostics/
```
