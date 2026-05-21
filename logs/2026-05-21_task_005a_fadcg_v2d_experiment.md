# 2026-05-21 task_005a FA-DCG V2d 实验记录

## 实验目标

task_005a 验证 Speckle-Robust Frequency Gate。

该实验基于 task_004 的结论：V2a 的保守频率门控对 FCN 有益，V2b 的动态 dilation 对 U-Net 有一定帮助，但 V2a/V2b/V2c 都没有同时超过 V1.1。V2d 尝试在 V2a 的稳定残差路径上进一步区分“有用边界高频”和“speckle-like 高频噪声”。

## 实现要点

- 新增 `FADCGV2d`。
- 保留 V2a 的 depthwise residual enhancement、channel gate 和 residual alpha。
- 将单一 frequency gate 拆为：
  - `boundary_gate`：来自 `local_high` 和 `multi_scale_consistency`。
  - `speckle_gate`：来自 `local_high`、`local_variance_proxy` 和 `multi_scale_consistency`。
- 输出形式：

```text
y = x + alpha * channel_gate(z) * boundary_gate * (1 - beta * speckle_gate) * z
```

- 初始化：
  - `boundary_gate` 初始接近 1.0。
  - `speckle_gate` 初始接近 0.018。
  - `beta` 初始约 0.1。
  - 因此初始 suppression ratio 约 0.998，起点接近 V2a。

## 验证

单元测试：

```text
python scripts/task_005a_test_fadcg_v2d.py
task_005a FA-DCG V2d tests passed
```

完整实验：

```text
python scripts/task_005a_fadcg_v2d_repro.py \
  --output-root exp/task_005a_fadcg_v2d_repro/20260521_fadcg_v2d_seed2024 \
  --models 'FCN + FA-DCG V2d' 'U-Net + FA-DCG V2d' \
  --seeds 2024 \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 2
```

## 结果

| Method | V1.1 mIoU | V2d mIoU | Delta | V1.1 Dice | V2d Dice | Delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FCN + FA-DCG V2d | 0.842804 | 0.845975 | +0.003171 | 0.905853 | 0.908181 | +0.002328 | v2d_better |
| U-Net + FA-DCG V2d | 0.892461 | 0.893472 | +0.001011 | 0.938098 | 0.938617 | +0.000519 | v2d_better |

速度与参数：

| Method | V1.1 latency ms | V2d latency ms | Params M |
| --- | ---: | ---: | ---: |
| FCN | 8.381 | 8.806 | 145.1444 |
| U-Net | 20.338 | 20.714 | 31.6253 |

## 与 task_004 对比

| Method | V2a mIoU | V2b mIoU | V2c mIoU | V2d mIoU | Best |
| --- | ---: | ---: | ---: | ---: | --- |
| FCN | 0.847235 | 0.842661 | 0.846577 | 0.845975 | V2a |
| U-Net | 0.888433 | 0.892354 | 0.891023 | 0.893472 | V2d |

V2d 没有超过 FCN 的 V2a，但仍超过 V1.1。U-Net 上 V2d 是目前最好的频率感知版本，也是第一个超过 U-Net V1.1 的 task_004/task_005 结构。

## 诊断观察

真实验证激活上的诊断：

| Method | Module | boundary gate | speckle gate | suppression ratio | beta |
| --- | --- | ---: | ---: | ---: | ---: |
| FCN | module0 | 0.993897 | 0.018564 | 0.998121 | 0.101224 |
| FCN | module1 | 1.000252 | 0.017964 | 0.998200 | 0.100181 |
| U-Net | module0 | 1.004199 | 0.018123 | 0.998206 | 0.099001 |

speckle suppression 始终非常保守，没有出现强抑制。这个结果支持一个更稳妥的解释：V2d 的收益不是靠粗暴压制高频，而是通过很小的 SAR speckle-aware 校正改善 U-Net 表征。

## 结论

V2d 是值得保留的候选：

- FCN：V2d 超过 V1.1，但不如 V2a，因此 FCN 当前仍推荐 V2a。
- U-Net：V2d 超过 V1.1、V2a、V2b、V2c，是当前 U-Net 最优频率感知版本。

后续 task_005b 应优先测试 boundary-aware auxiliary supervision：

- FCN：可在 V2a 与 V2d 上对比。
- U-Net：优先以 V2d 为结构基线。
