# task_005c Placement Experiment Summary

## Request

Run task_005c as a best insertion-position experiment for FA-DCG V1.1 and V2d.

## Implementation

- Added `SAR_FEM1/models/improved/placement_fadcg.py`.
- Added `scripts/task_005c_placement_repro.py`.
- Tested one FA-DCG block per model to isolate placement.
- FCN placements: after conv3 and after conv4.
- U-Net placements: after encoder stage 2 and after encoder stage 3.
- Blocks: V1.1 `VectorizedLightFADC` and V2d `FADCGV2d`.
- Weak initialization: `alpha_init=0.25`.

## Run

```bash
python scripts/task_005c_placement_repro.py \
  --output-root exp/task_005c_placement_repro/20260521_v1_1_v2d_placement_seed2024 \
  --models 'FCN + V1.1 after conv3' 'FCN + V1.1 after conv4' \
           'FCN + V2d after conv3' 'FCN + V2d after conv4' \
           'U-Net + V1.1 after encoder2' 'U-Net + V1.1 after encoder3' \
           'U-Net + V2d after encoder2' 'U-Net + V2d after encoder3' \
  --seeds 2024 --epochs 50 --batch-size 8 --num-workers 2 --alpha-init 0.25
```

## Results

| Method | mIoU | Dice | Best epoch |
| --- | ---: | ---: | ---: |
| FCN + V1.1 after conv3 | 0.653409 | 0.756980 | 36 |
| FCN + V1.1 after conv4 | 0.651189 | 0.756623 | 48 |
| FCN + V2d after conv3 | 0.660667 | 0.765494 | 48 |
| FCN + V2d after conv4 | 0.653649 | 0.757387 | 49 |
| U-Net + V1.1 after encoder2 | 0.885331 | 0.933787 | 46 |
| U-Net + V1.1 after encoder3 | 0.888685 | 0.936085 | 38 |
| U-Net + V2d after encoder2 | 0.880935 | 0.931215 | 35 |
| U-Net + V2d after encoder3 | 0.888610 | 0.936217 | 41 |

## Best Placements

- FCN + V1.1: after conv3, but far below the accepted original deep V1.1 result.
- FCN + V2d: after conv3, but far below the original deep V2d result.
- U-Net + V1.1: after encoder stage 3, comparable to but below the accepted original bottleneck V1.1 result.
- U-Net + V2d: after encoder stage 3, comparable to but below the original bottleneck V2d result.

## Interpretation

Early/mid placement is not a better main configuration for this dataset.
The ablation supports the paper story that FA-DCG is plug-and-play, but insertion
position is not arbitrary. FCN strongly prefers the original high-level FA-DCG
path, while U-Net can tolerate encoder stage 3 placement but still does not beat
the bottleneck/deep reference.

## Why the Original Bottleneck/Deep Placement Is Best

The result suggests that FA-DCG is most useful after the backbone has already
formed semantic water/non-water representations. In raw or shallow SAR feature
maps, high-frequency responses are a mixture of real flood boundaries, residual
speckle, building shadows, dark non-water surfaces, and acquisition texture.
Applying FA-DCG too early gives the module a weak semantic basis for deciding
which frequency cues are useful, so the residual enhancement can amplify
ambiguous low-level texture rather than flood structure.

The FCN results make this especially clear. FCN has no skip-reconstruction path
and depends heavily on its deepest classifier features. Moving FA-DCG from the
original deep path to a single conv3/conv4 block removes high-level modulation
near the classifier and leaves the network with enhanced mid-level texture but
insufficient semantic correction. This explains why all FCN early/mid variants
fall back near plain-FCN-level performance, despite stable training.

U-Net is more tolerant because its encoder-decoder structure can absorb and
reconstruct mid-level features through skip connections. Encoder stage 3 is
better than encoder stage 2 because it is late enough to contain stronger object
and region context, while still preserving more spatial detail than the
bottleneck. However, it still underperforms the original bottleneck/deep
placement, indicating that the final high-level representation remains the
cleanest location for frequency-aware modulation on this small SAR dataset.

V2d does not change the placement conclusion. Its speckle-robust gate helps
slightly in FCN after-conv3 compared with V1.1 after-conv3, but the large gap to
the original V2d configuration remains. This implies that conservative speckle
suppression cannot fully compensate for missing semantic abstraction at early
feature stages.

The paper-facing conclusion is:

> FA-DCG is plug-and-play, but the optimal insertion point is task- and
> backbone-dependent. For SAR flood segmentation, the bottleneck/deep placement
> provides the best balance between semantic reliability and frequency-aware
> enhancement, while earlier placements are more exposed to speckle and
> low-level texture ambiguity.

This can be presented as a placement ablation supporting the chosen main
architecture rather than as a failed variant.

## Artifacts

- Report: `exp/task_005c_placement_repro/20260521_v1_1_v2d_placement_seed2024/task_005c_report.md`
- Per-seed metrics: `exp/task_005c_placement_repro/20260521_v1_1_v2d_placement_seed2024/metrics_per_seed.csv`
- Summary metrics: `exp/task_005c_placement_repro/20260521_v1_1_v2d_placement_seed2024/metrics_summary.csv`
- Best-placement summary: `exp/task_005c_placement_repro/20260521_v1_1_v2d_placement_seed2024/best_placement_summary.csv`
- Diagnostics: `exp/task_005c_placement_repro/20260521_v1_1_v2d_placement_seed2024/figures/fadcg_v2_diagnostics/`
