# task_005: Next FA-DCG Experiments after V2a/V2b/V2c

## Background

task_004 explored three frequency-aware FA-DCG variants:

1. **V2a: SAR Frequency-Gated V1.1**
   - FCN: mIoU 0.847235, Dice 0.909229
   - U-Net: mIoU 0.888433, Dice 0.935235
2. **V2b: Frequency-Biased Dilation Preference**
   - FCN: mIoU 0.842661, Dice 0.906207
   - U-Net: mIoU 0.892354, Dice 0.937855
3. **V2c: Backbone-Aware Selected Configuration with U-Net Skip Calibration**
   - FCN: mIoU 0.846577, Dice 0.908685
   - U-Net: mIoU 0.891023, Dice 0.936816

The accepted baseline remains **FA-DCG V1.1**:

```text
FCN + FA-DCG V1.1:
  mIoU = 0.842804
  Dice = 0.905853

U-Net + FA-DCG V1.1:
  mIoU = 0.892461
  Dice = 0.938098
```

## Lessons from task_004

### 1. Frequency awareness is useful for the story, but not sufficient alone

V2a/V2b/V2c make the model more interpretable from a frequency-aware SAR
perspective, but none of them clearly surpasses V1.1 on both FCN and U-Net.

The strongest task_004 findings are:

```text
FCN prefers V2a:
  conservative frequency gating works better than dynamic dilation.

U-Net prefers V2b:
  frequency-biased dilation is useful, but only when the architecture can absorb
  it through bottleneck context and skip reconstruction.
```

### 2. Continuing to add complex dilation branches is not the best direction

V2b did not collapse, but it also did not create a decisive gain. V2c showed
that adding a skip frequency block did not improve over V2b.

Therefore task_005 should avoid:

- more dilation branches;
- larger dynamic kernels;
- official FADC-style heavy adaptive convolution;
- multi-branch designs without a direct SAR error target.

### 3. The remaining gap is task-specific SAR error handling

The likely unresolved errors are:

- residual speckle after Lee filtering;
- weak or noisy water-land boundaries;
- small water bodies and narrow channels;
- false alarms from building shadows or dark non-water regions;
- insufficient boundary supervision from the mask-only BCE objective.

task_005 should target these issues directly.

## Overall task_005 Goal

Design experiments that preserve the useful frequency-aware narrative from
task_004 while targeting SAR flood segmentation failure modes more directly.

The main question:

```text
Can FA-DCG improve beyond V1.1 by distinguishing useful boundary high-frequency
signals from SAR speckle/noise, or by supervising boundary-sensitive frequency
features more explicitly?
```

## Recommended Experiment Sequence

```text
task_005a = Speckle-Robust Frequency Gate
task_005b = Boundary-Aware Auxiliary Supervision
task_005c = Early/Mid Placement Ablation
```

Run task_005a first.

Reason:

- it is the smallest structural change after V2a;
- it directly addresses SAR-specific speckle and false alarms;
- it preserves FCN stability better than dynamic dilation;
- it gives a strong paper narrative: frequency awareness is not just attention,
  but speckle-robust useful-frequency selection.

## task_005a: Speckle-Robust Frequency Gate

### Purpose

V2a uses frequency information but does not distinguish useful high-frequency
boundaries from noisy high-frequency speckle.

task_005a should build a **Speckle-Robust Frequency Gate** on top of V2a.

### Core Idea

```text
useful_boundary_gate = useful high-frequency / multi-scale consistency
speckle_suppression_gate = unstable local variance / isolated high-frequency noise

z = depthwise residual enhancement
y = x + alpha * channel_gate(z) * boundary_gate * (1 - beta * speckle_gate) * z
```

### Suggested Structure

Start from V2a:

```text
local_low = avgpool3(x)
local_context = avgpool7(x)
local_high = abs(x - local_low)
multi_scale_consistency = abs(local_low - local_context)
```

Add a lightweight variance proxy:

```text
local_var = avgpool3(x * x) - avgpool3(x) * avgpool3(x)
speckle_proxy = local_high * normalize(local_var)
```

Two gates:

```text
boundary_gate = sigmoid(depthwise_head([local_high, multi_scale_consistency])) * 2
speckle_gate = sigmoid(depthwise_head([local_high, local_var, multi_scale_consistency]))
```

Final residual:

```text
y = x + alpha * channel_gate(z) * boundary_gate * (1 - beta * speckle_gate) * z
```

Initialization:

```text
boundary_gate starts near 1.0
speckle_gate starts near 0.0
beta starts small, e.g. 0.1 or learnable sigmoid-scaled value
alpha starts at 0.5
```

Important:

- The module must start close to V2a/V1.1.
- Speckle suppression must be conservative.
- Avoid suppressing all high-frequency information, because true flood
  boundaries are also high-frequency.

### Suggested Names

```text
FADCGV2d
FCNWithFADCGV2d
UNetWithFADCGV2d
```

### Suggested Files

```text
SAR_FEM1/models/improved/fadcg_v2d.py
SAR_FEM1/models/improved/fcn_fadcg_v2d.py
SAR_FEM1/models/improved/unet_fadcg_v2d.py
scripts/task_005a_test_fadcg_v2d.py
scripts/task_005a_fadcg_v2d_repro.py
```

### Mandatory Experiment

| Model | Seed | Epochs |
| --- | ---: | ---: |
| FCN + FA-DCG V2d | 2024 | 50 |
| U-Net + FA-DCG V2d | 2024 | 50 |

### Required Diagnostics

Save under:

```text
figures/fadcg_v2_diagnostics/
```

Required fields:

```text
boundary_gate_mean
boundary_gate_std
speckle_gate_mean
speckle_gate_std
suppression_ratio
local_high_descriptor_mean
local_variance_proxy_mean
multi_scale_consistency_mean
alpha_value
beta_value
channel_gate_mean
channel_gate_std
```

### Success Criteria

At minimum:

- no NaNs;
- no FCN collapse;
- latency no worse than 1.5x V1.1;
- parameter increase remains small;
- mIoU should match V1.1 within 0.005 on both FCN and U-Net.

Strong success:

```text
FCN improves over V2a or V1.1
and/or
U-Net improves over V2b or V1.1
```

### Interpretation

If V2d improves:

```text
SAR frequency awareness must separate boundary frequency from speckle-like
frequency noise.
```

If V2d does not improve:

```text
The dataset may already be sufficiently denoised by preprocessing, or the model
needs stronger supervision rather than another feature gate.
```

## task_005b: Boundary-Aware Auxiliary Supervision

### Purpose

V2a/V2b add frequency-aware modules, but the training objective is still mainly
mask-level BCE. The model may not receive enough direct supervision to learn
boundary-sensitive frequency features.

task_005b should test whether a lightweight boundary objective makes frequency
features useful.

### Core Idea

Keep the model structure conservative, then add boundary supervision:

```text
main_loss = BCEWithLogitsLoss(mask_logits, mask)
boundary_loss = BCE/Dice(boundary_logits, boundary_target)
total_loss = main_loss + lambda_boundary * boundary_loss
```

### Boundary Target Generation

Generate boundary target from the binary mask using local morphology:

```text
boundary = dilate(mask) - erode(mask)
```

or with pooling:

```text
max_pool(mask) - min_pool(mask)
```

Use a small width first:

```text
kernel_size = 3
```

### Model Options

Start with one of:

```text
FCN + V2a + boundary head
U-Net + V2b + boundary head
```

The boundary head should be small:

```text
selected feature -> 1x1 conv -> boundary logits
```

Do not add a large decoder.

### Suggested Names

```text
FADCGV2Boundary
task_005b_boundary_supervision
```

### Mandatory Experiment

| Model | Seed | Epochs |
| --- | ---: | ---: |
| FCN + selected FA-DCG + boundary loss | 2024 | 50 |
| U-Net + selected FA-DCG + boundary loss | 2024 | 50 |

Start with:

```text
lambda_boundary = 0.1
```

Optional if stable:

```text
lambda_boundary = 0.2
```

### Required Diagnostics

```text
boundary_loss
main_loss
boundary_precision
boundary_recall
boundary_f1
mask_miou
mask_dice
```

### Success Criteria

Strong success:

- U-Net mIoU/Dice surpasses V1.1;
- FCN does not lose V2a's improvement;
- boundary diagnostics improve without reducing water recall.

### Interpretation

If boundary supervision improves:

```text
The frequency-aware module needed a boundary-sensitive learning signal.
```

If it does not:

```text
Boundary error may not be the dominant metric bottleneck, or generated boundary
targets may be too noisy for this dataset.
```

## task_005c: Early/Mid Placement Ablation

### Purpose

The user pointed out an important architectural concern:

```text
FADC is intended to affect feature extraction, so the insertion position should
be early or middle rather than only deep/bottleneck.
```

V2c tested U-Net highest-level skip calibration, but not true early/mid feature
placement.

task_005c should isolate insertion position.

### Core Idea

Do not invent a new module. Use the most stable block:

```text
V2a-small
```

Insert it at one early/mid location only.

### Candidate Placements

FCN:

```text
after conv3
after conv4
```

U-Net:

```text
after encoder stage 2
after encoder stage 3
```

Start with one location per model:

```text
FCN after conv4
U-Net after encoder stage 3
```

### Stability Constraints

Use weaker initialization:

```text
alpha_init = 0.1 or 0.25
```

Only one inserted module per experiment.

Avoid stacking multiple early modules in the first run.

### Mandatory Experiment

| Model | Placement | Seed | Epochs |
| --- | --- | ---: | ---: |
| FCN + V2a-small | after conv4 | 2024 | 50 |
| U-Net + V2a-small | after encoder stage 3 | 2024 | 50 |

### Success Criteria

- no FCN instability;
- improves over V1.1 or matches it within 0.005 mIoU;
- if early/mid placement helps, run a follow-up placement ablation.

### Interpretation

If early/mid placement improves:

```text
The original insertion position was a key limitation, and FA-DCG should be
framed as a feature-extraction enhancement rather than only a bottleneck/deep
feature module.
```

If it underperforms:

```text
The current small SAR dataset may be too sensitive to early feature perturbation,
and deep conservative insertion is safer.
```

## Optional task_005d: Frequency Regularization without Inference Cost

### Purpose

If all structural variants fail to improve metrics, preserve V1.1's strong
structure and add a training-time frequency regularization term.

### Core Idea

Use feature-level local high-frequency response:

```text
feature_high = abs(feature - avgpool(feature))
```

Encourage:

```text
boundary region -> stronger useful high-frequency response
homogeneous water interior -> weaker noisy high-frequency response
```

This adds no inference cost if used only during training.

### Risk

This is easier to frame as a training strategy than as a module contribution.
Use it only if task_005a and task_005b do not produce useful gains.

## Reporting Requirements

Each task_005 subtask should produce:

```text
task_005a_report.md
task_005b_report.md
task_005c_report.md
```

Each report must include:

1. implementation summary;
2. relation to V1.1 and task_004;
3. SAR-specific motivation;
4. lightweight design justification;
5. speed benchmark;
6. seed 2024 results;
7. comparison with V1.1, V2a, V2b, and V2c;
8. diagnostics;
9. recommendation for the next step.

Also write an interaction/experiment summary to:

```text
logs/YYYY-MM-DD_task_005*_*.md
```

## Final Recommendation

Start with:

```text
task_005a: Speckle-Robust Frequency Gate
```

Do not continue increasing dynamic dilation complexity unless task_005a or
task_005b provides evidence that a specific SAR error pattern benefits from it.
