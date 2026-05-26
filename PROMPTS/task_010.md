````markdown id="task_010"
# task_010.md - Targeted Optimization of V2d-noLocalVar for SAR Flood Segmentation

## 0. Background

Task 009 showed that the strongest current U-Net candidate is:

```text
U-Net + V2d-noLocalVar
test mIoU = 0.878237
test Dice = 0.893934
delta vs U-Net + CBAM = +0.002670 mIoU
```

This advantage is positive but weak. The result suggests that SAR-aware frequency gating may help, but the evidence is not yet strong enough to support a broad claim that the proposed SAR-specific module clearly outperforms CBAM.

The current best variant, `V2d-noLocalVar`, removes the local variance descriptor while retaining:

```text
channel=True
boundary=True
speckle=True
msc=True
local_var=False
```

This task performs two targeted follow-up experiments, both motivated by the task_009 findings:

1. decouple the boundary gate from the multi-scale consistency descriptor;
2. initialize the depthwise frequency residual convolution with an explicit high-pass or edge prior.

Do not edit the LaTeX paper in this task. The goal is to produce raw evidence and a concise technical report.

## 1. Repository and Branch

Work in the current repository:

```bash
/home/superws/2026_Projects/FA_DCG
```

Before running experiments, record:

```bash
git branch --show-current
git rev-parse HEAD
```

Save this information in the final report.

## 2. Dataset

Use the real SAR flood dataset:

```bash
/home/superws/dataset/HISEA1_flooding_dataset
```

Expected split layout:

```text
train/image
train/label_1D
val/image
val/label_1D
test/image
test/label_1D
```

Use:

* `train` for training;
* `val` for checkpoint selection;
* `test` for final evaluation only.

Do not merge `val` into training. Do not tune on `test`.

## 3. Main Questions

Answer the following experimentally:

1. In `V2d-noLocalVar`, is the improvement mainly from the boundary gate, from the multi-scale consistency descriptor, or from their combination?
2. Does explicitly initializing the depthwise residual convolution as a high-pass or edge filter improve stability or final segmentation accuracy?
3. Can either targeted modification increase the margin over U-Net + CBAM enough to make the SAR-aware frequency-gating story stronger?

## 4. Baselines To Reuse Or Rerun

Reuse task_009 results if the protocol and metric definitions are identical.

Minimum baselines:

1. `U-Net + CBAM`
2. `U-Net + FADC-aligned`
3. `U-Net + SRFG` (`V2d-noMSC`)
4. `U-Net + V2d-noLocalVar`

If reused, clearly record:

* original result directory;
* per-seed metrics;
* whether the checkpoint was reused or retrained.

## 5. Target Experiments

Use U-Net as the primary backbone. Insert the module at the U-Net bottleneck, matching task_009.

### 5.1 Boundary and MSC Decoupling

The current `V2d-noLocalVar` keeps both boundary and multi-scale consistency:

```text
channel=True
boundary=True
speckle=True
msc=True
local_var=False
```

Task 009 also tested `V2d-noBoundary-noLocalVar`, but that variant disabled both boundary and MSC:

```text
channel=True
boundary=False
speckle=True
msc=False
local_var=False
```

This confounds the effect of the boundary gate with the effect of the MSC descriptor.

Implement explicit variants:

```python
"no_boundary_keep_msc_no_local_var": {
    "channel": True,
    "boundary": False,
    "speckle": True,
    "msc": True,
    "local_var": False,
}

"boundary_no_msc_no_local_var": {
    "channel": True,
    "boundary": True,
    "speckle": True,
    "msc": False,
    "local_var": False,
}
```

Run both variants and compare against:

```python
UNetWithFADCGV2dAblation(variant="no_local_var")
UNetWithFADCGV2dAblation(variant="no_boundary_no_local_var")
```

The expected interpretation should be:

* if `no_boundary_keep_msc_no_local_var` remains strong, MSC is likely more important than the boundary gate;
* if `boundary_no_msc_no_local_var` remains strong, the boundary gate itself is likely useful;
* if only `no_local_var` is strong, the boundary gate and MSC may be complementary;
* if neither is strong, the task_009 gain may be seed noise.

### 5.2 High-Pass Initialization for the Depthwise Frequency Residual

The current depthwise residual filter is randomly initialized:

```python
self.weight = nn.Parameter(torch.randn(channels, 1, kernel_size, kernel_size))
```

This weakens the frequency-domain story because the block is not explicitly initialized as a frequency or edge operator.

Add an initialization option for the depthwise residual convolution, for example:

```python
freq_init="random"      # current behavior
freq_init="laplacian"   # 3x3 high-pass / edge prior
freq_init="sobel_mix"   # optional if implemented cleanly
```

Required first implementation:

```text
freq_init="laplacian"
```

Recommended 3x3 Laplacian-like kernel:

```text
 0 -1  0
-1  4 -1
 0 -1  0
```

or the stronger 8-neighbor version:

```text
-1 -1 -1
-1  8 -1
-1 -1 -1
```

Choose one, document it, and initialize every depthwise channel with the same normalized kernel. Keep the filter learnable after initialization.

Test at least:

```python
UNetWithFADCGV2dAblation(variant="no_local_var", freq_init="laplacian")
```

If time allows, also test:

```python
UNetWithFADCGV2dAblation(variant="no_boundary_keep_msc_no_local_var", freq_init="laplacian")
UNetWithFADCGV2dAblation(variant="boundary_no_msc_no_local_var", freq_init="laplacian")
```

## 6. Training Protocol

Use the same protocol as task_009:

* input size: `256 x 256`
* input channel: `1`
* batch size: `8`
* epochs: `50`
* optimizer: Adam
* initial learning rate: `1e-4`
* learning-rate schedule: cosine annealing to `1e-6`
* loss: BCEWithLogitsLoss
* augmentation:
  * random horizontal flip
  * random vertical flip
  * random rotation within +/-10 degrees

Use the following seeds:

```text
42, 123, 2026
```

If compute is limited, first run seed `2026` for all new variants, then confirm the best one or two variants on seeds `42` and `123`.

## 7. Required Metrics

For each model and each seed, report:

* validation mIoU and Dice at the selected checkpoint;
* test mIoU and Dice;
* best epoch;
* checkpoint path;
* parameter count;
* MACs/FLOPs under the same tool convention as task_009;
* latency and FPS under the same benchmark protocol as task_009.

Use the same metric implementation as task_009. Do not change metric definitions.

## 8. Diagnostics

For each V2d variant, collect and report diagnostics if available:

* `alpha_value`
* `beta_value`
* `channel_gate_mean/std`
* `boundary_gate_mean/std`
* `speckle_gate_mean/std`
* `suppression_ratio`
* `combined_gate_mean/std`
* `local_high_descriptor_mean`
* `multi_scale_consistency_mean`

For high-pass initialized models, additionally report:

* initialization type;
* exact kernel used;
* whether the depthwise kernel remains learnable;
* final depthwise kernel mean/std if easy to compute.

## 9. Output Files

Create a new directory:

```bash
exp/task_010_v2d_nolocalvar_targeted/<timestamp>/
```

Save:

```text
raw_metrics_per_seed.csv
summary_mean_std.csv
complexity_raw.csv
diagnostics_raw.csv
environment.txt
implementation_audit.md
task_010_report.md
logs/
checkpoints/
```

The report must include:

1. exact git branch and commit hash;
2. exact dataset path and split sizes;
3. exact model configs;
4. whether baselines were reused or rerun;
5. per-seed raw results;
6. mean/std summary;
7. comparison against task_009 `U-Net + CBAM` and `U-Net + V2d-noLocalVar`;
8. a concise conclusion about whether the result strengthens the SAR-aware frequency-gating story.

## 10. Decision Rule

Use the following interpretation:

* If a new variant improves over CBAM by at least `0.005` mIoU mean over three seeds, treat it as a potentially publishable method candidate.
* If the gain is between `0.002` and `0.005` mIoU, treat it as weak positive evidence requiring more datasets or stronger analysis.
* If the gain is below `0.002` mIoU or inconsistent across seeds, stop architecture exploration and focus on analysis/negative-result framing.

Do not edit the LaTeX paper in this task.
````
