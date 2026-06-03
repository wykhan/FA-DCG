````markdown id="task_012"
# task_012.md - Fair FCN Residual-Control Experiment

## 0. Background

The U-Net core experiment has converged: `V2d-noBoundary-noLocalVar` is stable and exceeds CBAM.

The FCN results need a stricter control. Previous FCN comparisons showed that our FCN variants were far better than basic FCN, SE, CBAM, and FADC-aligned. However, those comparisons are not fully fair because the proposed FCN variants contain an explicit residual enhancement path, while SE, CBAM, FADC-aligned, and the plain FCN do not.

Therefore, the key FCN question is no longer:

```text
Does FCN + V2d-noBoundary-noLocalVar beat FCN, SE, CBAM, and FADC?
```

The correct question is:

```text
Does FCN + V2d-noBoundary-noLocalVar beat FCN + residual-only under the same residual insertion path?
```

This task designs and runs that fair residual-control experiment.

Do not edit the LaTeX paper in this task. Only implement the control if missing, run the experiments, and produce raw evidence plus a concise report.

## 1. Repository and Branch

Work in the current repository:

```bash
/home/superws/2026_Projects/FA_DCG
```

Before running experiments, record:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

Save this information in the final report.

## 2. Dataset

Use the same dataset and split protocol as task_010 and task_011:

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

## 3. Main Hypothesis

This task tests whether the FCN gain comes mainly from the residual structure or from the proposed V2d frequency/speckle-aware modulation.

Interpretation:

* If `FCN + V2d-noBoundary-noLocalVar` clearly beats `FCN + residual-only`, then the FCN result supports the proposed module beyond residual learning.
* If the two models are statistically similar, then the FCN improvement should be attributed mainly to the residual enhancement path, and FCN should not be used as strong evidence that the V2d mechanism itself is superior.
* If `FCN + residual-only` beats V2d, then the FCN part of the paper should be reframed as a residual-control negative result, and the main method evidence should rely on U-Net.

## 4. Models

Compare only the following two primary models.

### 4.1 FCN + Residual-Only

Implement this control if it does not already exist.

The residual-only model must match the FCN V2d integration path as closely as possible:

```text
conv1 -> conv2 -> conv3 -> conv4 -> conv5
-> residual block at the same 512-channel stage
-> relu6/drop6/up_dim
-> residual block at the same 4096-channel stage
-> relu7/drop7/score
```

Use the same insertion positions as `FCNWithFADCGV2dAblation`.

The residual-only block should keep only the residual enhancement path:

```python
z = depthwise_conv(x)
out = x + alpha * z
```

Requirements:

* use learnable depthwise convolution weights;
* use the same kernel sizes as FCN V2d ablation:
  * `kernel_size=7` for the 512-channel block;
  * `kernel_size=1` for the 4096-channel block;
* use a learnable scalar `alpha`, initialized to `0.5`;
* do not use channel gate;
* do not use boundary gate;
* do not use speckle gate;
* do not use multi-scale consistency;
* do not use local variance;
* do not use frequency descriptors;
* keep parameter initialization documented.

Name this model clearly, for example:

```text
FCN + Residual-only
```

Recommended implementation names:

```python
ResidualOnlyBlock
FCNWithResidualOnly
```

### 4.2 FCN + V2d-noBoundary-noLocalVar

Use the same final U-Net-winning configuration, adapted to FCN:

```text
channel=True
boundary=False
speckle=True
msc=False
local_var=False
freq_init=random
```

In the local code this should correspond to:

```python
FCNWithFADCGV2dAblation(variant="no_boundary_no_local_var")
```

Verify the actual `VARIANT_CONFIGS` before running. If the local name differs, document the exact variant name and config.

## 5. Optional Context Baselines

The primary conclusion must be based only on:

```text
FCN + Residual-only
FCN + V2d-noBoundary-noLocalVar
```

For context, the report may also include reused task_008/task_010 FCN results if the protocol is identical:

```text
FCN
FCN + CBAM
FCN + FADC-aligned
FCN + SRFG / V2d-noMSC
```

These optional baselines must be clearly marked as `context only`. They must not replace the residual-control comparison.

## 6. Training Protocol

Use the same protocol as task_010 and task_011:

* FCN backbone
* input size: `256 x 256`
* input channel: `1`
* batch size: `8`
* epochs: `50`
* optimizer: Adam
* initial learning rate: `1e-4`
* learning-rate schedule: cosine annealing to `1e-6`
* loss: BCEWithLogitsLoss
* validation set for checkpoint selection
* test set for final evaluation only
* augmentation:
  * random horizontal flip
  * random vertical flip
  * random rotation within +/-10 degrees

Use at least three seeds:

```text
42, 123, 2026
```

If the three-seed result is close, extend to six seeds:

```text
42, 123, 2026, 2027, 2028, 2029
```

Use paired seeds: both models must be trained with exactly the same seed list.

## 7. Required Metrics

For each model and seed, report:

* validation mIoU and Dice at the selected checkpoint;
* test mIoU and Dice;
* best epoch;
* checkpoint path;
* parameter count;
* MACs/FLOPs under the same tool convention as task_010;
* latency and FPS under the same benchmark protocol as task_010.

Use the existing metric implementation. Do not change metric definitions.

## 8. Residual-Control Audit

Before training, create an implementation audit confirming:

* both models use the same FCN backbone;
* both models insert enhancement blocks at the same two FCN stages;
* both models use the same output resolution and interpolation settings;
* `FCN + Residual-only` contains no channel, boundary, speckle, MSC, local variance, or frequency-descriptor gates;
* `FCN + V2d-noBoundary-noLocalVar` uses the intended config:

```text
channel=True
boundary=False
speckle=True
msc=False
local_var=False
freq_init=random
```

Also record parameter deltas:

```text
Params(FCN + Residual-only) - Params(FCN)
Params(FCN + V2d-noBoundary-noLocalVar) - Params(FCN)
Params(V2d) - Params(Residual-only)
```

## 9. Diagnostics

For `FCN + Residual-only`, collect if easy:

* `alpha_value` for both residual blocks;
* depthwise weight mean/std for both residual blocks.

For `FCN + V2d-noBoundary-noLocalVar`, collect the same diagnostics as task_010 where available:

* `alpha_value`
* `beta_value`
* `channel_gate_mean/std`
* `speckle_gate_mean/std`
* `suppression_ratio`
* `combined_gate_mean/std`
* depthwise weight mean/std

If diagnostics are not available for one model, do not change training logic just to force them. Report the missing diagnostics clearly.

## 10. Output Files

Create a new directory:

```bash
exp/task_012_fcn_residual_control/<timestamp>/
```

Save:

```text
raw_metrics_per_seed.csv
summary_mean_std.csv
paired_seed_delta.csv
complexity_raw.csv
diagnostics_raw.csv
environment.txt
implementation_audit.md
task_012_report.md
logs/
checkpoints/
```

The final report must include:

1. exact git branch, commit hash, and dirty-worktree status;
2. exact dataset path and split sizes;
3. exact residual-only implementation;
4. exact V2d variant config;
5. per-seed raw results;
6. mean/std summary;
7. paired seed deltas:

```text
delta_mIoU(seed) = mIoU(V2d-noBoundary-noLocalVar, seed) - mIoU(Residual-only, seed)
delta_Dice(seed) = Dice(V2d-noBoundary-noLocalVar, seed) - Dice(Residual-only, seed)
```

8. mean paired delta and standard deviation;
9. complexity and latency comparison;
10. a concise conclusion about whether FCN evidence survives the residual-control test.

## 11. Decision Rule

Use the following rule for the FCN claim:

* If `V2d-noBoundary-noLocalVar` improves over `Residual-only` by at least `0.005` mean test mIoU over three paired seeds, and the delta is positive on all or most seeds, treat the FCN result as positive evidence for the proposed module.
* If the mean gain is between `0.002` and `0.005` mIoU, treat it as weak positive evidence and extend to six seeds before making any paper claim.
* If the mean gain is below `0.002` mIoU, or if seed-level signs are inconsistent, do not claim that the FCN improvement is mainly caused by V2d. Attribute the FCN gain primarily to the residual enhancement path.
* If `Residual-only` is better, keep the result as a negative control and use it to make the paper more rigorous.

Do not edit the LaTeX paper in this task.
````
