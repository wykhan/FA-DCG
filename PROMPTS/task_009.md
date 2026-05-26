````markdown id="task_009"
# task_009.md — FADC-aligned + SRFG Hybrid Experiments for Real SAR Flood Segmentation

## 0. Background

The previous main-module comparison on the real SAR dataset showed that the current SRFG candidate is not sufficiently strong on the U-Net backbone:

```text
Dataset: /home/superws/dataset/HISEA1_flooding_dataset
Seed: 2026

U-Net + CBAM          test mIoU = 0.879219, test Dice = 0.894610
U-Net + FADC-aligned  test mIoU = 0.878748, test Dice = 0.894914
U-Net + SRFG          test mIoU = 0.875874, test Dice = 0.892199
```

SRFG remains useful on FCN, but the FCN gain is partly entangled with the classifier-path residual reparameterization. For U-Net, SRFG does not currently outperform CBAM or FADC-aligned on the real SAR flood segmentation dataset.

This task investigates a more realistic SAR-specific direction:

> Use **FADC-aligned as the main adaptive frequency/dilation operator**, and use **SRFG ideas only as reliability modulation**, especially speckle reliability and SAR-guided spatial masking.

The goal is not to produce a final paper table immediately. The goal is to generate reliable raw experimental evidence for which hybrid design can beat CBAM on U-Net.

Do not edit the LaTeX paper in this task.

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

## 3. Main Question

Answer the following experimentally:

1. Can FADC-aligned become stronger when SRFG is used as a **speckle reliability modulator** rather than as a separate residual block?
2. Does adding a CBAM-style spatial mask, guided by SAR frequency/speckle descriptors, close the gap or surpass CBAM?
3. Are historical SRFG variants such as `V2d-noLocalVar` better suited to real SAR U-Net than the current `V2d-noMSC` final candidate?

## 4. Baselines To Reuse Or Rerun

Reuse task_008 results if the protocol and metric definitions are identical. Otherwise rerun.

Minimum U-Net baselines:

1. `U-Net`
2. `U-Net + CBAM`
3. `U-Net + FADC-aligned`
4. `U-Net + SRFG` (`V2d-noMSC`)

Optional FCN baselines may be reused for secondary sanity checks, but the primary target is U-Net.

## 5. Target Hybrid Models

Implement and test the following U-Net variants. Insert the main hybrid block at the U-Net bottleneck unless otherwise stated.

### 5.1 `U-Net + V2d-noLocalVar`

Historical evidence suggests `V2d-noLocalVar` may be stronger than `V2d-noMSC` on U-Net.

Use the existing V2d ablation implementation if available:

```python
UNetWithFADCGV2dAblation(variant="no_local_var")
```

This is a high-priority candidate.

### 5.2 `U-Net + V2d-noBoundary-noLocalVar`

Implement if not already present.

Rationale:

* historical ablation showed boundary gate is weak or redundant on U-Net;
* local variance may overfit or inject unstable speckle-like noise;
* this variant keeps the residual response, channel gate, and speckle gate, while removing two potentially unstable descriptors.

If implementation requires changing `VARIANT_CONFIGS`, add a new explicit variant:

```python
"no_boundary_no_local_var"
```

Expected config:

```text
channel=True
boundary=False
speckle=True
msc=True or False; prefer False for first run if matching noMSC simplification
local_var=False
```

Document the exact config in the report.

### 5.3 `U-Net + FADC-aligned + SRFG-SR`

SRFG-SR means **speckle reliability modulation**.

Use FADC-aligned as the main operator. Do not add a second large residual convolution path.

Required behavior:

* compute FADC-aligned frequency-selected features as usual;
* compute a speckle reliability map from SAR descriptors;
* use that map to modulate FADC high-frequency or adaptive-dilation responses.

Recommended descriptors:

```text
local_high = abs(x - avgpool3(x))
local_var_proxy = avgpool3(x*x) - avgpool3(x)^2
local_var_norm = local_var_proxy / mean(local_var_proxy)
```

Recommended reliability form:

```text
speckle_score = Conv(local_high [, local_var_norm])
reliability = 1 - beta * sigmoid(speckle_score)
```

Implementation constraints:

* initialize `speckle_score` weights to zero;
* initialize `beta` small, e.g. `0.05` or `0.1`;
* initial behavior should be close to identity or close to original FADC-aligned;
* report learned `beta` and reliability mean/std diagnostics.

Test at least two versions:

1. `FADC-SR-high`: reliability modulates only high-frequency selected components.
2. `FADC-SR-branch`: reliability modulates adaptive dilation branch responses before branch mixing.

### 5.4 `U-Net + FADC-aligned + SAR-Spatial`

Add a CBAM-style spatial mask, but make the mask SAR-aware.

Do not use a generic decorative attention block without SAR descriptors. The mask should use:

```text
avg_over_channels(x)
max_over_channels(x)
avg_over_channels(local_high)
avg_over_channels(local_var_norm)  # optional; include one variant without it
```

Recommended spatial mask:

```python
mask = sigmoid(conv7x7(concat(descriptors)))
out = x * (1 + gamma * (mask - 0.5) * 2)
```

Initialization:

* initialize the final spatial conv to zero;
* initialize `gamma` to `0.0` or a very small value;
* the block should initially behave like identity.

Test at least two versions:

1. `FADC-SARSpatial-noVar`: descriptors exclude local variance.
2. `FADC-SARSpatial-withVar`: descriptors include local variance.

### 5.5 `U-Net + FADC-aligned + SRFG-SR + SAR-Spatial`

Combine the most conservative speckle reliability modulation with the SAR-guided spatial mask.

Recommended first combined candidate:

```text
FADC-SR-high + SARSpatial-noVar
```

Rationale:

* FADC supplies adaptive frequency/dilation modeling;
* SRFG supplies SAR speckle reliability;
* SARSpatial supplies the missing CBAM-like spatial selection;
* excluding local variance first reduces the risk of overfitting unstable SAR texture.

### 5.6 `U-Net + FADC-aligned + CBAM-SpatialOnly`

Control experiment.

Use standard CBAM spatial attention only:

```text
avg_over_channels(x), max_over_channels(x) -> 7x7 conv -> sigmoid spatial mask
```

No SAR descriptors.

This checks whether the benefit comes from generic spatial attention or from SAR-guided descriptors.

## 6. Prioritized Experiment Matrix

Run in this order.

### Priority A: must run

1. `U-Net + V2d-noLocalVar`
2. `U-Net + FADC-SR-high`
3. `U-Net + FADC-SARSpatial-noVar`
4. `U-Net + FADC-SR-high + SARSpatial-noVar`
5. `U-Net + FADC-CBAM-SpatialOnly`

### Priority B: run if Priority A finishes cleanly

6. `U-Net + V2d-noBoundary-noLocalVar`
7. `U-Net + FADC-SR-branch`
8. `U-Net + FADC-SARSpatial-withVar`
9. `U-Net + FADC-SR-branch + SARSpatial-noVar`

### Priority C: secondary FCN sanity check

Run only for the best one or two U-Net candidates:

1. `FCN + best hybrid`
2. `FCN + second-best hybrid`

The FCN check is secondary. Do not let it delay completing the U-Net matrix.

## 7. Training Protocol

Use the same protocol as task_008 unless explicitly noted:

* input size: `256 x 256`
* input channel: `1`
* batch size: `8`
* epochs: `50`
* optimizer: Adam
* initial learning rate: `1e-4`
* learning-rate schedule: cosine annealing to `1e-6`
* loss: BCEWithLogitsLoss
* data augmentation:
  * random horizontal flip
  * random vertical flip
  * random rotation within +/-10 degrees

Use seed:

```text
2026
```

If two or more candidates beat `U-Net + CBAM` on test mIoU under seed 2026, run a confirmation phase for those candidates plus `U-Net + CBAM` and `U-Net + FADC-aligned` using:

```text
42, 123, 2024, 2026
```

Do not claim a robust improvement from a single seed. The seed-2026 run is for screening.

## 8. Required Metrics

For every model and seed, report:

* best validation mIoU;
* best validation Dice;
* best epoch selected by validation mIoU;
* final test mIoU from the selected checkpoint;
* final test Dice from the selected checkpoint;
* checkpoint path;
* training status and notes.

Use the existing repository metric definition:

```text
sigmoid threshold = 0.5
mIoU = average of foreground IoU and background IoU
Dice = foreground Dice
```

Do not change metric definitions.

## 9. Complexity And Diagnostics

For each model, report:

* learnable parameters;
* parameter delta vs U-Net;
* MACs/FLOPs using the same tool and convention as task_008;
* latency ms;
* FPS;
* GPU name;
* PyTorch and CUDA versions.

Also collect module diagnostics for hybrid models:

### SRFG-SR diagnostics

* learned `beta`;
* reliability mean/std;
* local_high descriptor mean;
* local_var descriptor mean if used;
* branch reliability mean/std if using branch modulation.

### SAR-Spatial diagnostics

* learned `gamma`;
* spatial mask mean/std;
* spatial mask min/max;
* descriptor means.

Save diagnostics as CSV files.

For the best candidate and for `U-Net + CBAM`, save a small qualitative panel on the same test samples:

* input SAR image;
* ground-truth mask;
* predicted probability map;
* binary prediction;
* error map;
* optional spatial/reliability mask visualization.

Use 8-16 fixed test images. Do not cherry-pick after seeing results; choose deterministic sample indices before visualization.

## 10. Output Files

Create a new output directory:

```bash
exp/task_009_fadc_srfg_hybrid/<timestamp>/
```

Save at least:

### 10.1 `task_009_report.md`

Must include:

* git branch;
* git commit hash;
* dataset path;
* train/val/test split sizes;
* exact model list;
* implementation summary for each new hybrid;
* insertion positions;
* training protocol;
* metric definitions;
* complexity protocol;
* diagnostics protocol;
* completed runs;
* failed or skipped runs with reasons;
* factual observations only.

Do not write final paper claims.

### 10.2 `raw_metrics_per_seed.csv`

Required columns:

```text
backbone,
method,
variant,
seed,
val_miou,
val_dice,
test_miou,
test_dice,
best_epoch,
checkpoint_path,
status,
notes
```

### 10.3 `summary_mean_std.csv`

Required columns:

```text
backbone,
method,
variant,
num_seeds,
val_miou_mean,
val_miou_std,
test_miou_mean,
test_miou_std,
test_dice_mean,
test_dice_std,
params_abs,
params_delta_vs_unet,
macs_abs,
latency_ms_mean,
fps,
notes
```

### 10.4 `complexity_raw.csv`

Use the same fields as task_008, plus `variant`.

### 10.5 `diagnostics_raw.csv`

One row per model, seed, epoch/checkpoint, and diagnostic source.

### 10.6 `implementation_audit.md`

Include:

* model class names;
* parameter counts;
* output shape checks;
* identity/near-identity initialization checks;
* confirmation that no lazy modules are created in `forward`;
* confirmation that FADC-aligned is a local PyTorch implementation inspired by FADC, not the official FADC implementation.

## 11. Acceptance Criteria

This task is complete when:

1. Priority A models have completed seed 2026 training and test evaluation.
2. Priority B models have either completed or are explicitly documented as skipped with reason.
3. `U-Net + CBAM`, `U-Net + FADC-aligned`, and `U-Net + SRFG` task_008 baselines are included or clearly referenced.
4. All output CSV files are present.
5. New hybrid modules pass smoke tests:
   * forward shape `[B, 1, 256, 256]`;
   * finite output;
   * no module creation in `forward`;
   * checkpoint can be saved and reloaded.
6. The report clearly states whether any candidate beats `U-Net + CBAM` on seed 2026 test mIoU.

## 12. Decision Rules

Use the following decision rules after the screening run:

* If no hybrid beats CBAM, the report should say so directly and identify the closest variant.
* If a hybrid beats CBAM by less than `0.002` mIoU on seed 2026, treat it as inconclusive and require multi-seed confirmation.
* If a hybrid beats CBAM by at least `0.003` mIoU on seed 2026, run the confirmation phase for that hybrid.
* If SAR-guided spatial mask does not beat standard CBAM spatial-only, do not claim SAR descriptor superiority.
* If local variance variants underperform no-var variants again, recommend removing local variance from the final U-Net SRFG path.

## 13. Expected Scientific Interpretation To Test

The working hypothesis is:

> On real SAR flood segmentation, U-Net already preserves spatial detail through skip connections. A standalone SRFG residual block is therefore less useful than a module that combines FADC-style adaptive frequency/dilation selection with explicit spatial reliability. Speckle-aware descriptors should modulate the reliability of high-frequency responses rather than introduce an independent heavy residual pathway.

This hypothesis must be tested by results, not assumed.
````
