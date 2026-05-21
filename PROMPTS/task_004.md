# task_004: Design FA-DCG V2.0 for SAR Flood Segmentation

## Background

The project now has three important reference points:

1. **FA-DCG V1.0**: the original `LightFADC` implementation. It contains
   serious implementation problems and does not truly realize differentiable
   frequency-aware dilation.
2. **FA-DCG V1.1**: the accepted practical baseline. It keeps the effective
   residual depthwise enhancement and channel-gating behavior of V1.0 while
   fixing major engineering issues.
3. **task_003 FADC-aligned module**: a local PyTorch implementation inspired by
   official FADC. It implemented FreqSelect, AdaDR approximation, and AdaKern
   approximation, but it did not outperform V1.1 overall.

Task_004 should design **FA-DCG V2.0**, not another official-FADC-aligned port.

FA-DCG V2.0 must:

- start from FA-DCG V1.1 as the practical baseline;
- borrow the most useful idea from FADC: **frequency awareness**;
- be tailored to single-channel SAR flood segmentation;
- remain lightweight for small labeled datasets;
- aim to improve validation metrics without severe speed or parameter cost.

## Lessons from Existing Experiments

### From FA-DCG V1.1

V1.1 is stable because it is conservative:

```text
depthwise spatial enhancement
+ response-selective channel gate
+ residual controlled fusion
```

It achieved:

```text
FCN + FA-DCG V1.1:
  mIoU = 0.842804
  Dice = 0.905853
  benchmark latency = 8.317 ms

U-Net + FA-DCG V1.1:
  mIoU = 0.892461
  Dice = 0.938098
  benchmark latency = 20.350 ms
```

V2.0 should preserve this conservative residual-gated structure.

### From task_003 FADC-Aligned Experiment

The local FADC-aligned implementation showed:

```text
FCN + FADC:
  mIoU = 0.663577
  Dice = 0.771850

U-Net + FADC:
  mIoU = 0.887575
  Dice = 0.934839
```

Main lessons:

- Direct official-FADC-style complexity does not automatically transfer to this
  SAR flood segmentation task.
- Spatial branch mixing over large dilation rates can disturb early/mid features.
- FCN is especially sensitive to early feature perturbation.
- Frequency selection must be more task-specific and conservative.
- U-Net can absorb frequency-aware modules better because skip connections help
  recover spatial detail.

## Design Principles for FA-DCG V2.0

### 1. SAR-Specific

The module should explicitly address:

- residual speckle after Lee filtering;
- weak texture in water regions;
- blurred water-land boundaries;
- small water bodies and narrow river channels;
- false alarms from building shadows or dark non-water regions.

The design should reflect the SAR frequency structure:

```text
homogeneous flood interiors -> low-frequency / large-context preference
boundaries and small water bodies -> high-frequency / small-context preference
speckle and shadows -> unstable high-frequency or false low-backscatter responses
```

### 2. Lightweight and Small-Sample Friendly

Because SAR labels are expensive and the dataset is small:

- avoid heavy explicit FFT/wavelet branches as default;
- avoid per-pixel high-dimensional dynamic kernels;
- avoid custom CUDA and MMCV/MMSeg dependencies;
- prefer depthwise, grouped, or low-rank operations;
- keep parameter increase close to or below V1.1 when possible;
- initialize close to V1.1 or identity so training is stable.

### 3. Frequency-Aware but Conservative

Unlike task_003, V2.0 should not aggressively mix large dilation branches early.

Frequency information should act as:

```text
a modulator of V1.1 enhancement strength and receptive-field preference
```

not as a full replacement of V1.1's stable feature path.

### 4. Multi-Round Validation

Task_004 should not try to finalize V2.0 in one step.

Instead:

```text
task_004  = overall V2.0 design plan
task_004a = first minimal candidate
task_004b = second candidate / ablation
task_004c = selected candidate refinement and confirmation
```

## Candidate FA-DCG V2.0 Structures

### Candidate A: SAR Frequency-Gated V1.1

Goal:

Add a lightweight SAR frequency descriptor to V1.1 and use it only to modulate
the residual enhancement strength.

Core idea:

```text
x
-> V1.1 depthwise enhancement z
-> SAR frequency descriptor f(x)
-> frequency gate g_f
-> y = x + alpha * channel_gate(z) * frequency_gate(f) * z
```

Frequency descriptor options:

```text
local_mean = avgpool(x)
local_high = abs(x - local_mean)
local_contrast = local_std or abs residual
```

Use a very small descriptor head:

```text
descriptor: [local_high, local_low/context]
head: depthwise/grouped 3x3 + sigmoid
```

Expected benefit:

- keeps V1.1's stable residual path;
- adds explicit frequency awareness;
- low risk of overfitting;
- likely best first candidate for task_004a.

Risk:

- may behave like another attention gate if descriptor is too weak.

### Candidate B: Boundary-Preserved Frequency Dilation Bias

Goal:

Use SAR frequency cues to softly bias dilation preference:

```text
high-frequency boundary cue -> prefer small dilation
low-frequency homogeneous cue -> allow larger dilation
```

Design:

```text
freq_map = lightweight local high/low descriptor
dilation_logits = channel_logits + lambda * spatial_freq_bias
branch_weights = softmax(dilation_logits)
z = sum_k branch_weights_k * depthwise_conv_dilation_k(x)
y = x + alpha * gate * z
```

Important constraints:

- large dilation branches must be conservative;
- initialize strongly toward dilation 1 or V1.1-equivalent behavior;
- optionally cap large-dilation contribution during early training;
- no deformable convolution;
- no heavy FFT.

Expected benefit:

- closer to FADC's frequency-aware receptive-field idea;
- more interpretable than V1.1;
- can test whether dynamic receptive field improves SAR segmentation.

Risk:

- FCN may degrade if early/mid features are over-smoothed;
- needs careful initialization and diagnostics.

### Candidate C: Speckle-Robust Dual Gate

Goal:

Separate useful high-frequency boundaries from noisy speckle-like activations.

Design:

```text
boundary_gate = local gradient / local contrast descriptor
speckle_suppression_gate = instability or high-frequency noise descriptor
enhancement = depthwise residual branch
y = x + alpha * channel_gate * boundary_gate * (1 - speckle_gate) * enhancement
```

Descriptor ideas:

```text
local_abs_residual = abs(x - avgpool(x))
local_variance_proxy = avgpool(x^2) - avgpool(x)^2
multi-scale consistency = abs(avg3(x) - avg7(x))
```

Expected benefit:

- directly targets SAR residual speckle and false alarms;
- may reduce shadow/speckle false positives.

Risk:

- if the speckle gate suppresses true water boundaries, recall may drop.

### Candidate D: U-Net Skip-Level Frequency Calibration

Goal:

Apply V2.0 not at the bottleneck but on U-Net skip features, where boundary and
small-object information is preserved.

Design:

```text
encoder stage 2 or 3 skip feature
-> lightweight frequency-gated V1.1 block
-> calibrated skip
-> decoder fusion
```

Expected benefit:

- matches U-Net's strength in preserving high-frequency details;
- likely better than applying heavy modules to FCN;
- useful for small water bodies and fuzzy boundaries.

Risk:

- may not transfer to FCN;
- needs separate integration logic.

## Proposed Multi-Round Experiment Plan

## task_004a: Minimal SAR Frequency-Gated V1.1

Purpose:

Validate whether explicit lightweight frequency gating can improve V1.1 without
destabilizing training.

Implement Candidate A:

```text
SARFreqGate + V1.1 depthwise residual path
```

Suggested files:

```text
SAR_FEM1/models/improved/fadcg_v2a.py
SAR_FEM1/models/improved/fcn_fadcg_v2a.py
SAR_FEM1/models/improved/unet_fadcg_v2a.py
scripts/task_004a_test_fadcg_v2a.py
scripts/task_004a_fadcg_v2a_repro.py
```

Insertion positions:

- FCN: first test V1.1-compatible deep path, not early conv3.
- U-Net: bottleneck first, then optional encoder stage 3 if stable.

Reason:

004a should isolate frequency gating while minimizing architecture disturbance.

Mandatory experiment:

| Model | Seed | Epochs |
| --- | ---: | ---: |
| FCN + FA-DCG V2a | 2024 | 50 |
| U-Net + FA-DCG V2a | 2024 | 50 |

Success criteria:

- no NaNs;
- no more than 0.005 mIoU drop vs V1.1;
- ideally improves at least one backbone;
- latency no worse than 1.5x V1.1.

## task_004b: Frequency-Biased Dilation Preference

Purpose:

Test whether frequency cues can improve receptive-field allocation beyond
V1.1's fixed effective dilation behavior.

Implement Candidate B:

```text
frequency descriptor -> spatial/channel dilation bias -> branch fusion
```

Key constraints:

- initialize close to V1.1 or dilation=1;
- restrict large-dilation weights early;
- save branch-weight diagnostics;
- compare with 004a.

Mandatory experiment:

| Model | Seed | Epochs |
| --- | ---: | ---: |
| FCN + FA-DCG V2b | 2024 | 50 |
| U-Net + FA-DCG V2b | 2024 | 50 |

Success criteria:

- branch weights should not collapse to uniform trivial weights;
- large-dilation weights should correlate with low-frequency regions;
- mIoU should match or improve V1.1 on at least U-Net;
- FCN should not collapse as in task_003 FADC.

## task_004c: Selected Candidate Refinement

Purpose:

Refine the better candidate from 004a/004b and prepare a stronger V2.0
candidate.

Possible refinements:

- add Candidate C speckle-robust dual gate if false positives remain high;
- add Candidate D U-Net skip calibration if U-Net gains are promising;
- tune alpha initialization and warmup;
- run insertion-position ablation;
- run three-seed confirmation only for the strongest candidate.

Mandatory first step:

Choose one:

```text
V2a -> refine frequency gate
V2b -> refine dilation bias
V2a + C -> add speckle suppression
V2b + D -> add U-Net skip calibration
```

Optional three-seed confirmation:

```text
seeds = 42, 123, 2024
epochs = 50
```

Promotion criteria for FA-DCG V2.0:

- mean mIoU improves over V1.1, or no metric drop with better interpretability;
- no severe speed penalty;
- parameter increase remains small;
- diagnostics support the frequency-aware narrative;
- FCN and U-Net both remain stable.

## Baselines

Primary baseline:

```text
FCN + FA-DCG V1.1:
  mIoU = 0.842804
  Dice = 0.905853
  benchmark latency = 8.317 ms

U-Net + FA-DCG V1.1:
  mIoU = 0.892461
  Dice = 0.938098
  benchmark latency = 20.350 ms
```

Historical reference:

```text
FCN + FA-DCG V1.0:
  mIoU = 0.844849
  Dice = 0.907507

U-Net + FA-DCG V1.0:
  mIoU = 0.896663
  Dice = 0.941023
```

Task_003 reference:

```text
FCN + FADC:
  mIoU = 0.663577
  Dice = 0.771850

U-Net + FADC:
  mIoU = 0.887575
  Dice = 0.934839
```

## Required Diagnostics for All 004 Experiments

Save diagnostics under:

```text
figures/fadcg_v2_diagnostics/
```

At minimum:

```text
frequency_gate_mean
frequency_gate_std
local_high_descriptor_mean
local_low_descriptor_mean
alpha_value
channel_gate_mean
channel_gate_std
```

For dilation-bias variants:

```text
branch_weight_mean_d1
branch_weight_mean_d2
branch_weight_mean_d3
branch_weight_mean_d4
small_dilation_weight
large_dilation_weight
```

For speckle-aware variants:

```text
boundary_gate_mean
speckle_gate_mean
suppression_ratio
```

## Required Reports

Each subtask should produce:

```text
task_004a_report.md
task_004b_report.md
task_004c_report.md
```

Each report must include:

1. implementation summary;
2. relation to V1.1;
3. FADC idea borrowed;
4. SAR-specific adaptation;
5. lightweight design justification;
6. speed benchmark;
7. seed 2024 results;
8. comparison with V1.1 and task_003 FADC;
9. diagnostics;
10. recommendation for the next subtask.

## Overall task_004 Acceptance Criteria

Task_004 is successful when:

1. at least three candidate V2.0 directions are clearly defined;
2. 004a/004b/004c experiment sequence is specified;
3. each candidate is grounded in SAR flood segmentation needs;
4. each candidate remains lightweight and small-sample-aware;
5. V1.1 remains the baseline until a candidate proves better;
6. diagnostics are required to support the frequency-aware interpretation;
7. task_004 does not repeat task_003 official FADC alignment.

## Recommended Starting Point

Start with **task_004a: SAR Frequency-Gated V1.1**.

Reason:

- it changes the fewest things relative to V1.1;
- it directly adds frequency awareness;
- it is lightweight;
- it has the lowest risk of reproducing task_003's FCN collapse;
- it provides a clean test of whether SAR frequency cues can improve the accepted
  baseline.
