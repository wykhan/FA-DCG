# task_006: Search for the Minimal Sufficient FA-DCG V2d

## Background

The current paper direction is to use **FA-DCG V2d** as the final proposed
module and use the local PyTorch **FADC-aligned** implementation as a comparison
method.

Existing seed-2024 validation results are:

```text
FCN:
  FCN baseline      mIoU = 0.663746, Dice = 0.768719
  FCN + FADC        mIoU = 0.663577, Dice = 0.771850
  FCN + V2d         mIoU = 0.845975, Dice = 0.908181

U-Net:
  U-Net baseline    mIoU = 0.892698, Dice = 0.938395
  U-Net + FADC      mIoU = 0.887575, Dice = 0.934839
  U-Net + V2d       mIoU = 0.893472, Dice = 0.938617
```

V2d clearly outperforms the FADC-aligned comparison. However, V2d has more
parameters than the FADC-aligned comparison:

```text
FCN + FADC:   134.3209M params
FCN + V2d:    145.1444M params

U-Net + FADC: 31.0913M params
U-Net + V2d:  31.6253M params
```

Therefore, the paper should not claim that V2d is lighter than FADC. The better
claim is:

```text
V2d is more suitable for single-channel SAR flood segmentation because it uses
task-specific speckle-robust frequency selection, achieving better accuracy
with acceptable plug-and-play overhead.
```

Before finalizing V2d, task_006 should test whether the full V2d design is
actually necessary or whether some components are redundant.

## Overall Goal

Find the **minimal sufficient V2d**:

```text
the simplest V2d-family structure that preserves the performance advantage of
full V2d while removing unnecessary gates, descriptors, or suppression terms.
```

This is not a search for a new V3 module. The scope is restricted to controlled
ablation of V2d internals.

The central questions are:

1. Which V2d components actually improve segmentation?
2. Are boundary gate, speckle gate, channel gate, and frequency descriptors all
   necessary?
3. Can a simpler V2d variant match or exceed full V2d with fewer parameters or
   a cleaner explanation?
4. If full V2d remains best, can the ablation prove that its components are not
   over-designed?

## Full V2d Reference

Full V2d is:

```text
z = depthwise_residual_enhancement(x)
C = channel_gate(z)
B = boundary_gate(local_high, multi_scale_consistency)
S = speckle_gate(local_high, local_var, multi_scale_consistency)

y = x + alpha * C * B * (1 - beta * S) * z
```

Frequency descriptors:

```text
local_low = avgpool3(x)
local_context = avgpool7(x)
local_high = abs(x - local_low)
multi_scale_consistency = abs(local_low - local_context)
local_var = avgpool3(x * x) - avgpool3(x) * avgpool3(x)
```

Component meanings:

| Component | Symbol | Hypothesis |
| --- | --- | --- |
| Depthwise residual enhancement | `z` | Provides local spatial enhancement |
| Channel gate | `C` | Selects useful channel responses |
| Boundary gate | `B` | Preserves useful structural high-frequency cues |
| Speckle gate | `S` | Suppresses speckle-like unstable high-frequency cues |
| Residual scale | `alpha` | Controls total module strength |
| Speckle scale | `beta` | Keeps suppression conservative |

## Design Rules

task_006 must follow these rules:

- Do not introduce official FADC mechanisms into the ablation variants.
- Do not add new dilation branches, FFT branches, wavelet branches, deformable
  convolutions, or external priors.
- Do not compare against V1.1 or V2a as paper-facing methods.
- Keep the same insertion positions as full V2d:
  - FCN: original deep FA-DCG positions.
  - U-Net: bottleneck position.
- Use the same training protocol as task_005a unless there is a clear bug.
- Use seed 2024 first for all variants.
- If a simplified variant matches or exceeds full V2d, later multi-seed
  confirmation can be run only for that simplified candidate and full V2d.

## Required Ablation Set

Run the following variants on both FCN and U-Net.

### A0: Full V2d

Purpose:

Use as the internal reference for all V2d ablations.

Structure:

```text
y = x + alpha * C * B * (1 - beta * S) * z
```

Existing task_005a results can be reused if the implementation and protocol are
unchanged. If the task_006 runner changes shared code, rerun A0.

### A1: V2d without Channel Gate

Purpose:

Test whether the inherited response-selective channel gate is necessary.

Structure:

```text
y = x + alpha * B * (1 - beta * S) * z
```

Interpretation:

- If A1 is close to full V2d, the channel gate may be unnecessary.
- If A1 drops, the channel gate is needed to stabilize or select enhanced
  responses.

### A2: V2d without Boundary Gate

Purpose:

Test whether the boundary-preserving high-frequency gate is necessary.

Structure:

```text
y = x + alpha * C * (1 - beta * S) * z
```

Notes:

- Keep the speckle gate unchanged.
- This variant tests whether suppressing speckle-like cues without explicitly
  preserving boundary cues is sufficient.

Interpretation:

- If A2 drops, boundary gate is necessary.
- If A2 matches full V2d, boundary gate may be redundant.

### A3: V2d without Speckle Gate

Purpose:

Test whether speckle suppression actually contributes.

Structure:

```text
y = x + alpha * C * B * z
```

Notes:

- Remove `S`, `beta`, and `local_var` from the computation.

Interpretation:

- If A3 drops, speckle gate is necessary.
- If A3 matches or exceeds full V2d, speckle gate may be over-designed or too
  weak to matter.

### A4: V2d without Boundary and Speckle Gates

Purpose:

Test whether the complete frequency double-gate is necessary.

Structure:

```text
y = x + alpha * C * z
```

Interpretation:

- If A4 is close to full V2d, the V2d frequency gates are not the main source
  of gain.
- If A4 drops, the boundary/speckle frequency path is meaningful.

### A5: V2d without All Gates

Purpose:

Test whether the module is mainly a depthwise residual enhancement block.

Structure:

```text
y = x + alpha * z
```

Interpretation:

- If A5 is close to full V2d, the gates are likely over-designed.
- If A5 drops strongly, at least one gate is important.

### A6: V2d without Multi-Scale Consistency

Purpose:

Test whether cross-scale structural consistency is necessary.

Structure:

```text
B = boundary_gate(local_high)
S = speckle_gate(local_high, local_var)
y = x + alpha * C * B * (1 - beta * S) * z
```

Interpretation:

- If A6 drops, `multi_scale_consistency` is useful for distinguishing real
  boundary structure from isolated local variation.
- If A6 matches full V2d, the descriptor can be simplified.

### A7: V2d without Local Variance

Purpose:

Test whether the SAR-specific variance proxy is necessary for speckle modeling.

Structure:

```text
B = boundary_gate(local_high, multi_scale_consistency)
S = speckle_gate(local_high, multi_scale_consistency)
y = x + alpha * C * B * (1 - beta * S) * z
```

Interpretation:

- If A7 drops, `local_var` is important for speckle-like high-frequency
  detection.
- If A7 matches full V2d, local variance may be unnecessary.

## Optional Follow-Up Variants

Only run these if the required ablations suggest a simpler structure is
competitive.

### M1: Boundary-Only Minimal V2d

```text
B = boundary_gate(local_high)
y = x + alpha * C * B * z
```

Use this if A3 is close to or better than full V2d and A6 shows
`multi_scale_consistency` is not important.

### M2: Speckle-Only Minimal V2d

```text
S = speckle_gate(local_var)
y = x + alpha * C * (1 - beta * S) * z
```

Use this if A2 is close to full V2d and A7 shows local variance is important.

### M3: Frequency Gates without Channel Gate

```text
y = x + alpha * B * (1 - beta * S) * z
```

This is the same as A1, but if it performs well, treat it as a candidate final
minimal V2d.

### M4: Fixed-Beta V2d

```text
beta = 0.1 fixed
y = x + alpha * C * B * (1 - beta * S) * z
```

Use this if full V2d works but learned `beta` remains close to initialization.

## Suggested Names and Files

Implementation should use explicit variant names so results are easy to audit:

```text
FADCGV2dAblation
FCNWithFADCGV2dAblation
UNetWithFADCGV2dAblation
```

Suggested files:

```text
SAR_FEM1/models/improved/fadcg_v2d_ablation.py
SAR_FEM1/models/improved/fcn_fadcg_v2d_ablation.py
SAR_FEM1/models/improved/unet_fadcg_v2d_ablation.py
scripts/task_006_test_v2d_ablation.py
scripts/task_006_v2d_ablation_repro.py
```

Suggested CLI variant names:

```text
full
no_channel
no_boundary
no_speckle
no_boundary_no_speckle
no_all_gates
no_msc
no_local_var
```

## Mandatory Experiments

For the first pass:

| Architecture | Variants | Seed | Epochs |
| --- | --- | ---: | ---: |
| FCN | A0-A7 | 2024 | 50 |
| U-Net | A0-A7 | 2024 | 50 |

Use the same data split, optimizer, loss, scheduler, image size, batch size, and
model insertion positions as task_005a.

Recommended command shape:

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

## Required Metrics

Save:

```text
metrics_per_seed.csv
metrics_summary.csv
task_006_report.md
```

Required metrics:

```text
architecture
variant
seed
miou
dice
params_m
inference_time_ms
gpu_memory_mb
best_epoch
delta_miou_vs_full_v2d
delta_dice_vs_full_v2d
delta_params_vs_full_v2d
status
```

## Required Diagnostics

For variants where a gate exists, save diagnostics under:

```text
figures/fadcg_v2d_ablation_diagnostics/
```

Diagnostics should include whichever fields apply to that variant:

```text
alpha_value
channel_gate_mean
channel_gate_std
boundary_gate_mean
boundary_gate_std
speckle_gate_mean
speckle_gate_std
beta_value
suppression_ratio
local_high_descriptor_mean
multi_scale_consistency_mean
local_variance_proxy_mean
```

For removed components, write `NA` in the diagnostics CSV rather than omitting
columns. This makes cross-variant comparison easier.

## Status Rules

Use full V2d as the reference inside each architecture.

Suggested thresholds:

```text
better_than_full:
  delta_miou_vs_full_v2d >= +0.001

equivalent_to_full:
  abs(delta_miou_vs_full_v2d) < 0.001

slightly_worse_but_simpler:
  -0.003 <= delta_miou_vs_full_v2d < -0.001
  and params or latency are reduced

important_drop:
  delta_miou_vs_full_v2d < -0.003
```

The exact threshold can be adjusted after observing seed variance, but do not
over-interpret differences smaller than 0.001 mIoU from a single seed.

## Decision Logic

After all A0-A7 variants finish, choose the final module by this logic:

### Case 1: Full V2d is clearly best

Condition:

```text
Full V2d beats every simplified variant by >= 0.003 mIoU on at least one
architecture and is not worse on the other architecture.
```

Conclusion:

```text
Keep full V2d. The ablation supports that the design is not over-engineered.
```

Paper interpretation:

```text
Both boundary preservation and speckle suppression are necessary for robust
frequency-aware SAR flood segmentation.
```

### Case 2: One simplified variant matches full V2d

Condition:

```text
A simplified variant is within 0.001 mIoU of full V2d on both architectures, or
is better on one architecture and no worse than 0.001 on the other.
```

Conclusion:

```text
Promote the simplified variant as the final minimal V2d candidate.
```

Next step:

```text
Run multi-seed confirmation only for full V2d and the simplified candidate.
```

Paper interpretation:

```text
The final module is a minimal sufficient SAR frequency selection block, not the
largest tested design.
```

### Case 3: Different architectures prefer different simplified variants

Condition:

```text
FCN and U-Net select different simplified variants.
```

Conclusion:

Prefer the simplest variant that is stable on both. Do not make the paper's
main method architecture-specific unless the performance gap is large and
consistent.

Paper interpretation:

```text
The module is plug-and-play, but the useful frequency correction strength can be
backbone-dependent.
```

### Case 4: All gates appear unnecessary

Condition:

```text
A5 no_all_gates is equivalent to full V2d.
```

Conclusion:

The current V2d explanation is too complex. Do not claim that boundary/speckle
frequency selection is the source of improvement unless diagnostics and
additional tests support it.

Possible next step:

```text
Rename the final method around residual depthwise enhancement rather than
speckle-robust frequency selection, or rerun stronger diagnostics.
```

## Paper-Facing Ablation Table

If task_006 confirms full V2d or a simplified minimal V2d, use a table like:

| Variant | Channel Gate | Boundary Gate | Speckle Gate | MSC | Local Var | Params M | mIoU | Dice |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Full V2d | yes | yes | yes | yes | yes | | | |
| w/o Channel | no | yes | yes | yes | yes | | | |
| w/o Boundary | yes | no | yes | no/yes | yes | | | |
| w/o Speckle | yes | yes | no | yes | no | | | |
| w/o B&S | yes | no | no | no | no | | | |
| w/o All Gates | no | no | no | no | no | | | |
| w/o MSC | yes | yes | yes | no | yes | | | |
| w/o Local Var | yes | yes | yes | yes | no | | | |

Use one table for FCN and one table for U-Net, or one combined table with
architecture as the first column.

## Expected Outcomes

The most useful outcomes are:

1. Full V2d wins clearly:
   - strong support for the current speckle-robust frequency-selection story.

2. `w/o Channel Gate` matches full V2d:
   - V2d can be simplified and made more distinct from earlier FA-DCG designs.

3. `w/o Speckle Gate` matches full V2d:
   - the speckle branch may be over-designed; use boundary-focused V2d instead.

4. `w/o Boundary Gate` matches full V2d:
   - speckle suppression may be the real useful component.

5. `w/o Local Var` matches full V2d:
   - the SAR speckle explanation is weaker; local variance may not be necessary.

6. `w/o MSC` matches full V2d:
   - multi-scale consistency can be removed for a simpler final module.

## Final Deliverable

task_006 should produce:

```text
exp/task_006_v2d_ablation_repro/<run_name>/metrics_summary.csv
exp/task_006_v2d_ablation_repro/<run_name>/metrics_per_seed.csv
exp/task_006_v2d_ablation_repro/<run_name>/task_006_report.md
logs/<date>_task_006_v2d_minimal_sufficient_experiment.md
```

The report must answer:

1. Which V2d component is most important?
2. Which component, if any, is redundant?
3. Is full V2d justified, or should the final paper version be simplified?
4. Does the conclusion hold for both FCN and U-Net?
5. What should be used as the final paper model name and structure?
