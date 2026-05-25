# task_007_GPT: Frequency-Conditioned Response Gate for FA-DCG

## Background

This repository is for the FA-DCG SAR flood segmentation project.

Previous experiments show that:

1. FA-DCG V1.1 is essentially:

   ```text
   z = depthwise_residual_enhancement(x)
   C = channel_response_gate(z)
   y = x + alpha * C * z
   ```

   It is a strong and stable baseline, but its channel response gate is generic and not sufficiently SAR-specific.

2. FA-DCG V2d-noMSC is currently the best candidate in the existing experiments:

   ```text
   z = depthwise_residual_enhancement(x)

   C = channel_response_gate(z)

   local_low = avgpool3(x)
   local_high = abs(x - local_low)
   local_var = avgpool3(x * x) - local_low * local_low

   B = boundary_gate(local_high)
   S = speckle_gate(local_high, local_var)

   y = x + alpha * C * B * (1 - beta * S) * z
   ```

3. However, the current experimental evidence shows a paper-story risk:

   ```text
   The main performance gain comes from residual enhancement + channel response gate.
   The independent high-frequency selection gate is more SAR-specific, but its overall mIoU gain is small.
   ```

Therefore, task_007 should not simply make `B` and `S` more complicated. The goal is not to create a larger stack of gates. The goal is to transform the existing generic channel response gate into a SAR-frequency-conditioned response gate.

## Core Motivation

The final module should align the implementation contribution with the paper story.

Instead of:

```text
residual enhancement + generic channel gate + independent high-frequency gate
```

task_007 should evaluate:

```text
residual enhancement + frequency-conditioned response gate
```

The frequency-conditioned response gate should use SAR-relevant local frequency descriptors to modulate channel response selection.

The key SAR descriptors are:

```text
local_high = abs(x - avgpool3(x))
local_var  = avgpool3(x * x) - avgpool3(x)^2
```

Interpretation:

- `local_high` captures local high-frequency residuals, which may correspond to water-land boundaries, small inundated regions, narrow channels, speckle, or shadow edges.
- `local_var` is a lightweight proxy for local scattering instability and speckle-like fluctuation.
- These descriptors should condition the response gate, not form an excessively complex standalone boundary/speckle gate.

## Design Principle

Do not describe or implement the new method as a generic attention block.

The paper-facing concept should be:

```text
Frequency-Conditioned Response Gate
```

or:

```text
Speckle-Robust Frequency-Conditioned Response Gate
```

The response gate should answer:

```text
Which residual enhancement channels are reliable under SAR local high-frequency and local-variance conditions?
```

This is different from:

```text
Which high-frequency pixels should be enhanced?
```

The channel response gate remains responsible for response selection. SAR frequency descriptors provide reliability conditioning.

## Scope

Allowed:

- Reuse the V1.1 residual enhancement path.
- Reuse the V2d-noMSC local descriptors: `local_high` and `local_var`.
- Design frequency-conditioned response gates.
- Compare against V1.1 and V2d-noMSC.
- Run seed 2024 first, then multi-seed for the best candidate.

Avoid:

- Do not use the term `SE` in code comments, reports, class names, or paper-facing descriptions.
- Do not overcomplicate `B` and `S`.
- Do not implement five or more large variants.
- Do not add FFT branches.
- Do not add wavelet branches.
- Do not add external priors such as DEM, water masks, hydrodynamic outputs, or multi-source inputs.
- Do not reintroduce multi-dilation or official FADC mechanisms.
- Do not use a final design that looks like a generic channel module plus unrelated frequency gates.

## Required Experimental Structures

### Structure 0: V1.1 Reference

Purpose: strong baseline showing residual enhancement plus the original channel response gate.

```text
z = depthwise_residual_enhancement(x)
C0 = channel_response_gate(z)
y = x + alpha * C0 * z
```

Use existing V1.1 results if available, but the script should support rerunning it.

### Structure 1: V2d-noMSC Reference

Purpose: current best candidate and main reference for V3.

```text
z = depthwise_residual_enhancement(x)
C0 = channel_response_gate(z)

local_low = avgpool3(x)
local_high = abs(x - local_low)
local_var = avgpool3(x * x) - local_low * local_low

B = boundary_gate(local_high)
S = speckle_gate(local_high, local_var)

y = x + alpha * C0 * B * (1 - beta * S) * z
```

Use existing V2d-noMSC results if available, but the script should support rerunning it.

### Structure 2: V3-FCRG-A, Direct Frequency-Conditioned Response Gate

Purpose: simplest fusion of channel response selection and SAR frequency descriptors.

Instead of producing a generic channel gate from `z` alone, generate the response gate from:

```text
GAP(z), GAP(local_high), GAP(local_var)
```

Structure:

```text
z = depthwise_residual_enhancement(x)

local_low = avgpool3(x)
local_high = abs(x - local_low)
local_var = avgpool3(x * x) - local_low * local_low
var_norm = local_var / (mean(local_var over H,W) + eps)

d = concat(
    GAP(z),
    GAP(local_high),
    GAP(var_norm)
)

Cf = frequency_conditioned_response_gate(d)

y = x + alpha * Cf * z
```

Implementation requirements:

- `Cf` shape should be `B x C x 1 x 1`.
- `Cf` should be generated by a lightweight MLP or 1x1-conv block.
- `Cf` should be initialized so that the module starts close to V1.1-like behavior, if practical.
- Do not call this structure a generic channel module in reports.

Paper-facing interpretation:

```text
The response gate selects residual enhancement channels using both residual semantic response and SAR local frequency reliability descriptors.
```

### Structure 3: V3-FCRG-B, Residual-Gate with Frequency Modulation

Purpose: highest-priority candidate.

It preserves the strong V1.1 gate as the base response gate and adds a learnable SAR-frequency modulation term. This is safer than replacing the gate completely.

Structure:

```text
z = depthwise_residual_enhancement(x)

local_low = avgpool3(x)
local_high = abs(x - local_low)
local_var = avgpool3(x * x) - local_low * local_low
var_norm = local_var / (mean(local_var over H,W) + eps)

C0 = channel_response_gate(z)

R = tanh(
    frequency_reliability_mlp(
        concat(GAP(local_high), GAP(var_norm))
    )
)

Cf = C0 * (1 + lambda * R)

y = x + alpha * Cf * z
```

Initialization requirements:

```text
lambda = 0 at initialization
```

This makes the model initially equivalent to V1.1. During training, the model can learn how much SAR frequency reliability should modulate the response gate.

Implementation requirements:

- `R` shape should be `B x C x 1 x 1`.
- `lambda` can be a learnable scalar or a learnable per-channel vector.
- Start with learnable scalar `lambda`.
- Clamp or regularize only if instability appears.
- This variant should be prioritized because it is conservative and directly addresses the paper-story risk.

Paper-facing interpretation:

```text
V1.1 provides a stable residual response gate. V3-FCRG-B makes this gate SAR-aware by conditioning it on local high-frequency residuals and local variance.
```

### Structure 4: V3-FCRG-C, Frequency-Modulated Response Gate with Lightweight Spatial Reliability

Purpose: optional variant testing whether a very light spatial reliability map adds value beyond channel-level frequency conditioning.

Structure:

```text
z = depthwise_residual_enhancement(x)

local_low = avgpool3(x)
local_high = abs(x - local_low)
local_var = avgpool3(x * x) - local_low * local_low
var_norm = local_var / (mean(local_var over H,W) + eps)

C0 = channel_response_gate(z)

R = tanh(
    frequency_reliability_mlp(
        concat(GAP(local_high), GAP(var_norm))
    )
)

Cf = C0 * (1 + lambda * R)

A = 1 + rho * tanh(
    depthwise_conv(
        concat(local_high, var_norm)
    )
)

y = x + alpha * Cf * A * z
```

Initialization requirements:

```text
lambda = 0
rho = 0
```

This makes the model initially close to V1.1.

Use this only after Structure 2 and Structure 3 have run. It should not be the first priority.

Paper-facing interpretation:

```text
The module mainly performs frequency-conditioned response gating. The optional spatial reliability term tests whether local high-frequency reliability should also spatially regulate residual enhancement.
```

## Recommended Experiment Order

### Stage A: Seed-2024 Screening

Run the following on both FCN and U-Net:

```text
V1.1 reference
V2d-noMSC reference
V3-FCRG-A
V3-FCRG-B
```

If compute is limited, run only:

```text
V2d-noMSC reference
V3-FCRG-B
```

Structure 4 should only be run if V3-FCRG-A or V3-FCRG-B is close to or better than V2d-noMSC.

### Stage B: Optional Spatial Reliability

Run:

```text
V3-FCRG-C
```

only if Stage A shows that frequency-conditioned response gating is promising.

### Stage C: Multi-Seed Confirmation

Run seeds:

```text
42, 123, 2024
```

for:

```text
best V3-FCRG candidate
V1.1 reference
V2d-noMSC reference
```

If V1.1 and V2d-noMSC existing multi-seed results are reused, state this clearly in the report.

## Acceptance Criteria

A V3 candidate is promising if it satisfies at least one of the following:

1. It exceeds V2d-noMSC on both FCN and U-Net.
2. It matches V2d-noMSC within a small tolerance while providing a cleaner paper explanation.
3. It improves U-Net and does not hurt FCN beyond a small tolerance.
4. It improves mechanism consistency: the main effective response gate is now frequency-conditioned instead of being a generic channel response gate.

Suggested seed-2024 tolerance:

```text
abs(delta mIoU) <= 0.0015
```

For multi-seed confirmation:

```text
mean mIoU should not be lower than V2d-noMSC by more than 0.0015
```

## Required Metrics

For each architecture and variant, report:

```text
architecture
variant
seed
mIoU
Dice
params_m
flops_g
inference_time_ms
gpu_memory_mb
best_epoch
delta_mIoU_vs_V1.1
delta_mIoU_vs_V2d_noMSC
delta_Dice_vs_V1.1
delta_Dice_vs_V2d_noMSC
status
```

Status labels:

```text
better_than_nomsc
comparable_to_nomsc
worse_than_nomsc
failed
```

## Required Diagnostics

Each V3 module must expose diagnostics.

For V3-FCRG-A:

```text
alpha_value
response_gate_mean
response_gate_std
local_high_gap_mean
local_var_gap_mean
```

For V3-FCRG-B:

```text
alpha_value
lambda_value
base_response_gate_mean
base_response_gate_std
frequency_modulation_mean
frequency_modulation_std
final_response_gate_mean
final_response_gate_std
local_high_gap_mean
local_var_gap_mean
```

For V3-FCRG-C:

```text
alpha_value
lambda_value
rho_value
base_response_gate_mean
base_response_gate_std
frequency_modulation_mean
frequency_modulation_std
spatial_reliability_mean
spatial_reliability_std
final_gate_mean
final_gate_std
local_high_gap_mean
local_var_gap_mean
```

The report should check:

1. The frequency modulation does not collapse to zero after training, unless performance is unchanged and the model chooses to ignore it.
2. The final gate does not collapse to all 0 or all 1.
3. The module remains close to V1.1 in cost.
4. The added frequency conditioning gives either measurable improvement or a cleaner, evidence-backed explanation.

## Suggested Files

New implementation files:

```text
SAR_FEM1/models/improved/fadcg_v3_fcrg.py
SAR_FEM1/models/improved/fcn_fadcg_v3_fcrg.py
SAR_FEM1/models/improved/unet_fadcg_v3_fcrg.py
```

New scripts:

```text
scripts/task_007_test_v3_fcrg.py
scripts/task_007_v3_fcrg_repro.py
```

Suggested output root:

```text
exp/task_007_v3_fcrg_repro/<YYYYMMDD_HHMMSS>/
```

Required outputs:

```text
task_007_report.md
metrics_per_seed.csv
metrics_summary.csv
model_complexity.csv
frequency_gate_diagnostics.csv
run_commands.sh
environment.txt
logs/
figures/
```

## Report Requirements

The report must include:

1. Executive summary.
2. Reminder that this is still validation-protocol evaluation, not final independent-test testing.
3. Explanation of why task_007 avoids overcomplicating B/S.
4. Description of the evaluated structures:
   - V1.1 reference
   - V2d-noMSC reference
   - V3-FCRG-A
   - V3-FCRG-B
   - V3-FCRG-C if run
5. Main result table.
6. Complexity and speed comparison.
7. Diagnostics table.
8. Interpretation:
   - Does frequency conditioning improve or match V2d-noMSC?
   - Is V3-FCRG-B better than direct fusion?
   - Does optional spatial reliability help?
   - Should the final paper method use V2d-noMSC or a V3-FCRG variant?
9. Recommendation for the final paper method.

## Paper-Facing Decision Rules

If V3-FCRG-B matches or exceeds V2d-noMSC:

```text
Use V3-FCRG-B as the final FA-DCG variant.
Main claim: speckle-robust frequency-conditioned response gating.
```

If V3-FCRG-A exceeds V3-FCRG-B:

```text
Use V3-FCRG-A only if the improvement is consistent and the initialization is stable.
Main claim: direct frequency-conditioned response gating.
```

If V3-FCRG-C exceeds both A and B:

```text
Use V3-FCRG-C only if the added cost is small and the diagnostics show that spatial reliability is meaningful.
```

If all V3-FCRG variants fail:

```text
Keep V2d-noMSC as the final method.
Paper story should state that the high-frequency gate complements the residual response gate, but its overall mIoU contribution is modest.
```

## Minimal First Command

Start with:

```bash
python scripts/task_007_v3_fcrg_repro.py \
  --output-root exp/task_007_v3_fcrg_repro/$(date +%Y%m%d_%H%M%S) \
  --architectures fcn unet \
  --variants v1_1 v2d_nomsc v3_fcrg_a v3_fcrg_b \
  --seeds 2024 \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 2
```

If V3-FCRG-A or V3-FCRG-B is promising, continue with:

```bash
python scripts/task_007_v3_fcrg_repro.py \
  --output-root exp/task_007_v3_fcrg_repro/$(date +%Y%m%d_%H%M%S) \
  --architectures fcn unet \
  --variants v3_fcrg_c \
  --seeds 2024 \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 2
```

Then run multi-seed confirmation for the best candidate.
