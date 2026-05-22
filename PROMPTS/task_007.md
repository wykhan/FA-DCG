# task_007: Channel-Integrated Speckle-Robust Frequency Gate

## Background

The current best candidate is **V2d-noMSC**. Its structure is:

```text
z = depthwise_residual_enhancement(x)
C = channel_gate(z)

local_low = avgpool3(x)
local_high = abs(x - local_low)
local_var = avgpool3(x * x) - local_low * local_low

B = boundary_gate(local_high)
S = speckle_gate(local_high, local_var)

y = x + alpha * C * B * (1 - beta * S) * z
```

where:

```text
speckle_robust_frequency_gate = B * (1 - beta * S)
```

This **speckle-robust frequency gate** is the intended core innovation for the
paper. It is well aligned with single-channel SAR flood segmentation because SAR
high-frequency responses are mixed:

- useful high-frequency responses: water-land boundaries, small water bodies,
  narrow channels;
- harmful high-frequency responses: residual speckle, locally unstable
  scattering, dark non-water textures and shadows.

The boundary gate `B` preserves useful structural high-frequency responses. The
speckle gate `S` suppresses speckle-like unstable high-frequency responses.

However, task_006 shows a paper-story risk:

```text
channel gate contributes more stably than the high-frequency selection gate.
```

Observed task_006 drops:

```text
no_channel:
  FCN   -0.009020 mIoU
  U-Net -0.003754 mIoU

no_boundary:
  FCN   -0.002020 mIoU
  U-Net +0.001162 mIoU

no_speckle:
  FCN   -0.001875 mIoU
  U-Net -0.005039 mIoU
```

If the final method remains:

```text
channel gate + speckle-robust frequency gate
```

then reviewers may conclude that the strongest contribution comes from the
inherited channel gate rather than from the proposed SAR-specific frequency
mechanism.

## Core Motivation

The goal of task_007 is to make the implementation contribution and the paper
innovation consistent.

中文动机：

```text
V2d-noMSC 是目前最好的模型，但它仍然由残差结构、通道门、speckle-robust frequency gate 三部分组成。
从论文角度，speckle-robust frequency gate 才是核心创新点。
从实验角度，通道门贡献更稳定。
因此 V3.0 应将通道门整合进 speckle-robust frequency gate，使论文创新点和实验精度提升来源统一。
```

The new paper-facing module should be:

```text
residual enhancement + channel-integrated speckle-robust frequency gate
```

not:

```text
residual enhancement + channel gate + frequency gate
```

## Overall Goal

Design and evaluate **FA-DCG V3.0**:

```text
y = x + alpha * G_srfg * z
```

where `G_srfg` is a **channel-integrated speckle-robust frequency gate**.

The independent channel gate `C = channel_gate(z)` should be removed. Its
response-selection ability should be absorbed into the boundary-preserving and
speckle-suppressing frequency selection process.

The central question is:

```text
Can the independent channel gate be integrated into the speckle-robust
frequency gate without losing the performance advantage of V2d-noMSC?
```

## Paper-Facing Statement

Recommended English statement:

```text
We propose a channel-integrated speckle-robust frequency gate that incorporates
channel response context into boundary-preserving and speckle-suppressing
frequency selection for single-channel SAR flood segmentation.
```

Recommended Chinese statement:

```text
本文提出通道整合的 speckle-robust frequency gate，将通道响应上下文融入边界保持与斑点抑制两个频率选择过程，从而在统一结构中完成通道选择和频率选择。
```

## Scope

task_007 is a V3 design task. It should not be treated as another minor V2d
ablation.

Allowed:

- reuse the V2d-noMSC residual enhancement path `z`;
- reuse the V2d-noMSC descriptors `local_high` and `local_var`;
- keep the B/S design of the speckle-robust frequency gate;
- inject channel response context into `B`, `S`, or both;
- compare against V2d-noMSC and full V2d;
- run seed 2024 first, then multi-seed for the best candidate.

Avoid:

- retaining an independent multiplicative `C * B * (1 - beta * S)` design as
  the final V3 model;
- replacing B/S with a generic frequency logit that loses the SAR-specific
  boundary-vs-speckle interpretation;
- FFT branches;
- wavelet branches;
- external priors such as water masks, DEM, hydrodynamic outputs, or
  multi-source inputs;
- official heavy FADC mechanisms;
- new complex multi-branch dilation designs.

## Reference Baselines

Use existing task_006 results as references.

### Full V2d

```text
FCN   mIoU = 0.845375, Dice = 0.907630
U-Net mIoU = 0.893472, Dice = 0.938617
```

### V2d-noMSC

```text
FCN   mIoU = 0.848363, Dice = 0.910157
U-Net mIoU = 0.895189, Dice = 0.939710
```

V3 should first aim to match or exceed V2d-noMSC on both FCN and U-Net.

## Shared Notation

For all V3 candidates:

```text
z = depthwise_residual_enhancement(x)

local_low = avgpool3(x)
local_high = abs(x - local_low)
local_var = avgpool3(x * x) - local_low * local_low
var_norm = local_var / (mean(local_var over H,W) + eps)
```

Channel context should be derived from the residual response:

```text
channel_context = MLP(GAP(z))
```

Important:

```text
channel_context is not an independent output gate.
```

It is evidence used by the speckle-robust frequency gate to decide which
frequency responses should be preserved or suppressed.

The output should follow:

```text
G_srfg = B_ci * (1 - beta * S_ci)
y = x + alpha * G_srfg * z
```

where:

- `B_ci` is a channel-integrated boundary-preserving frequency gate;
- `S_ci` is a channel-integrated speckle-suppression frequency gate;
- `ci` means channel-integrated.

Initialization rule:

```text
B_ci starts near 1.0
S_ci starts near 0.0
G_srfg starts near 1.0
```

This keeps the module close to conservative residual enhancement at the start of
training.

## Candidate V3 Variants

### V3-SRFG-A: Logit-Injected Channel-Integrated B/S Gate

This is the highest-priority candidate.

Purpose:

Preserve the V2d-noMSC B/S structure while integrating channel response context
directly into both boundary preservation and speckle suppression.

Structure:

```text
z = depthwise_residual_enhancement(x)

channel_context = MLP(GAP(z))

L_b_local = boundary_head(local_high)
L_s_local = speckle_head(local_high, var_norm)

L_b_channel = channel_boundary_head(channel_context)
L_s_channel = channel_speckle_head(channel_context)

B_ci = 2 * sigmoid(L_b_local + L_b_channel)
S_ci = sigmoid(L_s_local + L_s_channel)

G_srfg = B_ci * (1 - beta * S_ci)

y = x + alpha * G_srfg * z
```

Implementation notes:

- `L_b_local` and `L_s_local` should be spatial maps, preferably generated by
  grouped/depthwise convolution.
- `L_b_channel` and `L_s_channel` should be `C x 1 x 1` maps and broadcast over
  `H x W`.
- Initialize `L_b_local`, `L_b_channel`, `L_s_local`, and `L_s_channel` so that:

```text
B_ci ~= 1.0
S_ci ~= 0.0
```

Suggested initialization:

- zero weights and zero bias for boundary logits;
- zero weights and negative bias, e.g. `-4.0`, for speckle logits.

Why this is the main candidate:

```text
Old V2d-noMSC:
  G = C * B * (1 - beta * S)

New V3-SRFG-A:
  G = B(channel context, local_high)
      * (1 - beta * S(channel context, local_high, local_var))
```

The independent channel gate disappears, but its evidence is integrated into the
speckle-robust frequency gate.

Expected paper story:

```text
The proposed SRFG uses channel response context to guide both useful
high-frequency preservation and speckle-like high-frequency suppression.
```

### V3-SRFG-B: Concatenation-Based Channel-Integrated B/S Gate

Purpose:

Test whether direct descriptor concatenation is better than logit injection.

Structure:

```text
z = depthwise_residual_enhancement(x)

channel_context = MLP(GAP(z))
channel_map = broadcast(channel_context)

B_ci = 2 * sigmoid(boundary_head([local_high, channel_map]))
S_ci = sigmoid(speckle_head([local_high, var_norm, channel_map]))

G_srfg = B_ci * (1 - beta * S_ci)

y = x + alpha * G_srfg * z
```

Interpretation:

- If B outperforms A, channel context may need local convolutional interaction
  with frequency descriptors instead of simple logit addition.
- If A matches or outperforms B, prefer A because it is cleaner and lighter.

### V3-SRFG-C: Channel Context Only in Boundary Gate

Purpose:

Test whether the channel gate mainly helps preserve useful high-frequency
responses.

Structure:

```text
channel_context = MLP(GAP(z))

B_ci = 2 * sigmoid(boundary_head(local_high, channel_context))
S = sigmoid(speckle_head(local_high, var_norm))

G_srfg = B_ci * (1 - beta * S)

y = x + alpha * G_srfg * z
```

Interpretation:

- If C is close to A, channel selection mainly supports boundary and useful
  structure enhancement.
- If C is weak, channel context should also affect speckle suppression.

### V3-SRFG-D: Channel Context Only in Speckle Gate

Purpose:

Test whether the channel gate mainly helps suppress harmful or unstable
high-frequency responses.

Structure:

```text
channel_context = MLP(GAP(z))

B = 2 * sigmoid(boundary_head(local_high))
S_ci = sigmoid(speckle_head(local_high, var_norm, channel_context))

G_srfg = B * (1 - beta * S_ci)

y = x + alpha * G_srfg * z
```

Interpretation:

- If D is close to A, channel selection mainly supports speckle-like response
  suppression.
- If D is weak, channel context should also guide boundary preservation.

### V3-SRFG-E: Shared Channel-Conditioned Beta

Purpose:

Test a more conservative integration strategy where channel context controls the
suppression strength rather than directly changing `S`.

Structure:

```text
channel_context = MLP(GAP(z))

B = 2 * sigmoid(boundary_head(local_high))
S = sigmoid(speckle_head(local_high, var_norm))

beta_c = sigmoid(beta_head(channel_context))

G_srfg = B * (1 - beta_c * S)

y = x + alpha * G_srfg * z
```

Interpretation:

- This variant preserves the original B/S local structure most strongly.
- It may be stable, but it integrates channel context less deeply than A/B.
- Use it as a fallback if A/B are unstable.

## Recommended Experiment Order

First-round seed-2024 screening:

```text
V3-SRFG-A
V3-SRFG-C
V3-SRFG-D
V2d-noMSC reference
```

If compute is limited, run only:

```text
V3-SRFG-A vs V2d-noMSC
```

If V3-SRFG-A is promising, run:

```text
V3-SRFG-B
V3-SRFG-E
```

Priority:

1. V3-SRFG-A
2. V3-SRFG-C
3. V3-SRFG-D
4. V3-SRFG-B
5. V3-SRFG-E

Reason:

- A is the cleanest match to the paper objective.
- C and D reveal where channel context helps: boundary preservation or speckle
  suppression.
- B tests whether stronger local interaction is needed.
- E is a conservative fallback.

## Required Comparisons

For each V3 candidate, compare against:

```text
FCN + V2d-noMSC
U-Net + V2d-noMSC
```

If compute permits, also include:

```text
FCN + full V2d
U-Net + full V2d
```

The main result table should include:

```text
architecture
variant
params
mIoU
Dice
delta_vs_v2d_nomsc
delta_vs_full_v2d
status
```

## Diagnostics

Each V3 module should expose diagnostics that prove the gate remains a
speckle-robust frequency gate and that channel context is integrated into it.

Required diagnostics:

```text
alpha_value
beta_value or beta_c_mean
boundary_gate_mean
boundary_gate_std
speckle_gate_mean
speckle_gate_std
srfg_gate_mean
srfg_gate_std
local_high_descriptor_mean
local_variance_proxy_mean
channel_context_mean
channel_context_std
```

For V3-SRFG-A, additionally record:

```text
boundary_local_logit_mean
boundary_channel_logit_mean
speckle_local_logit_mean
speckle_channel_logit_mean
```

For V3-SRFG-C/D, record whether channel context is injected into boundary or
speckle:

```text
channel_in_boundary = true/false
channel_in_speckle = true/false
```

These diagnostics are needed for paper figures and ablations. They should show:

1. `B_ci` is not collapsed to a constant;
2. `S_ci` remains conservative rather than over-suppressing all high frequency;
3. channel context affects the SRFG instead of acting as a detached separate
   attention block.

## Acceptance Criteria

A V3 candidate is promising if it satisfies at least one of the following:

1. It exceeds V2d-noMSC on both FCN and U-Net.
2. It matches V2d-noMSC within a small tolerance while removing the independent
   channel gate and giving a cleaner paper explanation.
3. It improves one backbone clearly and does not hurt the other more than a
   small tolerance.

Suggested seed-2024 tolerance:

```text
abs(delta mIoU) <= 0.0015 can be treated as roughly equivalent.
```

Multi-seed confirmation should be run for:

- the best V3-SRFG candidate;
- V2d-noMSC;
- full V2d if needed for the paper.

Recommended seeds:

```text
42, 123, 2024
```

## Paper-Facing Decision Rules

If V3-SRFG-A matches or exceeds V2d-noMSC:

```text
Use V3-SRFG-A as FA-DCG V3.0.
Main claim: channel-integrated speckle-robust frequency gating.
```

If V3-SRFG-C is close to A:

```text
Channel response context mainly supports useful high-frequency preservation.
This can be used to explain that boundary-related frequency selection absorbs
most of the channel gate's benefit.
```

If V3-SRFG-D is close to A:

```text
Channel response context mainly supports speckle-like high-frequency
suppression.
This can strengthen the SAR-specific speckle robustness story.
```

If V3-SRFG-B outperforms A:

```text
Use B only if the added parameters are acceptable.
Explain that local interaction between channel context and frequency
descriptors is necessary.
```

If all V3-SRFG variants fail:

```text
Keep V2d-noMSC as final method, but revise the paper story:
the channel gate should be described as a response-selection component coupled
with the speckle-robust frequency gate, not as an unrelated inherited module.
```

## Suggested Files

New implementation files:

```text
SAR_FEM1/models/improved/fadcg_v3_srfg.py
SAR_FEM1/models/improved/fcn_fadcg_v3_srfg.py
SAR_FEM1/models/improved/unet_fadcg_v3_srfg.py
```

New experiment/test files:

```text
scripts/task_007_test_v3_srfg.py
scripts/task_007_v3_srfg_repro.py
```

Suggested output root:

```text
exp/task_007_v3_srfg_repro/20260522_seed2024
```

## Minimal First Implementation

Start with V3-SRFG-A.

Required first command target:

```text
python scripts/task_007_v3_srfg_repro.py \
  --output-root exp/task_007_v3_srfg_repro/20260522_seed2024 \
  --architectures fcn unet \
  --variants v3_srfg_a \
  --seeds 2024 \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 2
```

If V3-SRFG-A is close to or better than V2d-noMSC, extend to:

```text
v3_srfg_c
v3_srfg_d
v3_srfg_b
v3_srfg_e
```

Then run multi-seed confirmation for the best V3-SRFG variant.

