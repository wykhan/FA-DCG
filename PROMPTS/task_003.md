# task_003: Implement an Officially Aligned FADC Version

## Background

The current repository contains FA-DCG V1.0 and FA-DCG V1.1:

- **FA-DCG V1.0**: the original `LightFADC` implementation. It behaves mainly
  as residual depthwise feature enhancement plus channel gating, and does not
  truly implement the official FADC paper's adaptive dilation mechanism.
- **FA-DCG V1.1**: the accepted optimized baseline. It keeps the effective V1.0
  computation path but fixes the main engineering issues through vectorized
  grouped convolution and constructor-defined modules.

The official FADC paper is stored locally:

```text
doc/Chen_Frequency-Adaptive_Dilated_Convolution_for_Semantic_Segmentation_CVPR_2024_paper.pdf
```

Official source code:

```text
https://github.com/ying-fu/FADC
```

The official FADC paper proposes three key strategies:

1. **AdaDR**: Adaptive Dilation Rate. Dynamically adjusts dilation rates
   spatially based on local frequency components.
2. **AdaKern**: Adaptive Kernel. Decomposes convolution weights into
   low-frequency and high-frequency components and adaptively adjusts their
   ratio.
3. **FreqSelect**: Frequency Selection. Reweights high- and low-frequency
   feature components in a spatially variant manner.

The official code includes:

```text
FrequencySelection
AdaptiveDilatedConv
AdaptiveDilatedDWConv
OmniAttention
```

It also depends on:

```text
mmcv.ops.ModulatedDeformConv2d
mmcv.ops.modulated_deform_conv2d
MMSegmentation registry/config ecosystem
```

## Important Separation from task_004

This task must only implement and evaluate an FADC version that is as aligned
as practical with the official paper and official code.

Do not design the SAR-specific lightweight improved version in this task.

The SAR-oriented and lightweight redesign should be deferred to:

```text
task_004: Design FA-DCG V2.0 for SAR flood segmentation
```

Therefore:

- task_003 = official-FADC-aligned implementation and feasibility validation;
- task_004 = SAR-specific lightweight FA-DCG V2.0 design and experiments.

Do not call the task_003 module FA-DCG V2.0.

Use names such as:

```text
FADC
OfficialFADC
FADCAligned
FCN + FADC
U-Net + FADC
```

## Feasibility Judgment

The task is feasible as a local PyTorch-compatible implementation.

Do not use the official MMCV/MMSeg implementation path in task_003. The official
repository should be used only as a design reference.

Do not install or depend on:

```text
mmcv-full
mmsegmentation
mmcv.ops.ModulatedDeformConv2d
mmcv.ops.modulated_deform_conv2d
```

Reason:

- the current project is a lightweight PyTorch FCN/U-Net codebase;
- MMCV custom CUDA operators are version-sensitive;
- adding MMSeg/MMCV would make this repository harder to maintain;
- task_003 should produce a self-contained implementation that can run with the
  current environment.

Implement a local PyTorch-compatible FADC module that follows the official
paper's structure as closely as possible while avoiding MMSeg registry logic
and custom CUDA operators.

Required alignment:

- implement Frequency Selection;
- implement adaptive dilation behavior;
- implement adaptive kernel decomposition or a clearly documented equivalent;
- use official code and paper as reference;
- document any deviations from the official implementation.

Because `ModulatedDeformConv2d` must not be used in task_003, approximate AdaDR
through differentiable spatial branch mixing over dilation rates. This must be
documented as an approximation of the official FADC adaptive dilation idea, not
as an exact port of the official implementation.

## Main Goal

Implement and evaluate an FADC-aligned module in the current flood segmentation
codebase.

The task should answer:

```text
Can an official-FADC-aligned module be made runnable in this repository, and how
does it compare with FA-DCG V1.1 under the current validation protocol?
```

The task must complete experiments for:

```text
FCN + FADC
U-Net + FADC
```

at seed 2024.

This is not yet the SAR-specific FA-DCG V2.0 task.

## Scope Control

- Do not delete FA-DCG V1.0 or V1.1.
- Do not rename FA-DCG V1.1.
- Do not claim task_003 is FA-DCG V2.0.
- Do not modify the dataset split.
- Do not claim independent test-set results.
- Do not introduce MMCV/MMSeg dependencies.
- Do not import `mmcv.ops` or rely on custom CUDA operators.
- Do not copy large parts of the official repository wholesale into this
  project.
- If official source snippets are adapted, keep them focused, cite the source
  in comments or report, and document the license compatibility.

## Required Files

Create:

```text
SAR_FEM1/models/improved/fadc_aligned.py
SAR_FEM1/models/improved/fcn_fadc_aligned.py
SAR_FEM1/models/improved/unet_fadc_aligned.py
scripts/task_003_test_fadc_aligned.py
scripts/task_003_fadc_aligned_repro.py
```

Recommended class names:

```python
class FrequencySelectionAligned(nn.Module):
    ...

class FADCAligned(nn.Module):
    ...

class FCNWithFADCAligned(FCN):
    ...

class UNetWithFADCAligned(UNet):
    ...
```

## Implementation Requirements

### 1. Frequency Selection

Implement a `FrequencySelectionAligned` module based on the official
`FrequencySelection` concept.

Support at least one stable mode:

```text
lp_type = "avgpool" or "laplacian"
```

Optional if practical:

```text
lp_type = "freq"
```

The module should decompose features into high/low frequency bands and learn
spatially variant weights for each band.

Minimum behavior:

```text
high_i = previous_low - low_i
low_final = final low-frequency component
x_selected = sum_i weight_i * high_i + weight_low * low_final
```

Requirements:

- all operations differentiable;
- no per-channel Python convolution loop;
- no `.item()` or integer conversion for learned selection;
- output shape equals input shape;
- expose diagnostics for high/low frequency weights.

### 2. Adaptive Dilation Rate

Official FADC uses deformable-convolution offsets to implement spatially
adaptive dilation.

Task_003 must not use `mmcv.ops.ModulatedDeformConv2d`. Implement a documented
local approximation that follows the official AdaDR motivation with spatially
adaptive dilation-branch mixing:

```text
dilation branches = [1, 2, 3, 4]
w = softmax(branch_weight_head(x_freq), dim=branch)
z = sum_k w_k * depthwise_conv_dilation_k(x_freq)
```

The report must clearly state:

```text
This is a local PyTorch FADC-aligned implementation that approximates official
AdaDR with spatial dilation-branch mixing because task_003 does not use
MMCV/MMSEG custom deformable convolution operators.
```

### 3. Adaptive Kernel

Implement an AdaKern-aligned component.

Minimum acceptable implementation:

```text
kernel_mean = mean(kernel over spatial dimensions)
kernel_high = kernel - kernel_mean
kernel_adapted = low_gate * kernel_mean + high_gate * kernel_high
```

Use a lightweight attention block similar to official `OmniAttention` to
generate low/high gates.

If adaptive per-sample kernels are too costly or awkward in the first working
version, implement AdaKern as an optional mode and document whether it was
enabled in the main experiment.

### 4. FADCAligned Output

The FADC-aligned block should return the same spatial and channel shape as its
input when used as a feature enhancement block.

Recommended wrapper behavior:

```text
y = projection_or_identity(x)
y = FADCAligned(y)
out = x + alpha * y
```

Where:

- `alpha` is learnable and initialized conservatively;
- residual fusion is used for stable integration into existing FCN/U-Net;
- any deviation from official FADC must be documented.

## Insertion Positions

Because task_003 is about FADC alignment, keep insertion positions conservative
and easy to compare.

Recommended first positions:

### U-Net

```text
after encoder stage 3: C=256
```

Optional:

```text
after encoder stage 2: C=128
bottleneck: C=1024
```

### FCN

```text
after conv3: C=256
```

Optional:

```text
after conv4: C=512
```

Do not insert FADC at the FCN 4096-channel classifier stage.

## Required Tests

Create:

```text
scripts/task_003_test_fadc_aligned.py
```

Test:

1. `FrequencySelectionAligned(C=128)`
2. `FADCAligned(C=128)`
3. `FADCAligned(C=256)`
4. `FCNWithFADCAligned`
5. `UNetWithFADCAligned`

Required checks:

- output shape equals expected shape;
- no NaN or Inf;
- frequency weights are finite;
- adaptive dilation weights or offsets are finite;
- branch weights sum to 1 if fallback branch mixing is used;
- `alpha` is trainable if residual fusion is used;
- frequency selection, adaptive dilation, adaptive kernel, and gate parameters
  receive gradients;
- no module is created inside `forward`;
- no `.item()` is used for learned adaptive behavior.

## Experiment Plan

Create:

```text
exp/task_003_fadc_aligned_repro/<YYYYMMDD_HHMMSS>/
```

Required files:

```text
task_003_report.md
implementation_audit.md
metrics_per_seed.csv
metrics_summary.csv
model_complexity.csv
speed_benchmark.csv
run_commands.sh
environment.txt
logs/
figures/
```

### Stage 0: Feasibility and Smoke Test

Run:

- source check confirming no MMCV/MMSeg dependency is imported;
- unit tests;
- one-epoch smoke training on a small subset;
- speed benchmark on random input.

### Stage 1: Seed 2024 Main Experiment

Mandatory run:

| Model | Seed | Epochs |
| --- | ---: | ---: |
| FCN + FADC | 2024 | 50 |
| U-Net + FADC | 2024 | 50 |

Use the current training protocol:

```text
optimizer = Adam
initial lr = 1e-4
min lr = 1e-6
scheduler = cosine annealing
loss = BCEWithLogitsLoss
batch size = 8 unless memory requires adjustment
input size = 256
best checkpoint selected by validation mIoU
evaluation on current validation split
```

This stage is required for task_003 completion. Do not stop after only
implementing the module or only running unit tests.

### Stage 2: Optional Three-Seed Confirmation

Only run if seed 2024 is stable and promising:

```text
seeds = 42, 123, 2024
epochs = 50
```

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

## Required Metrics

`metrics_per_seed.csv`:

```text
method,implementation,seed,miou,dice,params_m,flops_g,inference_time_ms,gpu_memory_mb,best_epoch,notes
```

`metrics_summary.csv`:

```text
method,implementation,miou_mean,miou_std,dice_mean,dice_std,params_m,inference_time_ms,gpu_memory_mb,delta_miou_vs_v1_1,delta_dice_vs_v1_1,delta_time_vs_v1_1,status
```

Suggested status labels:

```text
aligned_and_better
aligned_and_comparable
aligned_but_slower
unstable
failed
```

## Required Diagnostics

Save under:

```text
figures/fadc_diagnostics/
```

At minimum:

```text
high_frequency_weight_mean
low_frequency_weight_mean
adaptive_dilation_mean_or_branch_weights
mask_mean_if_deform_conv_used
alpha_value
gate_mean
gate_std
```

If fallback branch mixing is used, also save:

```text
branch_weight_mean_d1
branch_weight_mean_d2
branch_weight_mean_d3
branch_weight_mean_d4
small_dilation_weight = d1 + d2
large_dilation_weight = d3 + d4
```

## Report Requirements

`task_003_report.md` must include:

1. Feasibility result.
2. Confirmation that the implementation is local PyTorch and does not depend on
   MMCV/MMSeg.
3. Explanation that official AdaDR is approximated with spatial branch mixing
   rather than deformable convolution.
4. Official FADC components implemented:

```text
FreqSelect
AdaDR
AdaKern
```

5. Deviations from official code.
6. Integration positions in FCN and U-Net.
7. Unit test results.
8. Speed benchmark.
9. Seed 2024 training results for both FCN + FADC and U-Net + FADC.
10. Comparison with FA-DCG V1.1.
11. Recommendation:

```text
Use FADC aligned version for further comparison
or
Keep FA-DCG V1.1 as practical baseline
```

12. Explicit note:

```text
SAR-specific FA-DCG V2.0 design is deferred to task_004.
```

## Git Hygiene

Do not commit:

```text
*.pt
*.pth
*.ckpt
datasets
large raw tensors
large generated images
```

Commit only:

- source code;
- scripts;
- prompt file;
- small CSV results;
- reports;
- small diagnostics.

Suggested commit message:

```text
task_003: add official FADC aligned implementation plan
```

## Acceptance Criteria

Task_003 is successful if:

1. it implements a runnable FADC-aligned module;
2. it is implemented locally in PyTorch without MMCV/MMSeg dependencies;
3. it clearly documents that AdaDR is approximated with spatial branch mixing;
4. it includes FreqSelect, AdaDR, and AdaKern-aligned behavior or explicitly
   documents any disabled component;
5. it integrates into FCN and U-Net without deleting FA-DCG V1.0/V1.1;
6. tests pass;
7. seed 2024 experiments complete for FCN + FADC and U-Net + FADC;
8. the report compares against FA-DCG V1.1;
9. FA-DCG V2.0 is not designed in this task and is left for task_004.
