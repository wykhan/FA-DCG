

````markdown
# task_002b: Optimize FA-DCG-v1 Implementation and Rerun FA-DCG Baselines

## Background

Task 002 completed a baseline reproduction under the current repository validation protocol.

However, the current FA-DCG implementation has several structural and engineering issues:

1. `LightFADC` performs per-channel Python-loop convolution.
2. It uses `dilation[c].item()` and integer dilation, making the dilation preference non-differentiable.
3. FCN + FA-DCG inserts a FA-DCG-like module at a 4096-channel layer, causing severe inference-time overhead.
4. `FCNWithLightFADC` creates `up_dim` lazily inside `forward`, which is unsafe for optimizer construction, checkpointing, and reproducibility.

Task 002b should fix these implementation issues while keeping the task scope limited to FA-DCG-v1 optimization. Do not implement FA-DCG-v2 frequency-conditioned design in this task.

## Main Goal

Implement a fast, differentiable, vectorized FA-DCG-v1 module, then rerun:

1. U-Net + Fast FA-DCG
2. FCN + Fast FA-DCG

under the current train/validation protocol.

This task is an implementation correction and speed/feasibility validation task, not a final manuscript experiment.

## Important Scope Control

- Do NOT introduce frequency-conditioned FA-DCG-v2 in this task.
- Do NOT add high/low-frequency descriptors in this task.
- Do NOT add FFT, wavelet, FreqSelect, or AdaKern-like modules.
- Do NOT delete the existing `LightFADC` implementations.
- Keep the old implementation available for reference.
- Create new fast implementations with clear names.
- Do NOT modify the dataset split.
- Do NOT claim independent test-set reproduction, because task_002 found no independent test directory or event-isolated split manifest.

## Expected Output Directory

Create:

```text
exp/task_002b_fadc_fast_repro/<YYYYMMDD_HHMMSS>/
````

Required files:

```text
task_002b_report.md
metrics_per_seed.csv
metrics_summary.csv
model_complexity.csv
speed_benchmark.csv
implementation_audit.md
run_commands.sh
environment.txt
logs/
figures/
```

Optional:

```text
figures/training_curves/
figures/prediction_examples/
figures/branch_weight_statistics/
```

---

# Part 1: Implement Fast FA-DCG-v1

## 1.1 Create a New Vectorized FA-DCG Module

Create a new module, for example:

```text
SAR_FEM1/models/improved/fadc_fast.py
```

Implement a class:

```python
class FastFADCG(nn.Module):
    ...
```

or:

```python
class FastLightFADC(nn.Module):
    ...
```

This module should replace the current per-channel loop implementation with vectorized multi-branch depthwise dilated convolutions.

### Required Design

Given input:

```text
X: B x C x H x W
```

Use candidate dilation rates:

```python
D = [1, 2, 3, 4]
```

Create K depthwise convolution branches:

```python
nn.Conv2d(
    in_channels=C,
    out_channels=C,
    kernel_size=kernel_size,
    padding=d,
    dilation=d,
    groups=C,
    bias=False
)
```

For each dilation branch:

```python
Z_k = Conv_dk(X)
```

Stack branch outputs:

```python
Z_stack: B x C x K x H x W
```

Fuse them using differentiable channel-wise soft weights:

```text
w: C x K
```

or, if implementation is easier:

```text
w: 1 x C x K x 1 x 1
```

Then:

```python
Z = sum_k w[:, k] * Z_k
```

### Required Differentiable Dilation Preference

Do not use:

```python
int(...)
.item()
```

for dilation selection.

Instead, implement continuous channel-wise dilation preference:

```python
d_cont = d_min + (d_max - d_min) * sigmoid(gamma)
```

where:

```text
d_min = 1
d_max = 4
gamma: learnable parameter with shape C
```

Use branch weights:

```text
w_{c,k} = softmax(theta_{c,k} - beta_c * abs(d_cont_c - d_k))
```

where:

```text
theta: learnable parameter, shape C x K
beta: non-negative learnable parameter, shape C
```

Use `softplus(beta_raw)` to constrain beta to be non-negative.

### Initialization

Initialize the module so that it behaves stably at the beginning of training.

Recommended:

```text
gamma initialized to a negative value so d_cont starts near 1
theta initialized to slightly favor dilation 1
beta initialized so softplus(beta_raw) is approximately 1.0
alpha initialized to 0.5
```

Document the exact initialization in `implementation_audit.md`.

### Required Gate

Keep a lightweight channel gate similar to FA-DCG-v1:

```python
gate = GAP(Z)
gate = Conv1x1(C -> C // r)
gate = ReLU
gate = Conv1x1(C // r -> C)
gate = Sigmoid
```

Recommended reduction ratio:

```text
r = 16
```

Use safe channel reduction:

```python
hidden = max(C // r, 1)
```

Final output:

```python
Y = X + alpha * gate * Z
```

where `alpha` is learnable and initialized to 0.5.

### Required Diagnostics

The module should expose a method or return optional diagnostics when requested.

At minimum, support collecting:

```text
branch_weights_mean
small_dilation_weight = w_d1 + w_d2
large_dilation_weight = w_d3 + w_d4
gate_mean
gate_std
alpha_value
d_cont_mean
d_cont_min
d_cont_max
```

These diagnostics can be collected in evaluation mode and saved to CSV.

---

# Part 2: Implement Optimized Wrapper Models

## 2.1 U-Net + Fast FA-DCG

Create a new wrapper, for example:

```text
SAR_FEM1/models/improved/unet_fadc_fast.py
```

Class name suggestion:

```python
class UNetWithFastFADCG(UNet):
    ...
```

Use the same integration position as the current U-Net + FA-DCG:

```text
after U-Net bottleneck
```

If bottleneck output channel is 1024, use:

```python
self.fast_fadc = FastFADCG(1024, kernel_size=3)
```

Do not change the baseline U-Net architecture except replacing the old `LightFADC` with `FastFADCG`.

## 2.2 FCN + Fast FA-DCG

Create a new wrapper, for example:

```text
SAR_FEM1/models/improved/fcn_fadc_fast.py
```

Class name suggestion:

```python
class FCNWithFastFADCG(FCN):
    ...
```

Do NOT insert FA-DCG at the 4096-channel `fc7` stage.

Instead, insert Fast FA-DCG at the 512-channel deep feature stage:

```text
conv1 -> conv2 -> conv3 -> conv4 -> conv5 -> FastFADCG(512) -> fc6 -> fc7 -> score -> upsample
```

This avoids the pathological 4096-channel FA-DCG overhead.

Do not create modules inside `forward`.

If any extra projection layer is needed, define it in `__init__`.

The preferred FCN design is:

```python
x = self.conv1(x)
x = self.conv2(x)
x = self.conv3(x)
x = self.conv4(x)
x = self.conv5(x)

x = self.fast_fadc(x)

x = self.fc6(x)
x = self.relu6(x)
x = self.drop6(x)

x = self.fc7(x)
x = self.relu7(x)
x = self.drop7(x)

x = self.score(x)
x = interpolate to input size
```

This means FCN + Fast FA-DCG is not identical to the old `FCNWithLightFADC`, but it is a corrected and more reasonable implementation.

Clearly document this change.

---

# Part 3: Unit Tests and Shape Tests

Create or extend a script for testing, for example:

```text
scripts/task_002b_test_fadc_fast.py
```

Test:

1. `FastFADCG(C=512, kernel_size=3)`
2. `FastFADCG(C=1024, kernel_size=3)`
3. `UNetWithFastFADCG`
4. `FCNWithFastFADCG`

Required checks:

* output shape equals expected shape;
* no NaN;
* branch weights sum to 1 over K;
* gate values are within [0, 1];
* `alpha` is trainable;
* `gamma`, `theta`, `beta_raw`, branch conv weights, and gate parameters receive gradients;
* no module is created inside `forward`;
* no `.item()` is used for dilation selection in the new Fast FA-DCG module.

Record results in:

```text
implementation_audit.md
```

---

# Part 4: Speed Benchmark Before Full Training

Create a speed benchmark script, or extend the task runner, to compare:

1. old U-Net + FA-DCG
2. new U-Net + Fast FA-DCG
3. old FCN + FA-DCG
4. new FCN + Fast FA-DCG
5. optionally FCN baseline and U-Net baseline

Use the same input size and batch size as task_002 where possible.

Recommended:

```text
input size = 256 x 256
batch size = 8
warmup iterations = 20
benchmark iterations = 100
torch.cuda.synchronize() before and after timing
model.eval()
torch.no_grad()
```

Save:

```text
speed_benchmark.csv
```

Columns:

```text
method,implementation,batch_size,img_size,params_m,inference_time_ms,gpu_memory_mb,notes
```

Expected result:

* FCN + Fast FA-DCG should be much faster than old FCN + FA-DCG.
* U-Net + Fast FA-DCG should be faster than old U-Net + FA-DCG.
* If speed is not improved, investigate and report why.

---

# Part 5: Main 002b Training Experiment

## Mandatory Stage A

Run optimized models with seed 2024:

| Model               | Seed | Epochs |
| ------------------- | ---: | -----: |
| U-Net + Fast FA-DCG | 2024 |     50 |
| FCN + Fast FA-DCG   | 2024 |     50 |

Use the same training settings as task_002:

```text
optimizer = Adam
initial lr = 1e-4
scheduler = cosine annealing
minimum lr = 1e-6
loss = BCEWithLogitsLoss
batch size = 8 unless GPU memory requires adjustment
input size = 256
best checkpoint selected by validation mIoU
evaluation on current validation split
```

## Conditional Stage B

If Stage A is stable and no major implementation bug is found, run three seeds:

| Model               | Seeds         | Epochs |
| ------------------- | ------------- | -----: |
| U-Net + Fast FA-DCG | 42, 123, 2024 |     50 |
| FCN + Fast FA-DCG   | 42, 123, 2024 |     50 |

If runtime is limited, run Stage A only and clearly document that Stage B was not run.

## Optional

Do not rerun old slow FA-DCG training unless necessary. Use task_002 results as old-reference values:

```text
U-Net + old FA-DCG, seed 2024: mIoU = 0.896663, Dice = 0.941023, time = 9.826 ms
FCN + old FA-DCG, seed 2024: mIoU = 0.844849, Dice = 0.907507, time = 31.819 ms
```

If reusing task_002 results, document that they are reused rather than rerun.

---

# Part 6: Metrics

Create:

```text
metrics_per_seed.csv
metrics_summary.csv
model_complexity.csv
speed_benchmark.csv
```

## metrics_per_seed.csv

Columns:

```text
method,implementation,seed,miou,dice,params_m,flops_g,inference_time_ms,gpu_memory_mb,best_epoch,notes
```

## metrics_summary.csv

Columns:

```text
method,implementation,miou_mean,miou_std,dice_mean,dice_std,params_m,inference_time_ms,gpu_memory_mb,delta_miou_vs_old,delta_dice_vs_old,delta_time_vs_old,status
```

Suggested status labels:

```text
faster_and_comparable
faster_but_worse
faster_and_better
not_faster
failed
```

Use the following rule:

* `faster_and_comparable`: inference time improved by at least 30%, and mIoU drop <= 0.01
* `faster_and_better`: inference time improved by at least 30%, and mIoU improved
* `faster_but_worse`: inference time improved by at least 30%, but mIoU drop > 0.01
* `not_faster`: inference time improvement < 30%
* `failed`: training failed or produced NaN

## model_complexity.csv

Columns:

```text
method,implementation,params_m,delta_params_m_vs_old,delta_params_percent_vs_old,flops_g,inference_time_ms,gpu_memory_mb
```

FLOPs are recommended. If FLOPs cannot be computed, record `NA` and explain why.

---

# Part 7: Report

Create:

```text
task_002b_report.md
```

The report must include:

1. Executive summary.
2. Reminder that this is still validation-protocol evaluation, not final independent-test reproduction.
3. Explanation of why the old implementation was inefficient.
4. New Fast FA-DCG implementation details.
5. FCN integration change:

   * old implementation inserted FA-DCG-like logic at 512 and 4096 channels;
   * new implementation inserts Fast FA-DCG only at the 512-channel deep feature stage after `conv5`.
6. U-Net integration change:

   * old `LightFADC` at bottleneck replaced by vectorized `FastFADCG`.
7. Unit and shape test results.
8. Speed benchmark results.
9. Training results.
10. Comparison with task_002 old FA-DCG results.
11. Whether the optimized implementation should replace the old FA-DCG implementation for task_003.
12. Remaining issues and recommendations.

Include this speed table:

| Model | Old time(ms) | New time(ms) | Speedup | Old Params(M) | New Params(M) | Notes |
| ----- | -----------: | -----------: | ------: | ------------: | ------------: | ----- |

Include this accuracy table:

| Model | Old mIoU | New mIoU | ΔmIoU | Old Dice | New Dice | ΔDice | Status |
| ----- | -------: | -------: | ----: | -------: | -------: | ----: | ------ |

Include this implementation audit table:

| Issue                        | Old implementation | New implementation | Fixed? |
| ---------------------------- | ------------------ | ------------------ | ------ |
| per-channel Python loop      | yes                | no                 | yes    |
| `.item()` integer dilation   | yes                | no                 | yes    |
| 4096-channel FA-DCG in FCN   | yes                | no                 | yes    |
| module created in forward    | yes                | no                 | yes    |
| differentiable branch fusion | no                 | yes                | yes    |

---

# Part 8: Git Hygiene

Before finishing:

```bash
git status
```

Do not commit:

```text
*.pt
*.pth
*.ckpt
large raw logs
datasets
```

Commit only:

* new source files;
* task_002b scripts;
* prompt file;
* report;
* small CSV summaries;
* small diagnostic figures if useful.

Suggested commit message:

```text
task_002b: optimize FA-DCG v1 implementation and rerun fast baselines
```

---

# Acceptance Criteria

Task 002b is successful if:

1. A vectorized, differentiable Fast FA-DCG-v1 module is implemented.
2. Old FA-DCG implementation remains available.
3. No new module is created inside `forward`.
4. No `.item()` is used for dilation selection in the new module.
5. U-Net + Fast FA-DCG passes shape/backward tests.
6. FCN + Fast FA-DCG passes shape/backward tests.
7. Speed benchmark shows whether the new implementation fixes the old bottleneck.
8. At least seed 2024 training is completed for:

   * U-Net + Fast FA-DCG
   * FCN + Fast FA-DCG
9. Metrics and report are saved under the task output directory.
10. The report clearly recommends whether Fast FA-DCG should be used as the base implementation for task_003.

```

---

补充建议：002b 完成后，**task_003 应直接基于 `FastFADCG` 做 frequency-conditioned 版本**，不要再继承旧 `LightFADC`。这样后续 FA-DCG-v2 的实验速度、可微性和论文叙事都会更稳。
```

