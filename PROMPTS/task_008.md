

````markdown id="task_008"
# task_008.md — Complete Raw Experimental Data for TABLE I Main Module Comparison

## 0. Background

This task supplements the raw experimental data required for TABLE I: Main module comparison on FCN and U-Net backbones.

The paper now treats SRFG as a plug-and-play module, not as a full segmentation architecture. Therefore, the comparison should be module-level and should include:

- Baseline backbone
- SE
- CBAM
- FADC-aligned
- SRFG

Do not include DeepLabV3+ or other complete segmentation architectures in this task.

The purpose of this task is only to generate reliable raw experimental data. Do not edit the LaTeX paper. Do not decide the final paper table format. Only output raw metrics and a concise report.

## 1. Repository and Branch

Work in the repository:

```bash
https://github.com/wykhan/FA-DCG
````

Use the current working branch:

```bash
codex/fadcg-v1.1-baseline
```

Before running experiments, record:

```bash
git branch --show-current
git rev-parse HEAD
```

Save this information in the final report.

## 2. Target Models

Run or collect results for the following complete matrix.

### FCN group

1. FCN
2. FCN + SE
3. FCN + CBAM
4. FCN + FADC-aligned
5. FCN + SRFG

### U-Net group

1. U-Net
2. U-Net + SE
3. U-Net + CBAM
4. U-Net + FADC-aligned
5. U-Net + SRFG

Important:

* SRFG means the final V2d-noMSC design selected for the paper.
* If the code still uses older names such as FA-DCG V2d-noMSC, map it to `SRFG` in the raw report.
* FADC-aligned means the local PyTorch implementation inspired by FADC. Do not claim it is the official FADC implementation.
* If SE or CBAM is not implemented for one backbone, implement the minimal required wrapper and insert it at the same semantic feature level used for other plug-in modules.

## 3. Insertion Locations

Use consistent insertion locations within each backbone.

### FCN

Insert plug-in modules at the same deep semantic feature stage used by the current SRFG/FA-DCG implementation.

### U-Net

Insert plug-in modules at the bottleneck or the same deep semantic feature stage used by the current SRFG/FA-DCG implementation.

If different modules currently use different insertion positions, document this explicitly. Prefer unifying the insertion position when it is technically reasonable.

## 4. Training Protocol

Use the same training configuration for all models:

* Input size: `256 x 256`
* Input channel: `1`
* Batch size: `8`
* Epochs: `50`
* Optimizer: Adam
* Initial learning rate: `1e-4`
* Learning-rate schedule: cosine annealing to `1e-6`
* Loss: BCEWithLogitsLoss
* Data augmentation:

  * random horizontal flip
  * random vertical flip
  * random rotation within ±10 degrees

Use the following seeds:

```text
42, 123, 2024
```

For each model and each seed, save the raw metrics.

If a model has already been trained under exactly the same protocol and the raw per-seed results are available, you may reuse them. However:

* clearly state that they are reused;
* provide the original file path;
* ensure the metrics definitions are identical;
* still compute Params, FLOPs, latency, and FPS under the unified protocol below.

If any doubt exists, rerun the model.

## 5. Required Accuracy Metrics

For each model and each seed, report:

* mIoU
* Dice
* best epoch if available
* final selected checkpoint path if available

Use the existing metric implementation in the repository. Do not change the metric definitions.

Output both:

1. per-seed raw results;
2. mean and standard deviation over seeds.

## 6. Required Complexity Metrics

For each model, report:

* absolute Params
* Params increment relative to the corresponding backbone
* absolute FLOPs
* FLOPs increment relative to the corresponding backbone
* latency in milliseconds
* FPS

### Params

Count learnable parameters after the model is fully initialized.

Important:

* Some older FCN variants may lazily create modules in `forward`.
* If lazy modules exist, run one warm-up forward pass before parameter counting.
* Report both absolute Params and delta Params relative to the backbone.

### FLOPs

Compute FLOPs using one consistent method for all models.

Input tensor:

```python
torch.randn(1, 1, 256, 256)
```

Recommended tools, in order:

1. `fvcore.nn.FlopCountAnalysis`, if installed;
2. `thop`, if installed;
3. another available FLOPs tool, but use it consistently for all models.

Document the FLOPs convention clearly:

* whether the tool counts MACs or FLOPs;
* whether one multiply-add is counted as one operation or two operations.

Report FLOPs in raw numeric form and in GFLOPs.

Also report FLOPs increment relative to the corresponding backbone.

### FPS and latency

Use the same inference benchmark for all models.

Benchmark protocol:

* model in `eval()` mode
* `torch.no_grad()`
* input shape: `[1, 1, 256, 256]`
* batch size: `1`
* no data loading time
* no preprocessing time
* same GPU for all models
* 50 warm-up iterations
* 300 timed iterations
* use `torch.cuda.synchronize()` before and after timing when CUDA is available

Report:

* mean latency in ms
* standard deviation of latency in ms if easy to compute
* FPS = `1000 / mean_latency_ms`
* GPU name
* PyTorch version
* CUDA version

If CUDA is unavailable, run CPU timing only and clearly mark `device=CPU`. Do not mix CPU and GPU timing in the final summary.

## 7. Output Files

Create a new directory:

```bash
exp/task_008_main_module_table/<timestamp>/
```

Save the following files:

### 1. `task_008_report.md`

Must include:

* git branch
* git commit hash
* dataset path
* model list
* training protocol
* insertion positions
* metric definitions
* FLOPs tool and convention
* FPS benchmark protocol
* summary of completed runs
* failed or skipped runs with reasons
* key observations, only factual and concise

Do not write paper-style interpretation.

### 2. `raw_metrics_per_seed.csv`

Required columns:

```text
backbone,
method,
module_type,
seed,
miou,
dice,
best_epoch,
checkpoint_path,
status,
notes
```

### 3. `complexity_raw.csv`

Required columns:

```text
backbone,
method,
module_type,
params_abs,
params_delta_vs_backbone,
flops_abs,
flops_delta_vs_backbone,
flops_unit,
flops_tool,
latency_ms_mean,
latency_ms_std,
fps,
device,
gpu_name,
input_shape,
notes
```

### 4. `summary_mean_std.csv`

Required columns:

```text
backbone,
method,
module_type,
num_seeds,
miou_mean,
miou_std,
dice_mean,
dice_std,
params_abs,
params_delta_vs_backbone,
flops_abs,
flops_delta_vs_backbone,
latency_ms_mean,
fps,
notes
```

### 5. `environment.txt`

Include:

```bash
python --version
pip list
nvidia-smi
```

If `nvidia-smi` is unavailable, record that explicitly.

### 6. `logs/`

Save training logs and benchmark logs for each model.

## 8. Required Checks

Before finalizing the task, verify:

1. All ten target model rows exist in `summary_mean_std.csv`.
2. All ten target model rows have Params.
3. All ten target model rows have FLOPs.
4. All ten target model rows have FPS.
5. All available accuracy metrics are based on the same data split and same metric implementation.
6. All three seeds are completed for each model, unless there is a clearly documented failure.
7. Baseline-relative deltas for Params and FLOPs are computed correctly within each backbone group.

## 9. Do Not Do

Do not edit the LaTeX paper.

Do not update tables in `SCI-latex/fa-dcg.tex`.

Do not include DeepLabV3+ or other full segmentation architectures.

Do not rename the repository or reorganize existing project directories.

Do not delete previous experiment results.

Do not claim FADC-aligned is the official FADC implementation.

Do not write polished paper paragraphs. Provide raw data only.

## 10. Acceptance Criteria

The task is complete only if the following files exist:

```text
exp/task_008_main_module_table/<timestamp>/task_008_report.md
exp/task_008_main_module_table/<timestamp>/raw_metrics_per_seed.csv
exp/task_008_main_module_table/<timestamp>/complexity_raw.csv
exp/task_008_main_module_table/<timestamp>/summary_mean_std.csv
exp/task_008_main_module_table/<timestamp>/environment.txt
```

The most important deliverable is `summary_mean_std.csv`, which must provide the raw data needed to fill TABLE I.

At the end of the task, print the absolute path of the experiment directory and the location of `summary_mean_std.csv`.

```
```

