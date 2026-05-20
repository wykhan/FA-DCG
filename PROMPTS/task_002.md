

---

## task_002 基准实验设计

### 核心目标

复现 latex 手稿中的主实验结果，至少包括：

| 模型             |        手稿 mIoU |        手稿 Dice |
| -------------- | -------------: | -------------: |
| FCN            | 0.6319 ± 0.018 | 0.7390 ± 0.015 |
| U-Net          | 0.7048 ± 0.021 | 0.7894 ± 0.017 |
| FCN + FA-DCG   | 0.8243 ± 0.006 | 0.8949 ± 0.005 |
| U-Net + FA-DCG | 0.8782 ± 0.008 | 0.9289 ± 0.007 |

这些数值来自当前手稿 Table I；文章还使用 663 个 Hisea-1 SAR flood patches、三随机种子 42/123/2024，以及 flood-event-isolated testing。

### 最小验收标准


````markdown
# task_002: Baseline Reproduction for FA-DCG

## Background

This repository is for the FA-DCG SAR flood segmentation project.

Task 001 has initialized the Git project structure. The original project code is expected to be preserved under `SAR_FEM1/` or another existing source-code directory. Do not delete or rewrite the original code. This task is to run and document a baseline reproduction experiment before any FA-DCG structural modification.

The current manuscript reports the following main results on a Hisea-1 single-channel SAR flood segmentation dataset with 663 annotated patches, using flood-event-isolated testing and three random seeds: 42, 123, and 2024.

Target manuscript results:

| Method | Params (M) | mIoU | Dice |
|---|---:|---:|---:|
| FCN | 134.27 | 0.6319 ± 0.018 | 0.7390 ± 0.015 |
| U-Net | 31.04 | 0.7048 ± 0.021 | 0.7894 ± 0.017 |
| FCN + FA-DCG | 142.83 | 0.8243 ± 0.006 | 0.8949 ± 0.005 |
| U-Net + FA-DCG | 31.58 | 0.8782 ± 0.008 | 0.9289 ± 0.007 |

This task must verify whether the current repository, environment, dataset paths, and training scripts can reproduce these baseline results.

## Objectives

1. Inspect the repository and identify the existing dataset, preprocessing, model, training, evaluation, and visualization scripts.
2. Verify that the current environment can run the original training/evaluation pipeline.
3. Run a smoke test on a tiny setting to confirm that training and evaluation execute without crashing.
4. Run the main baseline reproduction experiment for the core models:
   - FCN
   - U-Net
   - FCN + FA-DCG
   - U-Net + FA-DCG
5. Use three random seeds:
   - 42
   - 123
   - 2024
6. Report per-seed and mean ± standard deviation metrics:
   - mIoU
   - Dice
   - parameter count
   - optional but recommended: FLOPs, inference time, GPU memory
7. Compare reproduced results against the manuscript target table.
8. Produce a complete experiment report.

## Important Constraints

- Do not modify FA-DCG model logic unless absolutely necessary to make the existing code runnable.
- If any code change is required, make the smallest possible change and document it clearly.
- Do not delete or overwrite the original code under `SAR_FEM1/`.
- Do not introduce the new FA-DCG-v2 structure in this task.
- Do not silently change dataset splits, preprocessing, loss function, optimizer, scheduler, input size, or random seeds.
- If the existing scripts do not exactly match the manuscript settings, document the mismatch instead of hiding it.
- Large checkpoints, raw logs, and generated temporary files should not be committed to Git unless they are small and necessary. Add or update `.gitignore` if needed.

## Expected Repository Outputs

Create the following output directory:

```text
exp/task_002_baseline_repro/<YYYYMMDD_HHMMSS>/
````

Inside it, save:

```text
task_002_report.md
metrics_per_seed.csv
metrics_summary.csv
environment.txt
dataset_audit.md
run_commands.sh
logs/
figures/
```

If training curves or qualitative predictions are easy to generate, save them under:

```text
figures/training_curves/
figures/prediction_examples/
```

## Step-by-Step Instructions

### Step 1: Repository and Environment Audit

Inspect the repository and identify:

* where the original source code is located;
* where model definitions are located;
* where FA-DCG is implemented;
* where dataset paths are configured;
* where train/val/test splits are defined;
* where metrics are computed;
* whether FCN, U-Net, FCN+FA-DCG, and U-Net+FA-DCG are already implemented;
* whether existing scripts already support seed control and repeated runs.

Write the findings to:

```text
exp/task_002_baseline_repro/<timestamp>/dataset_audit.md
```

Also create:

```text
exp/task_002_baseline_repro/<timestamp>/environment.txt
```

It should include:

* OS information;
* Python version;
* PyTorch version;
* CUDA version;
* GPU name and memory;
* major package versions;
* current Git commit hash;
* current working directory.

### Step 2: Dataset Audit

Verify and report:

* total number of patches;
* number of train/val/test patches;
* image size;
* number of channels;
* label values;
* whether labels are binary;
* whether the split is flood-event-isolated;
* whether the split matches the manuscript description.

If possible, compute water-pixel ratios for train/val/test.

If the dataset is missing or the path is wrong, do not invent results. Report the issue clearly in `task_002_report.md`.

### Step 3: Smoke Test

Run a tiny smoke test before full training.

Suggested smoke-test setting:

* models: U-Net and U-Net+FA-DCG first; FCN if feasible;
* seed: 42;
* epochs: 1 or 2;
* batch size: the default manuscript batch size if possible;
* small subset if supported;
* verify forward pass, backward pass, validation, checkpoint saving, and metric computation.

The smoke test is successful if:

* training starts;
* loss decreases or at least remains finite;
* validation runs;
* mIoU and Dice are computed;
* no CUDA, shape, dtype, label, or path errors occur.

Record commands and outputs.

### Step 4: Main Baseline Reproduction

Run the following core models with seeds 42, 123, and 2024:

1. FCN
2. U-Net
3. FCN + FA-DCG
4. U-Net + FA-DCG

Use the manuscript settings whenever possible:

* input size: 256 × 256;
* optimizer: Adam;
* initial learning rate: 1e-4;
* scheduler: cosine annealing;
* minimum learning rate: 1e-6;
* loss: BCEWithLogitsLoss;
* epochs: 50;
* batch size: 8, unless GPU memory requires adjustment;
* validation every 5 epochs;
* select best model by validation mIoU;
* evaluate final model on the independent test set.

If the current code uses different settings, record the actual settings and explain the difference.

### Step 5: Optional Comparison Models

If the code already supports them and runtime is reasonable, also run:

* FCN + SE
* U-Net + CBAM
* DeepLabV3+ Light
* DeepLabV3+ Light Scratch

Do not spend excessive time implementing missing comparison models in this task. The priority is to reproduce the core baseline and FA-DCG models.

### Step 6: Metrics and Tables

Create:

```text
metrics_per_seed.csv
metrics_summary.csv
```

`metrics_per_seed.csv` should contain:

```text
method,seed,miou,dice,params_m,flops_g,inference_time_ms,gpu_memory_mb,best_epoch,notes
```

`metrics_summary.csv` should contain:

```text
method,miou_mean,miou_std,dice_mean,dice_std,params_m,flops_g,inference_time_ms,gpu_memory_mb,delta_miou_vs_manuscript,delta_dice_vs_manuscript,status
```

Use the following status labels:

* `reproduced`: mIoU difference <= 0.03;
* `partial`: mIoU difference > 0.03 and <= 0.06;
* `not_reproduced`: mIoU difference > 0.06;
* `not_run`: model was not run.

### Step 7: Report

Create:

```text
task_002_report.md
```

The report must include:

1. Executive summary.
2. Repository and code audit.
3. Environment audit.
4. Dataset audit.
5. Smoke-test results.
6. Main reproduction results.
7. Comparison with manuscript results.
8. Any deviations from manuscript settings.
9. Failure cases or unresolved issues.
10. Recommendations for task_003.

The report should contain at least one table comparing reproduced results with the manuscript target results.

Use this table format:

| Method | Manuscript mIoU | Reproduced mIoU | ΔmIoU | Manuscript Dice | Reproduced Dice | ΔDice | Status |
| ------ | --------------: | --------------: | ----: | --------------: | --------------: | ----: | ------ |

### Step 8: Git Hygiene

Before finishing:

* run `git status`;
* ensure large checkpoints and raw logs are not accidentally staged;
* commit only useful code fixes, scripts, prompt files, reports, and small CSV summaries;
* do not commit large `.pt`, `.pth`, `.ckpt`, `.npy`, or raw dataset files unless explicitly required.

Suggested commit message:

```text
task_002: reproduce FA-DCG baseline experiments
```

## Acceptance Criteria

This task is considered successful if:

1. The environment and dataset have been audited.
2. At least one smoke test runs successfully.
3. The four core models are run or a clear blocking reason is documented.
4. Per-seed and summary metrics are saved.
5. A complete `task_002_report.md` is produced.
6. The report clearly states whether the manuscript results are reproduced, partially reproduced, or not reproduced.
7. No new FA-DCG-v2 structure is introduced in this task.

```


