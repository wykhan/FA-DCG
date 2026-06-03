````markdown id="task_013"
# task_013.md - Three-Seed FA-DCG V1.1 Results for FCN and U-Net

## 0. Background

Task 012 established a fair FCN residual-control comparison:

```text
FCN + V2d-noBoundary-noLocalVar
vs
FCN + Residual-only
```

The six-seed result showed that V2d still improves over the residual-only control on FCN. To complete the comparison table and maintain a consistent reference baseline, this task supplements fresh three-seed results for FA-DCG V1.1 on both FCN and U-Net.

Earlier V1.1 results exist in the repository, but several of them used validation-only reporting or older protocol variants. This task reruns V1.1 under the current train/val/test protocol used by task_010, task_011, and task_012.

Do not edit the LaTeX paper in this task. Only generate raw evidence and a concise report.

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

Use the same dataset and split protocol as task_010 through task_012:

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

## 3. Models

Run exactly:

```text
FCN + FA-DCG V1.1
U-Net + FA-DCG V1.1
```

Use the accepted optimized V1.1 implementation:

```python
VectorizedLightFADC
FCNWithOptimizedFADCG
UNetWithOptimizedFADCG
```

In the report, call the models:

```text
FCN + FA-DCG V1.1
U-Net + FA-DCG V1.1
```

Do not use the old loop-based FA-DCG V1.0 implementation.

## 4. Training Protocol

Use the same protocol as task_012:

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

Use three seeds:

```text
42, 123, 2026
```

Both models must use exactly the same seed list.

## 5. Required Metrics

For each model and seed, report:

* validation mIoU and Dice at the selected checkpoint;
* test mIoU and Dice;
* best epoch;
* checkpoint path;
* parameter count;
* MACs/FLOPs under the same tool convention as task_012;
* latency and FPS under the same benchmark protocol as task_012.

Use the existing metric implementation. Do not change metric definitions.

## 6. Implementation Audit

Before or after training, create an implementation audit confirming:

* `FCN + FA-DCG V1.1` uses two `VectorizedLightFADC` blocks on the original deep FCN path:

```text
conv1 -> conv2 -> conv3 -> conv4 -> conv5
-> VectorizedLightFADC(512, kernel_size=7)
-> relu6/drop6/up_dim
-> VectorizedLightFADC(4096, kernel_size=1)
-> relu7/drop7/score
```

* `U-Net + FA-DCG V1.1` uses one `VectorizedLightFADC` block at the U-Net bottleneck.
* both models output `[N, 1, 256, 256]`;
* no lazy modules are created in `forward`;
* V1.1 uses the vectorized implementation, not V1.0.

## 7. Diagnostics

For each V1.1 block, collect if easy:

* `alpha_value`;
* `gate_mean/std`;
* depthwise weight mean/std;
* dilation parameter mean/std.

If diagnostics are not available for one model, report missing diagnostics clearly.

## 8. Output Files

Create a new directory:

```bash
exp/task_013_v1_1_3seed/<timestamp>/
```

Save:

```text
raw_metrics_per_seed.csv
summary_mean_std.csv
complexity_raw.csv
diagnostics_raw.csv
environment.txt
implementation_audit.md
task_013_report.md
logs/
checkpoints/
```

The final report must include:

1. exact git branch, commit hash, and dirty-worktree status;
2. exact dataset path and split sizes;
3. exact V1.1 implementation;
4. per-seed raw results;
5. mean/std summary;
6. complexity and latency comparison;
7. a concise note that these are fresh three-seed train/val/test V1.1 results.

Do not edit the LaTeX paper in this task.
````
