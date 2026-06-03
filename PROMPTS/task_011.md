````markdown id="task_011"
# task_011.md - Six-Seed Confirmation: CBAM vs V2d-noBoundary-noLocalVar

## 0. Background

Task 010 found that the strongest U-Net variant was:

```text
U-Net + V2d-noBoundary-noLocalVar
test mIoU = 0.878788 ± 0.002885 over seeds 42, 123, 2026
delta vs U-Net + CBAM = +0.003221 mIoU
```

The margin over CBAM is positive but small. This task extends the comparison to six seeds to check whether the advantage remains stable.

Do not edit the LaTeX paper in this task. Only produce raw experimental evidence and a concise comparison report.

## 1. Dataset and Protocol

Use the same dataset and protocol as task_010:

```bash
/home/superws/dataset/HISEA1_flooding_dataset
```

Training protocol:

* U-Net backbone
* input size: `256 x 256`
* batch size: `8`
* epochs: `50`
* optimizer: Adam
* learning rate: `1e-4`
* cosine annealing to `1e-6`
* loss: BCEWithLogitsLoss
* validation set for checkpoint selection
* test set for final evaluation only

## 2. Models

Compare only:

1. `U-Net + CBAM`
2. `U-Net + V2d-noBoundary-noLocalVar`

For `V2d-noBoundary-noLocalVar`, use:

```text
channel=True
boundary=False
speckle=True
msc=False
local_var=False
freq_init=random
```

## 3. Seeds

Reuse existing results for:

```text
42, 123, 2026
```

Run new experiments for:

```text
2027, 2028, 2029
```

The final comparison should include all six seeds:

```text
42, 123, 2026, 2027, 2028, 2029
```

## 4. Output

Create:

```bash
exp/task_011_cbam_vs_v2d_6seed/20260527_extra_seeds/
```

Save:

```text
raw_metrics_per_seed.csv
summary_mean_std.csv
combined_6seed_raw_metrics.csv
combined_6seed_summary.csv
combined_6seed_report.md
logs/
checkpoints/
```

The final report must state:

* per-seed test mIoU and Dice;
* 6-seed mean and standard deviation;
* delta of `V2d-noBoundary-noLocalVar` versus `U-Net + CBAM`;
* whether the advantage remains stable after increasing from three seeds to six seeds.
````
