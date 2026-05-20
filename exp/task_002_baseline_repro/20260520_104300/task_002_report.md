# task_002 Baseline Reproduction Report

## Executive Summary

- Completed runs recorded: 8/8 under the final protocol: FCN and U-Net used seeds 42/123/2024; FCN + FA-DCG and U-Net + FA-DCG used seed 2024 only per the user update during execution.
- Evaluation uses the available validation split because no independent test directory or event-isolated split manifest exists in the local dataset.
- Status labels therefore describe reproduction against manuscript numbers under the current repository/data constraints, not a strict independent-test reproduction.

## Repository and Code Audit

- Original source code: `SAR_FEM1/`.
- Model definitions: `SAR_FEM1/models/baseline/` and `SAR_FEM1/models/improved/`.
- FA-DCG implementations: `SAR_FEM1/models/improved/fcn_fadc_light.py` and `SAR_FEM1/models/improved/unet_fadc_light.py`.
- Dataset loader code: `SAR_FEM1/data/dataset.py` and `SAR_FEM1/data/dataset_pt.py`.
- Training script in tracked source is not present; `SAR_FEM1/train_cursive_T4.py` is a plotting script for checkpoint CSVs.
- Task driver used for this report: `scripts/task_002_baseline_repro.py`.

## Environment Audit

See `environment.txt`.

## Dataset Audit

- Data root: `/home/superws/2026_Projects/FA_DCG/SAR_FEM1/data/flood_dataset`.
- Total image files found: 663.
- Matches manuscript 663 patches: True.
- Flood-event-isolated split: unknown; no event metadata or split manifest found.
- Source PNG files report three bands, but the dataset loader converts every image to grayscale (`L`) before training, so the model input is one channel.

See `dataset_audit.md` for split counts, label values, and water-pixel ratios.

## Smoke Test Results

- Smoke tests are logged in `logs/*_smoke.log` when run.
- A smoke test is considered successful when epochs complete, losses remain finite, validation metrics are computed, and a checkpoint is written under the local ignored `checkpoints/` directory.
- Completed smoke tests: U-Net seed 42 and U-Net + FA-DCG seed 42, both for one reduced epoch.

## Main Reproduction Results

| Method | Manuscript mIoU | Reproduced mIoU | Delta mIoU | Manuscript Dice | Reproduced Dice | Delta Dice | Status |
| ------ | --------------: | --------------: | ---------: | --------------: | --------------: | ---------: | ------ |
| FCN | 0.6319 | 0.656403 | 0.024503 | 0.7390 | 0.762888 | 0.023888 | reproduced |
| U-Net | 0.7048 | 0.883401 | 0.178601 | 0.7894 | 0.932461 | 0.143061 | not_reproduced |
| FCN + FA-DCG | 0.8243 | 0.844849 | 0.020549 | 0.8949 | 0.907507 | 0.012607 | reproduced |
| U-Net + FA-DCG | 0.8782 | 0.896663 | 0.018463 | 0.9289 | 0.941023 | 0.012123 | reproduced |

## Deviations from Manuscript Settings

- No independent `test/` split is present locally; validation metrics are reported.
- No event metadata or split manifest is present, so flood-event isolation cannot be verified.
- FLOPs are reported as `NA`; parameter counts, inference time, and GPU memory are recorded.
- Checkpoints are generated locally for reproducibility but are ignored by Git and not committed.
- FCN and U-Net were run with three seeds. FA-DCG variants were run with seed 2024 only after the user narrowed the remaining FA-DCG experiment scope during execution.

## Failure Cases or Unresolved Issues

- Strict manuscript reproduction remains blocked until the exact independent test split and flood-event-isolated metadata are available.
- `FCNWithLightFADC` creates `up_dim` lazily inside `forward`; the task runner performs a warmup forward before optimizer creation so the parameter is trainable without editing original source.

## Recommendations for task_003

- Add an explicit train/val/test split manifest with event IDs.
- Move the training pipeline into a versioned script with CLI arguments for seed, split, and model.
- Replace lazy module creation in model forward paths with constructor-defined modules.
- Add deterministic metric tests for binary mIoU and Dice.
