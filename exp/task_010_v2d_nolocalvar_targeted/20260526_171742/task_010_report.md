# Task 010 V2d-noLocalVar Targeted Optimization Report

- git branch: `codex/fadcg-v1.1-baseline`
- git commit hash: `490e20fbe3ca68a33346caf02181c1424146a77b`
- dataset path: `/home/superws/dataset/HISEA1_flooding_dataset`
- split sizes: train=1404, val=468, test=468
- output directory: `/home/superws/2026_Projects/FA_DCG/exp/task_010_v2d_nolocalvar_targeted/20260526_171742`
- PyTorch: `2.0.1+cu118`; CUDA: `11.8`

## Model List

- Reused baselines: U-Net, U-Net + CBAM, U-Net + FADC-aligned, U-Net + SRFG, U-Net + V2d-noLocalVar, and available V2d-noBoundary-noLocalVar rows from task_009 when available.
- U-Net + V2d-noBoundary-noLocalVar (v2d_no_boundary_no_local_var, priority A): comparison variant without boundary, MSC, or local variance; seed 2026 is reused from task_009 when available
- U-Net + V2d-noBoundaryKeepMSC-noLocalVar (v2d_no_boundary_keep_msc_no_local_var, priority A): decoupling variant: disables boundary gate but keeps MSC descriptor and speckle gate
- U-Net + V2d-BoundaryNoMSC-noLocalVar (v2d_boundary_no_msc_no_local_var, priority A): decoupling variant: keeps boundary gate but removes MSC and local variance descriptors
- U-Net + V2d-noLocalVar-Laplacian (v2d_no_local_var_laplacian, priority A): best task_009 V2d-noLocalVar with L1-normalized 4-neighbor Laplacian depthwise initialization
- U-Net + V2d-noBoundaryKeepMSC-noLocalVar-Laplacian (v2d_no_boundary_keep_msc_no_local_var_laplacian, priority A): boundary/MSC decoupling variant with Laplacian depthwise initialization
- U-Net + V2d-BoundaryNoMSC-noLocalVar-Laplacian (v2d_boundary_no_msc_no_local_var_laplacian, priority A): boundary-only frequency gate variant with Laplacian depthwise initialization

## Protocol

- data: train for training, val for checkpoint selection, test for final metrics
- input size: 256 x 256; batch size: 8; epochs: 50
- optimizer: Adam, lr=0.0001, cosine eta_min=1e-06; loss=BCEWithLogitsLoss
- seeds: 42, 123, 2026
- augmentation: random horizontal flip, random vertical flip, random rotation within +/-10 degrees
- metrics: sigmoid threshold 0.5; mIoU averages foreground/background IoU; Dice is foreground Dice
- Laplacian initialization: L1-normalized 4-neighbor high-pass kernel; depthwise residual kernels remain learnable
- complexity: THOP MACs when available; latency uses batch size 1, eval mode, no data loading

## Completed Runs

- completed or reused rows: 29
- complexity rows: 11

## Failed Or Skipped Runs

- None.

## Raw Per-Seed Results

| method | variant | seed | status | val mIoU | test mIoU | test Dice | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net | baseline | 2026 | reused | 0.835508 | 0.856289 | 0.873615 | 31 |
| U-Net + CBAM | cbam | 42 | reused | 0.851224 | 0.866560 | 0.883349 | 35 |
| U-Net + CBAM | cbam | 123 | reused | 0.854195 | 0.880922 | 0.896710 | 44 |
| U-Net + CBAM | cbam | 2026 | reused | 0.848976 | 0.879219 | 0.894610 | 31 |
| U-Net + FADC-aligned | fadc_aligned | 42 | reused | 0.835614 | 0.862594 | 0.879863 | 32 |
| U-Net + FADC-aligned | fadc_aligned | 123 | reused | 0.844207 | 0.869415 | 0.885330 | 15 |
| U-Net + FADC-aligned | fadc_aligned | 2026 | reused | 0.847281 | 0.878748 | 0.894914 | 41 |
| U-Net + SRFG | srfg_v2d_nomsc | 2026 | reused | 0.848554 | 0.875874 | 0.892199 | 41 |
| U-Net + V2d-BoundaryNoMSC-noLocalVar | v2d_boundary_no_msc_no_local_var | 42 | completed | 0.845804 | 0.870132 | 0.886648 | 38 |
| U-Net + V2d-BoundaryNoMSC-noLocalVar | v2d_boundary_no_msc_no_local_var | 123 | completed | 0.846743 | 0.877634 | 0.893498 | 28 |
| U-Net + V2d-BoundaryNoMSC-noLocalVar | v2d_boundary_no_msc_no_local_var | 2026 | completed | 0.855068 | 0.874823 | 0.889777 | 31 |
| U-Net + V2d-BoundaryNoMSC-noLocalVar-Laplacian | v2d_boundary_no_msc_no_local_var_laplacian | 42 | completed | 0.831377 | 0.856329 | 0.874531 | 32 |
| U-Net + V2d-BoundaryNoMSC-noLocalVar-Laplacian | v2d_boundary_no_msc_no_local_var_laplacian | 123 | completed | 0.848257 | 0.872689 | 0.889140 | 46 |
| U-Net + V2d-BoundaryNoMSC-noLocalVar-Laplacian | v2d_boundary_no_msc_no_local_var_laplacian | 2026 | completed | 0.847508 | 0.873346 | 0.889182 | 27 |
| U-Net + V2d-noBoundary-noLocalVar | v2d_no_boundary_no_local_var | 42 | completed | 0.859120 | 0.878555 | 0.894385 | 38 |
| U-Net + V2d-noBoundary-noLocalVar | v2d_no_boundary_no_local_var | 123 | completed | 0.854573 | 0.881781 | 0.897475 | 46 |
| U-Net + V2d-noBoundary-noLocalVar | v2d_no_boundary_no_local_var | 2026 | reused | 0.852858 | 0.876026 | 0.891790 | 31 |
| U-Net + V2d-noBoundaryKeepMSC-noLocalVar | v2d_no_boundary_keep_msc_no_local_var | 42 | completed | 0.847787 | 0.870832 | 0.887727 | 38 |
| U-Net + V2d-noBoundaryKeepMSC-noLocalVar | v2d_no_boundary_keep_msc_no_local_var | 123 | completed | 0.851866 | 0.880677 | 0.896095 | 25 |
| U-Net + V2d-noBoundaryKeepMSC-noLocalVar | v2d_no_boundary_keep_msc_no_local_var | 2026 | completed | 0.848154 | 0.877044 | 0.892731 | 35 |
| U-Net + V2d-noBoundaryKeepMSC-noLocalVar-Laplacian | v2d_no_boundary_keep_msc_no_local_var_laplacian | 42 | completed | 0.830035 | 0.854528 | 0.872866 | 38 |
| U-Net + V2d-noBoundaryKeepMSC-noLocalVar-Laplacian | v2d_no_boundary_keep_msc_no_local_var_laplacian | 123 | completed | 0.842549 | 0.869226 | 0.885732 | 41 |
| U-Net + V2d-noBoundaryKeepMSC-noLocalVar-Laplacian | v2d_no_boundary_keep_msc_no_local_var_laplacian | 2026 | completed | 0.848954 | 0.870845 | 0.886051 | 27 |
| U-Net + V2d-noLocalVar | v2d_no_local_var | 42 | reused | 0.851109 | 0.876423 | 0.892520 | 38 |
| U-Net + V2d-noLocalVar | v2d_no_local_var | 123 | reused | 0.850405 | 0.878534 | 0.894097 | 46 |
| U-Net + V2d-noLocalVar | v2d_no_local_var | 2026 | reused | 0.856994 | 0.879753 | 0.895185 | 31 |
| U-Net + V2d-noLocalVar-Laplacian | v2d_no_local_var_laplacian | 42 | completed | 0.830419 | 0.856736 | 0.874516 | 38 |
| U-Net + V2d-noLocalVar-Laplacian | v2d_no_local_var_laplacian | 123 | completed | 0.841940 | 0.874432 | 0.890823 | 30 |
| U-Net + V2d-noLocalVar-Laplacian | v2d_no_local_var_laplacian | 2026 | completed | 0.848862 | 0.873420 | 0.888984 | 27 |

## Summary

| method | variant | seeds | test mIoU | test Dice | delta vs CBAM | delta vs V2d-noLocalVar | params | MACs | latency ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | baseline | 1 | 0.856289 | 0.873615 | -0.019278 | -0.021948 | 31042369 | 54662266880 | 2.884045 | 346.735280 |
| U-Net + CBAM | cbam | 3 | 0.875567 | 0.891556 | 0.000000 | -0.002670 | 31173539 | 54662817280 | 3.092413 | 323.372104 |
| U-Net + FADC-aligned | fadc_aligned | 3 | 0.870252 | 0.886702 | -0.005315 | -0.007985 | 31483786 | 54682781696 | 4.223656 | 236.761725 |
| U-Net + SRFG | srfg_v2d_nomsc | 1 | 0.875874 | 0.892199 | 0.000307 | -0.002363 | 31606851 | 54670918656 | 2.863488 | 349.224419 |
| U-Net + V2d-BoundaryNoMSC-noLocalVar | v2d_boundary_no_msc_no_local_var | 3 | 0.874196 | 0.889974 | -0.001371 | -0.004041 | 31597635 | 54668559360 | 2.853670 | 350.425946 |
| U-Net + V2d-BoundaryNoMSC-noLocalVar-Laplacian | v2d_boundary_no_msc_no_local_var_laplacian | 3 | 0.867455 | 0.884285 | -0.008112 | -0.010782 | 31597635 | 54668559360 | 2.864793 | 349.065380 |
| U-Net + V2d-noBoundary-noLocalVar | v2d_no_boundary_no_local_var | 3 | 0.878788 | 0.894550 | 0.003221 | 0.000551 | 31587395 | 54666200064 | 2.823539 | 354.165426 |
| U-Net + V2d-noBoundaryKeepMSC-noLocalVar | v2d_no_boundary_keep_msc_no_local_var | 3 | 0.876184 | 0.892184 | 0.000617 | -0.002053 | 31596611 | 54668559360 | 2.846958 | 351.252091 |
| U-Net + V2d-noBoundaryKeepMSC-noLocalVar-Laplacian | v2d_no_boundary_keep_msc_no_local_var_laplacian | 3 | 0.864866 | 0.881550 | -0.010701 | -0.013371 | 31596611 | 54668559360 | 2.846851 | 351.265354 |
| U-Net + V2d-noLocalVar | v2d_no_local_var | 3 | 0.878237 | 0.893934 | 0.002670 | 0.000000 | 31616067 | 54673277952 | 2.841889 | 351.878679 |
| U-Net + V2d-noLocalVar-Laplacian | v2d_no_local_var_laplacian | 3 | 0.868196 | 0.884775 | -0.007371 | -0.010041 | 31616067 | 54673277952 | 2.866714 | 348.831446 |

## Decision Notes

- Best seed-screening test mIoU: U-Net + V2d-noBoundary-noLocalVar (v2d_no_boundary_no_local_var) = 0.878788.
- Delta vs U-Net + CBAM: 0.003221 mIoU.
- Delta vs task_009 U-Net + V2d-noLocalVar: 0.000551 mIoU.
- Decision rule: weak positive evidence; requires more datasets or stronger analysis.

## Interpretation

- Boundary/MSC decoupling: no-boundary/no-MSC reached 0.878788 mIoU, no-boundary/keep-MSC reached 0.876184, and boundary/no-MSC reached 0.874196.
- This suggests the original boundary and MSC descriptors are not complementary under the current U-Net bottleneck placement; removing both is slightly better than keeping either one alone.
- High-pass initialization: the best Laplacian variant was U-Net + V2d-noLocalVar-Laplacian at 0.868196 mIoU, below the corresponding random-initialized V2d variants.
- The explicit high-pass prior did not improve the metrics; in this setup it appears to constrain or slow useful adaptation rather than strengthen the frequency-gating story.
- Factual raw data only; no LaTeX paper edits were made.
