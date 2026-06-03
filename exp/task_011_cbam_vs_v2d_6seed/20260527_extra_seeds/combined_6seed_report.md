# 6-Seed CBAM vs V2d-noBoundary-noLocalVar Comparison

- seeds: 42, 123, 2026, 2027, 2028, 2029
- existing seeds 42/123/2026 are reused from task_010/task_009.
- extra seeds 2027/2028/2029 were trained in this directory.

## Per-Seed Results

| method | seed | val mIoU | test mIoU | test Dice | best epoch | source |
|---|---:|---:|---:|---:|---:|---|
| U-Net + CBAM | 42 | 0.851224 | 0.866560 | 0.883349 | 35 | task_010_existing |
| U-Net + CBAM | 123 | 0.854195 | 0.880922 | 0.896710 | 44 | task_010_existing |
| U-Net + CBAM | 2026 | 0.848976 | 0.879219 | 0.894610 | 31 | task_010_existing |
| U-Net + CBAM | 2027 | 0.855020 | 0.873478 | 0.890031 | 33 | task_011_extra |
| U-Net + CBAM | 2028 | 0.853242 | 0.877505 | 0.893762 | 32 | task_011_extra |
| U-Net + CBAM | 2029 | 0.855939 | 0.876579 | 0.892598 | 32 | task_011_extra |
| U-Net + V2d-noBoundary-noLocalVar | 42 | 0.859120 | 0.878555 | 0.894385 | 38 | task_010_existing |
| U-Net + V2d-noBoundary-noLocalVar | 123 | 0.854573 | 0.881781 | 0.897475 | 46 | task_010_existing |
| U-Net + V2d-noBoundary-noLocalVar | 2026 | 0.852858 | 0.876026 | 0.891790 | 31 | task_010_existing |
| U-Net + V2d-noBoundary-noLocalVar | 2027 | 0.853410 | 0.877605 | 0.894177 | 33 | task_011_extra |
| U-Net + V2d-noBoundary-noLocalVar | 2028 | 0.853453 | 0.882947 | 0.898849 | 49 | task_011_extra |
| U-Net + V2d-noBoundary-noLocalVar | 2029 | 0.846947 | 0.875799 | 0.892297 | 30 | task_011_extra |

## Summary

| method | seeds | test mIoU mean±std | test Dice mean±std |
|---|---:|---:|---:|
| U-Net + CBAM | 6 | 0.875711 ± 0.005141 | 0.891843 ± 0.004712 |
| U-Net + V2d-noBoundary-noLocalVar | 6 | 0.878786 ± 0.002976 | 0.894829 ± 0.002808 |

## Delta

- V2d-noBoundary-noLocalVar minus CBAM: `0.003075` test mIoU, `0.002986` test Dice.
