# Task 009 Implementation Audit

## U-Net + CBAM

- variant: `cbam`
- class: `UNetWithCBAM`
- params: `31173539`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- FADC-aligned is a local PyTorch implementation inspired by FADC, not the official implementation.

## U-Net + V2d-noBoundary-noLocalVar

- variant: `v2d_no_boundary_no_local_var`
- class: `UNetWithFADCGV2dAblation`
- params: `31587395`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- FADC-aligned is a local PyTorch implementation inspired by FADC, not the official implementation.

