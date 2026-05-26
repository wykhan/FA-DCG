# Task 009 Implementation Audit

## U-Net + V2d-noLocalVar

- variant: `v2d_no_local_var`
- class: `UNetWithFADCGV2dAblation`
- params: `31616067`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- FADC-aligned is a local PyTorch implementation inspired by FADC, not the official implementation.

## U-Net + FADC-SR-high

- variant: `fadc_sr_high`
- class: `UNetWithFADCSRFGHybrid`
- params: `31494027`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- hybrid initialization: frequency heads and optional reliability/spatial heads are zero-initialized; alpha starts at 0.1; spatial gamma starts at 0.0.
- FADC-aligned is a local PyTorch implementation inspired by FADC, not the official implementation.

## U-Net + FADC-SARSpatial-noVar

- variant: `fadc_sar_spatial_no_var`
- class: `UNetWithFADCSRFGHybrid`
- params: `31483935`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- hybrid initialization: frequency heads and optional reliability/spatial heads are zero-initialized; alpha starts at 0.1; spatial gamma starts at 0.0.
- FADC-aligned is a local PyTorch implementation inspired by FADC, not the official implementation.

## U-Net + FADC-SR-high + SARSpatial-noVar

- variant: `fadc_sr_high_sar_spatial_no_var`
- class: `UNetWithFADCSRFGHybrid`
- params: `31494176`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- hybrid initialization: frequency heads and optional reliability/spatial heads are zero-initialized; alpha starts at 0.1; spatial gamma starts at 0.0.
- FADC-aligned is a local PyTorch implementation inspired by FADC, not the official implementation.

## U-Net + FADC-CBAM-SpatialOnly

- variant: `fadc_cbam_spatial_only`
- class: `UNetWithFADCSRFGHybrid`
- params: `31483886`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- hybrid initialization: frequency heads and optional reliability/spatial heads are zero-initialized; alpha starts at 0.1; spatial gamma starts at 0.0.
- FADC-aligned is a local PyTorch implementation inspired by FADC, not the official implementation.

## U-Net + V2d-noBoundary-noLocalVar

- variant: `v2d_no_boundary_no_local_var`
- class: `UNetWithFADCGV2dAblation`
- params: `31587395`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- FADC-aligned is a local PyTorch implementation inspired by FADC, not the official implementation.

## U-Net + FADC-SR-branch

- variant: `fadc_sr_branch`
- class: `UNetWithFADCSRFGHybrid`
- params: `31494027`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- hybrid initialization: frequency heads and optional reliability/spatial heads are zero-initialized; alpha starts at 0.1; spatial gamma starts at 0.0.
- FADC-aligned is a local PyTorch implementation inspired by FADC, not the official implementation.

## U-Net + FADC-SARSpatial-withVar

- variant: `fadc_sar_spatial_with_var`
- class: `UNetWithFADCSRFGHybrid`
- params: `31483984`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- hybrid initialization: frequency heads and optional reliability/spatial heads are zero-initialized; alpha starts at 0.1; spatial gamma starts at 0.0.
- FADC-aligned is a local PyTorch implementation inspired by FADC, not the official implementation.

## U-Net + FADC-SR-branch + SARSpatial-noVar

- variant: `fadc_sr_branch_sar_spatial_no_var`
- class: `UNetWithFADCSRFGHybrid`
- params: `31494176`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- hybrid initialization: frequency heads and optional reliability/spatial heads are zero-initialized; alpha starts at 0.1; spatial gamma starts at 0.0.
- FADC-aligned is a local PyTorch implementation inspired by FADC, not the official implementation.

