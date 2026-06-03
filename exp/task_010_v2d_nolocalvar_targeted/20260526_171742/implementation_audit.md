# Task 010 Implementation Audit

- Laplacian initialization uses the L1-normalized 4-neighbor kernel `[[0,-1,0],[-1,4,-1],[0,-1,0]] / 8`.
- Depthwise frequency residual kernels remain learnable after initialization.

## U-Net + V2d-noBoundary-noLocalVar

- variant: `v2d_no_boundary_no_local_var`
- class: `UNetWithFADCGV2dAblation`
- params: `31587395`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- V2d variant: `no_boundary_no_local_var`
- frequency initialization: `random`
- depthwise kernel learnable: `True`
- config: `{'channel': True, 'boundary': False, 'speckle': True, 'msc': False, 'local_var': False}`

## U-Net + V2d-noBoundaryKeepMSC-noLocalVar

- variant: `v2d_no_boundary_keep_msc_no_local_var`
- class: `UNetWithFADCGV2dAblation`
- params: `31596611`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- V2d variant: `no_boundary_keep_msc_no_local_var`
- frequency initialization: `random`
- depthwise kernel learnable: `True`
- config: `{'channel': True, 'boundary': False, 'speckle': True, 'msc': True, 'local_var': False}`

## U-Net + V2d-BoundaryNoMSC-noLocalVar

- variant: `v2d_boundary_no_msc_no_local_var`
- class: `UNetWithFADCGV2dAblation`
- params: `31597635`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- V2d variant: `boundary_no_msc_no_local_var`
- frequency initialization: `random`
- depthwise kernel learnable: `True`
- config: `{'channel': True, 'boundary': True, 'speckle': True, 'msc': False, 'local_var': False}`

## U-Net + V2d-noLocalVar-Laplacian

- variant: `v2d_no_local_var_laplacian`
- class: `UNetWithFADCGV2dAblation`
- params: `31616067`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- V2d variant: `no_local_var`
- frequency initialization: `laplacian`
- depthwise kernel learnable: `True`
- config: `{'channel': True, 'boundary': True, 'speckle': True, 'msc': True, 'local_var': False}`

## U-Net + V2d-noBoundaryKeepMSC-noLocalVar-Laplacian

- variant: `v2d_no_boundary_keep_msc_no_local_var_laplacian`
- class: `UNetWithFADCGV2dAblation`
- params: `31596611`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- V2d variant: `no_boundary_keep_msc_no_local_var`
- frequency initialization: `laplacian`
- depthwise kernel learnable: `True`
- config: `{'channel': True, 'boundary': False, 'speckle': True, 'msc': True, 'local_var': False}`

## U-Net + V2d-BoundaryNoMSC-noLocalVar-Laplacian

- variant: `v2d_boundary_no_msc_no_local_var_laplacian`
- class: `UNetWithFADCGV2dAblation`
- params: `31597635`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- V2d variant: `boundary_no_msc_no_local_var`
- frequency initialization: `laplacian`
- depthwise kernel learnable: `True`
- config: `{'channel': True, 'boundary': True, 'speckle': True, 'msc': False, 'local_var': False}`

