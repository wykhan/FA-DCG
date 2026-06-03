# Task 012 Implementation Audit

## Required Variant Config

- `no_boundary_no_local_var`: `{'channel': True, 'boundary': False, 'speckle': True, 'msc': False, 'local_var': False}`
- required config: `channel=True, boundary=False, speckle=True, msc=False, local_var=False, freq_init=random`

## FCN + Residual-only

- variant: `residual_only`
- class: `FCNWithResidualOnly`
- params: `136402371`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- path: `conv1 -> conv2 -> conv3 -> conv4 -> conv5 -> block512 -> relu6/drop6/up_dim -> block4096 -> relu7/drop7/score -> interpolate`
- enhancement modules: `['ResidualOnlyBlock', 'ResidualOnlyBlock']`
  - module0 channels: `512`; kernel_size: `7`
  - gates: `none`; descriptors: `none`; residual formula: `out = x + alpha * depthwise_conv(x)`
  - module1 channels: `4096`; kernel_size: `1`
  - gates: `none`; descriptors: `none`; residual formula: `out = x + alpha * depthwise_conv(x)`

## FCN + V2d-noBoundary-noLocalVar

- variant: `no_boundary_no_local_var`
- class: `FCNWithFADCGV2dAblation`
- params: `144973893`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- path: `conv1 -> conv2 -> conv3 -> conv4 -> conv5 -> block512 -> relu6/drop6/up_dim -> block4096 -> relu7/drop7/score -> interpolate`
- enhancement modules: `['FADCGV2dAblation', 'FADCGV2dAblation']`
  - module0 channels: `512`; kernel_size: `7`
  - config: `{'channel': True, 'boundary': False, 'speckle': True, 'msc': False, 'local_var': False}`
  - freq_init: `random`
  - module1 channels: `4096`; kernel_size: `1`
  - config: `{'channel': True, 'boundary': False, 'speckle': True, 'msc': False, 'local_var': False}`
  - freq_init: `random`

## Parameter Deltas

- FCN: params=134271937, delta_vs_fcn=0, MACs=20518027264, MACs_delta_vs_fcn=0
- FCN + Residual-only: params=136402371, delta_vs_fcn=2130434, MACs=20174340096, MACs_delta_vs_fcn=-343687168
- FCN + V2d-noBoundary-noLocalVar: params=144973893, delta_vs_fcn=10701956, MACs=20186698240, MACs_delta_vs_fcn=-331329024
- Params(V2d) - Params(Residual-only) = `8571522`
