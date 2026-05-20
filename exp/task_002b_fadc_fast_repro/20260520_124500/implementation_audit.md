# Implementation Audit

## Fast FA-DCG-v1 Changes

| Issue | Old implementation | New implementation | Fixed? |
| --- | --- | --- | --- |
| per-channel Python loop | yes | no, uses vectorized depthwise branch convolutions | yes |
| `.item()` integer dilation | yes | no, uses differentiable soft branch fusion | yes |
| 4096-channel FA-DCG in FCN | yes | no, only after 512-channel conv5 | yes |
| module created in forward | yes | no, all modules are constructor-defined | yes |
| differentiable branch fusion | no | yes, softmax over dilation branches | yes |

## Initialization

- `gamma = -4.0`, so continuous dilation starts near 1.
- `theta = 0` with a +0.25 bias for dilation 1.
- `beta_raw = 0.5413248546`, so `softplus(beta_raw)` is approximately 1.
- `alpha = 0.5`.
- Branch depthwise convolutions use Kaiming normal initialization.
- Gate hidden channels use `max(C // 16, 1)`.

## Unit and Shape Test Output

```text
{'module': 'FastFADCG(512)', 'output_shape': (2, 512, 16, 16), 'weights_sum_min': 0.9999999403953552, 'weights_sum_max': 0.9999999403953552, 'gate_min': 0.4506959021091461, 'gate_max': 0.5508518815040588, 'd_cont_mean': 1.053958773612976}
{'module': 'FastFADCG(1024)', 'output_shape': (2, 1024, 16, 16), 'weights_sum_min': 0.9999999403953552, 'weights_sum_max': 0.9999999403953552, 'gate_min': 0.45727697014808655, 'gate_max': 0.5401746034622192, 'd_cont_mean': 1.053958773612976}
{'module': 'UNetWithFastFADCG', 'output_shape': (1, 1, 256, 256), 'fast_modules': 1}
{'module': 'FCNWithFastFADCG', 'output_shape': (1, 1, 256, 256), 'fast_modules': 1}
task_002b FastFADCG tests passed
```
