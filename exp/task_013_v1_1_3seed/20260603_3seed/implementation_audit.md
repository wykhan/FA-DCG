# Task 013 Implementation Audit

- V1.1 block class: `VectorizedLightFADC`.
- V1.0 loop-based implementation was not used.

## FCN + FA-DCG V1.1

- variant: `v1_1`
- class: `FCNWithOptimizedFADCG`
- params: `144932419`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- VectorizedLightFADC blocks: `2`
  - block0: channels=`512`, kernel_size=`7`
  - block1: channels=`4096`, kernel_size=`1`
- path: `conv1 -> conv2 -> conv3 -> conv4 -> conv5 -> V1.1(512,k=7) -> relu6/drop6/up_dim -> V1.1(4096,k=1) -> relu7/drop7/score -> interpolate`

## U-Net + FA-DCG V1.1

- variant: `v1_1`
- class: `UNetWithOptimizedFADCG`
- params: `31578178`
- output shape: `(2, 1, 256, 256)`
- finite output: `True`
- no lazy modules in forward: `True`
- VectorizedLightFADC blocks: `1`
  - block0: channels=`1024`, kernel_size=`3`
- path: `U-Net bottleneck -> V1.1(1024,k=3) -> decoder`

## Parameter Deltas

- FCN: params=134271937, delta_vs_backbone=0, MACs=20518027264, MACs_delta_vs_backbone=0
- FCN + FA-DCG V1.1: params=144932419, delta_vs_backbone=10660482, MACs=20183159296, MACs_delta_vs_backbone=-334867968
- U-Net: params=31042369, delta_vs_backbone=0, MACs=54662266880, MACs_delta_vs_backbone=0
- U-Net + FA-DCG V1.1: params=31578178, delta_vs_backbone=535809, MACs=54663054336, MACs_delta_vs_backbone=787456
