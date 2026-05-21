# Local FADC-Aligned Underperformance Analysis

Date: 2026-05-21

## Context

Task 003 implemented a local PyTorch FADC-aligned module without relying on
MMCV/MMSeg custom operators. The implementation followed the official FADC
paper at the mechanism level:

- FreqSelect: high/low frequency band reweighting.
- AdaDR: approximated with spatially adaptive dilation branch mixing.
- AdaKern: low/high depthwise kernel decomposition with attention gates.

The implementation was evaluated on the current validation protocol with seed
2024.

## Results

| Model | V1.1 mIoU | FADC mIoU | Delta | V1.1 Dice | FADC Dice | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FCN + FADC | 0.842804 | 0.663577 | -0.179227 | 0.905853 | 0.771850 | underperforming |
| U-Net + FADC | 0.892461 | 0.887575 | -0.004886 | 0.938098 | 0.934839 | aligned_and_comparable |

The FCN result is clearly poor relative to FA-DCG V1.1. The U-Net result is
close to V1.1, but it does not exceed V1.1 from this single-seed run.

## Why the Local FADC Version Underperformed

### 1. The local implementation is not equivalent to official FADC

Official FADC uses deformable convolution offsets to dynamically adjust sampling
locations, which is how it realizes spatially adaptive dilation.

Task 003 intentionally avoided MMCV/MMSeg and custom CUDA operators. Therefore,
AdaDR was approximated as:

```text
dilation branches = [1, 2, 3, 4]
w = softmax(branch_weight_head(x_freq), dim=branch)
z = sum_k w_k * depthwise_conv_dilation_k(x_freq)
```

This captures the idea of local receptive-field selection, but it cannot fully
replicate deformable adaptive dilation. It mixes fixed dilation responses rather
than changing the sampling geometry of each kernel point.

### 2. FreqSelect learned only weak frequency separation

The diagnostics showed frequency weights close to identity:

```text
high_frequency_weight_mean ~= 0.99 - 1.00
low_frequency_weight_mean  ~= 1.01
```

This suggests that the frequency selection branch did not learn a strong
high/low-frequency separation in the current training setting. As a result, the
module behaved less like a frequency-adaptive operator and more like a
multi-dilation residual block.

### 3. Large dilation branches can harm SAR boundary features

The dilation branch diagnostics were approximately:

```text
d1 ~= 0.475
d2 ~= 0.174
d3 ~= 0.174
d4 ~= 0.176
small dilation weight ~= 0.65
large dilation weight ~= 0.35
```

Even though dilation 1 received the highest weight, the large-dilation branches
still contributed about one third of the response. For SAR flood segmentation,
water boundaries, small water bodies, narrow rivers, and building-shadow
confusions are sensitive to local detail. Large dilation in early/mid features
can blur or disturb these signals.

### 4. FCN is especially sensitive to early feature perturbation

The task_003 insertion point was:

```text
FCN: after conv3, C=256
U-Net: after encoder stage 3, C=256
```

This follows the idea that FADC should operate in feature extraction stages.
However, FCN-32s has no strong decoder or skip-fusion path to recover disturbed
spatial detail. If the early/mid feature distribution is perturbed by the
multi-dilation FADC approximation, FCN cannot easily repair it later.

This likely explains why FCN + FADC only reached mIoU 0.663577, close to the
plain FCN baseline and far below FA-DCG V1.1.

### 5. FA-DCG V1.1 and FADC solve different practical problems

FA-DCG V1.1 is not a true official FADC implementation, but it preserves the
effective part of V1.0:

```text
depthwise spatial enhancement
+ channel gate
+ residual controlled fusion
```

This is conservative and stable. The task_003 FADC-aligned module is more
complex:

```text
frequency selection
+ spatial dilation branch mixing
+ adaptive low/high kernel gates
+ residual fusion
```

The added mechanisms are conceptually closer to official FADC, but they also
introduce more ways to disrupt feature distributions.

### 6. Training was not specially tuned for FADC

Task 003 reused the current baseline protocol:

```text
Adam
lr = 1e-4
CosineAnnealingLR
BCEWithLogitsLoss
50 epochs
batch size = 8
```

A frequency/dilation adaptive module may need stabilization strategies:

- smaller learning rate for FADC parameters;
- alpha warmup;
- stronger identity initialization;
- entropy or sparsity control for branch weights;
- staged training;
- insertion-point ablation.

These were intentionally left for later analysis rather than hidden inside
task_003.

## Interpretation

The task_003 result does not prove that the official FADC idea is unsuitable for
SAR flood segmentation. It shows that a local no-MMCV approximation of FADC:

- is runnable;
- is stable on U-Net;
- does not transfer cleanly to FCN;
- does not outperform FA-DCG V1.1 under the current protocol.

The key lesson is that official FADC mechanisms should not be copied directly
into the SAR flood setting without adaptation. They need SAR-specific design
constraints, especially around boundary preservation, conservative residual
fusion, and lightweight early/mid-stage feature enhancement.

## Recommendation

Keep **FA-DCG V1.1** as the current practical baseline.

Use task_003 as a mechanism study and feasibility check for official FADC-style
ideas.

Design **FA-DCG V2.0** in task_004 as a SAR-specific lightweight module rather
than treating task_003 FADC as the final SAR solution.
