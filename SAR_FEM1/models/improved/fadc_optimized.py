import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorizedLightFADC(nn.Module):
    """FA-DCG V1.1 behavior-compatible vectorized block.

    FA-DCG V1.0 initialized all channel dilation values to 1 and
    converted them to Python integers, so dilation was effectively fixed during
    training. FA-DCG V1.1 preserves that behavior while replacing the
    per-channel Python loop with one depthwise grouped convolution.
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation_param = nn.Parameter(torch.ones(channels) * 1.0)
        hidden = max(channels // 4, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.weight = nn.Parameter(torch.randn(channels, 1, kernel_size, kernel_size))

    def forward(self, x):
        identity = x
        padding = (self.kernel_size - 1) // 2
        x_dilated = F.conv2d(x, self.weight, padding=padding, groups=self.channels)
        gate = self.gate(x_dilated)
        return identity + self.alpha * gate * x_dilated

    def extra_repr(self):
        return f"channels={self.channels}, kernel_size={self.kernel_size}, dilation=1"
