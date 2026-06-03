import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.baseline.fcn import FCN


class ResidualOnlyBlock(nn.Module):
    """Depthwise residual enhancement without channel or frequency gates."""

    def __init__(self, channels, kernel_size=3, alpha_init=0.5):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        nn.init.normal_(self.weight)
        self.last_diagnostics = {}

    def forward(self, x, collect_diagnostics=False):
        padding = (self.kernel_size - 1) // 2
        z = F.conv2d(x, self.weight, padding=padding, groups=self.channels)
        out = x + self.alpha * z
        if collect_diagnostics:
            self.last_diagnostics = self._diagnostics()
        return out

    def _diagnostics(self):
        return {
            "alpha_value": self.alpha.detach().cpu(),
            "depthwise_weight_mean": self.weight.detach().mean().cpu(),
            "depthwise_weight_std": self.weight.detach().std(unbiased=False).cpu(),
        }

    def collect_input_diagnostics(self, x):
        with torch.no_grad():
            _ = self.forward(x, collect_diagnostics=True)
        return self.last_diagnostics


class FCNWithResidualOnly(FCN):
    """FCN residual-only control with the same path as FCN V2d ablation."""

    def __init__(self, in_channels=1, num_classes=1):
        super().__init__(in_channels, num_classes)
        self.fc6_light = ResidualOnlyBlock(512, kernel_size=7)
        self.up_dim = nn.Conv2d(512, 4096, 1)
        self.fc7_light = ResidualOnlyBlock(4096, kernel_size=1)
        self.score = nn.Conv2d(4096, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.fc6_light(x)
        x = self.relu6(x)
        x = self.drop6(x)
        x = self.up_dim(x)

        x = self.fc7_light(x)
        x = self.relu7(x)
        x = self.drop7(x)
        x = self.score(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)


def test():
    x = torch.randn(2, 1, 256, 256)
    model = FCNWithResidualOnly(in_channels=1, num_classes=1)
    out = model(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"FCN + Residual-only: {params:.2f}M")
    print(f"Input: {x.shape}, Output: {out.shape}")


if __name__ == "__main__":
    test()
