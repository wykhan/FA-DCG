import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.baseline.fcn import FCN
from models.improved.fadc_optimized import VectorizedLightFADC


class FCNWithOptimizedFADCG(FCN):
    """FCN + FA-DCG V1.1 with the original integration path and vectorized blocks."""

    def __init__(self, in_channels=1, num_classes=1):
        super().__init__(in_channels, num_classes)
        self.fc6_light = VectorizedLightFADC(512, kernel_size=7)
        self.up_dim = nn.Conv2d(512, 4096, 1)
        self.fc7_light = VectorizedLightFADC(4096, kernel_size=1)
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
    model = FCNWithOptimizedFADCG(in_channels=1, num_classes=1)
    out = model(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"FCN + Optimized FA-DCG: {params:.2f}M")
    print(f"Input: {x.shape}, Output: {out.shape}")


if __name__ == "__main__":
    test()
