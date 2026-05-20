import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.baseline.fcn import FCN
from models.improved.fadc_fast import FastFADCG


class FCNWithFastFADCG(FCN):
    """FCN with Fast FA-DCG-v1 after conv5, before the classifier."""

    def __init__(self, in_channels=1, num_classes=1):
        super().__init__(in_channels, num_classes)
        self.fast_fadc = FastFADCG(512, kernel_size=3)

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.fast_fadc(x)

        x = self.fc6(x)
        x = self.relu6(x)
        x = self.drop6(x)

        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)

        x = self.score(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)


def test():
    x = torch.randn(2, 1, 256, 256)
    model = FCNWithFastFADCG(in_channels=1, num_classes=1)
    out = model(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"FCN + Fast FA-DCG: {params:.2f}M")
    print(f"Input: {x.shape}, Output: {out.shape}")


if __name__ == "__main__":
    test()
