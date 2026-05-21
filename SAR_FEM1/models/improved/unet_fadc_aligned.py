import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.baseline.unet import UNet
from models.improved.fadc_aligned import FADCAligned


class UNetWithFADCAligned(UNet):
    """U-Net with local PyTorch FADC after encoder stage 3."""

    def __init__(self, in_channels=1, num_classes=1, features=[64, 128, 256, 512]):
        super().__init__(in_channels, num_classes, features)
        self.fadc_stage = 2
        self.fadc = FADCAligned(features[self.fadc_stage], kernel_size=3, lp_type="avgpool", spatial_group=1)

    def forward(self, x):
        skip_connections = []

        for idx, encoder in enumerate(self.encoder):
            x = encoder(x)
            if idx == self.fadc_stage:
                x = self.fadc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for idx, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            x = torch.cat([x, skip_connections[idx]], dim=1)
            x = decoder(x)

        return self.final_conv(x)


def test():
    x = torch.randn(2, 1, 256, 256)
    model = UNetWithFADCAligned(in_channels=1, num_classes=1)
    out = model(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"U-Net + FADC: {params:.2f}M")
    print(f"Input: {x.shape}, Output: {out.shape}")


if __name__ == "__main__":
    test()
