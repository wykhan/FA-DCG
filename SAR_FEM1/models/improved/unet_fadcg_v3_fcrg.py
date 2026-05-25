import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.baseline.unet import UNet
from models.improved.fadcg_v3_fcrg import FADCGV3FCRG


class UNetWithFADCGV3FCRG(UNet):
    """U-Net + FA-DCG V3 frequency-conditioned response gate."""

    def __init__(self, in_channels=1, num_classes=1, features=[64, 128, 256, 512], variant="v3_fcrg_b"):
        super().__init__(in_channels, num_classes, features)
        self.variant = variant
        self.light_fadc = FADCGV3FCRG(features[-1] * 2, kernel_size=3, variant=variant)

    def forward(self, x):
        skip_connections = []

        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.light_fadc(x)

        skip_connections = skip_connections[::-1]
        for idx, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            x = torch.cat([x, skip_connections[idx]], dim=1)
            x = decoder(x)

        return self.final_conv(x)


def test():
    x = torch.randn(2, 1, 256, 256)
    model = UNetWithFADCGV3FCRG(in_channels=1, num_classes=1, variant="v3_fcrg_b")
    out = model(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"U-Net + FA-DCG V3 FCRG: {params:.2f}M")
    print(f"Input: {x.shape}, Output: {out.shape}")


if __name__ == "__main__":
    test()

