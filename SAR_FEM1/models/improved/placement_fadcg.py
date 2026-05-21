import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.baseline.fcn import FCN
from models.baseline.unet import UNet
from models.improved.fadc_optimized import VectorizedLightFADC
from models.improved.fadcg_v2d import FADCGV2d


def _make_block(block_type, channels, kernel_size=3, alpha_init=0.25):
    if block_type == "v1.1":
        block = VectorizedLightFADC(channels, kernel_size=kernel_size)
        with torch.no_grad():
            block.alpha.fill_(float(alpha_init))
        return block
    if block_type == "v2d":
        return FADCGV2d(channels, kernel_size=kernel_size, alpha_init=alpha_init)
    raise ValueError(f"Unsupported FA-DCG block type: {block_type}")


class FCNWithPlacementFADCG(FCN):
    """FCN with one FA-DCG block inserted in an early/mid feature stage."""

    CHANNELS_BY_PLACEMENT = {
        "after_conv3": 256,
        "after_conv4": 512,
    }

    def __init__(self, in_channels=1, num_classes=1, block_type="v1.1", placement="after_conv4", alpha_init=0.25):
        if placement not in self.CHANNELS_BY_PLACEMENT:
            raise ValueError(f"Unsupported FCN placement: {placement}")
        super().__init__(in_channels, num_classes)
        self.block_type = block_type
        self.placement = placement
        self.placement_fadcg = _make_block(
            block_type,
            self.CHANNELS_BY_PLACEMENT[placement],
            kernel_size=3,
            alpha_init=alpha_init,
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.placement == "after_conv3":
            x = self.placement_fadcg(x)
        x = self.conv4(x)
        if self.placement == "after_conv4":
            x = self.placement_fadcg(x)
        x = self.conv5(x)

        x = self.fc6(x)
        x = self.relu6(x)
        x = self.drop6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)
        x = self.score(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)


class UNetWithPlacementFADCG(UNet):
    """U-Net with one FA-DCG block inserted after an encoder stage."""

    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        features=[64, 128, 256, 512],
        block_type="v1.1",
        placement="after_encoder3",
        alpha_init=0.25,
    ):
        if placement not in {"after_encoder2", "after_encoder3"}:
            raise ValueError(f"Unsupported U-Net placement: {placement}")
        super().__init__(in_channels, num_classes, features)
        self.block_type = block_type
        self.placement = placement
        stage_idx = 1 if placement == "after_encoder2" else 2
        self.placement_stage_idx = stage_idx
        self.placement_fadcg = _make_block(block_type, features[stage_idx], kernel_size=3, alpha_init=alpha_init)

    def forward(self, x):
        skip_connections = []

        for idx, encoder in enumerate(self.encoder):
            x = encoder(x)
            if idx == self.placement_stage_idx:
                x = self.placement_fadcg(x)
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
    configs = [
        ("fcn", "v1.1", "after_conv3", FCNWithPlacementFADCG),
        ("fcn", "v2d", "after_conv4", FCNWithPlacementFADCG),
        ("unet", "v1.1", "after_encoder2", UNetWithPlacementFADCG),
        ("unet", "v2d", "after_encoder3", UNetWithPlacementFADCG),
    ]
    for arch, block, placement, factory in configs:
        model = factory(in_channels=1, num_classes=1, block_type=block, placement=placement)
        out = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{arch} {block} {placement}: params={params:.4f}M, output={tuple(out.shape)}")


if __name__ == "__main__":
    test()
