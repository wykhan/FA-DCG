import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.baseline.unet import UNet


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation 通道注意力模块
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


class UNetWithFEM(UNet):
    """
    U-Net + SE模块 (Frequency Enhancement Module)
    在bottleneck后插入SE模块，增强通道特征
    """

    def __init__(self, in_channels=1, num_classes=1, features=[64, 128, 256, 512], reduction=16):
        super().__init__(in_channels, num_classes, features)

        # 在bottleneck后添加SE模块
        # bottleneck输出通道数为 features[-1] * 2 = 1024
        bottleneck_channels = features[-1] * 2
        self.se = SELayer(bottleneck_channels, reduction=reduction)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # SE模块增强
        x = self.se(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            x = torch.cat([x, skip_connections[idx]], dim=1)
            x = decoder(x)

        x = self.final_conv(x)
        return x


def test():
    """测试模型"""
    x = torch.randn(2, 1, 256, 256)

    # 测试U-Net baseline
    model_unet = UNet(in_channels=1, num_classes=1)
    params_unet = sum(p.numel() for p in model_unet.parameters()) / 1e6

    # 测试U-Net + SE
    model = UNetWithFEM(in_channels=1, num_classes=1)
    out = model(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6

    print("=" * 60)
    print("U-Net + SE (通道注意力)")
    print("=" * 60)
    print(f"U-Net (baseline):        {params_unet:.2f}M")
    print(f"U-Net + SE:              {params:.2f}M")
    print(f"参数量增加:              {params - params_unet:.2f}M")
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")


if __name__ == '__main__':
    test()