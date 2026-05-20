import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.baseline.unet import UNet


class LightFADC(nn.Module):
    """
    轻量级频率感知卷积模块（Lightweight Frequency-Aware Dilated Convolution）
    与FCN版本保持一致
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        # 可学习膨胀率（每个通道一个值）
        self.dilation_param = nn.Parameter(torch.ones(channels) * 1.0)

        # 频率感知门控（通道级）
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        # 可学习平衡系数
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # 卷积核（每个通道独立）
        self.weight = nn.Parameter(torch.randn(channels, 1, kernel_size, kernel_size))

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # 限制膨胀率范围
        dilation = torch.clamp(self.dilation_param, 1.0, 4.0)

        # 为每个通道做膨胀卷积
        outputs = []
        for c in range(C):
            weight_c = self.weight[c:c + 1]  # [1, 1, K, K]
            x_c = x[:, c:c + 1, :, :]

            d = int(dilation[c].item())
            padding = (self.kernel_size - 1) * d // 2

            out_c = F.conv2d(x_c, weight_c, padding=padding, dilation=d, groups=1)
            outputs.append(out_c)

        x_dilated = torch.cat(outputs, dim=1)  # [B, C, H, W]

        # 频率感知门控
        gate = self.gate(x_dilated)

        # 融合输出
        out = identity + self.alpha * gate * x_dilated

        return out


class UNetWithLightFADC(UNet):
    """
    U-Net + 轻量级频率感知卷积模块
    在bottleneck后插入Light-FADC模块
    """

    def __init__(self, in_channels=1, num_classes=1, features=[64, 128, 256, 512]):
        super().__init__(in_channels, num_classes, features)

        # 在bottleneck后添加Light-FADC模块
        # bottleneck输出通道数为 features[-1] * 2 = 1024
        bottleneck_channels = features[-1] * 2
        self.light_fadc = LightFADC(bottleneck_channels, kernel_size=3)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Light-FADC增强
        x = self.light_fadc(x)

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

    # 测试U-Net + LightFADC
    model = UNetWithLightFADC(in_channels=1, num_classes=1)
    out = model(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6

    print("=" * 60)
    print("U-Net + Light-FADC")
    print("=" * 60)
    print(f"U-Net (baseline):        {params_unet:.2f}M")
    print(f"U-Net + LightFADC:       {params:.2f}M")
    print(f"参数量增加:              {params - params_unet:.2f}M")
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")


if __name__ == '__main__':
    test()