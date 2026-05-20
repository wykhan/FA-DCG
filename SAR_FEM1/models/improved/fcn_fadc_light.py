import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.baseline.fcn import FCN


class LightFADC(nn.Module):
    """
    FA-DCG V1.0 轻量级频域自适应卷积模块
    输入输出通道数相同
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        # 可学习膨胀率（每个通道一个值）
        self.dilation_param = nn.Parameter(torch.ones(channels) * 1.0)

        # 频域门控（通道级）
        self.freq_gate = nn.Sequential(
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
            # 计算膨胀后的有效padding，保证输出尺寸不变
            padding = (self.kernel_size - 1) * d // 2

            out_c = F.conv2d(x_c, weight_c, padding=padding, dilation=d, groups=1)
            outputs.append(out_c)

        x_dilated = torch.cat(outputs, dim=1)  # [B, C, H, W]

        # 频域门控
        gate = self.freq_gate(x_dilated)

        # 融合输出
        out = identity + self.alpha * gate * x_dilated

        return out


class FCNWithLightFADC(FCN):
    """
    FCN + FA-DCG V1.0
    """

    def __init__(self, in_channels=1, num_classes=1):
        super().__init__(in_channels, num_classes)

        # 替换fc6: 输入512 → 输出4096
        # 注意：fc6的输入是512（conv5的输出），输出是4096
        self.fc6_light = LightFADC(512, kernel_size=7)

        # 替换fc7: 输入4096 → 输出4096
        self.fc7_light = LightFADC(4096, kernel_size=1)

        # 输出层
        self.score = nn.Conv2d(4096, num_classes, 1)

        # 重新注册模块，覆盖父类的fc6和fc7
        # 但为了不破坏父类结构，我们直接在forward中使用新模块

    def forward(self, x):
        input_size = x.shape[-2:]

        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # 使用LightFADC替代fc6
        x = self.fc6_light(x)  # [B, 512, H, W] → [B, 512, H, W]
        x = self.relu6(x)
        x = self.drop6(x)

        # 注意：这里需要先升维到4096，因为fc6_light输出是512
        # 所以加一个1x1卷积升维
        if not hasattr(self, 'up_dim'):
            self.up_dim = nn.Conv2d(512, 4096, 1)
            self.up_dim = self.up_dim.to(x.device)
        x = self.up_dim(x)

        # 使用LightFADC替代fc7
        x = self.fc7_light(x)  # [B, 4096, H, W] → [B, 4096, H, W]
        x = self.relu7(x)
        x = self.drop7(x)

        x = self.score(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        return x


def test():
    """测试模型"""
    x = torch.randn(2, 1, 256, 256)

    # 测试FCN baseline
    model_fcn = FCN(in_channels=1, num_classes=1)
    params_fcn = sum(p.numel() for p in model_fcn.parameters()) / 1e6

    # 测试FCN + 轻量级FADC
    model = FCNWithLightFADC(in_channels=1, num_classes=1)
    out = model(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6

    print("=" * 60)
    print("轻量级FADC - 适配小样本")
    print("=" * 60)
    print(f"FCN (baseline):        {params_fcn:.2f}M")
    print(f"FCN + 轻量FADC:        {params:.2f}M")
    print(f"参数量增加:            {params - params_fcn:.2f}M")

    # 检查输出
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")


if __name__ == '__main__':
    test()
