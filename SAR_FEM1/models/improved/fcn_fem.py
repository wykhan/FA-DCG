import torch
import torch.nn as nn
import torch.nn.functional as F
from models.baseline.fcn import FCN


class FEM(nn.Module):
    """频域增强模块"""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.gate(x)
        return x * weight


class FCNWithFEM(FCN):
    """FCN + FEM 模块"""

    def __init__(self, in_channels=1, num_classes=1):
        super().__init__(in_channels, num_classes)
        # 在分类器后添加FEM
        self.fem = FEM(4096)  # fc7输出是4096通道

    def forward(self, x):
        # 记录输入尺寸
        input_size = x.shape[-2:]

        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Classifier
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.drop6(x)

        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)

        # FEM模块 - 改进点
        x = self.fem(x)

        x = self.score(x)

        # 修复：使用目标尺寸
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        return x


def test():
    x = torch.randn(2, 1, 256, 256)

    model_base = FCN(in_channels=1, num_classes=1)
    out_base = model_base(x)
    params_base = sum(p.numel() for p in model_base.parameters()) / 1e6

    model_improved = FCNWithFEM(in_channels=1, num_classes=1)
    out_improved = model_improved(x)
    params_improved = sum(p.numel() for p in model_improved.parameters()) / 1e6

    print(f"FCN           - 参数量: {params_base:.2f}M, 输出: {out_base.shape}")
    print(f"FCN+FEM       - 参数量: {params_improved:.2f}M, 输出: {out_improved.shape}")
    print(f"尺寸匹配: {x.shape == out_base.shape}")


if __name__ == '__main__':
    test()