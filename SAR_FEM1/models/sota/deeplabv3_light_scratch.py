import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class DeepLabV3PlusLightScratch(nn.Module):
    """
    DeepLabV3+ Light - 从零训练，不使用ImageNet预训练
    用于公平对比Light-FADC
    """

    def __init__(self, n_classes=1):
        super(DeepLabV3PlusLightScratch, self).__init__()

        # 不使用预训练 (pretrained=False)
        self.model = deeplabv3_mobilenet_v3_large(pretrained=False)

        # 修改输出通道
        in_channels = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = nn.Conv2d(in_channels, n_classes, kernel_size=1)

        # 如果使用辅助分类器，也修改
        if hasattr(self.model, 'aux_classifier') and self.model.aux_classifier is not None:
            aux_in = self.model.aux_classifier[-1].in_channels
            self.model.aux_classifier[-1] = nn.Conv2d(aux_in, n_classes, kernel_size=1)

        self.n_classes = n_classes

        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"DeepLabV3+ Light (Scratch) initialized")
        print(f"  Total parameters: {total_params:.2f}M")
        print(f"  Pretrained: False")

    def forward(self, x):
        # 单通道SAR图像 -> 3通道（复制）
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 前向传播
        out = self.model(x)

        # 返回分割结果
        if isinstance(out, dict):
            out = out['out']

        return out


def test():
    """测试模型"""
    x = torch.randn(2, 1, 256, 256)

    # 测试从零训练版本
    model = DeepLabV3PlusLightScratch(n_classes=1)
    out = model(x)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")

    # 检查梯度
    loss = out.sum()
    loss.backward()
    print("✅ Forward and backward pass successful")


if __name__ == '__main__':
    test()