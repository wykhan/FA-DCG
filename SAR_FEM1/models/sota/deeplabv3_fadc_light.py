import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class LightFADC(nn.Module):
    """
    轻量级频域自适应卷积模块
    输入输出通道数相同
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        # 可学习膨胀率（每个通道一个值）
        self.dilation_param = nn.Parameter(torch.ones(channels) * 1.0)

        # 响应选择性门控（通道级）
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

        # 响应选择性门控
        gate = self.freq_gate(x_dilated)

        # 融合输出
        out = identity + self.alpha * gate * x_dilated

        return out


class DeepLabV3PlusLightWithFADC(nn.Module):
    """
    DeepLabV3+ Light + Light-FADC模块
    正确访问DeepLabV3的内部结构
    """

    def __init__(self, n_classes=1, pretrained=True):
        super(DeepLabV3PlusLightWithFADC, self).__init__()

        # 使用torchvision预训练的DeepLabV3+ MobileNetV3
        if pretrained:
            self.model = deeplabv3_mobilenet_v3_large(pretrained=True)
        else:
            self.model = deeplabv3_mobilenet_v3_large(pretrained=False)

        # 保存原始分类器的输入通道数
        # DeepLabV3的结构: backbone -> classifier
        # classifier包含: ASPP + 1x1卷积
        # 我们需要获取ASPP的输出通道数

        # 方法：直接获取classifier的输入通道（即backbone的输出通道）
        # MobileNetV3 backbone的输出通道是960
        backbone_out_channels = 960

        # 获取ASPP的输出通道（通常是256）
        # 从classifier的第一个模块（ASPP）获取输出通道
        if hasattr(self.model.classifier, '0'):
            # classifier是一个ModuleList，第一个是ASPP
            aspp_module = self.model.classifier[0]
            # ASPP的输出通道
            aspp_out_channels = 256
        else:
            aspp_out_channels = 256

        # 插入Light-FADC模块（在ASPP之后）
        self.light_fadc = LightFADC(aspp_out_channels, kernel_size=3)

        # 修改分类器的最后一层（输出通道改为n_classes）
        # 原classifier的结构: ASPP -> Conv2d(256, 256) -> Conv2d(256, 21)
        # 我们需要保留ASPP和第一个Conv，只修改最后的输出卷积
        self.original_classifier = self.model.classifier

        # 创建新的分类器，在ASPP之后插入Light-FADC
        self.new_classifier = nn.Sequential()

        # 添加ASPP部分（classifier的前两层）
        if isinstance(self.model.classifier, nn.Sequential):
            # 保存ASPP部分（通常是第一个模块）
            self.aspp = self.model.classifier[0]
            # 保存中间的1x1卷积（如果有）
            if len(self.model.classifier) > 1:
                self.mid_conv = self.model.classifier[1]
            else:
                self.mid_conv = nn.Identity()
        else:
            self.aspp = self.model.classifier
            self.mid_conv = nn.Identity()

        # 最终的输出卷积
        self.final_conv = nn.Conv2d(256, n_classes, kernel_size=1)

        self.n_classes = n_classes

        print(f"DeepLabV3+ Light + Light-FADC initialized")
        print(f"  ASPP output channels: 256")
        print(f"  Light-FADC channels: 256")

    def forward(self, x):
        # 单通道SAR图像 -> 3通道（复制）
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        input_shape = x.shape[-2:]

        # 通过backbone
        features = self.model.backbone(x)

        # backbone输出可能是dict或tensor
        if isinstance(features, dict):
            features = features['out']

        # 通过ASPP
        aspp_out = self.aspp(features)

        # 通过中间的1x1卷积（如果有）
        if hasattr(self, 'mid_conv') and not isinstance(self.mid_conv, nn.Identity):
            aspp_out = self.mid_conv(aspp_out)

        # ========== 插入Light-FADC ==========
        aspp_out = self.light_fadc(aspp_out)

        # 最终分类卷积
        out = self.final_conv(aspp_out)

        # 上采样到原尺寸
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=True)

        return out


class DeepLabV3PlusLightWithFADC_Simple(nn.Module):
    """
    DeepLabV3+ Light + Light-FADC模块（简化版本）
    完全重建classifier，避免访问内部结构问题
    """

    def __init__(self, n_classes=1, pretrained=True):
        super(DeepLabV3PlusLightWithFADC_Simple, self).__init__()

        # 使用torchvision预训练的DeepLabV3+ MobileNetV3
        if pretrained:
            self.model = deeplabv3_mobilenet_v3_large(pretrained=True)
        else:
            self.model = deeplabv3_mobilenet_v3_large(pretrained=False)

        # 获取backbone的输出通道数
        # MobileNetV3 Large的输出是960
        backbone_out_channels = 960

        # 重新构建ASPP（与原始一致）
        from torchvision.models.segmentation.deeplabv3 import ASPP
        self.aspp = ASPP(backbone_out_channels, [12, 24, 36])

        # 插入Light-FADC
        self.light_fadc = LightFADC(256, kernel_size=3)

        # 最终分类卷积
        self.final_conv = nn.Conv2d(256, n_classes, kernel_size=1)

        self.n_classes = n_classes

        # 复制预训练权重（如果可用）
        if pretrained:
            self._copy_pretrained_weights()

        print(f"DeepLabV3+ Light + Light-FADC (Simple) initialized")

    def _copy_pretrained_weights(self):
        """尝试复制预训练权重"""
        try:
            # 复制backbone权重
            self.model.backbone.load_state_dict(self.model.backbone.state_dict())
            # 复制ASPP权重
            if hasattr(self.model, 'classifier') and hasattr(self.model.classifier, '0'):
                self.aspp.load_state_dict(self.model.classifier[0].state_dict())
            print("  ✅ Pretrained weights loaded")
        except Exception as e:
            print(f"  ⚠️ Could not copy pretrained weights: {e}")

    def forward(self, x):
        # 单通道SAR图像 -> 3通道（复制）
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        input_shape = x.shape[-2:]

        # 通过backbone
        features = self.model.backbone(x)

        if isinstance(features, dict):
            features = features['out']

        # 通过ASPP
        aspp_out = self.aspp(features)

        # ========== 插入Light-FADC ==========
        aspp_out = self.light_fadc(aspp_out)

        # 最终分类卷积
        out = self.final_conv(aspp_out)

        # 上采样到原尺寸
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=True)

        return out


def test():
    """测试模型"""
    x = torch.randn(2, 1, 256, 256)

    # 测试原始DeepLabV3+ Light
    try:
        model_base = deeplabv3_mobilenet_v3_large(pretrained=False)
        model_base.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
        params_base = sum(p.numel() for p in model_base.parameters()) / 1e6
        print(f"DeepLabV3+ Light (baseline):    {params_base:.2f}M")

        # 测试前向传播
        with torch.no_grad():
            out_base = model_base(x)
            if isinstance(out_base, dict):
                out_base = out_base['out']
        print(f"  Baseline output shape: {out_base.shape}")
    except Exception as e:
        print(f"Baseline test failed: {e}")

    # 测试版本1（直接访问结构）
    try:
        model_v1 = DeepLabV3PlusLightWithFADC(n_classes=1, pretrained=False)
        params_v1 = sum(p.numel() for p in model_v1.parameters()) / 1e6
        out_v1 = model_v1(x)
        print(f"DeepLabV3+ Light + FADC (V1):   {params_v1:.2f}M")
        print(f"  V1 output shape: {out_v1.shape}")
    except Exception as e:
        print(f"V1 test failed: {e}")

    # 测试版本2（简化版本）
    try:
        model_v2 = DeepLabV3PlusLightWithFADC_Simple(n_classes=1, pretrained=False)
        params_v2 = sum(p.numel() for p in model_v2.parameters()) / 1e6
        out_v2 = model_v2(x)
        print(f"DeepLabV3+ Light + FADC (Simple): {params_v2:.2f}M")
        print(f"  Simple output shape: {out_v2.shape}")

        # 检查梯度
        loss = out_v2.sum()
        loss.backward()
        print("  ✅ Forward and backward pass successful")
    except Exception as e:
        print(f"Simple version test failed: {e}")

    print("\n" + "=" * 60)
    print("✅ 测试完成")


if __name__ == '__main__':
    test()