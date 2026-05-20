import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class DeepLabV3PlusLight(nn.Module):
    """
    DeepLabV3+ Light - 使用MobileNetV3作为骨干网络
    参数量约12M，适合有限数据训练
    """

    def __init__(self, n_classes=1, pretrained=True):
        super(DeepLabV3PlusLight, self).__init__()

        # 使用torchvision预训练的DeepLabV3+ MobileNetV3
        if pretrained:
            self.model = deeplabv3_mobilenet_v3_large(pretrained=True)
        else:
            self.model = deeplabv3_mobilenet_v3_large(pretrained=False)

        # 修改输出通道（从21类改为二分类）
        # DeepLabV3+的classifier结构： (256, 256) -> (256, n_classes)
        in_channels = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = nn.Conv2d(in_channels, n_classes, kernel_size=1)

        # 如果使用辅助分类器，也修改
        if hasattr(self.model, 'aux_classifier') and self.model.aux_classifier is not None:
            aux_in = self.model.aux_classifier[-1].in_channels
            self.model.aux_classifier[-1] = nn.Conv2d(aux_in, n_classes, kernel_size=1)

        self.n_classes = n_classes

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


class DeepLabV3PlusLightSimple(nn.Module):
    """
    简化版DeepLabV3+ Light，不依赖torchvision的aux_classifier
    避免某些版本兼容性问题
    """

    def __init__(self, n_classes=1, pretrained=True):
        super(DeepLabV3PlusLightSimple, self).__init__()

        from torchvision.models import mobilenet_v3_large

        # 加载骨干网络
        if pretrained:
            backbone = mobilenet_v3_large(pretrained=True)
        else:
            backbone = mobilenet_v3_large(pretrained=False)

        # 提取特征层
        self.features = backbone.features

        # ASPP模块（简化的DeepLabV3+）
        self.aspp = ASPP(960, 256)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1)
        )

        self.n_classes = n_classes

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 编码
        features = self.features(x)

        # ASPP
        aspp_out = self.aspp(features)

        # 解码
        out = self.decoder(aspp_out)

        # 上采样到原尺寸
        out = F.interpolate(out, size=(x.shape[2], x.shape[3]),
                            mode='bilinear', align_corners=True)

        return out


class ASPP(nn.Module):
    """空洞空间金字塔池化模块"""

    def __init__(self, in_channels, out_channels, dilations=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[0], dilation=dilations[0], bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(x)))
        out3 = self.relu(self.bn3(self.conv3(x)))
        out4 = self.relu(self.bn4(self.conv4(x)))

        out5 = self.global_pool(x)
        out5 = F.interpolate(out5, size=(h, w), mode='bilinear', align_corners=True)

        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.relu(self.bn_out(self.conv_out(out)))

        return out