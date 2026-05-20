import torch
from torch.utils.data import Dataset
import os


class FloodDatasetPT(Dataset):
    """加载预处理后的.pt格式数据"""

    def __init__(self, data_dir, img_size=256):
        """
        data_dir: 包含 .pt 文件的目录
        .pt 文件格式: 假设为 [C, H, W]，C=2 (image, mask)
        """
        self.data_dir = data_dir
        self.img_size = img_size

        # 获取所有.pt文件
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])

        print(f"加载数据集: {len(self.files)} 个样本")
        print(f"  目录: {data_dir}")

        # 打印第一个文件的信息（调试）
        if len(self.files) > 0:
            sample = torch.load(os.path.join(data_dir, self.files[0]))
            if isinstance(sample, torch.Tensor):
                print(f"  数据格式: tensor, shape={sample.shape}")
            elif isinstance(sample, dict):
                print(f"  数据格式: dict, keys={list(sample.keys())}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 加载.pt文件
        data = torch.load(os.path.join(self.data_dir, self.files[idx]))

        # 解析数据格式
        if isinstance(data, torch.Tensor):
            # 假设 tensor 形状为 [2, H, W] 或 [C, H, W]
            if data.shape[0] == 2:
                image = data[0:1]  # [1, H, W]
                mask = data[1:2]  # [1, H, W]
            else:
                # 默认第一个通道是图像
                image = data[0:1]
                mask = data[1:2] if data.shape[0] > 1 else data[0:1]
        elif isinstance(data, dict):
            # 字典格式
            image = data['image']
            mask = data['mask']
        elif isinstance(data, (tuple, list)):
            # 元组或列表格式
            image, mask = data[0], data[1]
        else:
            raise ValueError(f"不支持的.pt格式: {type(data)}")

        # 确保是 [C, H, W] 格式
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # 调整尺寸
        if image.shape[-2] != self.img_size or image.shape[-1] != self.img_size:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear'
            ).squeeze(0)
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).float(), size=(self.img_size, self.img_size), mode='nearest'
            ).squeeze(0)

        # 确保mask是0/1
        mask = (mask > 0.5).float()

        return image, mask