import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class FloodDataset(Dataset):
    """洪水分割数据集（支持数据增强）"""

    def __init__(self, img_dir, mask_dir, img_size=256, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

        # 获取所有图像文件
        self.images = sorted([f for f in os.listdir(img_dir)
                              if f.endswith(('.png', '.jpg', '.tif'))])

        print(f"加载数据集: {len(self.images)} 个样本")
        print(f"  图像目录: {img_dir}")
        print(f"  标签目录: {mask_dir}")
        if augment:
            print(f"  数据增强: 启用")

        # 定义数据增强变换
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        else:
            self.transform = None

        # 调试标志
        self._debug_printed = False

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 加载图像（单通道）
        image = Image.open(img_path).convert('L')
        image = image.resize((self.img_size, self.img_size))
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]

        # 加载标签（文件名相同）
        mask_path = os.path.join(self.mask_dir, img_name)

        # 确保标签文件存在
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"标签文件不存在: {mask_path}")

        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.img_size, self.img_size))
        mask = np.array(mask, dtype=np.float32)

        # 二值化：255 -> 1, 0 -> 0
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

        # 数据增强（同时对图像和mask做相同变换）
        if self.augment and self.transform:
            # 使用相同的随机种子保证图像和mask变换一致
            seed = torch.random.initial_seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)

        # 调试：打印第一个样本的信息
        if not self._debug_printed:
            print(f"\n[调试] 样本 {img_name}:")
            print(f"  图像范围: [{image.min():.3f}, {image.max():.3f}]")
            print(f"  标签范围: [{mask.min():.3f}, {mask.max():.3f}]")
            print(f"  洪水像素比例: {(mask == 1).float().mean():.4f}")
            print(f"  标签唯一值: {torch.unique(mask)}")
            self._debug_printed = True

        return image, mask