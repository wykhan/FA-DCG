"""
双架构特征响应对比图 — FCN + U-Net
论文 Figure: FA-DCG Feature Enhancement across Architectures
(Feature-Level Frequency-Aware Dilated Convolution Module with Adaptive Gating)
"""
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# ===== 配置 =====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_PATH = r'F:\SAR_flood\SAR_FEM1\data\flood_dataset\val\images\HS601.png'
GT_PATH    = r'F:\SAR_flood\SAR_FEM1\data\flood_dataset\val\labels\HS601.png'

# FCN
WEIGHT_FCN       = r'F:\SAR_flood\SAR_FEM1\checkpoints\fcn_best.pth'
WEIGHT_FCN_FADC  = r'F:\SAR_flood\SAR_FEM1\checkpoints\fcn_fadc_light_best.pth'
# U-Net
WEIGHT_UNET      = r'F:\SAR_flood\SAR_FEM1\checkpoints\unet_best.pth'
WEIGHT_UNET_FADC = r'F:\SAR_flood\SAR_FEM1\checkpoints\unet_fadc_light_best.pth'

SAVE_DIR = Path(r'F:\SAR_flood\SAR_FEM1\picture')
SAVE_DIR.mkdir(exist_ok=True)

# ===== 导入 =====
import sys
sys.path.append(r'F:\SAR_flood\SAR_FEM')

from models.baseline.fcn import FCN
from models.baseline.unet import UNet
from models.improved.fcn_fadc_light import FCNWithLightFADC
from models.improved.unet_fadc_light import UNetWithLightFADC

# ===== Hook =====
features = {}
def get_hook(name):
    def hook_fn(module, input, output):
        features[name] = output.detach()
    return hook_fn

# ===== 加载 =====
def load_model(model_class, weight_path, model_name):
    print(f"Loading {model_name}...")
    model = model_class(in_channels=1, num_classes=1).to(DEVICE)
    ckpt = torch.load(weight_path, map_location=DEVICE)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {params:.2f}M")
    return model

fcn       = load_model(FCN,              WEIGHT_FCN,       "FCN")
# FA-DCG: Feature-Level Frequency-Aware Dilated Convolution Module with Adaptive Gating (Plug-and-Play)
fcn_fadc  = load_model(FCNWithLightFADC, WEIGHT_FCN_FADC,  "FCN+FA-DCG")
unet      = load_model(UNet,             WEIGHT_UNET,      "U-Net")
unet_fadc = load_model(UNetWithLightFADC,WEIGHT_UNET_FADC, "U-Net+FA-DCG")

# ===== 注册 Hook =====
# FCN: fc6 (4096ch, 10×10) vs fc6_light (512ch, 16×16)
# FA-DCG: Plug-and-Play Feature-Level Frequency-Aware module
# 注：频率感知仅作用于特征图级别，非原始频谱域操作
dict(fcn.named_modules())['fc6'].register_forward_hook(get_hook('fcn_base'))
dict(fcn_fadc.named_modules())['fc6_light'].register_forward_hook(get_hook('fcn_fadc'))
print("  ✅ FCN hooks: fc6 / fc6_light (FA-DCG)")

# U-Net: bottleneck (512ch, 16×16) vs light_fadc (512ch, 16×16)
# FA-DCG: Plug-and-Play Feature-Level Frequency-Aware module
dict(unet.named_modules())['bottleneck'].register_forward_hook(get_hook('unet_base'))
dict(unet_fadc.named_modules())['light_fadc'].register_forward_hook(get_hook('unet_fadc'))
print("  ✅ U-Net hooks: bottleneck / light_fadc (FA-DCG)")

# ===== 加载图像 =====
print(f"\nLoading: {Path(IMAGE_PATH).name}")
image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
gt = cv2.imread(GT_PATH, cv2.IMREAD_GRAYSCALE)
if gt is None:
    gt = np.zeros_like(image)

h, w = image.shape
image_float = image.astype(np.float32) / 255.0
image_tensor = torch.from_numpy(image_float).unsqueeze(0).unsqueeze(0).to(DEVICE)
gt_binary = (gt > 127).astype(np.uint8)

# ===== 前向传播 =====
print("Forward pass (4 models)...")
with torch.no_grad():
    _ = fcn(image_tensor)
    _ = fcn_fadc(image_tensor)
    _ = unet(image_tensor)
    _ = unet_fadc(image_tensor)

# 打印特征图形状
print(f"  FCN fc6:              {features['fcn_base'].shape}")    # (1, 4096, 10, 10)
print(f"  FCN fc6_light (FA-DCG):{features['fcn_fadc'].shape}")   # (1, 512, 16, 16)
print(f"  UNet bottleneck:      {features['unet_base'].shape}")   # (1, 512, 16, 16)
print(f"  UNet light_fadc (FA-DCG):{features['unet_fadc'].shape}") # (1, 512, 16, 16)

# ===== 计算激活图 =====
def get_activation(feat_key):
    act = features[feat_key].mean(dim=1).squeeze().cpu().numpy()
    act = cv2.resize(act, (w, h), interpolation=cv2.INTER_LINEAR)
    return (act - act.min()) / (act.max() - act.min() + 1e-8)

act_fcn_base   = get_activation('fcn_base')
act_fcn_fadc   = get_activation('fcn_fadc')
act_unet_base  = get_activation('unet_base')
act_unet_fadc  = get_activation('unet_fadc')

# ===== ROI =====
roi_size = 70
cy, cx = h//2, w//2
rois = [
    (cy - roi_size - 30, cx - roi_size//2, cy - 30, cx + roi_size//2),          # ROI-1: 上方水体
    (cy - roi_size//2, cx - roi_size//2, cy + roi_size//2, cx + roi_size//2),    # ROI-2: 中心边界
    (cy + 30, cx - roi_size//2, cy + roi_size + 30, cx + roi_size//2),           # ROI-3: 下方散斑
]
rois = [(max(0,y1),max(0,x1),min(h,y2),min(w,x2)) for y1,x1,y2,x2 in rois]

# ===== 绘图: 4行 × 6列 =====
print("Plotting 4×6 grid...")
fig, axes = plt.subplots(4, 6, figsize=(24, 14))

row_titles = [
    'Full Scene',
    'Water Interior\n(Low-Frequency)',
    'Water Boundary\n(High-Frequency)',
    'Speckle Region\n(Residual Interference)'
]
col_titles = [
    '(a) SAR Image',
    '(b) Ground Truth',
    '(c) FCN\nBaseline',
    '(d) FCN\n+ FA-DCG',
    '(e) U-Net\nBaseline',
    '(f) U-Net\n+ FA-DCG'
]

act_maps  = [act_fcn_base, act_fcn_fadc, act_unet_base, act_unet_fadc]
act_cols  = [2, 3, 4, 5]

# 行1：完整场景
axes[0,0].imshow(image_float, cmap='gray')
axes[0,1].imshow(gt_binary, cmap='gray')
for col_idx, act in zip(act_cols, act_maps):
    axes[0,col_idx].imshow(image_float, cmap='gray')
    axes[0,col_idx].imshow(act, cmap='jet', alpha=0.45)

# ROI框
roi_colors = ['cyan', 'yellow', 'magenta']
for i, (y1, x1, y2, x2) in enumerate(rois):
    for col_idx in [0] + act_cols:
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                             fill=False, color=roi_colors[i],
                             linewidth=2, linestyle='--')
        axes[0,col_idx].add_patch(rect)
    axes[0,0].annotate(f'ROI-{i+1}', (x1-3, y1-3),
                       color=roi_colors[i], fontsize=8, fontweight='bold',
                       ha='right', va='bottom')

# 行2-4：ROI放大
for i, (y1, x1, y2, x2) in enumerate(rois):
    row = i + 1
    axes[row,0].imshow(image_float[y1:y2, x1:x2], cmap='gray')
    axes[row,1].imshow(gt_binary[y1:y2, x1:x2], cmap='gray')
    for col_idx, act in zip(act_cols, act_maps):
        axes[row,col_idx].imshow(image_float[y1:y2, x1:x2], cmap='gray')
        axes[row,col_idx].imshow(act[y1:y2, x1:x2], cmap='jet', alpha=0.45)

# 格式
for i in range(4):
    for j in range(6):
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([])
    axes[i,0].set_ylabel(row_titles[i], fontsize=11, fontweight='bold')

for j in range(6):
    axes[0,j].set_title(col_titles[j], fontsize=11, fontweight='bold')

# FCN组和U-Net组之间的分隔线
for i in range(4):
    axes[i,3].spines['right'].set_linewidth(2.5)
    axes[i,3].spines['right'].set_color('black')

# 标注
annotations = [
    (1, 'Uneven', 'orange', 'Uniform', 'green'),
    (3, 'False', 'red', 'Suppr.', 'green'),
]
for row_idx, label_bad, color_bad, label_good, color_good in annotations:
    for col_base, col_fadc in [(2,3), (4,5)]:
        axes[row_idx,col_base].annotate(label_bad, xy=(0.72, 0.12), xycoords='axes fraction',
                            fontsize=6.5, color='white', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor=color_bad, alpha=0.8))
        axes[row_idx,col_fadc].annotate(label_good, xy=(0.72, 0.12), xycoords='axes fraction',
                            fontsize=6.5, color='white', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor=color_good, alpha=0.8))

# 分组标签
fig.text(0.34, 0.98, 'FCN Architecture', ha='center', fontsize=14, fontweight='bold', color='#d62728')
fig.text(0.74, 0.98, 'U-Net Architecture', ha='center', fontsize=14, fontweight='bold', color='#1f77b4')

# FA-DCG 模块说明（Feature-Level, Plug-and-Play）
fig.text(0.5, 0.015,
         'FA-DCG: Feature-Level Frequency-Aware Dilated Convolution Module with Adaptive Gating (Plug-and-Play)',
         ha='center', fontsize=9, fontstyle='italic', color='gray')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(SAVE_DIR / 'feature_visualization_dual.pdf', dpi=300, bbox_inches='tight')
plt.savefig(SAVE_DIR / 'feature_visualization_dual.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✅ Saved to: {SAVE_DIR / 'feature_visualization_dual.pdf'}")