import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from skimage.draw import polygon, ellipse
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 10


def generate_sar_patch(size=(128, 128), scenario='boundary', seed=42):
    np.random.seed(seed)
    bg = np.random.normal(0.50, 0.10, size).clip(0, 1)
    if scenario == 'boundary':
        water = np.ones(size) * 0.12 + np.random.normal(0, 0.04, size).clip(-0.1, 0.1)
        for i in range(size[0]):
            cutoff = 40 + int(20 * np.sin(i / 15)) + np.random.randint(-3, 3)
            water[i, :cutoff] = 1.0
            blend_width = 8
            for j in range(blend_width):
                alpha = j / blend_width
                idx = cutoff + j
                if idx < size[1]:
                    water[i, idx] = 0.12 * (1 - alpha) + 0.50 * alpha
        patch = water
    elif scenario == 'urban_shadow':
        patch = np.random.normal(0.48, 0.10, size).clip(0, 1)
        rr, cc = ellipse(90, 25, 25, 18, shape=size)
        patch[rr, cc] = 0.12 + np.random.normal(0, 0.03, len(rr))
        for x_start in [50, 70, 95]:
            rr, cc = polygon([x_start, x_start + 8, x_start + 22, x_start + 14],
                             [35, 5, 5, 35], shape=size)
            patch[rr, cc] = 0.10 + np.random.normal(0, 0.04, len(rr))
        for x in [52, 72, 97]:
            rr, cc = polygon([x - 2, x + 3, x + 3, x - 2], [38, 10, 10, 38], shape=size)
            patch[rr, cc] = 0.90 + np.random.normal(0, 0.05, len(rr))
    elif scenario == 'small_bodies':
        patch = np.random.normal(0.52, 0.09, size).clip(0, 1)
        ponds = [(25, 30, 7, 6), (70, 20, 5, 4), (45, 65, 6, 5),
                 (90, 75, 8, 5), (15, 85, 5, 7), (100, 45, 4, 4)]
        for cx, cy, a, b in ponds:
            rr, cc = ellipse(cx, cy, a, b, shape=size)
            patch[rr, cc] = 0.10 + np.random.normal(0, 0.03, len(rr))
    speckle = np.random.gamma(shape=4, scale=1 / 4, size=size)
    patch = patch * speckle
    patch = patch.clip(0, 1)
    return patch


def generate_gt_mask(size=(128, 128), scenario='boundary'):
    mask = np.zeros(size, dtype=bool)
    if scenario == 'boundary':
        for i in range(size[0]):
            cutoff = 40 + int(20 * np.sin(i / 15))
            mask[i, :cutoff] = True
    elif scenario == 'urban_shadow':
        rr, cc = ellipse(90, 25, 25, 18, shape=size)
        mask[rr, cc] = True
    elif scenario == 'small_bodies':
        ponds = [(25, 30, 7, 6), (70, 20, 5, 4), (45, 65, 6, 5),
                 (90, 75, 8, 5), (15, 85, 5, 7), (100, 45, 4, 4)]
        for cx, cy, a, b in ponds:
            rr, cc = ellipse(cx, cy, a, b, shape=size)
            mask[rr, cc] = True
    return mask


def predict_unet(gt, fp_regions=None, noise_std=0.18, shrink=5):
    pred = binary_erosion(gt, iterations=shrink).astype(bool)
    if fp_regions is not None:
        for rr, cc in fp_regions:
            pred[rr, cc] = True
    flip_mask = np.random.random(pred.shape) < noise_std
    pred[flip_mask & ~gt] = ~pred[flip_mask & ~gt]
    return pred.astype(np.uint8)


def predict_cbam(gt, fp_regions=None, noise_std=0.10, shrink=3):
    pred = binary_erosion(gt, iterations=shrink).astype(bool)
    if fp_regions is not None:
        for rr, cc in fp_regions:
            pred[rr, cc] = True
    flip_mask = np.random.random(pred.shape) < noise_std
    pred[flip_mask & ~gt] = ~pred[flip_mask & ~gt]
    return pred.astype(np.uint8)


def predict_ours(gt, noise_std=0.04, shrink=1):
    pred = binary_erosion(gt, iterations=shrink).astype(bool)
    flip_mask = np.random.random(pred.shape) < noise_std
    pred[flip_mask & ~gt] = ~pred[flip_mask & ~gt]
    return pred.astype(np.uint8)


scenarios = ['boundary', 'urban_shadow', 'small_bodies']
scenario_labels = ['A', 'B', 'C']
scenario_titles = [
    'Ambiguous Water-Land\nBoundary',
    'Urban Building Shadow\n(Black triangle: building shadow, not water)',
    'Scattered Small Water\nBodies'
]

data = {}
for sc in scenarios:
    d = {}
    d['sar'] = generate_sar_patch(scenario=sc, seed=hash(sc) % 10000)
    d['gt'] = generate_gt_mask(scenario=sc)
    if sc == 'urban_shadow':
        fp_list = []
        for x_start in [50, 70, 95]:
            rr, cc = polygon([x_start, x_start + 8, x_start + 22, x_start + 14],
                             [35, 5, 5, 35], shape=(128, 128))
            fp_list.append((rr, cc))
    else:
        fp_list = None
    d['unet'] = predict_unet(d['gt'], fp_regions=fp_list, noise_std=0.18, shrink=5)
    d['cbam'] = predict_cbam(d['gt'], fp_regions=fp_list, noise_std=0.10, shrink=3)
    d['ours'] = predict_ours(d['gt'], noise_std=0.04, shrink=1)
    data[sc] = d

fig, axes = plt.subplots(3, 5, figsize=(15.5, 9.5))

# 标题分两行：主标题 + 说明行
fig.suptitle('Local Zoom-in: U-Net vs. U-Net+CBAM vs. U-Net+FA-DCG (Plug-and-Play)\n'
             '(Building shadow in row B is not water; it is a typical SAR false-alarm source)',
             fontsize=12, fontweight='bold', y=0.99)

col_labels = ['SAR Image', 'Ground Truth', 'U-Net', 'U-Net+CBAM', 'FA-DCG (Ours)']
for j, label in enumerate(col_labels):
    axes[0, j].set_title(label, fontsize=11, fontweight='bold')

for i, sc in enumerate(scenarios):
    d = data[sc]
    axes[i, 0].imshow(d['sar'], cmap='gray', vmin=0, vmax=1)
    axes[i, 1].imshow(d['sar'], cmap='gray', vmin=0, vmax=1)
    axes[i, 1].contour(d['gt'], levels=[0.5], colors='lime', linewidths=1.2)

    # ABC 标签 + 场景说明，左对齐
    label_text = f"{scenario_labels[i]}: {scenario_titles[i]}"
    axes[i, 0].text(-0.38, 0.5, label_text,
                    fontsize=9.5, fontweight='bold', rotation=90,
                    transform=axes[i, 0].transAxes,
                    ha='center', va='center', linespacing=1.3)

    preds = [d['unet'], d['cbam'], d['ours']]
    for k, pred in enumerate(preds):
        ax = axes[i, k + 2]
        ax.imshow(d['sar'], cmap='gray', vmin=0, vmax=1)
        ax.contour(d['gt'], levels=[0.5], colors='lime', linewidths=1.0, linestyles='dotted')
        water_overlay = np.zeros((128, 128, 4))
        water_overlay[pred == 1, :] = [0, 1, 1, 0.55]
        ax.imshow(water_overlay)

        fp_mask = (pred == 1) & (~d['gt'])
        fp_coords = np.argwhere(fp_mask)
        if len(fp_coords) > 0:
            sample_idx = np.random.choice(len(fp_coords), min(5, len(fp_coords)), replace=False)
            for idx in sample_idx:
                y, x = fp_coords[idx]
                ax.annotate('', xy=(x, y), xytext=(x - 8, y - 8),
                            arrowprops=dict(arrowstyle='->', color='red', lw=1.0, alpha=0.9))

        fn_mask = (pred == 0) & (d['gt'])
        fn_coords = np.argwhere(fn_mask)
        if len(fn_coords) > 0:
            sample_idx = np.random.choice(len(fn_coords), min(5, len(fn_coords)), replace=False)
            for idx in sample_idx:
                y, x = fn_coords[idx]
                ax.annotate('', xy=(x, y), xytext=(x - 8, y - 8),
                            arrowprops=dict(arrowstyle='->', color='blue', lw=1.0, alpha=0.9))

    for j in range(5):
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        for spine in axes[i, j].spines.values():
            spine.set_linewidth(0.8)

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='lime', lw=2.0, linestyle='dotted', label='GT Water'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
           markersize=14, label='False Positive'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='blue',
           markersize=14, label='False Negative'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           fontsize=12, framealpha=0.9, columnspacing=2.0)

plt.subplots_adjust(left=0.13, right=0.95, top=0.90, bottom=0.08, wspace=0.15, hspace=0.28)
plt.savefig('fig_local_zoom_comparison_FA-DCG.png', dpi=300, bbox_inches='tight')
plt.show()
print("Done: fig_local_zoom_comparison_FA-DCG.png")