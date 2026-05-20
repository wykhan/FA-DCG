import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import AutoMinorLocator
import os
import glob

# 设置全局字体，使论文更专业
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


# ===== 平滑函数（可调整平滑程度）=====
def smooth(y, weight=0.85):
    """
    指数移动平均平滑曲线
    weight: 平滑权重，越大越平滑（建议0.85-0.95）
    """
    smoothed = []
    last = y[0]
    for val in y:
        last = last * weight + (1 - weight) * val
        smoothed.append(last)
    return np.array(smoothed)


def find_latest_csv(pattern):
    """查找最新的CSV文件"""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)


def load_csv(path):
    """加载CSV文件"""
    try:
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        return data[:, 0], data[:, 3]
    except Exception as e:
        print(f"错误: 读取文件 {path} 时出错: {e}")
        return None, None


# ===== 定义模型配置（请根据实际文件名更新）=====
# FA-DCG: Feature-Level Frequency-Aware Dilated Convolution Module with Adaptive Gating
models_config = {
    'FCN': {
        'path': 'checkpoints/fcn_20260324_105935.csv',
        'color': '#2E86AB',  # 深蓝色
        'linestyle': '-',
        'linewidth': 2.2,
        'group': 'fcn_baseline'
    },
    'FCN+FEM': {
        'path': 'checkpoints/fcn_fem_20260324_112221.csv',
        'color': '#A23B72',  # 紫罗兰色
        'linestyle': '-',
        'linewidth': 2.0,
        'group': 'fcn_variants'
    },
    'FCN+FA-DCG': {
        'path': 'checkpoints/fcn_fadc_light_20260325_163249.csv',  # 如文件名已改请更新
        'color': '#F18F01',  # 橙色
        'linestyle': '-',
        'linewidth': 3.5,
        'group': 'fcn_variants'
    },
    'UNet': {
        'path': 'checkpoints/unet_20260328_170332.csv',
        'color': '#73AB84',  # 薄荷绿
        'linestyle': '--',
        'linewidth': 2.0,
        'group': 'unet_baseline'
    },
    'UNet+CBAM': {
        'path': 'checkpoints/unet_cbam_20260330_094230.csv',
        'color': '#BF4E30',  # 红褐色
        'linestyle': '--',
        'linewidth': 2.0,
        'group': 'unet_variants'
    },
    'UNet+FA-DCG': {
        'path': 'checkpoints/unet_fadc_light_20260328_195515.csv',  # 如文件名已改请更新
        'color': '#5D2E8C',  # 紫色
        'linestyle': '--',
        'linewidth': 3.5,
        'alpha': 0.95,
        'group': 'unet_variants'
    },
    'UNet+FEM': {
        'path': 'checkpoints/unet_fem_20260328_213558.csv',
        'color': '#D96C6C',  # 珊瑚色
        'linestyle': '--',
        'linewidth': 2.0,
        'group': 'unet_variants'
    }
}

# 自动查找所有CSV文件
print("正在搜索checkpoints目录下的CSV文件...")
print("-" * 60)

all_csv_files = glob.glob('checkpoints/*.csv')
print(f"找到 {len(all_csv_files)} 个CSV文件:")
for f in all_csv_files:
    print(f"  - {os.path.basename(f)}")

print("\n" + "=" * 60)

# ===== 读取所有数据 =====
model_data = {}
loaded_models = []

print("加载模型数据:")
print("-" * 60)

for name, config in models_config.items():
    print(f"正在加载 {name}...", end=' ')
    epochs, values = load_csv(config['path'])
    if epochs is not None and values is not None:
        # 平滑处理
        smoothed = smooth(values, weight=0.88)
        model_data[name] = {
            'epochs': epochs,
            'values': values,
            'smoothed': smoothed,
            'config': config
        }
        loaded_models.append(name)
        print(f"✓ 成功 (Final mIoU = {smoothed[-1]:.4f}, Best = {np.max(smoothed):.4f})")
    else:
        print(f"✗ 失败 - 请检查文件路径: {config['path']}")
        print(f"   提示: 请根据上面列出的CSV文件更新路径")

print("\n" + "=" * 60)

# 检查是否有数据被加载
if len(model_data) == 0:
    print("错误: 没有找到任何数据文件，请检查文件路径！")
    print("\n请执行以下操作之一:")
    print("1. 修改上面的 models_config 中的文件路径")
    print("2. 确保所有模型都已训练并生成CSV文件")
    exit()

print(f"\n成功加载 {len(model_data)}/{len(models_config)} 个模型")
if len(model_data) < len(models_config):
    missing = set(models_config.keys()) - set(model_data.keys())
    print(f"缺失的模型: {', '.join(missing)}")

# ===== 创建图形 =====
fig, ax = plt.subplots(figsize=(14, 8))

# ===== 绘制所有曲线 =====
for name, data in model_data.items():
    config = data['config']
    epochs = data['epochs']
    smoothed = data['smoothed']

    # 绘制主曲线
    ax.plot(epochs, smoothed,
            color=config['color'],
            label=name,
            linestyle=config['linestyle'],
            linewidth=config['linewidth'],
            alpha=0.9,
            zorder=2)

    # 添加轻微阴影填充增强视觉效果
    ax.fill_between(epochs, smoothed - 0.015, smoothed + 0.015,
                    color=config['color'], alpha=0.08, linewidth=0, zorder=1)

# ===== 设置坐标轴 =====
ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax.set_ylabel('mIoU', fontsize=13, fontweight='bold')

# 更新标题，使用新的模块名称
ax.set_title('Training Convergence: FCN vs UNet with FA-DCG Integration',
             fontsize=14, fontweight='bold', pad=15)

# 设置坐标轴范围
ax.set_xlim(0, 50)
# 动态设置y轴范围
all_values = []
for data in model_data.values():
    all_values.extend(data['smoothed'])
y_min = max(0.3, min(all_values) - 0.05)
y_max = min(0.9, max(all_values) + 0.05)
ax.set_ylim(y_min, y_max)

# 设置坐标轴刻度
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

# 设置网格
ax.grid(True, which='major', linestyle='--', alpha=0.3, linewidth=0.8)
ax.grid(True, which='minor', linestyle=':', alpha=0.2, linewidth=0.5)

# 设置边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# ===== 图例设置 =====
legend = ax.legend(loc='lower right', frameon=True, fancybox=False,
                   edgecolor='gray', framealpha=0.95, ncol=1,
                   fontsize=12,
                   columnspacing=0.8,
                   handlelength=2.5,
                   handleheight=1.5,
                   borderpad=0.8,
                   labelspacing=0.5)
legend.get_frame().set_linewidth(0.8)

# ===== 添加FA-DCG模块说明注释 =====
# 在图表下方添加模块全称说明
fig.text(0.5, 0.01,
         'FA-DCG: Feature-Level Frequency-Aware Dilated Convolution Module with Adaptive Gating',
         ha='center', fontsize=9, fontstyle='italic', alpha=0.7)

# ===== 调整布局 =====
plt.tight_layout(rect=[0, 0.03, 1, 1])  # 为底部注释留出空间

# ===== 保存图片 =====
output_name = 'all_models_comparison_fa_dcg'
plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.savefig(f'{output_name}.svg', dpi=300, bbox_inches='tight', pad_inches=0.05)

print(f"\n图片已保存为: {output_name}.png, {output_name}.pdf, {output_name}.svg")

plt.show()

# ===== 打印详细统计信息 =====
print("\n" + "=" * 70)
print("训练结果完整统计（按最终mIoU排序）:")
print("FA-DCG: Feature-Level Frequency-Aware Dilated Convolution Module with Adaptive Gating")
print("=" * 70)
print(f"{'Rank':<4} {'Model':<25} {'Final mIoU':<12} {'Best mIoU':<12} {'vs FCN':<10}")
print("-" * 70)

if 'FCN' in model_data:
    baseline_final = model_data['FCN']['smoothed'][-1]
else:
    baseline_final = None

sorted_models = sorted(model_data.items(),
                       key=lambda x: x[1]['smoothed'][-1],
                       reverse=True)

for rank, (name, data) in enumerate(sorted_models, 1):
    final_miou = data['smoothed'][-1]
    best_miou = np.max(data['smoothed'])
    if baseline_final is not None and name != 'FCN':
        imp = (final_miou - baseline_final) / baseline_final * 100
        improvement_str = f"+{imp:.1f}%"
    elif name == 'FCN' and baseline_final is not None:
        improvement_str = "baseline"
    else:
        improvement_str = "N/A"
    print(f"{rank:<4} {name:<25} {final_miou:.4f}      {best_miou:.4f}      {improvement_str:<10}")

print("\n" + "=" * 70)
print("架构组对比分析:")
print("=" * 70)

# 分组统计
fcn_group = {name: data for name, data in model_data.items() if 'FCN' in name}
unet_group = {name: data for name, data in model_data.items() if 'UNet' in name}

if fcn_group:
    fcn_best = max(fcn_group.items(), key=lambda x: x[1]['smoothed'][-1])
    fcn_avg = np.mean([d['smoothed'][-1] for d in fcn_group.values()])
    print(f"\nFCN架构组:")
    print(f"  - 模型数量: {len(fcn_group)}")
    print(f"  - 最佳模型: {fcn_best[0]} (mIoU={fcn_best[1]['smoothed'][-1]:.4f})")
    print(f"  - 平均性能: {fcn_avg:.4f}")

    if 'FCN' in fcn_group and fcn_best[0] != 'FCN':
        fcn_baseline = fcn_group['FCN']['smoothed'][-1]
        fcn_imp = (fcn_best[1]['smoothed'][-1] - fcn_baseline) / fcn_baseline * 100
        print(f"  - 最佳提升: +{fcn_imp:.2f}% (相对于FCN基线)")

if unet_group:
    unet_best = max(unet_group.items(), key=lambda x: x[1]['smoothed'][-1])
    unet_avg = np.mean([d['smoothed'][-1] for d in unet_group.values()])
    print(f"\nUNet架构组:")
    print(f"  - 模型数量: {len(unet_group)}")
    print(f"  - 最佳模型: {unet_best[0]} (mIoU={unet_best[1]['smoothed'][-1]:.4f})")
    print(f"  - 平均性能: {unet_avg:.4f}")

    if 'UNet' in unet_group and unet_best[0] != 'UNet':
        unet_baseline = unet_group['UNet']['smoothed'][-1]
        unet_imp = (unet_best[1]['smoothed'][-1] - unet_baseline) / unet_baseline * 100
        print(f"  - 最佳提升: +{unet_imp:.2f}% (相对于UNet基线)")

# 跨架构对比
if fcn_group and unet_group:
    best_fcn = max(fcn_group.values(), key=lambda x: x['smoothed'][-1])
    best_unet = max(unet_group.values(), key=lambda x: x['smoothed'][-1])
    print(f"\n跨架构对比:")
    print(f"  - 最佳FCN变体: {best_fcn['smoothed'][-1]:.4f}")
    print(f"  - 最佳UNet变体: {best_unet['smoothed'][-1]:.4f}")

    if best_unet['smoothed'][-1] > best_fcn['smoothed'][-1]:
        improvement = (best_unet['smoothed'][-1] - best_fcn['smoothed'][-1]) / best_fcn['smoothed'][-1] * 100
        print(f"  - UNet架构优势: +{improvement:.2f}%")
    else:
        improvement = (best_fcn['smoothed'][-1] - best_unet['smoothed'][-1]) / best_unet['smoothed'][-1] * 100
        print(f"  - FCN架构优势: +{improvement:.2f}%")

# ===== FA-DCG 模块专项分析 =====
print("\n" + "=" * 70)
print("FA-DCG 模块效果专项分析:")
print("=" * 70)

fadcg_models = {name: data for name, data in model_data.items() if 'FA-DCG' in name}
if fadcg_models:
    print("\nFA-DCG (Feature-Level Frequency-Aware Dilated Convolution Module with Adaptive Gating)")
    print("Plug-and-Play feature-level frequency enhancement module")
    print("-" * 50)

    for name, data in fadcg_models.items():
        base_arch = 'FCN' if 'FCN' in name else 'UNet'
        if base_arch in model_data:
            baseline_final = model_data[base_arch]['smoothed'][-1]
            fadcg_final = data['smoothed'][-1]
            improvement = (fadcg_final - baseline_final) / baseline_final * 100
            print(f"  {name}:")
            print(f"    - 相对于{base_arch}基线提升: +{improvement:.2f}%")
            print(f"    - 最终mIoU: {fadcg_final:.4f} vs 基线: {baseline_final:.4f}")

        # 对比其他增强方法
        other_enhancements = []
        if base_arch == 'FCN':
            other_enhancements = ['FCN+FEM']
        else:
            other_enhancements = ['UNet+CBAM', 'UNet+FEM']

        for other_name in other_enhancements:
            if other_name in model_data:
                other_final = model_data[other_name]['smoothed'][-1]
                diff = (fadcg_final - other_final) / other_final * 100
                if diff > 0:
                    print(f"    - 优于{other_name}: +{diff:.2f}%")
                else:
                    print(f"    - 低于{other_name}: {diff:.2f}%")

    print("\n关键特性:")
    print("  • Plug-and-Play: 即插即用，可直接集成到现有分割架构")
    print("  • Feature-Level Frequency-Aware: 在特征层面进行频域感知，捕捉多尺度频率信息")
    print("  • Adaptive Gating: 自适应门控机制，动态调整频域特征的重要性")
    print("  • Dilated Convolution: 膨胀卷积扩大感受野，保持分辨率")