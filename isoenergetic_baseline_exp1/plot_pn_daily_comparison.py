#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制Pn每日对比柱状图
对比原始PPFD和调整后PPFD(+50)的Pn均值
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pathlib import Path

# 获取脚本所在目录
script_dir = Path(__file__).parent

# 设置适合发表的图表样式 - 统一参数
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Arial Unicode MS', 'DejaVu Sans', 'Helvetica']
plt.rcParams['font.serif'] = ['Arial', 'Arial Unicode MS', 'DejaVu Sans', 'Helvetica']
plt.rcParams["font.size"] = "22"
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.unicode_minus'] = False
# 设置数学文本使用加粗字体
plt.rcParams['mathtext.default'] = 'bf'  # 'bf' = boldface，确保数学符号也加粗
plt.rcParams['mathtext.fontset'] = 'stix'

# 文件路径（使用绝对路径）
input_file = script_dir / 'pn_daily_comparison_simple.csv'

# 读取数据
df = pd.read_csv(input_file, comment='#')

# 过滤掉平均行和注释行
df = df[df['日期'].str.match(r'^\d{4}-\d{2}-\d{2}$', na=False)].copy()

# 提取数据（注意：原始=LightFARM传感器数据，新计算=Rule-based固定450）
dates = df['日期'].values
pn_lightfarm = df['Pn均值_原始'].values  # LightFARM (传感器PPFD)
pn_rulebased = df['Pn均值_新计算'].values  # Rule-based (固定450 PPFD)

# 创建天数标签（只用数字 1, 2, 3...）
num_days = len(dates)
day_labels = [f'{i+1}' for i in range(num_days)]

# 创建图形
fig, ax = plt.subplots(figsize=(12, 6))

# 设置柱子位置
x = np.arange(len(day_labels))
width = 0.35  # 柱子宽度

# 颜色设置
color_rulebased = '#C2E2FA'  # Rule-based 蓝色（纯色）
color_lightfarm = '#FE6244'  # LightFARM 粉红色/玫瑰色（纯色）

# Rule-based (蓝色纯色) - 固定450 PPFD (先绘制，在左侧)
bars1 = ax.bar(
    x - width / 2,
    pn_rulebased,
    width,
    label='_nolegend_',
    facecolor=color_rulebased,
    edgecolor='black',
    linewidth=1.0,
    alpha=0.8,
)

# LightFARM (粉红色纯色) - 传感器PPFD (后绘制，在右侧)
bars2 = ax.bar(
    x + width / 2,
    pn_lightfarm,
    width,
    label='_nolegend_',
    facecolor=color_lightfarm,
    edgecolor='black',
    linewidth=1.0,
    alpha=0.8,
)

# 设置坐标轴标签 - 加粗
ax.set_xlabel('Day', fontsize=22, fontweight='bold')
# 对于包含数学符号的 Y 轴标签，使用原始字符串并确保加粗
# 使用 \mathbf{} 或 \boldsymbol{} 来确保数学符号也加粗
ylabel = ax.set_ylabel(r'Pn (μmol CO$_2$ m$\mathbf{^{-2}}$ s$\mathbf{^{-1}}$)', 
                       fontsize=22, fontweight='bold')
# 确保标签对象本身也设置为加粗
ylabel.set_fontweight('bold')
ylabel.set_fontsize(22)

# 设置X轴刻度
ax.set_xticks(x)
ax.set_xticklabels(day_labels, rotation=0)

# 设置Y轴范围和刻度 - 从0到20，每格5
y_min = 0
y_max = 20

ax.set_ylim(y_min, y_max)
# 设置Y轴刻度: 0, 5, 10, 15, 20
ax.set_yticks(np.arange(0, y_max + 1, 5))

# 设置刻度标签字体加粗
ax.tick_params(axis='y', labelsize=20, which='major')
ax.tick_params(axis='x', labelsize=20, which='major')
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# 设置网格
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# 设置图例 - 放在左上角，字体加粗，使用纯色
rule_patch = Patch(
    facecolor=color_rulebased,
    edgecolor='black',
    linewidth=1.0,
    label='Rule-based',
    alpha=0.8,
)
lightfarm_patch = Patch(
    facecolor=color_lightfarm,
    edgecolor='black',
    linewidth=1.0,
    label='LightFARM',
    alpha=0.8,
)

legend_handles = [rule_patch, lightfarm_patch]
legend = ax.legend(
    handles=legend_handles,
    loc='upper left',
    fontsize=16,
    frameon=True,
    fancybox=False,
    shadow=False,
    framealpha=0.9,
    edgecolor='black',
    facecolor='white',
    handlelength=2,
    handleheight=1.2,
)
# 设置图例文字加粗
for text in legend.get_texts():
    text.set_fontweight('bold')

# 设置坐标轴样式
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)

plt.tight_layout()

# 保存为高质量PNG
output_file = script_dir / 'pn_time.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"图表已保存到: {output_file.name} (300 DPI)")

# 保存为PDF格式（矢量图）
output_file_pdf = script_dir / 'pn_time.pdf'
plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"PDF图表已保存到: {output_file_pdf.name} (矢量图)")

# 不显示图表，只保存文件
# plt.show()

# 打印数据摘要
print("\n数据摘要:")
print(f"Rule-based (固定450 PPFD) - 平均Pn: {pn_rulebased.mean():.2f}, 范围: {pn_rulebased.min():.2f} - {pn_rulebased.max():.2f}")
print(f"LightFARM (传感器PPFD) - 平均Pn: {pn_lightfarm.mean():.2f}, 范围: {pn_lightfarm.min():.2f} - {pn_lightfarm.max():.2f}")
# 计算差异: 1 - (LightFARM/Rule-based)
diff_pct = (1 - pn_lightfarm.mean() / pn_rulebased.mean()) * 100
print(f"LightFARM相对Rule-based的节能差异: {diff_pct:.2f}% (1 - {pn_lightfarm.mean():.2f}/{pn_rulebased.mean():.2f})")

