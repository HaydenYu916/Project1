#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制MPPI控制日志中的pred_pn数据
只显示每天9am到次日1am之间的数据，其他时间段设为0
"""

import csv
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取MPPI控制日志数据
csv_path = '/home/pi/Desktop/LED_MPPI_Controller/logs/mppi_v2_control_log.csv'

timestamps_mppi = []
pn_filtered_mppi = []
all_timestamps_mppi = []
all_pn_mppi = []
all_data_mppi = []

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 跳过空行或无效数据
        if not row['timestamp'] or not row['timestamp'].strip():
            continue
        
        try:
            # 解析时间戳
            ts = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
            
            # 读取pred_pn值
            pn = float(row['pred_pn'])
            
            # 记录所有数据
            all_data_mppi.append((ts, pn))
            all_timestamps_mppi.append(ts)
            all_pn_mppi.append(pn)
                
        except (ValueError, KeyError):
            # 跳过无法解析的行
            continue

# 处理MPPI数据：在1am-9am之间插入NaN来断开连线
from datetime import timedelta

for i, (ts, pn) in enumerate(all_data_mppi):
    hour = ts.hour
    
    # 只添加有效时段的数据点（9:00到23:59，或0:00到1:00）
    if (hour >= 9) or (hour <= 0):
        # 检查是否需要断开连线：与上一个点的时间差超过2小时
        if len(timestamps_mppi) > 0 and timestamps_mppi[-1] is not None:
            time_diff = (ts - timestamps_mppi[-1]).total_seconds() / 3600  # 小时
            if time_diff > 2:  # 超过2小时，说明跨越了休眠期
                timestamps_mppi.append(None)
                pn_filtered_mppi.append(None)
        
        timestamps_mppi.append(ts)
        pn_filtered_mppi.append(pn)

# 读取PPFD计算数据
ppfd_csv_path = '/home/pi/Desktop/ppfd_calculations_corrected.csv'

timestamps_ppfd = []
pn_filtered_ppfd = []
all_timestamps_ppfd = []
all_pn_ppfd = []
all_data_ppfd = []

with open(ppfd_csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not row['timestamp'] or not row['timestamp'].strip():
            continue
        
        try:
            # 解析ISO格式时间戳（去掉Z后缀）
            ts_str = row['timestamp'].replace('Z', '')
            ts = datetime.fromisoformat(ts_str)
            
            # 读取predicted_pn值
            pn = float(row['predicted_pn'])
            
            # 记录所有数据
            all_data_ppfd.append((ts, pn))
            all_timestamps_ppfd.append(ts)
            all_pn_ppfd.append(pn)
                
        except (ValueError, KeyError) as e:
            # 跳过无法解析的行
            continue

# 处理PPFD数据：在1am-9am之间插入NaN来断开连线
for i, (ts, pn) in enumerate(all_data_ppfd):
    hour = ts.hour
    
    # 只添加有效时段的数据点（9:00到23:59，或0:00到1:00）
    if (hour >= 9) or (hour <= 0):
        # 检查是否需要断开连线：与上一个点的时间差超过2小时
        if len(timestamps_ppfd) > 0 and timestamps_ppfd[-1] is not None:
            time_diff = (ts - timestamps_ppfd[-1]).total_seconds() / 3600  # 小时
            if time_diff > 2:  # 超过2小时，说明跨越了休眠期
                timestamps_ppfd.append(None)
                pn_filtered_ppfd.append(None)
        
        timestamps_ppfd.append(ts)
        pn_filtered_ppfd.append(pn)

# 过滤数据：只保留10月14、15、16日
from datetime import date as dt_date

target_dates = [dt_date(2025, 10, 14), dt_date(2025, 10, 15), dt_date(2025, 10, 16)]

# 过滤MPPI数据
timestamps_mppi_filtered = []
pn_mppi_filtered = []
for ts, pn in zip(timestamps_mppi, pn_filtered_mppi):
    if ts is None:
        timestamps_mppi_filtered.append(None)
        pn_mppi_filtered.append(None)
    elif ts.date() in target_dates:
        timestamps_mppi_filtered.append(ts)
        pn_mppi_filtered.append(pn)

# 过滤PPFD数据
timestamps_ppfd_filtered = []
pn_ppfd_filtered = []
for ts, pn in zip(timestamps_ppfd, pn_filtered_ppfd):
    if ts is None:
        timestamps_ppfd_filtered.append(None)
        pn_ppfd_filtered.append(None)
    elif ts.date() in target_dates:
        timestamps_ppfd_filtered.append(ts)
        pn_ppfd_filtered.append(pn)

# 绘图
fig, ax = plt.subplots(figsize=(14, 6))

# 绘制MPPI控制数据
ax.plot(timestamps_mppi_filtered, pn_mppi_filtered, 
        linewidth=1.5, marker='o', markersize=3, 
        color='#2E86AB', label='MPPI Control PN', alpha=0.8)

# 绘制PPFD计算数据
ax.plot(timestamps_ppfd_filtered, pn_ppfd_filtered, 
        linewidth=1.0, marker='s', markersize=2, 
        color='#E63946', label='PPFD Calculated PN', alpha=0.6)

# 设置标签和标题
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted PN (μmol/m²/s)', fontsize=12, fontweight='bold')
ax.set_title('Predicted Net Photosynthesis Rate Comparison\n(9:00 AM - 1:00 AM Daily)', 
             fontsize=14, fontweight='bold', pad=20)

# 设置网格
ax.grid(True, alpha=0.3, linestyle='--')

# 格式化x轴，只显示日期
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator())

# 旋转x轴标签
plt.xticks(rotation=45, ha='right')

# 添加图例
ax.legend(loc='upper right', framealpha=0.9)

# 设置y轴范围
ax.set_ylim(bottom=0)

# 调整布局
plt.tight_layout()

# 保存图片
output_path = '/home/pi/Desktop/mppi_pn_filtered.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图表已保存到: {output_path}")

# 计算每天的平均值
from collections import defaultdict

# MPPI数据按天分组
daily_pn_mppi = defaultdict(list)
for ts, pn in zip(timestamps_mppi, pn_filtered_mppi):
    if ts is not None and pn is not None:
        date_key = ts.date()
        daily_pn_mppi[date_key].append(pn)

# PPFD数据按天分组
daily_pn_ppfd = defaultdict(list)
for ts, pn in zip(timestamps_ppfd, pn_filtered_ppfd):
    if ts is not None and pn is not None:
        date_key = ts.date()
        daily_pn_ppfd[date_key].append(pn)

# 打印统计信息
valid_pn_mppi = [p for p in pn_filtered_mppi if p is not None]
valid_pn_ppfd = [p for p in pn_filtered_ppfd if p is not None]

print(f"\n=== MPPI控制数据统计 ===")
print(f"总数据点数: {len(all_timestamps_mppi)}")
print(f"有效数据点数 (9am-1am): {len(valid_pn_mppi)}")
print(f"过滤掉的数据点数 (1am-9am): {len(all_timestamps_mppi) - len(valid_pn_mppi)}")
if valid_pn_mppi:
    print(f"PN平均值 (有效时段): {sum(valid_pn_mppi)/len(valid_pn_mppi):.2f}")
    print(f"PN最大值: {max(valid_pn_mppi):.2f}")
    print(f"PN最小值: {min(valid_pn_mppi):.2f}")
if all_timestamps_mppi:
    print(f"数据时间范围: {min(all_timestamps_mppi)} 到 {max(all_timestamps_mppi)}")

print(f"\n=== PPFD计算数据统计 ===")
print(f"总数据点数: {len(all_timestamps_ppfd)}")
print(f"有效数据点数 (9am-1am): {len(valid_pn_ppfd)}")
print(f"过滤掉的数据点数 (1am-9am): {len(all_timestamps_ppfd) - len(valid_pn_ppfd)}")
if valid_pn_ppfd:
    print(f"PN平均值 (有效时段): {sum(valid_pn_ppfd)/len(valid_pn_ppfd):.2f}")
    print(f"PN最大值: {max(valid_pn_ppfd):.2f}")
    print(f"PN最小值: {min(valid_pn_ppfd):.2f}")
if all_timestamps_ppfd:
    print(f"数据时间范围: {min(all_timestamps_ppfd)} 到 {max(all_timestamps_ppfd)}")

# 打印每天的平均值
print(f"\n{'='*60}")
print(f"每天PN平均值对比 (9am-1am时段)")
print(f"{'='*60}")
print(f"{'日期':<15} {'MPPI控制':<20} {'PPFD计算':<20}")
print(f"{'-'*60}")

# 获取所有日期并排序
all_dates = sorted(set(list(daily_pn_mppi.keys()) + list(daily_pn_ppfd.keys())))

for date in all_dates:
    mppi_avg = ""
    ppfd_avg = ""
    
    if date in daily_pn_mppi:
        mppi_values = daily_pn_mppi[date]
        mppi_avg = f"{sum(mppi_values)/len(mppi_values):.2f} ({len(mppi_values)}点)"
    else:
        mppi_avg = "无数据"
    
    if date in daily_pn_ppfd:
        ppfd_values = daily_pn_ppfd[date]
        ppfd_avg = f"{sum(ppfd_values)/len(ppfd_values):.2f} ({len(ppfd_values)}点)"
    else:
        ppfd_avg = "无数据"
    
    print(f"{str(date):<15} {mppi_avg:<20} {ppfd_avg:<20}")

print(f"{'='*60}")

