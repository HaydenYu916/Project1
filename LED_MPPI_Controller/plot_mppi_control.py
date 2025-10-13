#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制MPPI控制日志的时间序列图
横轴：时间
纵轴：r_pwm + b_pwm的总和
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_mppi_control_data():
    """绘制MPPI控制数据的时间序列图"""
    
    # 读取数据
    file_path = '/home/pi/Desktop/LED_MPPI_Controller/logs/mppi_v2_control_log.csv'
    df = pd.read_csv(file_path)
    
    # 转换时间戳
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 计算r_pwm + b_pwm的总和
    df['total_pwm'] = df['r_pwm'] + df['b_pwm']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 绘制时间序列图
    ax.plot(df['timestamp'], df['total_pwm'], 'b-', linewidth=1.5, label='r_pwm + b_pwm 总和')
    
    # 添加数据点
    ax.scatter(df['timestamp'], df['total_pwm'], c='red', s=20, alpha=0.6, zorder=5)
    
    # 设置标签和标题
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('r_pwm + b_pwm 总和', fontsize=12)
    ax.set_title('MPPI控制日志 - PWM总和时间序列图', fontsize=14, fontweight='bold')
    
    # 设置时间轴格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    ax.legend()
    
    # 添加统计信息文本
    total_mean = df['total_pwm'].mean()
    total_std = df['total_pwm'].std()
    total_max = df['total_pwm'].max()
    total_min = df['total_pwm'].min()
    
    stats_text = f'统计信息:\n均值: {total_mean:.2f}\n标准差: {total_std:.2f}\n最大值: {total_max:.2f}\n最小值: {total_min:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = '/home/pi/Desktop/LED_MPPI_Controller/mppi_pwm_sum_timeseries.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()
    
    # 打印数据概览
    print(f"\n数据概览:")
    print(f"数据点数量: {len(df)}")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"PWM总和统计:")
    print(f"  均值: {total_mean:.2f}")
    print(f"  标准差: {total_std:.2f}")
    print(f"  最大值: {total_max:.2f}")
    print(f"  最小值: {total_min:.2f}")
    print(f"  中位数: {df['total_pwm'].median():.2f}")

if __name__ == "__main__":
    plot_mppi_control_data()

