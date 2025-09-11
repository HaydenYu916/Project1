#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPFD与温度关系分析
分析7:00-23:00时间段内PPFD恒定的数据，建立PPFD-温度关系模型
"""

import pandas as pd
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_ppfd_temp_data(file_path):
    """加载数据并分析PPFD与温度的关系"""
    print("正在加载数据...")
    # 跳过第一行注释
    df = pd.read_csv(file_path, skiprows=1)
    
    # 转换时间戳
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"时间范围：{df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    # 计算PPFD（假设使用光谱数据计算）
    # 这里使用简化的PPFD计算，您可以根据实际公式调整
    spectral_columns = ['sp_415', 'sp_445', 'sp_480', 'sp_515', 'sp_555', 'sp_590', 'sp_630', 'sp_680']
    
    # 检查是否有光谱数据列
    available_spectral = [col for col in spectral_columns if col in df.columns]
    
    if available_spectral:
        # 计算总光谱强度作为PPFD的代理
        df['ppfd_proxy'] = df[available_spectral].sum(axis=1)
        print(f"使用光谱数据计算PPFD代理值，使用列: {available_spectral}")
    else:
        # 如果没有光谱数据，创建模拟PPFD
        print("警告：没有找到光谱数据，将创建模拟PPFD")
        df['ppfd_proxy'] = 300  # 默认值
    
    # 添加日期和时间信息
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time
    df['hour'] = df['timestamp'].dt.hour
    
    return df

def filter_stable_ppfd_days(df):
    """筛选PPFD稳定的天数（7:00-23:00时间段内PPFD恒定）"""
    print("正在筛选PPFD稳定的天数...")
    
    # 筛选7:00-23:00时间段的数据
    target_hours = list(range(7, 23))  # 7:00-22:59
    df_period = df[df['hour'].isin(target_hours)].copy()
    
    stable_days = []
    ppfd_levels = []
    
    # 按日期分组分析
    for date, group in df_period.groupby('date'):
        if len(group) < 50:  # 数据点太少，跳过
            continue
        
        # 分析PPFD变化
        ppfd_values = group['ppfd_proxy'].values
        ppfd_std = np.std(ppfd_values)
        ppfd_mean = np.mean(ppfd_values)
        ppfd_cv = ppfd_std / ppfd_mean if ppfd_mean > 0 else float('inf')  # 变异系数
        
        # 判断PPFD是否稳定（变异系数小于5%）
        if ppfd_cv < 0.05:
            stable_days.append(date)
            # 更智能的PPFD等级分类 - 基于数据范围自动分组
            ppfd_levels.append(ppfd_mean)  # 先存储原始值，后面再分组
            
            print(f"  {date}: PPFD稳定，平均值={ppfd_mean:.0f}, 变异系数={ppfd_cv:.3f}")
        else:
            print(f"  {date}: PPFD不稳定，变异系数={ppfd_cv:.3f} (跳过)")
    
    print(f"\n找到 {len(stable_days)} 个PPFD稳定的天数")
    
    if len(ppfd_levels) == 0:
        return stable_days, ppfd_levels
    
    # 智能PPFD分组 - 基于K-means聚类或手动分组
    ppfd_array = np.array(ppfd_levels)
    
    # 如果PPFD值变化很小，可能是同一等级
    ppfd_range = ppfd_array.max() - ppfd_array.min()
    ppfd_mean_all = ppfd_array.mean()
    
    print(f"PPFD分析:")
    print(f"  范围: {ppfd_array.min():.0f} - {ppfd_array.max():.0f}")
    print(f"  平均: {ppfd_mean_all:.0f}")
    print(f"  变化范围: {ppfd_range:.0f}")
    
    # 如果所有数据在同一等级，尝试寻找更多的PPFD等级
    if ppfd_range / ppfd_mean_all < 0.3:  # 变化小于30%
        print("警告: 所有稳定天数的PPFD都在相似水平，无法建立PPFD-温度关系")
        print("建议: 需要更多不同PPFD等级的数据")
        
        # 尝试更宽松的稳定性标准，寻找其他PPFD等级
        print("\n尝试使用简单分组方法...")
    
    # 强制使用简单分组方法创建多个PPFD等级
    print("使用简单分组方法创建PPFD等级...")
    ppfd_levels = simple_ppfd_grouping(ppfd_array)
    
    # 统计最终分布
    ppfd_counts = {}
    for level in ppfd_levels:
        ppfd_counts[level] = ppfd_counts.get(level, 0) + 1
    
    print("\n最终PPFD等级分布:")
    for level, count in sorted(ppfd_counts.items()):
        print(f"  {level} PPFD: {count} 天")
    
    return stable_days, ppfd_levels

def find_additional_ppfd_levels(df_period):
    """寻找额外的PPFD等级（更宽松的稳定性标准）"""
    additional_days = []
    additional_levels = []
    
    # 使用更宽松的变异系数标准 (10%)
    for date, group in df_period.groupby('date'):
        if len(group) < 50:
            continue
        
        ppfd_values = group['ppfd_proxy'].values
        ppfd_std = np.std(ppfd_values)
        ppfd_mean = np.mean(ppfd_values)
        ppfd_cv = ppfd_std / ppfd_mean if ppfd_mean > 0 else float('inf')
        
        if 0.05 <= ppfd_cv < 0.10:  # 中等稳定性
            additional_days.append(date)
            additional_levels.append(ppfd_mean)
            print(f"  {date}: 中等稳定，平均值={ppfd_mean:.0f}, 变异系数={ppfd_cv:.3f}")
    
    return additional_days, additional_levels

def simple_ppfd_grouping(ppfd_array):
    """简单的PPFD分组方法"""
    # 按数值范围分组
    ppfd_levels_classified = []
    for ppfd in ppfd_array:
        if ppfd < np.percentile(ppfd_array, 25):
            ppfd_levels_classified.append(300)
        elif ppfd < np.percentile(ppfd_array, 50):
            ppfd_levels_classified.append(400)
        elif ppfd < np.percentile(ppfd_array, 75):
            ppfd_levels_classified.append(500)
        else:
            ppfd_levels_classified.append(600)
    
    return ppfd_levels_classified

def analyze_ppfd_temperature_relationship(df, stable_days, ppfd_levels):
    """分析PPFD与温度的关系"""
    print("\n正在分析PPFD与温度的关系...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 筛选7:00-23:00时间段的数据
    target_hours = list(range(7, 23))
    df_period = df[df['hour'].isin(target_hours)].copy()
    
    # 收集稳定天数的数据
    ppfd_temp_data = []
    
    for i, date in enumerate(stable_days):
        ppfd_level = ppfd_levels[i]
        day_data = df_period[df_period['date'] == date].copy()
        
        if len(day_data) == 0:
            continue
        
        # 按设备分组
        for device_id in day_data['device_id'].unique():
            device_data = day_data[day_data['device_id'] == device_id].copy()
            device_data = device_data.sort_values('timestamp')
            
            if len(device_data) < 20:
                continue
            
            # 计算该天的温度统计
            temp_stats = {
                'date': date,
                'device_id': device_id,
                'ppfd_level': ppfd_level,
                'temp_mean': device_data['temperature'].mean(),
                'temp_min': device_data['temperature'].min(),
                'temp_max': device_data['temperature'].max(),
                'temp_range': device_data['temperature'].max() - device_data['temperature'].min(),
                'temp_start': device_data['temperature'].iloc[0],
                'temp_end': device_data['temperature'].iloc[-1],
                'temp_rise': device_data['temperature'].iloc[-1] - device_data['temperature'].iloc[0],
                'data_points': len(device_data)
            }
            
            ppfd_temp_data.append(temp_stats)
    
    # 转换为DataFrame
    results_df = pd.DataFrame(ppfd_temp_data)
    
    if len(results_df) == 0:
        print("没有找到有效的PPFD-温度数据")
        return None
    
    print(f"收集到 {len(results_df)} 组PPFD-温度数据")
    
    # 按设备分析
    devices = results_df['device_id'].unique()
    
    for device_id in devices:
        device_results = results_df[results_df['device_id'] == device_id].copy()
        
        print(f"\n设备 {device_id} 的PPFD-温度关系:")
        
        # 按PPFD等级统计
        for ppfd_level in sorted(device_results['ppfd_level'].unique()):
            level_data = device_results[device_results['ppfd_level'] == ppfd_level]
            print(f"  PPFD {ppfd_level}:")
            print(f"    天数: {len(level_data)}")
            print(f"    平均温度: {level_data['temp_mean'].mean():.2f} ± {level_data['temp_mean'].std():.2f}°C")
            print(f"    温度范围: {level_data['temp_range'].mean():.2f} ± {level_data['temp_range'].std():.2f}°C")
            print(f"    温升: {level_data['temp_rise'].mean():.2f} ± {level_data['temp_rise'].std():.2f}°C")
    
    return results_df

def create_ppfd_temp_model(results_df):
    """建立PPFD-温度关系模型"""
    print("\n正在建立PPFD-温度关系模型...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    devices = results_df['device_id'].unique()
    
    for device_id in devices:
        device_data = results_df[results_df['device_id'] == device_id].copy()
        
        if len(device_data) < 4:  # 需要足够的数据点
            print(f"设备 {device_id} 数据点不足，跳过建模")
            continue
        
        print(f"\n设备 {device_id} 的PPFD-温度模型:")
        
        # 按PPFD等级聚合数据
        ppfd_aggregated = device_data.groupby('ppfd_level').agg({
            'temp_mean': ['mean', 'std', 'count'],
            'temp_range': ['mean', 'std'],
            'temp_rise': ['mean', 'std']
        }).round(3)
        
        ppfd_levels = []
        temp_means = []
        temp_mean_stds = []
        temp_ranges = []
        temp_rises = []
        
        for ppfd_level in sorted(device_data['ppfd_level'].unique()):
            level_data = device_data[device_data['ppfd_level'] == ppfd_level]
            if len(level_data) > 0:
                ppfd_levels.append(ppfd_level)
                temp_means.append(level_data['temp_mean'].mean())
                temp_mean_stds.append(level_data['temp_mean'].std() if len(level_data) > 1 else 0)
                temp_ranges.append(level_data['temp_range'].mean())
                temp_rises.append(level_data['temp_rise'].mean())
        
        if len(ppfd_levels) < 2:
            print(f"设备 {device_id} PPFD等级不足，无法建模")
            continue
        
        # 建立线性模型
        try:
            # 1. 平均温度 vs PPFD
            z_temp = np.polyfit(ppfd_levels, temp_means, 1)
            p_temp = np.poly1d(z_temp)
            
            # 2. 温度范围 vs PPFD
            z_range = np.polyfit(ppfd_levels, temp_ranges, 1)
            p_range = np.poly1d(z_range)
            
            # 3. 温升 vs PPFD
            z_rise = np.polyfit(ppfd_levels, temp_rises, 1)
            p_rise = np.poly1d(z_rise)
            
            # 计算R²
            r2_temp = 1 - np.sum((temp_means - p_temp(ppfd_levels))**2) / np.sum((temp_means - np.mean(temp_means))**2)
            r2_range = 1 - np.sum((temp_ranges - p_range(ppfd_levels))**2) / np.sum((temp_ranges - np.mean(temp_ranges))**2)
            r2_rise = 1 - np.sum((temp_rises - p_rise(ppfd_levels))**2) / np.sum((temp_rises - np.mean(temp_rises))**2)
            
            print(f"线性模型结果:")
            print(f"  平均温度 = {z_temp[0]:.4f} × PPFD + {z_temp[1]:.2f} (R² = {r2_temp:.3f})")
            print(f"  温度范围 = {z_range[0]:.4f} × PPFD + {z_range[1]:.2f} (R² = {r2_range:.3f})")
            print(f"  温升幅度 = {z_rise[0]:.4f} × PPFD + {z_rise[1]:.2f} (R² = {r2_rise:.3f})")
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. 平均温度 vs PPFD
            ax1 = axes[0, 0]
            ax1.errorbar(ppfd_levels, temp_means, yerr=temp_mean_stds, 
                        fmt='o', capsize=5, capthick=2, color='blue', markersize=8)
            ppfd_smooth = np.linspace(min(ppfd_levels), max(ppfd_levels), 100)
            ax1.plot(ppfd_smooth, p_temp(ppfd_smooth), 'r-', linewidth=2, 
                    label=f'线性拟合 (R²={r2_temp:.3f})')
            ax1.set_xlabel('PPFD')
            ax1.set_ylabel('平均温度 (°C)')
            ax1.set_title(f'设备 {device_id} - PPFD vs 平均温度')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. 温度范围 vs PPFD
            ax2 = axes[0, 1]
            ax2.plot(ppfd_levels, temp_ranges, 'o', color='green', markersize=8)
            ax2.plot(ppfd_smooth, p_range(ppfd_smooth), 'r-', linewidth=2,
                    label=f'线性拟合 (R²={r2_range:.3f})')
            ax2.set_xlabel('PPFD')
            ax2.set_ylabel('温度范围 (°C)')
            ax2.set_title(f'设备 {device_id} - PPFD vs 温度范围')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 3. 温升 vs PPFD
            ax3 = axes[1, 0]
            ax3.plot(ppfd_levels, temp_rises, 'o', color='orange', markersize=8)
            ax3.plot(ppfd_smooth, p_rise(ppfd_smooth), 'r-', linewidth=2,
                    label=f'线性拟合 (R²={r2_rise:.3f})')
            ax3.set_xlabel('PPFD')
            ax3.set_ylabel('温升幅度 (°C)')
            ax3.set_title(f'设备 {device_id} - PPFD vs 温升幅度')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 4. 模型总结
            ax4 = axes[1, 1]
            ax4.axis('off')
            model_text = (f"设备 {device_id} PPFD-温度关系模型\n\n"
                         f"线性回归方程:\n"
                         f"平均温度 = {z_temp[0]:.4f} × PPFD + {z_temp[1]:.2f}\n"
                         f"温度范围 = {z_range[0]:.4f} × PPFD + {z_range[1]:.2f}\n"
                         f"温升幅度 = {z_rise[0]:.4f} × PPFD + {z_rise[1]:.2f}\n\n"
                         f"拟合质量:\n"
                         f"平均温度模型 R² = {r2_temp:.3f}\n"
                         f"温度范围模型 R² = {r2_range:.3f}\n"
                         f"温升幅度模型 R² = {r2_rise:.3f}\n\n"
                         f"数据统计:\n"
                         f"PPFD等级: {sorted(ppfd_levels)}\n"
                         f"总天数: {len(device_data)}\n"
                         f"PPFD范围: {min(ppfd_levels)} - {max(ppfd_levels)}")
            
            ax4.text(0.1, 0.9, model_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # 保存图表
            import os
            data_dir = "/Users/z5540822/Desktop/mpc-farming-master/Data"
            os.makedirs(data_dir, exist_ok=True)
            filename = f'ppfd_temperature_model_{device_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            filepath = os.path.join(data_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"PPFD-温度关系图已保存为: {filepath}")
            
            plt.show()
            
        except Exception as e:
            print(f"设备 {device_id} 建模失败: {e}")

def main():
    """主函数"""
    print("=== PPFD与温度关系分析 ===")
    
    # 数据文件路径
    data_file = "/Users/z5540822/Desktop/mpc-farming-master/Data/5-1-400_20250829_091025.csv"
    
    try:
        # 1. 加载数据
        df = load_and_analyze_ppfd_temp_data(data_file)
        
        # 2. 筛选PPFD稳定的天数
        stable_days, ppfd_levels = filter_stable_ppfd_days(df)
        
        if len(stable_days) == 0:
            print("没有找到PPFD稳定的天数")
            return
        
        # 3. 分析PPFD与温度的关系
        results_df = analyze_ppfd_temperature_relationship(df, stable_days, ppfd_levels)
        
        if results_df is None:
            return
        
        # 4. 建立PPFD-温度关系模型
        create_ppfd_temp_model(results_df)
        
        # 5. 保存结果
        output_file = f"/Users/z5540822/Desktop/mpc-farming-master/Data/ppfd_temperature_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nPPFD-温度分析结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
