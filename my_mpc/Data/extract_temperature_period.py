#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取特定时间段的温度数据并进行滤波处理
时间段：9月6日11:00到9月8日7:00
"""

import pandas as pd
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_filter_temperature_data(file_path, start_time, end_time):
    """加载并滤波指定时间段的温度数据"""
    print("正在加载数据...")
    # 跳过第一行注释，从第二行开始读取
    df = pd.read_csv(file_path, skiprows=1)
    
    # 转换时间戳
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 按时间排序
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"原始数据加载完成，共 {len(df)} 条记录")
    print(f"时间范围：{df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    # 提取指定时间段的数据
    print(f"正在提取时间段：{start_time} 到 {end_time}")
    mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
    period_df = df[mask].copy().reset_index(drop=True)
    
    if len(period_df) == 0:
        print("指定时间段内没有数据！")
        return None
    
    print(f"提取到 {len(period_df)} 条记录")
    
    # 按30秒分组
    print("正在按30秒分组...")
    period_df['time_group'] = period_df['timestamp'].dt.floor('30S')
    grouped = period_df.groupby(['device_id', 'time_group'])
    print(f"共分为 {len(grouped)} 个30秒组")
    
    # 应用滤波
    print("正在应用滤波...")
    filtered_data = []
    
    for (dev_id, time_group), group in grouped:
        if len(group) < 2:  # 如果组内数据点太少，跳过
            continue
            
        # 对温度数据进行多种滤波
        temp_data = group['temperature'].values
        
        # 1. 移动平均滤波
        window_size = min(5, len(temp_data))
        if window_size > 1:
            ma_filtered = uniform_filter1d(temp_data, size=window_size, mode='nearest')
        else:
            ma_filtered = temp_data
        
        # 2. 中值滤波
        kernel_size = min(3, len(temp_data))
        if kernel_size % 2 == 0:
            kernel_size = max(1, kernel_size - 1)
        median_filtered = signal.medfilt(temp_data, kernel_size=kernel_size)
        
        # 3. 高斯滤波
        if len(temp_data) > 3:
            sigma = min(1.0, len(temp_data) / 10)
            gaussian_filtered = gaussian_filter1d(temp_data, sigma=sigma)
        else:
            gaussian_filtered = temp_data
        
        # 4. 低通滤波（去除高频噪声）
        if len(temp_data) > 10:  # 需要足够的数据点进行滤波
            try:
                # 设计低通滤波器
                nyquist = 0.5  # 假设采样频率为1Hz
                cutoff = 0.1   # 截止频率
                b, a = signal.butter(2, cutoff / nyquist, btype='low')
                lowpass_filtered = signal.filtfilt(b, a, temp_data)
            except:
                # 如果滤波失败，使用原始数据
                lowpass_filtered = temp_data
        else:
            lowpass_filtered = temp_data
        
        # 创建滤波后的数据框
        group_filtered = group.copy()
        group_filtered['temp_ma'] = ma_filtered
        group_filtered['temp_median'] = median_filtered
        group_filtered['temp_gaussian'] = gaussian_filtered
        group_filtered['temp_lowpass'] = lowpass_filtered
        
        # 计算30秒内的统计信息
        group_filtered['temp_mean_30s'] = temp_data.mean()
        group_filtered['temp_std_30s'] = temp_data.std()
        group_filtered['temp_min_30s'] = temp_data.min()
        group_filtered['temp_max_30s'] = temp_data.max()
        
        filtered_data.append(group_filtered)
    
    if filtered_data:
        result_df = pd.concat(filtered_data, ignore_index=True)
        result_df = result_df.sort_values('timestamp').reset_index(drop=True)
        return result_df
    else:
        return pd.DataFrame()

def plot_temperature_period(df, start_time, end_time, save_plots=True):
    """绘制指定时间段的温度对比图"""
    print("正在生成温度对比图...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if len(df) == 0:
        print("没有数据可绘制")
        return
    
    # 格式化时间显示
    start_str = start_time.strftime("%m月%d日%H:%M")
    end_str = end_time.strftime("%m月%d日%H:%M")
    time_range_str = f"{start_str}-{end_str}"
    
    # 按设备分组绘制
    devices = df['device_id'].unique()
    
    for dev_id in devices:
        dev_data = df[df['device_id'] == dev_id].copy()
        dev_data = dev_data.sort_values('timestamp')
        
        # 检查时间间隔，如果间隔过大则不连线
        time_diffs = dev_data['timestamp'].diff()
        large_gaps = time_diffs > pd.Timedelta(minutes=5)  # 5分钟以上的间隔
        
        plt.figure(figsize=(16, 12))
        
        # 分段绘制，避免大间隔连线
        segments = []
        current_segment = []
        for i, (idx, row) in enumerate(dev_data.iterrows()):
            if i > 0 and large_gaps.iloc[i]:
                if current_segment:
                    segments.append(pd.DataFrame(current_segment))
                current_segment = [row]
            else:
                current_segment.append(row)
        if current_segment:
            segments.append(pd.DataFrame(current_segment))
        
        # 移动平均滤波
        plt.subplot(2, 2, 1)
        for segment in segments:
            plt.plot(segment['timestamp'], segment['temperature'], 'b-', alpha=0.7, label='原始数据' if segment is segments[0] else "")
            plt.plot(segment['timestamp'], segment['temp_ma'], 'r-', linewidth=2, label='移动平均' if segment is segments[0] else "")
        
        plt.title(f'设备 {dev_id} - 移动平均滤波 ({time_range_str})')
        plt.xlabel('时间')
        plt.ylabel('温度 (°C)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 中值滤波
        plt.subplot(2, 2, 2)
        for segment in segments:
            plt.plot(segment['timestamp'], segment['temperature'], 'b-', alpha=0.7, label='原始数据' if segment is segments[0] else "")
            plt.plot(segment['timestamp'], segment['temp_median'], 'g-', linewidth=2, label='中值滤波' if segment is segments[0] else "")
        
        plt.title(f'设备 {dev_id} - 中值滤波 ({time_range_str})')
        plt.xlabel('时间')
        plt.ylabel('温度 (°C)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 高斯滤波
        plt.subplot(2, 2, 3)
        for segment in segments:
            plt.plot(segment['timestamp'], segment['temperature'], 'b-', alpha=0.7, label='原始数据' if segment is segments[0] else "")
            plt.plot(segment['timestamp'], segment['temp_gaussian'], 'm-', linewidth=2, label='高斯滤波' if segment is segments[0] else "")
        
        plt.title(f'设备 {dev_id} - 高斯滤波 ({time_range_str})')
        plt.xlabel('时间')
        plt.ylabel('温度 (°C)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 低通滤波
        plt.subplot(2, 2, 4)
        for segment in segments:
            plt.plot(segment['timestamp'], segment['temperature'], 'b-', alpha=0.7, label='原始数据' if segment is segments[0] else "")
            plt.plot(segment['timestamp'], segment['temp_lowpass'], 'c-', linewidth=2, label='低通滤波' if segment is segments[0] else "")
        
        plt.title(f'设备 {dev_id} - 低通滤波 ({time_range_str})')
        plt.xlabel('时间')
        plt.ylabel('温度 (°C)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            # 保存到Data文件夹
            import os
            data_dir = "/Users/z5540822/Desktop/mpc-farming-master/Data"
            os.makedirs(data_dir, exist_ok=True)
            # 生成动态文件名
            start_file_str = start_time.strftime("%m-%d_%H-%M")
            end_file_str = end_time.strftime("%m-%d_%H-%M")
            filename = f'temperature_period_{start_file_str}_to_{end_file_str}_{dev_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            filepath = os.path.join(data_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"图表已保存为: {filepath}")
        
        plt.show()

def fit_temperature_curves(df, start_time, end_time, save_plots=True):
    """对温度数据进行曲线拟合"""
    print("正在进行温度曲线拟合...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if len(df) == 0:
        print("没有数据可拟合")
        return
    
    # 定义拟合函数
    def polynomial_func(x, a, b, c, d):
        """多项式函数：ax³ + bx² + cx + d"""
        return a * x**3 + b * x**2 + c * x + d
    
    def sine_func(x, a, b, c, d):
        """正弦函数：a * sin(b * x + c) + d"""
        return a * np.sin(b * x + c) + d
    
    def exponential_func(x, a, b, c):
        """指数函数：a * exp(b * x) + c"""
        return a * np.exp(b * x) + c
    
    def gaussian_func(x, a, b, c, d):
        """高斯函数：a * exp(-((x - b) / c)²) + d"""
        return a * np.exp(-((x - b) / c)**2) + d
    
    # 格式化时间显示
    start_str = start_time.strftime("%m月%d日%H:%M")
    end_str = end_time.strftime("%m月%d日%H:%M")
    time_range_str = f"{start_str}-{end_str}"
    
    # 按设备分组拟合
    devices = df['device_id'].unique()
    
    for dev_id in devices:
        dev_data = df[df['device_id'] == dev_id].copy()
        dev_data = dev_data.sort_values('timestamp')
        
        if len(dev_data) < 10:
            print(f"设备 {dev_id} 数据点太少，跳过拟合")
            continue
        
        # 将时间转换为数值（小时）
        time_start = dev_data['timestamp'].min()
        dev_data['time_hours'] = (dev_data['timestamp'] - time_start).dt.total_seconds() / 3600
        
        # 准备拟合数据
        x_data = dev_data['time_hours'].values
        y_data = dev_data['temperature'].values
        
        # 创建拟合结果存储
        fit_results = {}
        
        plt.figure(figsize=(20, 15))
        
        # 1. 多项式拟合
        plt.subplot(3, 3, 1)
        try:
            popt_poly, pcov_poly = curve_fit(polynomial_func, x_data, y_data, maxfev=5000)
            y_fit_poly = polynomial_func(x_data, *popt_poly)
            r2_poly = 1 - np.sum((y_data - y_fit_poly)**2) / np.sum((y_data - np.mean(y_data))**2)
            
            plt.plot(dev_data['timestamp'], y_data, 'b-', alpha=0.7, label='原始数据')
            plt.plot(dev_data['timestamp'], y_fit_poly, 'r-', linewidth=2, label=f'多项式拟合 (R²={r2_poly:.3f})')
            plt.title(f'设备 {dev_id} - 多项式拟合')
            plt.xlabel('时间')
            plt.ylabel('温度 (°C)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            fit_results['polynomial'] = {
                'params': popt_poly,
                'r2': r2_poly,
                'func': polynomial_func
            }
        except Exception as e:
            plt.text(0.5, 0.5, f'多项式拟合失败\n{str(e)}', transform=plt.gca().transAxes, ha='center')
            plt.title(f'设备 {dev_id} - 多项式拟合')
        
        # 2. 正弦拟合
        plt.subplot(3, 3, 2)
        try:
            # 初始参数估计
            amplitude = (y_data.max() - y_data.min()) / 2
            period = (x_data.max() - x_data.min()) / 2
            phase = 0
            offset = np.mean(y_data)
            p0 = [amplitude, 2*np.pi/period, phase, offset]
            
            popt_sine, pcov_sine = curve_fit(sine_func, x_data, y_data, p0=p0, maxfev=5000)
            y_fit_sine = sine_func(x_data, *popt_sine)
            r2_sine = 1 - np.sum((y_data - y_fit_sine)**2) / np.sum((y_data - np.mean(y_data))**2)
            
            plt.plot(dev_data['timestamp'], y_data, 'b-', alpha=0.7, label='原始数据')
            plt.plot(dev_data['timestamp'], y_fit_sine, 'g-', linewidth=2, label=f'正弦拟合 (R²={r2_sine:.3f})')
            plt.title(f'设备 {dev_id} - 正弦拟合')
            plt.xlabel('时间')
            plt.ylabel('温度 (°C)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            fit_results['sine'] = {
                'params': popt_sine,
                'r2': r2_sine,
                'func': sine_func
            }
        except Exception as e:
            plt.text(0.5, 0.5, f'正弦拟合失败\n{str(e)}', transform=plt.gca().transAxes, ha='center')
            plt.title(f'设备 {dev_id} - 正弦拟合')
        
        # 3. 指数拟合
        plt.subplot(3, 3, 3)
        try:
            # 确保数据为正数
            y_shifted = y_data - y_data.min() + 1
            p0 = [1, 0.1, y_data.min()]
            popt_exp, pcov_exp = curve_fit(exponential_func, x_data, y_shifted, p0=p0, maxfev=5000)
            y_fit_exp = exponential_func(x_data, *popt_exp)
            r2_exp = 1 - np.sum((y_data - y_fit_exp)**2) / np.sum((y_data - np.mean(y_data))**2)
            
            plt.plot(dev_data['timestamp'], y_data, 'b-', alpha=0.7, label='原始数据')
            plt.plot(dev_data['timestamp'], y_fit_exp, 'm-', linewidth=2, label=f'指数拟合 (R²={r2_exp:.3f})')
            plt.title(f'设备 {dev_id} - 指数拟合')
            plt.xlabel('时间')
            plt.ylabel('温度 (°C)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            fit_results['exponential'] = {
                'params': popt_exp,
                'r2': r2_exp,
                'func': exponential_func
            }
        except Exception as e:
            plt.text(0.5, 0.5, f'指数拟合失败\n{str(e)}', transform=plt.gca().transAxes, ha='center')
            plt.title(f'设备 {dev_id} - 指数拟合')
        
        # 4. 高斯拟合
        plt.subplot(3, 3, 4)
        try:
            # 初始参数估计
            amplitude = y_data.max() - y_data.min()
            center = x_data[np.argmax(y_data)]
            width = (x_data.max() - x_data.min()) / 4
            offset = y_data.min()
            p0 = [amplitude, center, width, offset]
            
            popt_gauss, pcov_gauss = curve_fit(gaussian_func, x_data, y_data, p0=p0, maxfev=5000)
            y_fit_gauss = gaussian_func(x_data, *popt_gauss)
            r2_gauss = 1 - np.sum((y_data - y_fit_gauss)**2) / np.sum((y_data - np.mean(y_data))**2)
            
            plt.plot(dev_data['timestamp'], y_data, 'b-', alpha=0.7, label='原始数据')
            plt.plot(dev_data['timestamp'], y_fit_gauss, 'c-', linewidth=2, label=f'高斯拟合 (R²={r2_gauss:.3f})')
            plt.title(f'设备 {dev_id} - 高斯拟合')
            plt.xlabel('时间')
            plt.ylabel('温度 (°C)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            fit_results['gaussian'] = {
                'params': popt_gauss,
                'r2': r2_gauss,
                'func': gaussian_func
            }
        except Exception as e:
            plt.text(0.5, 0.5, f'高斯拟合失败\n{str(e)}', transform=plt.gca().transAxes, ha='center')
            plt.title(f'设备 {dev_id} - 高斯拟合')
        
        # 5. 移动平均对比
        plt.subplot(3, 3, 5)
        plt.plot(dev_data['timestamp'], y_data, 'b-', alpha=0.7, label='原始数据')
        plt.plot(dev_data['timestamp'], dev_data['temp_ma'], 'r-', linewidth=2, label='移动平均')
        plt.title(f'设备 {dev_id} - 移动平均滤波')
        plt.xlabel('时间')
        plt.ylabel('温度 (°C)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 6. 最佳拟合对比
        plt.subplot(3, 3, 6)
        if fit_results:
            best_fit = max(fit_results.items(), key=lambda x: x[1]['r2'])
            fit_name, fit_data = best_fit
            y_fit_best = fit_data['func'](x_data, *fit_data['params'])
            
            plt.plot(dev_data['timestamp'], y_data, 'b-', alpha=0.7, label='原始数据')
            plt.plot(dev_data['timestamp'], y_fit_best, 'orange', linewidth=3, 
                    label=f'最佳拟合 ({fit_name}, R²={fit_data["r2"]:.3f})')
            plt.title(f'设备 {dev_id} - 最佳拟合')
            plt.xlabel('时间')
            plt.ylabel('温度 (°C)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 7. 残差分析
        plt.subplot(3, 3, 7)
        if fit_results:
            best_fit = max(fit_results.items(), key=lambda x: x[1]['r2'])
            fit_name, fit_data = best_fit
            y_fit_best = fit_data['func'](x_data, *fit_data['params'])
            residuals = y_data - y_fit_best
            
            plt.plot(dev_data['timestamp'], residuals, 'purple', alpha=0.7)
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            plt.title(f'设备 {dev_id} - 残差分析')
            plt.xlabel('时间')
            plt.ylabel('残差 (°C)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 8. 拟合质量评估
        plt.subplot(3, 3, 8)
        if fit_results:
            fit_names = list(fit_results.keys())
            r2_values = [fit_results[name]['r2'] for name in fit_names]
            
            bars = plt.bar(fit_names, r2_values, color=['red', 'green', 'magenta', 'cyan'])
            plt.title(f'设备 {dev_id} - 拟合质量 (R²)')
            plt.ylabel('R² 值')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, r2_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 9. 温度趋势分析
        plt.subplot(3, 3, 9)
        temp_diff = np.diff(y_data)
        plt.plot(dev_data['timestamp'][1:], temp_diff, 'orange', alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.title(f'设备 {dev_id} - 温度变化率')
        plt.xlabel('时间')
        plt.ylabel('温度变化 (°C/点)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            # 保存到Data文件夹
            import os
            data_dir = "/Users/z5540822/Desktop/mpc-farming-master/Data"
            os.makedirs(data_dir, exist_ok=True)
            # 生成动态文件名
            start_file_str = start_time.strftime("%m-%d_%H-%M")
            end_file_str = end_time.strftime("%m-%d_%H-%M")
            filename = f'temperature_curve_fitting_{start_file_str}_to_{end_file_str}_{dev_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            filepath = os.path.join(data_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"曲线拟合图已保存为: {filepath}")
        
        plt.show()
        
        # 打印拟合结果
        print(f"\n设备 {dev_id} 拟合结果:")
        for fit_name, fit_data in fit_results.items():
            print(f"  {fit_name}: R² = {fit_data['r2']:.4f}")
            print(f"    参数: {fit_data['params']}")

def create_continuous_curve(df, start_time, end_time, save_plots=True):
    """创建从开始到结束的连续拟合曲线"""
    print("正在创建连续拟合曲线...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if len(df) == 0:
        print("没有数据可拟合")
        return
    
    # 定义拟合函数
    def polynomial_func(x, a, b, c, d):
        """多项式函数：ax³ + bx² + cx + d"""
        return a * x**3 + b * x**2 + c * x + d
    
    def sine_func(x, a, b, c, d):
        """正弦函数：a * sin(b * x + c) + d"""
        return a * np.sin(b * x + c) + d
    
    def exponential_func(x, a, b, c):
        """指数函数：a * exp(b * x) + c"""
        return a * np.exp(b * x) + c
    
    def gaussian_func(x, a, b, c, d):
        """高斯函数：a * exp(-((x - b) / c)²) + d"""
        return a * np.exp(-((x - b) / c)**2) + d
    
    def logistic_func(x, a, b, c, d):
        """逻辑函数：a / (1 + exp(-b * (x - c))) + d"""
        return a / (1 + np.exp(-b * (x - c))) + d
    
    # 格式化时间显示
    start_str = start_time.strftime("%m月%d日%H:%M")
    end_str = end_time.strftime("%m月%d日%H:%M")
    time_range_str = f"{start_str}-{end_str}"
    
    # 按设备分组拟合
    devices = df['device_id'].unique()
    
    for dev_id in devices:
        dev_data = df[df['device_id'] == dev_id].copy()
        dev_data = dev_data.sort_values('timestamp')
        
        if len(dev_data) < 10:
            print(f"设备 {dev_id} 数据点太少，跳过拟合")
            continue
        
        # 将时间转换为数值（小时）
        time_start = dev_data['timestamp'].min()
        dev_data['time_hours'] = (dev_data['timestamp'] - time_start).dt.total_seconds() / 3600
        
        # 准备拟合数据
        x_data = dev_data['time_hours'].values
        y_data = dev_data['temperature'].values
        
        # 创建拟合结果存储
        fit_results = {}
        
        plt.figure(figsize=(20, 12))
        
        # 1. 多项式拟合
        plt.subplot(2, 3, 1)
        try:
            popt_poly, pcov_poly = curve_fit(polynomial_func, x_data, y_data, maxfev=5000)
            y_fit_poly = polynomial_func(x_data, *popt_poly)
            r2_poly = 1 - np.sum((y_data - y_fit_poly)**2) / np.sum((y_data - np.mean(y_data))**2)
            
            plt.plot(dev_data['timestamp'], y_data, 'b-', alpha=0.7, label='原始数据')
            plt.plot(dev_data['timestamp'], y_fit_poly, 'r-', linewidth=3, label=f'多项式拟合 (R²={r2_poly:.3f})')
            plt.title(f'设备 {dev_id} - 多项式拟合曲线')
            plt.xlabel('时间')
            plt.ylabel('温度 (°C)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            fit_results['polynomial'] = {
                'params': popt_poly,
                'r2': r2_poly,
                'func': polynomial_func
            }
        except Exception as e:
            plt.text(0.5, 0.5, f'多项式拟合失败\n{str(e)}', transform=plt.gca().transAxes, ha='center')
            plt.title(f'设备 {dev_id} - 多项式拟合曲线')
        
        # 2. 正弦拟合
        plt.subplot(2, 3, 2)
        try:
            # 初始参数估计
            amplitude = (y_data.max() - y_data.min()) / 2
            period = (x_data.max() - x_data.min()) / 2
            phase = 0
            offset = np.mean(y_data)
            p0 = [amplitude, 2*np.pi/period, phase, offset]
            
            popt_sine, pcov_sine = curve_fit(sine_func, x_data, y_data, p0=p0, maxfev=5000)
            y_fit_sine = sine_func(x_data, *popt_sine)
            r2_sine = 1 - np.sum((y_data - y_fit_sine)**2) / np.sum((y_data - np.mean(y_data))**2)
            
            plt.plot(dev_data['timestamp'], y_data, 'b-', alpha=0.7, label='原始数据')
            plt.plot(dev_data['timestamp'], y_fit_sine, 'g-', linewidth=3, label=f'正弦拟合 (R²={r2_sine:.3f})')
            plt.title(f'设备 {dev_id} - 正弦拟合曲线')
            plt.xlabel('时间')
            plt.ylabel('温度 (°C)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            fit_results['sine'] = {
                'params': popt_sine,
                'r2': r2_sine,
                'func': sine_func
            }
        except Exception as e:
            plt.text(0.5, 0.5, f'正弦拟合失败\n{str(e)}', transform=plt.gca().transAxes, ha='center')
            plt.title(f'设备 {dev_id} - 正弦拟合曲线')
        
        # 3. 逻辑函数拟合
        plt.subplot(2, 3, 3)
        try:
            # 初始参数估计
            a = y_data.max() - y_data.min()
            b = 1.0
            c = x_data.mean()
            d = y_data.min()
            p0 = [a, b, c, d]
            
            popt_log, pcov_log = curve_fit(logistic_func, x_data, y_data, p0=p0, maxfev=5000)
            y_fit_log = logistic_func(x_data, *popt_log)
            r2_log = 1 - np.sum((y_data - y_fit_log)**2) / np.sum((y_data - np.mean(y_data))**2)
            
            plt.plot(dev_data['timestamp'], y_data, 'b-', alpha=0.7, label='原始数据')
            plt.plot(dev_data['timestamp'], y_fit_log, 'm-', linewidth=3, label=f'逻辑函数拟合 (R²={r2_log:.3f})')
            plt.title(f'设备 {dev_id} - 逻辑函数拟合曲线')
            plt.xlabel('时间')
            plt.ylabel('温度 (°C)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            fit_results['logistic'] = {
                'params': popt_log,
                'r2': r2_log,
                'func': logistic_func
            }
        except Exception as e:
            plt.text(0.5, 0.5, f'逻辑函数拟合失败\n{str(e)}', transform=plt.gca().transAxes, ha='center')
            plt.title(f'设备 {dev_id} - 逻辑函数拟合曲线')
        
        # 4. 高斯拟合
        plt.subplot(2, 3, 4)
        try:
            # 初始参数估计
            amplitude = y_data.max() - y_data.min()
            center = x_data[np.argmax(y_data)]
            width = (x_data.max() - x_data.min()) / 4
            offset = y_data.min()
            p0 = [amplitude, center, width, offset]
            
            popt_gauss, pcov_gauss = curve_fit(gaussian_func, x_data, y_data, p0=p0, maxfev=5000)
            y_fit_gauss = gaussian_func(x_data, *popt_gauss)
            r2_gauss = 1 - np.sum((y_data - y_fit_gauss)**2) / np.sum((y_data - np.mean(y_data))**2)
            
            plt.plot(dev_data['timestamp'], y_data, 'b-', alpha=0.7, label='原始数据')
            plt.plot(dev_data['timestamp'], y_fit_gauss, 'c-', linewidth=3, label=f'高斯拟合 (R²={r2_gauss:.3f})')
            plt.title(f'设备 {dev_id} - 高斯拟合曲线')
            plt.xlabel('时间')
            plt.ylabel('温度 (°C)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            fit_results['gaussian'] = {
                'params': popt_gauss,
                'r2': r2_gauss,
                'func': gaussian_func
            }
        except Exception as e:
            plt.text(0.5, 0.5, f'高斯拟合失败\n{str(e)}', transform=plt.gca().transAxes, ha='center')
            plt.title(f'设备 {dev_id} - 高斯拟合曲线')
        
        # 5. 最佳拟合曲线
        plt.subplot(2, 3, 5)
        if fit_results:
            best_fit = max(fit_results.items(), key=lambda x: x[1]['r2'])
            fit_name, fit_data = best_fit
            y_fit_best = fit_data['func'](x_data, *fit_data['params'])
            
            plt.plot(dev_data['timestamp'], y_data, 'b-', alpha=0.7, label='原始数据')
            plt.plot(dev_data['timestamp'], y_fit_best, 'orange', linewidth=4, 
                    label=f'最佳拟合曲线 ({fit_name}, R²={fit_data["r2"]:.3f})')
            plt.title(f'设备 {dev_id} - 最佳拟合曲线')
            plt.xlabel('时间')
            plt.ylabel('温度 (°C)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 6. 拟合质量对比
        plt.subplot(2, 3, 6)
        if fit_results:
            fit_names = list(fit_results.keys())
            r2_values = [fit_results[name]['r2'] for name in fit_names]
            
            bars = plt.bar(fit_names, r2_values, color=['red', 'green', 'magenta', 'cyan'])
            plt.title(f'设备 {dev_id} - 拟合质量对比')
            plt.ylabel('R² 值')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, r2_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            # 保存到Data文件夹
            import os
            data_dir = "/Users/z5540822/Desktop/mpc-farming-master/Data"
            os.makedirs(data_dir, exist_ok=True)
            # 生成动态文件名
            start_file_str = start_time.strftime("%m-%d_%H-%M")
            end_file_str = end_time.strftime("%m-%d_%H-%M")
            filename = f'temperature_continuous_curve_{start_file_str}_to_{end_file_str}_{dev_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            filepath = os.path.join(data_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"连续拟合曲线图已保存为: {filepath}")
        
        # 打印拟合结果
        print(f"\n设备 {dev_id} 连续拟合结果:")
        for fit_name, fit_data in fit_results.items():
            print(f"  {fit_name}: R² = {fit_data['r2']:.4f}")
            print(f"    参数: {fit_data['params']}")
        
        if fit_results:
            best_fit = max(fit_results.items(), key=lambda x: x[1]['r2'])
            print(f"  最佳拟合: {best_fit[0]} (R² = {best_fit[1]['r2']:.4f})")

def create_single_curve(df, start_time, end_time, save_plots=True):
    """创建单一连续拟合曲线"""
    print("正在创建单一连续拟合曲线...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if len(df) == 0:
        print("没有数据可拟合")
        return
    
    # 定义拟合函数
    def polynomial_func(x, a, b, c, d):
        """多项式函数：ax³ + bx² + cx + d"""
        return a * x**3 + b * x**2 + c * x + d
    
    def logistic_func(x, a, b, c, d):
        """逻辑函数：a / (1 + exp(-b * (x - c))) + d"""
        return a / (1 + np.exp(-b * (x - c))) + d
    
    def gaussian_func(x, a, b, c, d):
        """高斯函数：a * exp(-((x - b) / c)²) + d"""
        return a * np.exp(-((x - b) / c)**2) + d
    
    # 格式化时间显示
    start_str = start_time.strftime("%m月%d日%H:%M")
    end_str = end_time.strftime("%m月%d日%H:%M")
    time_range_str = f"{start_str}-{end_str}"
    
    # 按设备分组拟合
    devices = df['device_id'].unique()
    
    for dev_id in devices:
        dev_data = df[df['device_id'] == dev_id].copy()
        dev_data = dev_data.sort_values('timestamp')
        
        if len(dev_data) < 10:
            print(f"设备 {dev_id} 数据点太少，跳过拟合")
            continue
        
        # 将时间转换为数值（小时）
        time_start = dev_data['timestamp'].min()
        dev_data['time_hours'] = (dev_data['timestamp'] - time_start).dt.total_seconds() / 3600
        
        # 准备拟合数据
        x_data = dev_data['time_hours'].values
        y_data = dev_data['temperature'].values
        
        # 创建拟合结果存储
        fit_results = {}
        
        # 尝试不同的拟合方法
        try:
            # 多项式拟合
            popt_poly, pcov_poly = curve_fit(polynomial_func, x_data, y_data, maxfev=5000)
            y_fit_poly = polynomial_func(x_data, *popt_poly)
            r2_poly = 1 - np.sum((y_data - y_fit_poly)**2) / np.sum((y_data - np.mean(y_data))**2)
            fit_results['polynomial'] = {'params': popt_poly, 'r2': r2_poly, 'func': polynomial_func}
        except:
            pass
        
        try:
            # 逻辑函数拟合
            a = y_data.max() - y_data.min()
            b = 1.0
            c = x_data.mean()
            d = y_data.min()
            p0 = [a, b, c, d]
            popt_log, pcov_log = curve_fit(logistic_func, x_data, y_data, p0=p0, maxfev=5000)
            y_fit_log = logistic_func(x_data, *popt_log)
            r2_log = 1 - np.sum((y_data - y_fit_log)**2) / np.sum((y_data - np.mean(y_data))**2)
            fit_results['logistic'] = {'params': popt_log, 'r2': r2_log, 'func': logistic_func}
        except:
            pass
        
        try:
            # 高斯拟合
            amplitude = y_data.max() - y_data.min()
            center = x_data[np.argmax(y_data)]
            width = (x_data.max() - x_data.min()) / 4
            offset = y_data.min()
            p0 = [amplitude, center, width, offset]
            popt_gauss, pcov_gauss = curve_fit(gaussian_func, x_data, y_data, p0=p0, maxfev=5000)
            y_fit_gauss = gaussian_func(x_data, *popt_gauss)
            r2_gauss = 1 - np.sum((y_data - y_fit_gauss)**2) / np.sum((y_data - np.mean(y_data))**2)
            fit_results['gaussian'] = {'params': popt_gauss, 'r2': r2_gauss, 'func': gaussian_func}
        except:
            pass
        
        # 选择最佳拟合
        if fit_results:
            best_fit = max(fit_results.items(), key=lambda x: x[1]['r2'])
            fit_name, fit_data = best_fit
            y_fit_best = fit_data['func'](x_data, *fit_data['params'])
            
            # 创建单一图表
            plt.figure(figsize=(16, 10))
            
            # 绘制原始数据和拟合曲线
            plt.plot(dev_data['timestamp'], y_data, 'b-', alpha=0.7, linewidth=1, label='原始温度数据')
            plt.plot(dev_data['timestamp'], y_fit_best, 'r-', linewidth=3, 
                    label=f'拟合曲线 ({fit_name}, R²={fit_data["r2"]:.4f})')
            
            # 设置图表属性
            plt.title(f'设备 {dev_id} - 温度连续拟合曲线 ({time_range_str})', fontsize=16, fontweight='bold')
            plt.xlabel('时间', fontsize=14)
            plt.ylabel('温度 (°C)', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # 设置坐标轴
            plt.tight_layout()
            
            # 添加拟合信息文本
            fit_info = f"拟合方法: {fit_name}\nR² = {fit_data['r2']:.4f}\n参数: {fit_data['params']}"
            plt.text(0.02, 0.98, fit_info, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10)
            
            if save_plots:
                # 保存到Data文件夹
                import os
                data_dir = "/Users/z5540822/Desktop/mpc-farming-master/Data"
                os.makedirs(data_dir, exist_ok=True)
                # 生成动态文件名
                start_file_str = start_time.strftime("%m-%d_%H-%M")
                end_file_str = end_time.strftime("%m-%d_%H-%M")
                filename = f'temperature_single_curve_{start_file_str}_to_{end_file_str}_{dev_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                filepath = os.path.join(data_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"单一拟合曲线图已保存为: {filepath}")
            
            plt.show()
            
            # 打印拟合结果
            print(f"\n设备 {dev_id} 单一拟合结果:")
            print(f"  最佳拟合方法: {fit_name}")
            print(f"  R² = {fit_data['r2']:.4f}")
            print(f"  拟合参数: {fit_data['params']}")
            
            # 计算拟合曲线的统计信息
            residuals = y_data - y_fit_best
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            print(f"  均方根误差 (RMSE): {rmse:.4f} °C")
            print(f"  平均绝对误差 (MAE): {mae:.4f} °C")
        else:
            print(f"设备 {dev_id} 无法进行拟合")

def create_complete_curve(df, start_time, end_time, save_plots=True):
    """创建包含起点和终点的完整连续温度曲线"""
    print("正在创建完整连续温度曲线...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if len(df) == 0:
        print("没有数据可绘制")
        return
    
    # 格式化时间显示
    start_str = start_time.strftime("%m月%d日%H:%M")
    end_str = end_time.strftime("%m月%d日%H:%M")
    time_range_str = f"{start_str}-{end_str}"
    
    # 按设备分组处理
    devices = df['device_id'].unique()
    
    for dev_id in devices:
        dev_data = df[df['device_id'] == dev_id].copy()
        dev_data = dev_data.sort_values('timestamp').reset_index(drop=True)
        
        if len(dev_data) < 2:
            print(f"设备 {dev_id} 数据点太少，跳过绘制")
            continue
        
        # 确保数据完整性 - 从起点到终点
        print(f"设备 {dev_id} 数据范围:")
        print(f"  实际起点: {dev_data['timestamp'].min()}")
        print(f"  实际终点: {dev_data['timestamp'].max()}")
        print(f"  数据点数: {len(dev_data)}")
        
        # 创建完整的连续曲线图
        plt.figure(figsize=(18, 10))
        
        # 绘制原始温度数据作为连续曲线
        plt.plot(dev_data['timestamp'], dev_data['temperature'], 
                color='blue', linewidth=2, alpha=0.8, 
                label=f'原始温度数据 ({len(dev_data)}个数据点)')
        
        # 绘制移动平均滤波后的平滑曲线
        plt.plot(dev_data['timestamp'], dev_data['temp_ma'], 
                color='red', linewidth=3, alpha=0.9,
                label='移动平均滤波曲线')
        
        # 标记起点和终点
        start_temp = dev_data['temperature'].iloc[0]
        end_temp = dev_data['temperature'].iloc[-1]
        start_timestamp = dev_data['timestamp'].iloc[0]
        end_timestamp = dev_data['timestamp'].iloc[-1]
        
        plt.scatter([start_timestamp], [start_temp], 
                   color='green', s=100, zorder=5, 
                   label=f'起点: {start_temp:.2f}°C')
        plt.scatter([end_timestamp], [end_temp], 
                   color='orange', s=100, zorder=5, 
                   label=f'终点: {end_temp:.2f}°C')
        
        # 添加起点和终点的文本标注
        plt.annotate(f'起点\n{start_temp:.2f}°C\n{start_timestamp.strftime("%H:%M")}', 
                    xy=(start_timestamp, start_temp), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='green'))
        
        plt.annotate(f'终点\n{end_temp:.2f}°C\n{end_timestamp.strftime("%H:%M")}', 
                    xy=(end_timestamp, end_temp), 
                    xytext=(-50, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='orange'))
        
        # 设置图表属性
        plt.title(f'设备 {dev_id} - 完整连续温度曲线 ({time_range_str})', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('时间', fontsize=14)
        plt.ylabel('温度 (°C)', fontsize=14)
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 设置Y轴范围，留出一些边距
        temp_min = dev_data['temperature'].min()
        temp_max = dev_data['temperature'].max()
        temp_range = temp_max - temp_min
        plt.ylim(temp_min - temp_range * 0.1, temp_max + temp_range * 0.1)
        
        # 添加统计信息文本框
        stats_info = (f"温度统计信息:\n"
                     f"最低温度: {temp_min:.2f}°C\n"
                     f"最高温度: {temp_max:.2f}°C\n"
                     f"温度范围: {temp_range:.2f}°C\n"
                     f"平均温度: {dev_data['temperature'].mean():.2f}°C\n"
                     f"温度变化: {end_temp - start_temp:.2f}°C\n"
                     f"时间跨度: {(end_timestamp - start_timestamp).total_seconds() / 3600:.1f}小时")
        
        plt.text(0.02, 0.98, stats_info, transform=plt.gca().transAxes, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=10)
        
        # 设置坐标轴
        plt.tight_layout()
        
        if save_plots:
            # 保存到Data文件夹
            import os
            data_dir = "/Users/z5540822/Desktop/mpc-farming-master/Data"
            os.makedirs(data_dir, exist_ok=True)
            # 生成动态文件名
            start_file_str = start_time.strftime("%m-%d_%H-%M")
            end_file_str = end_time.strftime("%m-%d_%H-%M")
            filename = f'temperature_complete_curve_{start_file_str}_to_{end_file_str}_{dev_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            filepath = os.path.join(data_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"完整连续曲线图已保存为: {filepath}")
        
        plt.show()
        
        # 打印详细信息
        print(f"\n设备 {dev_id} 完整连续曲线信息:")
        print(f"  起点时间: {start_timestamp}")
        print(f"  终点时间: {end_timestamp}")
        print(f"  起点温度: {start_temp:.2f}°C")
        print(f"  终点温度: {end_temp:.2f}°C")
        print(f"  温度总变化: {end_temp - start_temp:.2f}°C")
        print(f"  时间跨度: {(end_timestamp - start_timestamp).total_seconds() / 3600:.1f}小时")
        print(f"  数据点数: {len(dev_data)}")
        print(f"  平均温度: {dev_data['temperature'].mean():.2f}°C")
        print(f"  温度范围: {temp_min:.2f}°C - {temp_max:.2f}°C")

def create_fixed_endpoint_curve(df, start_time, end_time, save_plots=True):
    """创建固定起点和终点的拟合曲线"""
    print("正在创建固定起终点的拟合曲线...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if len(df) == 0:
        print("没有数据可拟合")
        return
    
    # 定义约束拟合函数
    def constrained_polynomial(x, a, b, c):
        """三次多项式，固定起点和终点"""
        # f(x) = ax³ + bx² + cx + d
        # 其中d由起点约束确定，系数由终点约束调整
        return a * x**3 + b * x**2 + c * x
    
    def constrained_sigmoid(x, a, b, c):
        """S型函数，固定起点和终点"""
        return a / (1 + np.exp(-b * (x - c)))
    
    def constrained_exponential(x, a, b):
        """指数函数，固定起点和终点"""
        return a * (1 - np.exp(-b * x))
    
    # 格式化时间显示
    start_str = start_time.strftime("%m月%d日%H:%M")
    end_str = end_time.strftime("%m月%d日%H:%M")
    time_range_str = f"{start_str}-{end_str}"
    
    # 按设备分组处理
    devices = df['device_id'].unique()
    
    for dev_id in devices:
        dev_data = df[df['device_id'] == dev_id].copy()
        dev_data = dev_data.sort_values('timestamp').reset_index(drop=True)
        
        if len(dev_data) < 10:
            print(f"设备 {dev_id} 数据点太少，跳过拟合")
            continue
        
        # 获取起点和终点
        start_temp = dev_data['temperature'].iloc[0]
        end_temp = dev_data['temperature'].iloc[-1]
        start_timestamp = dev_data['timestamp'].iloc[0]
        end_timestamp = dev_data['timestamp'].iloc[-1]
        
        print(f"设备 {dev_id} 固定约束:")
        print(f"  起点: {start_timestamp.strftime('%H:%M')} - {start_temp:.2f}°C")
        print(f"  终点: {end_timestamp.strftime('%H:%M')} - {end_temp:.2f}°C")
        
        # 将时间转换为数值（0到1标准化）
        time_start = dev_data['timestamp'].min()
        dev_data['time_normalized'] = (dev_data['timestamp'] - time_start).dt.total_seconds()
        dev_data['time_normalized'] = dev_data['time_normalized'] / dev_data['time_normalized'].max()
        
        # 准备拟合数据（将温度也标准化到起点-终点范围）
        x_data = dev_data['time_normalized'].values
        y_data = dev_data['temperature'].values
        
        # 标准化温度数据以便拟合
        temp_range = end_temp - start_temp
        y_normalized = (y_data - start_temp) / temp_range if temp_range != 0 else y_data - start_temp
        
        # 尝试不同的拟合方法
        fit_results = {}
        
        # 1. 三次多项式拟合（约束起终点）
        try:
            # 构建约束条件：f(0) = 0, f(1) = 1
            def poly_with_constraints(x, a, b):
                # f(x) = ax³ + bx² + (1-a-b)x，确保f(0)=0, f(1)=1
                c = 1 - a - b
                return a * x**3 + b * x**2 + c * x
            
            popt_poly, _ = curve_fit(poly_with_constraints, x_data, y_normalized, maxfev=5000)
            y_fit_norm = poly_with_constraints(x_data, *popt_poly)
            y_fit_poly = y_fit_norm * temp_range + start_temp
            
            # 确保起终点精确匹配
            y_fit_poly[0] = start_temp
            y_fit_poly[-1] = end_temp
            
            r2_poly = 1 - np.sum((y_data - y_fit_poly)**2) / np.sum((y_data - np.mean(y_data))**2)
            fit_results['polynomial'] = {
                'y_fit': y_fit_poly,
                'r2': r2_poly,
                'params': popt_poly,
                'name': '约束多项式'
            }
        except Exception as e:
            print(f"多项式拟合失败: {e}")
        
        # 2. 贝塞尔曲线拟合
        try:
            from scipy.interpolate import interp1d
            
            # 使用三次样条插值，固定端点
            f_spline = interp1d(x_data, y_data, kind='cubic', bounds_error=False, fill_value='extrapolate')
            y_fit_spline = f_spline(x_data)
            
            # 强制固定起终点
            y_fit_spline[0] = start_temp
            y_fit_spline[-1] = end_temp
            
            r2_spline = 1 - np.sum((y_data - y_fit_spline)**2) / np.sum((y_data - np.mean(y_data))**2)
            fit_results['spline'] = {
                'y_fit': y_fit_spline,
                'r2': r2_spline,
                'params': [],
                'name': '三次样条'
            }
        except Exception as e:
            print(f"样条拟合失败: {e}")
        
        # 3. 自定义S型曲线
        try:
            def custom_sigmoid(x, k, x0):
                # S型函数，固定起终点
                sigmoid = 1 / (1 + np.exp(-k * (x - x0)))
                # 标准化到0-1范围
                sigmoid_norm = (sigmoid - sigmoid[0]) / (sigmoid[-1] - sigmoid[0])
                return sigmoid_norm
            
            popt_sig, _ = curve_fit(custom_sigmoid, x_data, y_normalized, 
                                  p0=[5, 0.5], maxfev=5000)
            y_fit_norm = custom_sigmoid(x_data, *popt_sig)
            y_fit_sigmoid = y_fit_norm * temp_range + start_temp
            
            # 确保起终点精确匹配
            y_fit_sigmoid[0] = start_temp
            y_fit_sigmoid[-1] = end_temp
            
            r2_sigmoid = 1 - np.sum((y_data - y_fit_sigmoid)**2) / np.sum((y_data - np.mean(y_data))**2)
            fit_results['sigmoid'] = {
                'y_fit': y_fit_sigmoid,
                'r2': r2_sigmoid,
                'params': popt_sig,
                'name': 'S型曲线'
            }
        except Exception as e:
            print(f"S型曲线拟合失败: {e}")
        
        # 选择最佳拟合
        if fit_results:
            best_fit = max(fit_results.items(), key=lambda x: x[1]['r2'])
            fit_name, fit_data = best_fit
            
            # 创建图表
            plt.figure(figsize=(16, 10))
            
            # 绘制原始数据
            plt.plot(dev_data['timestamp'], y_data, 'b-', alpha=0.6, linewidth=1, 
                    label=f'原始数据 ({len(dev_data)}个点)')
            
            # 绘制拟合曲线
            plt.plot(dev_data['timestamp'], fit_data['y_fit'], 'r-', linewidth=4, 
                    label=f'拟合曲线 ({fit_data["name"]}, R²={fit_data["r2"]:.4f})')
            
            # 强调起点和终点
            plt.scatter([start_timestamp], [start_temp], color='green', s=150, zorder=10, 
                       label=f'固定起点: {start_temp:.2f}°C')
            plt.scatter([end_timestamp], [end_temp], color='orange', s=150, zorder=10, 
                       label=f'固定终点: {end_temp:.2f}°C')
            
            # 添加起终点标注
            plt.annotate(f'起点\n{start_temp:.2f}°C', 
                        xy=(start_timestamp, start_temp), 
                        xytext=(20, 20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))
            
            plt.annotate(f'终点\n{end_temp:.2f}°C', 
                        xy=(end_timestamp, end_temp), 
                        xytext=(-40, 20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=2))
            
            # 设置图表属性
            plt.title(f'设备 {dev_id} - 固定起终点拟合曲线 ({time_range_str})', 
                     fontsize=18, fontweight='bold', pad=20)
            plt.xlabel('时间', fontsize=14)
            plt.ylabel('温度 (°C)', fontsize=14)
            plt.legend(fontsize=12, loc='best')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # 添加拟合信息
            fit_info = (f"拟合方法: {fit_data['name']}\n"
                       f"R² = {fit_data['r2']:.4f}\n"
                       f"起点固定: {start_temp:.2f}°C\n"
                       f"终点固定: {end_temp:.2f}°C\n"
                       f"温度变化: {end_temp - start_temp:.2f}°C")
            
            plt.text(0.02, 0.98, fit_info, transform=plt.gca().transAxes, 
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9),
                    fontsize=11)
            
            plt.tight_layout()
            
            if save_plots:
                # 保存到Data文件夹
                import os
                data_dir = "/Users/z5540822/Desktop/mpc-farming-master/Data"
                os.makedirs(data_dir, exist_ok=True)
                start_file_str = start_time.strftime("%m-%d_%H-%M")
                end_file_str = end_time.strftime("%m-%d_%H-%M")
                filename = f'temperature_fixed_curve_{start_file_str}_to_{end_file_str}_{dev_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                filepath = os.path.join(data_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"固定起终点拟合曲线已保存为: {filepath}")
            
            plt.show()
            
            # 打印结果
            print(f"\n设备 {dev_id} 固定起终点拟合结果:")
            print(f"  最佳拟合方法: {fit_data['name']}")
            print(f"  R² = {fit_data['r2']:.4f}")
            print(f"  起点温度: {start_temp:.2f}°C (固定)")
            print(f"  终点温度: {end_temp:.2f}°C (固定)")
            print(f"  拟合验证 - 起点: {fit_data['y_fit'][0]:.2f}°C, 终点: {fit_data['y_fit'][-1]:.2f}°C")
        else:
            print(f"设备 {dev_id} 无法进行约束拟合")

def create_smooth_fixed_curve(df, start_time, end_time, save_plots=True):
    """创建固定起终点的平滑拟合曲线（只显示拟合曲线）"""
    print("正在创建平滑拟合曲线...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if len(df) == 0:
        print("没有数据可拟合")
        return
    
    # 格式化时间显示
    start_str = start_time.strftime("%m月%d日%H:%M")
    end_str = end_time.strftime("%m月%d日%H:%M")
    time_range_str = f"{start_str}-{end_str}"
    
    # 按设备分组处理
    devices = df['device_id'].unique()
    
    for dev_id in devices:
        dev_data = df[df['device_id'] == dev_id].copy()
        dev_data = dev_data.sort_values('timestamp').reset_index(drop=True)
        
        if len(dev_data) < 10:
            print(f"设备 {dev_id} 数据点太少，跳过拟合")
            continue
        
        # 获取起点和终点
        start_temp = dev_data['temperature'].iloc[0]
        end_temp = dev_data['temperature'].iloc[-1]
        start_timestamp = dev_data['timestamp'].iloc[0]
        end_timestamp = dev_data['timestamp'].iloc[-1]
        
        print(f"设备 {dev_id}:")
        print(f"  起点: {start_timestamp.strftime('%H:%M')} - {start_temp:.2f}°C")
        print(f"  终点: {end_timestamp.strftime('%H:%M')} - {end_temp:.2f}°C")
        
        # 将时间转换为数值（小时）
        time_start = dev_data['timestamp'].min()
        time_hours = (dev_data['timestamp'] - time_start).dt.total_seconds() / 3600
        
        # 使用贝塞尔曲线或高阶多项式创建平滑曲线
        # 创建更密集的时间点用于绘制平滑曲线
        time_dense = np.linspace(0, time_hours.max(), 1000)  # 1000个点创建平滑曲线
        timestamp_dense = start_timestamp + pd.to_timedelta(time_dense, unit='h')
        
        # 使用三次多项式拟合，但固定起终点
        from scipy.optimize import minimize
        
        def polynomial_objective(params, x_data, y_data, start_y, end_y):
            """多项式目标函数，约束起终点"""
            a, b, c = params
            # 计算d使得f(0) = start_y
            d = start_y
            # 计算多项式值
            x_max = x_data[-1]
            predicted = a * (x_data/x_max)**3 + b * (x_data/x_max)**2 + c * (x_data/x_max) + d
            
            # 添加终点约束
            end_predicted = a + b + c + d
            end_constraint = (end_predicted - end_y)**2 * 1000  # 强约束
            
            # 拟合误差
            fit_error = np.sum((predicted - y_data)**2)
            
            return fit_error + end_constraint
        
        # 优化多项式参数
        try:
            x_data = time_hours.values
            y_data = dev_data['temperature'].values
            x_max = x_data[-1]
            
            # 多次尝试不同的初始猜测
            temp_change = end_temp - start_temp
            initial_guesses = [
                [temp_change * 0.1, temp_change * 0.3, temp_change * 0.6],
                [temp_change * 0.2, temp_change * 0.5, temp_change * 0.3],
                [temp_change * 0.0, temp_change * 0.2, temp_change * 0.8],
                [temp_change * 0.5, temp_change * 0.1, temp_change * 0.4]
            ]
            
            best_result = None
            best_error = float('inf')
            
            for initial_guess in initial_guesses:
                try:
                    result = minimize(polynomial_objective, initial_guess, 
                                    args=(x_data, y_data, start_temp, end_temp),
                                    method='L-BFGS-B')
                    
                    if result.success and result.fun < best_error:
                        best_result = result
                        best_error = result.fun
                except:
                    continue
            
            if best_result is not None:
                a, b, c = best_result.x
                d = start_temp
                
                # 生成平滑曲线
                x_smooth = time_dense / time_dense.max()
                y_smooth = a * x_smooth**3 + b * x_smooth**2 + c * x_smooth + d
                
                # 确保起终点精确匹配
                y_smooth[0] = start_temp
                y_smooth[-1] = end_temp
                
                # 创建图表
                plt.figure(figsize=(14, 8))
                
                # 只绘制平滑拟合曲线
                plt.plot(timestamp_dense, y_smooth, 'red', linewidth=4, 
                        label=f'平滑拟合曲线 (三次多项式)')
                
                # 标记起点和终点
                plt.scatter([start_timestamp], [start_temp], color='green', s=200, zorder=10, 
                           edgecolor='darkgreen', linewidth=2,
                           label=f'起点: {start_temp:.2f}°C')
                plt.scatter([end_timestamp], [end_temp], color='orange', s=200, zorder=10, 
                           edgecolor='darkorange', linewidth=2,
                           label=f'终点: {end_temp:.2f}°C')
                
                # 添加起终点标注
                plt.annotate(f'起点\n{start_temp:.2f}°C\n{start_timestamp.strftime("%H:%M")}', 
                            xy=(start_timestamp, start_temp), 
                            xytext=(30, 30), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9),
                            arrowprops=dict(arrowstyle='->', color='green', lw=2),
                            fontsize=11, fontweight='bold')
                
                plt.annotate(f'终点\n{end_temp:.2f}°C\n{end_timestamp.strftime("%H:%M")}', 
                            xy=(end_timestamp, end_temp), 
                            xytext=(-60, 30), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightsalmon', alpha=0.9),
                            arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                            fontsize=11, fontweight='bold')
                
                # 设置图表属性
                plt.title(f'设备 {dev_id} - 平滑拟合曲线 ({time_range_str})', 
                         fontsize=18, fontweight='bold', pad=20)
                plt.xlabel('时间', fontsize=14)
                plt.ylabel('温度 (°C)', fontsize=14)
                plt.legend(fontsize=12, loc='best')
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.xticks(rotation=45)
                
                # 设置Y轴范围
                temp_margin = (end_temp - start_temp) * 0.1
                plt.ylim(start_temp - temp_margin, end_temp + temp_margin)
                
                # 添加曲线信息
                curve_info = (f"拟合类型: 三次多项式\n"
                             f"起点固定: {start_temp:.2f}°C\n"
                             f"终点固定: {end_temp:.2f}°C\n"
                             f"温度变化: {end_temp - start_temp:.2f}°C\n"
                             f"时间跨度: {time_hours.max():.1f}小时")
                
                plt.text(0.02, 0.98, curve_info, transform=plt.gca().transAxes, 
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
                        fontsize=11)
                
                plt.tight_layout()
                
                if save_plots:
                    # 保存到Data文件夹
                    import os
                    data_dir = "/Users/z5540822/Desktop/mpc-farming-master/Data"
                    os.makedirs(data_dir, exist_ok=True)
                    start_file_str = start_time.strftime("%m-%d_%H-%M")
                    end_file_str = end_time.strftime("%m-%d_%H-%M")
                    filename = f'temperature_smooth_curve_{start_file_str}_to_{end_file_str}_{dev_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                    filepath = os.path.join(data_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    print(f"平滑拟合曲线已保存为: {filepath}")
                
                plt.show()
                
                # 打印结果
                print(f"平滑拟合完成:")
                print(f"  多项式参数: a={a:.6f}, b={b:.6f}, c={c:.6f}, d={d:.6f}")
                print(f"  起点验证: {y_smooth[0]:.2f}°C")
                print(f"  终点验证: {y_smooth[-1]:.2f}°C")
                
            else:
                print(f"设备 {dev_id} 拟合优化失败")
                
        except Exception as e:
            print(f"设备 {dev_id} 拟合过程出错: {e}")

def create_thermal_model_curve(df, start_time, end_time, save_plots=True):
    """基于LED升温热力学模型的拟合曲线"""
    print("正在创建LED升温热力学模型拟合曲线...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if len(df) == 0:
        print("没有数据可拟合")
        return
    
    # 热力学模型函数
    def led_thermal_model(t, T0, T_max, tau, alpha, beta):
        """
        LED升温热力学模型
        T(t) = T0 + (T_max - T0) * (1 - exp(-t/tau)) + alpha*t + beta*t^2
        
        参数说明:
        T0: 初始温度 (起点温度)
        T_max: 最大理论温度
        tau: 热时间常数 (升温速度参数)
        alpha: 线性升温系数
        beta: 二次升温系数
        """
        return T0 + (T_max - T0) * (1 - np.exp(-t/tau)) + alpha * t + beta * t**2
    
    def newton_cooling_model(t, T0, T_ambient, R_th, P_led, tau):
        """
        牛顿冷却定律模型 + LED功率加热
        T(t) = T_ambient + (T0 - T_ambient) * exp(-t/tau) + P_led * R_th * (1 - exp(-t/tau))
        
        参数说明:
        T0: 初始温度
        T_ambient: 环境温度 (终点温度)
        R_th: 热阻
        P_led: LED功率
        tau: 热时间常数
        """
        return T_ambient + (T0 - T_ambient) * np.exp(-t/tau) + P_led * R_th * (1 - np.exp(-t/tau))
    
    def exponential_heating_model(t, T0, T_final, tau1, tau2, k):
        """
        双指数升温模型
        T(t) = T0 + (T_final - T0) * (1 - k*exp(-t/tau1) - (1-k)*exp(-t/tau2))
        
        参数说明:
        T0: 初始温度
        T_final: 最终温度
        tau1: 快速升温时间常数
        tau2: 慢速升温时间常数
        k: 快速成分权重
        """
        return T0 + (T_final - T0) * (1 - k*np.exp(-t/tau1) - (1-k)*np.exp(-t/tau2))
    
    # 格式化时间显示
    start_str = start_time.strftime("%m月%d日%H:%M")
    end_str = end_time.strftime("%m月%d日%H:%M")
    time_range_str = f"{start_str}-{end_str}"
    
    # 按设备分组处理
    devices = df['device_id'].unique()
    
    for dev_id in devices:
        dev_data = df[df['device_id'] == dev_id].copy()
        dev_data = dev_data.sort_values('timestamp').reset_index(drop=True)
        
        if len(dev_data) < 20:
            print(f"设备 {dev_id} 数据点太少，跳过拟合")
            continue
        
        # 获取起点和终点
        start_temp = dev_data['temperature'].iloc[0]
        end_temp = dev_data['temperature'].iloc[-1]
        start_timestamp = dev_data['timestamp'].iloc[0]
        end_timestamp = dev_data['timestamp'].iloc[-1]
        
        print(f"设备 {dev_id} LED升温模型:")
        print(f"  起点: {start_timestamp.strftime('%H:%M')} - {start_temp:.2f}°C")
        print(f"  终点: {end_timestamp.strftime('%H:%M')} - {end_temp:.2f}°C")
        print(f"  温升: {end_temp - start_temp:.2f}°C")
        
        # 将时间转换为数值（小时）
        time_start = dev_data['timestamp'].min()
        time_hours = (dev_data['timestamp'] - time_start).dt.total_seconds() / 3600
        
        # 准备数据
        t_data = time_hours.values
        T_data = dev_data['temperature'].values
        
        # 尝试不同的热力学模型
        models = {}
        
        # 1. LED升温热力学模型
        try:
            # 初始参数估计
            T0_guess = start_temp
            T_max_guess = end_temp + (end_temp - start_temp) * 0.5  # 理论最大温度
            tau_guess = t_data.max() / 3  # 时间常数
            alpha_guess = (end_temp - start_temp) / t_data.max()  # 线性系数
            beta_guess = 0.0  # 二次系数
            
            p0 = [T0_guess, T_max_guess, tau_guess, alpha_guess, beta_guess]
            
            # 拟合约束：T0必须等于起点温度
            from scipy.optimize import curve_fit
            
            def constrained_led_model(t, T_max, tau, alpha, beta):
                return led_thermal_model(t, start_temp, T_max, tau, alpha, beta)
            
            popt, pcov = curve_fit(constrained_led_model, t_data, T_data, 
                                 p0=p0[1:], maxfev=5000)
            
            T_max_fit, tau_fit, alpha_fit, beta_fit = popt
            
            # 验证终点约束
            T_final_predicted = constrained_led_model(t_data[-1], *popt)
            
            # 调整参数以匹配终点
            adjustment = end_temp - T_final_predicted
            T_max_fit += adjustment
            
            models['led_thermal'] = {
                'func': lambda t: led_thermal_model(t, start_temp, T_max_fit, tau_fit, alpha_fit, beta_fit),
                'params': [start_temp, T_max_fit, tau_fit, alpha_fit, beta_fit],
                'name': 'LED升温热力学模型',
                'equation': 'T(t) = T₀ + (T_max - T₀)(1 - e^(-t/τ)) + αt + βt²'
            }
            
        except Exception as e:
            print(f"LED升温模型拟合失败: {e}")
        
        # 2. 双指数升温模型
        try:
            # 初始参数估计
            tau1_guess = t_data.max() / 10  # 快速时间常数
            tau2_guess = t_data.max() / 2   # 慢速时间常数
            k_guess = 0.3  # 快速成分权重
            
            def constrained_exp_model(t, tau1, tau2, k):
                return exponential_heating_model(t, start_temp, end_temp, tau1, tau2, k)
            
            popt, pcov = curve_fit(constrained_exp_model, t_data, T_data,
                                 p0=[tau1_guess, tau2_guess, k_guess], 
                                 bounds=([0.1, 0.1, 0.0], [t_data.max()*2, t_data.max()*5, 1.0]),
                                 maxfev=5000)
            
            tau1_fit, tau2_fit, k_fit = popt
            
            models['double_exp'] = {
                'func': lambda t: exponential_heating_model(t, start_temp, end_temp, tau1_fit, tau2_fit, k_fit),
                'params': [start_temp, end_temp, tau1_fit, tau2_fit, k_fit],
                'name': '双指数升温模型',
                'equation': 'T(t) = T₀ + (T_f - T₀)(1 - ke^(-t/τ₁) - (1-k)e^(-t/τ₂))'
            }
            
        except Exception as e:
            print(f"双指数模型拟合失败: {e}")
        
        # 选择最佳模型（基于物理意义和拟合效果）
        if models:
            # 评估每个模型
            best_model = None
            best_rmse = float('inf')
            
            for model_name, model_data in models.items():
                try:
                    T_pred = model_data['func'](t_data)
                    rmse = np.sqrt(np.mean((T_data - T_pred)**2))
                    
                    # 检查起终点约束
                    start_error = abs(T_pred[0] - start_temp)
                    end_error = abs(T_pred[-1] - end_temp)
                    
                    print(f"  {model_data['name']}: RMSE={rmse:.4f}, 起点误差={start_error:.4f}, 终点误差={end_error:.4f}")
                    
                    if rmse < best_rmse and start_error < 0.1 and end_error < 1.0:
                        best_model = model_name
                        best_rmse = rmse
                        
                except Exception as e:
                    print(f"  {model_data['name']}: 评估失败 - {e}")
            
            if best_model:
                model_data = models[best_model]
                
                # 创建密集时间点用于绘制平滑曲线
                t_dense = np.linspace(0, t_data.max(), 1000)
                timestamp_dense = start_timestamp + pd.to_timedelta(t_dense, unit='h')
                T_dense = model_data['func'](t_dense)
                
                # 确保起终点精确匹配
                T_dense[0] = start_temp
                T_dense[-1] = end_temp
                
                # 创建图表
                plt.figure(figsize=(14, 10))
                
                # 绘制热力学模型曲线
                plt.plot(timestamp_dense, T_dense, 'red', linewidth=4, 
                        label=f'{model_data["name"]}')
                
                # 标记起点和终点
                plt.scatter([start_timestamp], [start_temp], color='green', s=200, zorder=10, 
                           edgecolor='darkgreen', linewidth=2,
                           label=f'起点: {start_temp:.2f}°C')
                plt.scatter([end_timestamp], [end_temp], color='orange', s=200, zorder=10, 
                           edgecolor='darkorange', linewidth=2,
                           label=f'终点: {end_temp:.2f}°C')
                
                # 添加起终点标注
                plt.annotate(f'起点\n{start_temp:.2f}°C\n{start_timestamp.strftime("%H:%M")}', 
                            xy=(start_timestamp, start_temp), 
                            xytext=(30, 40), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9),
                            arrowprops=dict(arrowstyle='->', color='green', lw=2),
                            fontsize=11, fontweight='bold')
                
                plt.annotate(f'终点\n{end_temp:.2f}°C\n{end_timestamp.strftime("%H:%M")}', 
                            xy=(end_timestamp, end_temp), 
                            xytext=(-60, 40), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightsalmon', alpha=0.9),
                            arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                            fontsize=11, fontweight='bold')
                
                # 设置图表属性
                plt.title(f'设备 {dev_id} - LED升温热力学模型 ({time_range_str})', 
                         fontsize=18, fontweight='bold', pad=20)
                plt.xlabel('时间', fontsize=14)
                plt.ylabel('温度 (°C)', fontsize=14)
                plt.legend(fontsize=12, loc='best')
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.xticks(rotation=45)
                
                # 设置Y轴范围
                temp_margin = (end_temp - start_temp) * 0.1
                plt.ylim(start_temp - temp_margin, end_temp + temp_margin)
                
                # 添加模型信息
                model_info = (f"热力学模型: {model_data['name']}\n"
                             f"方程: {model_data['equation']}\n"
                             f"起点温度: {start_temp:.2f}°C\n"
                             f"终点温度: {end_temp:.2f}°C\n"
                             f"升温幅度: {end_temp - start_temp:.2f}°C\n"
                             f"RMSE: {best_rmse:.4f}°C\n"
                             f"时间跨度: {t_data.max():.1f}小时")
                
                plt.text(0.02, 0.98, model_info, transform=plt.gca().transAxes, 
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.95),
                        fontsize=10)
                
                plt.tight_layout()
                
                if save_plots:
                    # 保存到Data文件夹
                    import os
                    data_dir = "/Users/z5540822/Desktop/mpc-farming-master/Data"
                    os.makedirs(data_dir, exist_ok=True)
                    start_file_str = start_time.strftime("%m-%d_%H-%M")
                    end_file_str = end_time.strftime("%m-%d_%H-%M")
                    filename = f'temperature_thermal_model_{start_file_str}_to_{end_file_str}_{dev_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                    filepath = os.path.join(data_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    print(f"LED升温热力学模型曲线已保存为: {filepath}")
                
                plt.show()
                
                # 打印模型参数
                print(f"\n最佳热力学模型: {model_data['name']}")
                print(f"模型方程: {model_data['equation']}")
                print(f"模型参数: {model_data['params']}")
                print(f"拟合精度 (RMSE): {best_rmse:.4f}°C")
                
            else:
                print(f"设备 {dev_id}: 无法找到合适的热力学模型")
        else:
            print(f"设备 {dev_id}: 所有热力学模型拟合都失败")

def analyze_period_data(df):
    """分析指定时间段的数据"""
    print("\n=== 指定时间段温度数据分析 ===")
    
    for device_id in df['device_id'].unique():
        device_data = df[df['device_id'] == device_id].copy()
        
        if len(device_data) == 0:
            continue
            
        # 计算温度变化率
        device_data = device_data.sort_values('timestamp')
        device_data['temp_diff'] = device_data['temperature'].diff()
        device_data['temp_diff_ma'] = device_data['temp_ma'].diff()
        
        # 统计信息
        print(f"\n设备 {device_id}:")
        print(f"  数据点数: {len(device_data)}")
        print(f"  时间跨度: {(device_data['timestamp'].max() - device_data['timestamp'].min()).total_seconds() / 3600:.1f} 小时")
        print(f"  平均温度: {device_data['temperature'].mean():.2f} °C")
        print(f"  温度标准差: {device_data['temperature'].std():.2f} °C")
        print(f"  温度范围: {device_data['temperature'].min():.2f} - {device_data['temperature'].max():.2f} °C")
        print(f"  温度变化率: {device_data['temp_diff'].mean():.4f} °C/点")
        print(f"  噪声水平: {device_data['temp_diff'].std():.4f} °C")

def main(start_time_str=None, end_time_str=None):
    """主函数"""
    print("=== 提取特定时间段温度数据 ===")
    
    # 数据文件路径
    data_file = "/Users/z5540822/Desktop/mpc-farming-master/Data/5-1-400_20250829_091025.csv"
    
    # 如果没有提供时间参数，使用默认值
    if start_time_str is None:
        start_time_str = "2025-09-06 23:00:00"
    if end_time_str is None:
        end_time_str = "2025-09-08 07:00:00"
    
    # 定义时间段
    start_time = pd.to_datetime(start_time_str)
    end_time = pd.to_datetime(end_time_str)
    
    print(f"提取时间段：{start_time} 到 {end_time}")
    
    try:
        # 加载并滤波数据
        filtered_df = load_and_filter_temperature_data(data_file, start_time, end_time)
        
        if filtered_df is None or len(filtered_df) == 0:
            print("没有找到指定时间段的数据")
            return
        
        # 分析数据
        analyze_period_data(filtered_df)
        
        # 基于LED升温热力学模型的拟合曲线
        create_thermal_model_curve(filtered_df, start_time, end_time, save_plots=True)
        
        # 保存滤波后的数据
        start_file_str = start_time.strftime("%m-%d_%H-%M")
        end_file_str = end_time.strftime("%m-%d_%H-%M")
        output_file = f"/Users/z5540822/Desktop/mpc-farming-master/Data/temperature_period_{start_file_str}_to_{end_file_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # 选择要保存的列
        columns_to_save = [
            'id', 'timestamp', 'device_id', 'temperature', 
            'temp_ma', 'temp_median', 'temp_gaussian', 'temp_lowpass',
            'temp_mean_30s', 'temp_std_30s', 'temp_min_30s', 'temp_max_30s'
        ]
        
        # 只保存存在的列
        available_columns = [col for col in columns_to_save if col in filtered_df.columns]
        df_to_save = filtered_df[available_columns].copy()
        
        df_to_save.to_csv(output_file, index=False)
        print(f"\n滤波后的数据已保存到: {output_file}")
        print(f"共保存 {len(df_to_save)} 条记录")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) == 3:
        # 如果提供了两个参数，使用它们作为开始和结束时间
        start_time_str = sys.argv[1]
        end_time_str = sys.argv[2]
        print(f"使用命令行参数：开始时间={start_time_str}, 结束时间={end_time_str}")
        main(start_time_str, end_time_str)
    else:
        # 否则使用默认时间
        print("使用默认时间范围，如需自定义请提供两个参数：开始时间 结束时间")
        print("示例：python extract_temperature_period.py '2025-09-06 23:00:00' '2025-09-08 07:00:00'")
        main()
