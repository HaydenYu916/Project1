#!/usr/bin/env python3
"""
简化热力学模型演示 - 15分钟步长

快速演示LED热力学模型的基本功能，包括：
1. 15分钟步长的温度变化
2. 不同功率下的温度响应
3. 基于Solar值的升温/降温判断

使用方法:
    python thermal_model_demo_simple.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from led import LedThermalParams, ThermalModelManager, create_model

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def simple_thermal_demo():
    """简单的热力学模型演示"""
    print("热力学模型演示 - 15分钟步长")
    print("=" * 40)
    
    # 参数设置
    dt = 15 * 60  # 15分钟 = 900秒
    total_hours = 4  # 总时间4小时
    total_steps = int(total_hours * 3600 / dt)
    
    print(f"步长: {dt/60:.0f} 分钟")
    print(f"总时间: {total_hours} 小时")
    print(f"总步数: {total_steps}")
    
    # 创建热力学模型
    params = LedThermalParams(
        base_ambient_temp=25.0,
        thermal_resistance=0.05,
        time_constant_s=7.5,
        thermal_mass=150.0,
        model_type="thermal",
        solar_threshold=1.4
    )
    
    model = ThermalModelManager(params)
    print(f"模型类型: {params.model_type}")
    print(f"基础环境温度: {params.base_ambient_temp}°C")
    print(f"Solar阈值: {params.solar_threshold}")
    
    # 测试场景1: 恒定功率
    print(f"\n=== 场景1: 恒定功率 50W ===")
    model.reset()
    
    time_hours = []
    temps_power = []
    powers = []
    
    for step in range(total_steps):
        time_hour = step * dt / 3600
        power = 50.0
        
        temp = model.step(power=power, dt=dt)
        
        time_hours.append(time_hour)
        temps_power.append(temp)
        powers.append(power)
        
        if step % 4 == 0:  # 每1小时打印一次
            print(f"时间: {time_hour:.1f}h, 功率: {power:.1f}W, 温度: {temp:.2f}°C")
    
    # 测试场景2: 基于Solar值的温度变化
    print(f"\n=== 场景2: 基于Solar值变化 ===")
    model.reset()
    
    temps_solar = []
    solar_values = []
    
    for step in range(total_steps):
        time_hour = step * dt / 3600
        
        # 模拟Solar值变化：前2小时升温，后2小时降温
        if time_hour < 2:
            solar = 2.0  # 高Solar值，升温
        else:
            solar = 0.8  # 低Solar值，降温
        
        temp = model.step(power=0.0, dt=dt, solar_vol=solar)
        
        temps_solar.append(temp)
        solar_values.append(solar)
        
        if step % 4 == 0:  # 每1小时打印一次
            phase = "升温" if solar > params.solar_threshold else "降温"
            print(f"时间: {time_hour:.1f}h, Solar: {solar:.2f}, 阶段: {phase}, 温度: {temp:.2f}°C")
    
    # 测试场景3: 功率+Solar组合
    print(f"\n=== 场景3: 功率+Solar组合 ===")
    model.reset()
    
    temps_combined = []
    
    for step in range(total_steps):
        time_hour = step * dt / 3600
        
        # 功率变化：正弦波
        power = 40.0 + 20.0 * np.sin(2 * np.pi * time_hour / 2)  # 2小时周期
        
        # Solar值变化
        if time_hour < 2:
            solar = 1.8
        else:
            solar = 1.0
        
        temp = model.step(power=power, dt=dt, solar_vol=solar)
        temps_combined.append(temp)
        
        if step % 4 == 0:  # 每1小时打印一次
            phase = "升温" if solar > params.solar_threshold else "降温"
            print(f"时间: {time_hour:.1f}h, 功率: {power:.1f}W, Solar: {solar:.2f}, 阶段: {phase}, 温度: {temp:.2f}°C")
    
    # 绘制结果
    plt.figure(figsize=(12, 8))
    
    # 子图1: 温度变化
    plt.subplot(2, 2, 1)
    plt.plot(time_hours, temps_power, 'b-', linewidth=2, label='恒定功率50W')
    plt.plot(time_hours, temps_solar, 'r-', linewidth=2, label='Solar值变化')
    plt.plot(time_hours, temps_combined, 'g-', linewidth=2, label='功率+Solar组合')
    plt.xlabel('时间 (小时)')
    plt.ylabel('温度 (°C)')
    plt.title('温度变化对比')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 子图2: 功率变化
    plt.subplot(2, 2, 2)
    plt.plot(time_hours, powers, 'b-', linewidth=2, label='恒定功率')
    plt.plot(time_hours, [40.0 + 20.0 * np.sin(2 * np.pi * t / 2) for t in time_hours], 
             'g-', linewidth=2, label='正弦波功率')
    plt.xlabel('时间 (小时)')
    plt.ylabel('功率 (W)')
    plt.title('功率变化')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 子图3: Solar值变化
    plt.subplot(2, 2, 3)
    plt.plot(time_hours, solar_values, 'r-', linewidth=2, label='Solar值')
    plt.axhline(y=params.solar_threshold, color='k', linestyle='--', alpha=0.7, label=f'阈值={params.solar_threshold}')
    plt.xlabel('时间 (小时)')
    plt.ylabel('Solar值')
    plt.title('Solar值变化')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 子图4: 温度变化率
    plt.subplot(2, 2, 4)
    temp_rates_power = np.diff(temps_power) / (dt/3600)  # °C/h
    temp_rates_solar = np.diff(temps_solar) / (dt/3600)
    temp_rates_combined = np.diff(temps_combined) / (dt/3600)
    
    plt.plot(time_hours[1:], temp_rates_power, 'b-', linewidth=2, label='恒定功率')
    plt.plot(time_hours[1:], temp_rates_solar, 'r-', linewidth=2, label='Solar值变化')
    plt.plot(time_hours[1:], temp_rates_combined, 'g-', linewidth=2, label='功率+Solar组合')
    plt.xlabel('时间 (小时)')
    plt.ylabel('温度变化率 (°C/h)')
    plt.title('温度变化率')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'thermal_demo_simple_15min.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存到: {save_path}")
    
    plt.show()
    
    # 打印结果摘要
    print(f"\n=== 结果摘要 ===")
    print(f"恒定功率场景:")
    print(f"  初始温度: {temps_power[0]:.2f}°C")
    print(f"  最终温度: {temps_power[-1]:.2f}°C")
    print(f"  温度变化: {temps_power[-1] - temps_power[0]:.2f}°C")
    
    print(f"Solar值变化场景:")
    print(f"  初始温度: {temps_solar[0]:.2f}°C")
    print(f"  最终温度: {temps_solar[-1]:.2f}°C")
    print(f"  温度变化: {temps_solar[-1] - temps_solar[0]:.2f}°C")
    
    print(f"功率+Solar组合场景:")
    print(f"  初始温度: {temps_combined[0]:.2f}°C")
    print(f"  最终温度: {temps_combined[-1]:.2f}°C")
    print(f"  温度变化: {temps_combined[-1] - temps_combined[0]:.2f}°C")

if __name__ == "__main__":
    simple_thermal_demo()
