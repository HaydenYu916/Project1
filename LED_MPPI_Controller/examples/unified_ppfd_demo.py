#!/usr/bin/env python3
"""
统一PPFD热力学模型使用示例

本示例展示如何使用新的统一PPFD温度差模型：
- 直接使用PPFD值进行温度预测
- 高精度拟合（R² = 0.9254）
- 支持连续时间预测
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from led import (
    Led, 
    LedThermalParams, 
    UnifiedPPFDThermalModel,
    PWMtoPPFDModel,
    PWMtoPowerModel,
    DEFAULT_CALIB_CSV
)


def demo_unified_ppfd_model():
    """演示统一PPFD模型的基本使用"""
    print("=== 统一PPFD热力学模型演示 ===\n")
    
    # 1. 创建统一PPFD模型
    params = LedThermalParams(base_ambient_temp=25.0)
    led = Led(model_type="unified_ppfd", params=params, initial_temp=25.0)
    
    print(f"模型类型: {led.model.__class__.__name__}")
    print(f"是否为PPFD模型: {led.is_ppfd_model}")
    print(f"初始温度: {led.temperature:.2f}°C")
    print()
    
    # 2. 测试不同PPFD值的稳态温度
    print("稳态温度预测:")
    ppfd_values = [0, 100, 200, 300, 400, 500]
    for ppfd in ppfd_values:
        target_temp = led.target_temperature(ppfd)
        print(f"PPFD {ppfd:3d} μmol/m²/s → 稳态温度: {target_temp:.2f}°C")
    print()
    
    # 3. 时间序列仿真
    print("时间序列仿真 (PPFD = 300 μmol/m²/s):")
    dt = 1.0  # 1秒时间步长
    total_time = 60.0  # 60秒
    times = np.arange(0, total_time + dt, dt)
    temperatures = []
    
    led.reset(25.0)  # 重置到环境温度
    target_ppfd = 300.0
    
    for t in times:
        temp = led.step_with_ppfd(target_ppfd, dt)
        temperatures.append(temp)
        if t % 10 == 0:  # 每10秒打印一次
            print(f"t = {t:3.0f}s: T = {temp:.2f}°C")
    
    print(f"最终温度: {temperatures[-1]:.2f}°C")
    print(f"稳态温度: {led.target_temperature(target_ppfd):.2f}°C")
    print()
    
    # 4. 模型状态信息
    print("模型状态信息:")
    model_info = led.get_model_info()
    for key, value in model_info.items():
        print(f"{key}: {value:.2f}")
    print()
    
    return times, temperatures, target_ppfd


def demo_ppfd_step_change():
    """演示PPFD阶跃变化"""
    print("=== PPFD阶跃变化演示 ===\n")
    
    led = Led(model_type="unified_ppfd", initial_temp=25.0)
    
    # 定义PPFD变化序列
    time_segments = [
        (0, 30, 0),      # 0-30s: PPFD = 0
        (30, 60, 200),   # 30-60s: PPFD = 200
        (60, 90, 400),   # 60-90s: PPFD = 400
        (90, 120, 0),    # 90-120s: PPFD = 0
    ]
    
    dt = 1.0
    times = []
    temperatures = []
    ppfd_values = []
    
    led.reset(25.0)
    
    for start_time, end_time, ppfd in time_segments:
        for t in np.arange(start_time, end_time, dt):
            temp = led.step_with_ppfd(ppfd, dt)
            times.append(t)
            temperatures.append(temp)
            ppfd_values.append(ppfd)
    
    print("阶跃变化结果:")
    print("时间(s)  PPFD(μmol/m²/s)  温度(°C)")
    print("-" * 40)
    for i in range(0, len(times), 10):  # 每10秒打印一次
        print(f"{times[i]:6.0f}  {ppfd_values[i]:12.0f}  {temperatures[i]:8.2f}")
    
    return times, temperatures, ppfd_values


def plot_results(times1, temps1, ppfd1, times2, temps2, ppfd2):
    """绘制结果图表"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 图1: 单PPFD值的时间响应
    ax1.plot(times1, temps1, 'b-', linewidth=2, label=f'PPFD = {ppfd1} μmol/m²/s')
    ax1.axhline(y=25.0, color='r', linestyle='--', alpha=0.7, label='环境温度')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('温度 (°C)')
    ax1.set_title('统一PPFD模型 - 单PPFD值时间响应')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 图2: PPFD阶跃变化响应
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(times2, temps2, 'b-', linewidth=2, label='温度')
    line2 = ax2_twin.plot(times2, ppfd2, 'r-', linewidth=2, label='PPFD')
    
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('温度 (°C)', color='b')
    ax2_twin.set_ylabel('PPFD (μmol/m²/s)', color='r')
    ax2.set_title('统一PPFD模型 - PPFD阶跃变化响应')
    ax2.grid(True, alpha=0.3)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('unified_ppfd_demo.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存为: unified_ppfd_demo.png")


def demo_with_calibration_data():
    """使用标定数据演示完整工作流程"""
    print("=== 使用标定数据演示 ===\n")
    
    # 检查标定数据文件是否存在
    calib_path = DEFAULT_CALIB_CSV
    if not os.path.exists(calib_path):
        print(f"标定数据文件不存在: {calib_path}")
        print("跳过标定数据演示")
        return
    
    try:
        # 加载PPFD和功率模型
        ppfd_model = PWMtoPPFDModel().fit(calib_path)
        power_model = PWMtoPowerModel().fit(calib_path)
        
        print("标定模型加载成功")
        print(f"可用标签: {ppfd_model.list_labels()}")
        print()
        
        # 创建LED实例
        led = Led(model_type="unified_ppfd", initial_temp=25.0)
        
        # 测试特定PWM设定
        test_pwm = [(50, 30), (70, 50), (90, 70)]  # (R_PWM, B_PWM)
        
        print("PWM → PPFD → 温度预测:")
        print("R_PWM  B_PWM  PPFD(μmol/m²/s)  稳态温度(°C)")
        print("-" * 50)
        
        for r_pwm, b_pwm in test_pwm:
            try:
                # 预测PPFD
                ppfd = ppfd_model.predict(r_pwm=r_pwm, b_pwm=b_pwm, key="5:1")
                # 预测稳态温度
                steady_temp = led.target_temperature(ppfd)
                print(f"{r_pwm:5.0f}  {b_pwm:5.0f}  {ppfd:12.1f}  {steady_temp:12.2f}")
            except Exception as e:
                print(f"{r_pwm:5.0f}  {b_pwm:5.0f}  预测失败: {e}")
        
    except Exception as e:
        print(f"标定数据演示失败: {e}")


if __name__ == "__main__":
    # 运行所有演示
    times1, temps1, ppfd1 = demo_unified_ppfd_model()
    times2, temps2, ppfd2 = demo_ppfd_step_change()
    demo_with_calibration_data()
    
    # 绘制结果
    try:
        plot_results(times1, temps1, ppfd1, times2, temps2, ppfd2)
    except ImportError:
        print("\n注意: matplotlib未安装，跳过图表绘制")
    
    print("\n=== 演示完成 ===")
    print("统一PPFD模型特点:")
    print("✓ 直接使用PPFD值进行温度预测")
    print("✓ 高精度拟合 (R² = 0.9254)")
    print("✓ 支持连续时间预测")
    print("✓ 物理意义明确的温度差模型")
    print("✓ 与现有PWM/功率模型兼容")
