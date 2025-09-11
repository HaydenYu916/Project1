#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红蓝LED系统创建工具
根据您的实验数据创建LED控制系统

使用方法:
1. 在 USER_DATA 列表中输入您的实验数据
2. 运行脚本进行MPPI优化
3. 获得红色和蓝色LED的最优PWM控制序列
"""

import numpy as np
from led_rb_control import RedBlueDataParser, RedBlueLEDModel
from mppi_rb_control import RedBlueLEDPlant, RedBlueMPPIController

# ================== 在此输入您的实验数据 ==================
# 格式: "红蓝比:PPFD-红色PWM:蓝色PWM"
# 例如: "1:1-100-9:15" 表示红蓝比1:1，PPFD=100，红色PWM=9%，蓝色PWM=15%

USER_DATA = [
    # 请在此处替换为您的真实实验数据
    "1:1-100-9:15",
    "1:1-200-15:25", 
    "1:1-300-26:51",
    "1:1-400-35:62",
    "1:1-500-43:88",
    "1:2-100-13:25",
    "1:2-200-26:52",
    "1:2-300-39:78",
    "1:2-400-52:105",
    "1:2-500-67:138",
    "1:3-100-16:35",
    "1:3-200-29:70",
    "1:3-300-43:115",
    "1:3-400-58:155",
    "1:3-500-71:200",
    # 在此添加更多数据...
]

# =================== 优化参数设置 ===================
SIMULATION_DURATION = 120  # 仿真时长(秒)
MPPI_SAMPLES = 500         # MPPI采样数量
HORIZON = 10               # 预测视界
TARGET_PPFD = 400          # 目标PPFD值

def create_led_system():
    """创建基于用户数据的LED系统"""
    
    print("🔧 创建红蓝LED控制系统")
    print("=" * 50)
    
    # 1. 解析用户数据
    print("📊 解析实验数据...")
    parser = RedBlueDataParser()
    df = parser.load_data_from_list(USER_DATA)
    
    print(f"✅ 成功解析 {len(df)} 个数据点")
    print("\n前5个数据点:")
    print(df.head())
    
    # 2. 创建插值器
    print("\n🧠 创建PPFD插值模型...")
    interpolator = parser.get_interpolator()
    
    # 3. 创建LED植物模型
    print("🌱 创建LED植物模型...")
    plant = RedBlueLEDPlant(interpolator, max_power=100.0)
    
    # 4. 创建MPPI控制器
    print("🎮 创建MPPI控制器...")
    controller = RedBlueMPPIController(
        plant, 
        horizon=HORIZON, 
        num_samples=MPPI_SAMPLES,
        dt=1.0
    )
    
    # 调整权重以优化光合作用
    controller.Q_photo = 15.0  # 增加光合作用权重
    controller.R_power = 0.05  # 减少功耗惩罚
    
    return plant, controller, df

def run_optimization():
    """运行LED优化控制"""
    
    # 创建系统
    plant, controller, data_df = create_led_system()
    
    print(f"\n🚀 开始 {SIMULATION_DURATION} 秒优化仿真")
    print("=" * 60)
    
    # 仿真参数
    dt = 1.0
    steps = int(SIMULATION_DURATION / dt)
    
    # 数据存储
    results = {
        'time': [],
        'red_pwm': [],
        'blue_pwm': [],
        'ppfd': [],
        'temp': [],
        'rb_ratio': [],
        'photosynthesis': [],
        'power': []
    }
    
    # 初始化
    plant.ambient_temp = 25.0
    mean_sequence = np.ones((controller.horizon, 2)) * 20.0  # [红PWM, 蓝PWM]
    
    print("时间 | 红PWM | 蓝PWM | PPFD | 温度 | R:B | 光合速率 | 功耗")
    print("-" * 70)
    
    # 仿真循环
    for k in range(steps):
        current_time = k * dt
        
        # MPPI优化
        red_pwm, blue_pwm, sequence, success, cost = controller.solve(
            plant.ambient_temp, mean_sequence
        )
        
        # 更新均值序列
        if len(sequence) > 1:
            mean_sequence = np.vstack([sequence[1:], sequence[-1:]])
        
        # 应用控制
        ppfd, temp, power, rb_ratio, photo_rate = plant.step(red_pwm, blue_pwm, dt)
        
        # 存储结果
        results['time'].append(current_time)
        results['red_pwm'].append(red_pwm)
        results['blue_pwm'].append(blue_pwm)
        results['ppfd'].append(ppfd)
        results['temp'].append(temp)
        results['rb_ratio'].append(rb_ratio)
        results['photosynthesis'].append(photo_rate)
        results['power'].append(power)
        
        # 定期打印进度
        if k % 10 == 0:
            print(f"{current_time:4.0f} | {red_pwm:5.1f} | {blue_pwm:5.1f} | "
                  f"{ppfd:4.0f} | {temp:4.1f} | {rb_ratio:4.2f} | "
                  f"{photo_rate:6.2f} | {power:5.1f}W")
    
    # 计算性能指标
    avg_photosynthesis = np.mean(results['photosynthesis'])
    total_photosynthesis = np.sum(results['photosynthesis'])
    avg_power = np.mean(results['power'])
    energy_efficiency = avg_photosynthesis / max(avg_power, 0.1)
    
    print(f"\n🎯 优化结果:")
    print(f"平均光合速率: {avg_photosynthesis:.2f} μmol/m²/s")
    print(f"总光合产量: {total_photosynthesis:.1f} μmol/m²/s·s")
    print(f"平均功耗: {avg_power:.1f}W")
    print(f"能效比: {energy_efficiency:.3f} (光合/功耗)")
    print(f"温度范围: {np.min(results['temp']):.1f}°C - {np.max(results['temp']):.1f}°C")
    
    return results

def get_optimal_control_table(results):
    """生成最优控制参数表"""
    
    print(f"\n📋 最优控制参数表:")
    print("=" * 60)
    print("时间(s) | 红PWM(%) | 蓝PWM(%) | PPFD | R:B比 | 光合速率")
    print("-" * 60)
    
    # 每10秒显示一次
    for i in range(0, len(results['time']), 10):
        t = results['time'][i]
        r_pwm = results['red_pwm'][i]
        b_pwm = results['blue_pwm'][i]
        ppfd = results['ppfd'][i]
        rb = results['rb_ratio'][i]
        pn = results['photosynthesis'][i]
        
        print(f"{t:6.0f} | {r_pwm:8.1f} | {b_pwm:8.1f} | {ppfd:4.0f} | "
              f"{rb:4.2f} | {pn:8.2f}")

def recommend_led_settings():
    """推荐LED设置"""
    
    results = run_optimization()
    get_optimal_control_table(results)
    
    # 找到最佳设置点
    best_idx = np.argmax(results['photosynthesis'])
    
    print(f"\n💡 推荐LED设置 (最高光合速率时刻):")
    print("=" * 50)
    print(f"红色LED PWM: {results['red_pwm'][best_idx]:.1f}%")
    print(f"蓝色LED PWM: {results['blue_pwm'][best_idx]:.1f}%")
    print(f"预期PPFD: {results['ppfd'][best_idx]:.0f} μmol/m²/s")
    print(f"红蓝比: {results['rb_ratio'][best_idx]:.2f}")
    print(f"预期光合速率: {results['photosynthesis'][best_idx]:.2f} μmol/m²/s")
    print(f"功耗: {results['power'][best_idx]:.1f}W")
    
    return results

if __name__ == "__main__":
    print("🌱 红蓝LED智能控制系统")
    print("基于实验数据的光合作用优化")
    print("=" * 60)
    
    # 运行优化并获得推荐设置
    optimal_results = recommend_led_settings()
    
    print(f"\n✅ 优化完成！")
    print(f"💾 您可以根据推荐设置配置您的LED系统")

