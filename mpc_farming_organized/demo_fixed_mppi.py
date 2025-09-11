#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复后的MPPI系统演示
展示光合作用预测模型和MPPI控制器的集成
"""

import sys
import os
sys.path.append('.')

from core.mppi import LEDPlant, LEDMPPIController, LEDMPPISimulation
from core.mppi_api import mppi_next_ppfd

def demo_photosynthesis_prediction():
    """演示光合作用预测功能"""
    print("🌱 光合作用预测演示")
    print("=" * 40)
    
    plant = LEDPlant()
    print(f"使用预测模型: {plant.use_photo_model}")
    
    # 测试不同条件下的光合作用预测
    test_conditions = [
        (100, 400, 20, 0.5, "低光照，低温"),
        (300, 400, 25, 0.75, "中等光照，最适温度"),
        (500, 600, 28, 0.8, "高光照，高CO2"),
        (200, 400, 22, 0.83, "您提供的实际条件"),
    ]
    
    for ppfd, co2, temp, rb, desc in test_conditions:
        if plant.use_photo_model:
            pn = plant.get_photosynthesis_rate(ppfd, temp, co2, rb)
        else:
            pn = plant.simple_photosynthesis_model(ppfd, temp)
        
        print(f"{desc}:")
        print(f"  PPFD={ppfd}, CO2={co2}, T={temp}°C, R:B={rb}")
        print(f"  预测光合作用速率: {pn:.3f} μmol/m²/s")
        print()

def demo_mppi_control():
    """演示MPPI控制功能"""
    print("🎛️ MPPI控制演示")
    print("=" * 40)
    
    # 创建植物模型
    plant = LEDPlant(
        base_ambient_temp=22.0,
        max_ppfd=500.0,
        max_power=100.0,
        thermal_resistance=1.2,
        thermal_mass=5.0,
    )
    
    # 创建控制器
    controller = LEDMPPIController(
        plant=plant,
        horizon=8,
        num_samples=200,
        dt=1.0,
        temperature=0.5
    )
    
    # 配置控制器
    controller.set_weights(Q_photo=10.0, R_pwm=0.001, R_dpwm=0.05, R_power=0.01)
    controller.set_constraints(pwm_min=0.0, pwm_max=70.0, temp_min=20.0, temp_max=29.0)
    
    print("控制器配置完成")
    print(f"预测时域: {controller.horizon}")
    print(f"采样数量: {controller.num_samples}")
    print(f"温度约束: {controller.temp_min}-{controller.temp_max}°C")
    print(f"PWM约束: {controller.pwm_min}-{controller.pwm_max}%")
    print()
    
    # 运行短时间仿真
    simulation = LEDMPPISimulation(plant, controller)
    print("开始MPPI仿真...")
    results = simulation.run_simulation(duration=15, dt=1.0)
    
    # 显示结果摘要
    print("\n📊 仿真结果摘要:")
    print(f"最终温度: {results['temp'][-1]:.1f}°C")
    print(f"最终PPFD: {results['ppfd'][-1]:.1f} μmol/m²/s")
    print(f"最终PWM: {results['pwm'][-1]:.1f}%")
    print(f"最终光合作用速率: {results['photosynthesis'][-1]:.2f} μmol/m²/s")
    print(f"平均光合作用速率: {results['photosynthesis'].mean():.2f} μmol/m²/s")
    print(f"总光合作用: {results['photosynthesis'].sum():.1f} μmol/m²")
    
    # 检查约束满足情况
    temp_violations = ((results['temp'] < controller.temp_min) | 
                      (results['temp'] > controller.temp_max)).sum()
    temp_satisfaction = 100 * (1 - temp_violations / len(results['temp']))
    print(f"温度约束满足率: {temp_satisfaction:.1f}%")

def demo_api_usage():
    """演示API使用"""
    print("\n🔌 MPPI API演示")
    print("=" * 40)
    
    # 测试API函数
    test_cases = [
        (100, 25, 400, 60, "低光照条件"),
        (300, 25, 400, 60, "中等光照条件"),
        (500, 25, 400, 60, "高光照条件"),
    ]
    
    for current_ppfd, temp, co2, humidity, desc in test_cases:
        next_ppfd = mppi_next_ppfd(current_ppfd, temp, co2, humidity)
        print(f"{desc}:")
        print(f"  当前PPFD: {current_ppfd} μmol/m²/s")
        print(f"  建议PPFD: {next_ppfd:.1f} μmol/m²/s")
        print(f"  温度: {temp}°C, CO2: {co2} ppm")
        print()

def main():
    """主演示函数"""
    print("🚀 修复后的MPPI系统演示")
    print("=" * 50)
    print("本演示展示修复后的MPPI系统，现在可以正确使用")
    print("基于实际数据校准的光合作用预测模型。")
    print()
    
    try:
        # 演示光合作用预测
        demo_photosynthesis_prediction()
        
        # 演示MPPI控制
        demo_mppi_control()
        
        # 演示API使用
        demo_api_usage()
        
        print("✅ 所有演示完成！")
        print("\n💡 系统现在可以:")
        print("  - 正确加载和使用光合作用预测模型")
        print("  - 运行MPPI优化控制")
        print("  - 通过API接口进行实时控制")
        print("  - 最大化植物光合作用效率")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
