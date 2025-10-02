#!/usr/bin/env python3
"""
MPPI动态mean_sequence演示
=======================

展示MPPI控制器中mean_sequence的动态更新过程：
1. 滚动时域更新
2. 控制量变化判断升温/降温
3. 完整的MPPI仿真循环
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mppi_v2 import LEDPlant, LEDMPPIController
from led import PWMtoPowerModel

def demonstrate_dynamic_mean_sequence():
    """演示动态mean_sequence的工作原理"""
    print("🔥 MPPI动态mean_sequence演示")
    print("=" * 50)
    
    # 创建简化的功率模型
    power_model = PWMtoPowerModel(include_intercept=True)
    
    # 创建LEDPlant
    plant = LEDPlant(
        base_ambient_temp=25.0,
        thermal_model_type="thermal",
        model_dir="Thermal/exported_models",
        power_model=power_model
    )
    
    # 创建MPPI控制器
    controller = LEDMPPIController(
        plant=plant,
        horizon=5,  # 预测时域
        num_samples=100,
        dt=60.0,
        temperature=0.5
    )
    
    print(f"MPPI控制器参数:")
    print(f"  预测时域: {controller.horizon}")
    print(f"  采样数量: {controller.num_samples}")
    print(f"  时间步长: {controller.dt}秒")
    
    # 🔥 模拟MPPI控制循环
    print(f"\n🔥 MPPI控制循环演示:")
    print("-" * 40)
    
    current_temp = 25.0
    mean_sequence = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # 初始参考序列
    
    for step in range(8):
        print(f"\n步骤 {step + 1}:")
        print(f"  当前温度: {current_temp:.2f}°C")
        print(f"  mean_sequence: {mean_sequence}")
        
        # 🔥 MPPI求解
        optimal_u, optimal_seq, success, cost, weights = controller.solve(
            current_temp=current_temp,
            mean_sequence=mean_sequence
        )
        
        print(f"  最优控制: {optimal_u:.3f}")
        print(f"  最优序列: {optimal_seq}")
        print(f"  求解代价: {cost:.2e}")
        
        # 🔥 执行控制
        solar_vol, new_temp, power, photo = plant.step(solar_vol=optimal_u, dt=controller.dt)
        
        # 🔥 滚动时域更新mean_sequence（关键！）
        if len(optimal_seq) > 1:
            # 去掉第一个元素，添加最后一个元素
            mean_sequence = np.concatenate([optimal_seq[1:], [optimal_seq[-1]]])
        else:
            mean_sequence = optimal_seq
        
        print(f"  执行结果: Solar={solar_vol:.3f}, 温度={new_temp:.2f}°C")
        print(f"  更新后mean_sequence: {mean_sequence}")
        
        # 更新状态
        current_temp = new_temp

def analyze_control_change_patterns():
    """分析控制量变化模式"""
    print(f"\n📊 控制量变化模式分析:")
    print("-" * 40)
    
    # 模拟不同的控制序列
    test_sequences = [
        {"name": "持续升温", "sequence": [1.0, 1.2, 1.4, 1.6, 1.8]},
        {"name": "持续降温", "sequence": [1.8, 1.6, 1.4, 1.2, 1.0]},
        {"name": "振荡控制", "sequence": [1.0, 1.5, 1.0, 1.5, 1.0]},
        {"name": "阶梯上升", "sequence": [1.0, 1.0, 1.5, 1.5, 2.0]},
        {"name": "平滑过渡", "sequence": [1.0, 1.1, 1.2, 1.1, 1.0]},
    ]
    
    for case in test_sequences:
        print(f"\n{case['name']}:")
        sequence = case['sequence']
        
        for i in range(len(sequence)):
            if i == 0:
                print(f"  步骤{i}: u0={sequence[i]:.1f} (初始)")
            else:
                delta_u = sequence[i] - sequence[i-1]
                phase = "升温" if delta_u > 0 else "降温" if delta_u < 0 else "保持"
                print(f"  步骤{i}: u0={sequence[i]:.1f}, Δu={delta_u:.1f} ({phase})")

def simulate_rolling_horizon():
    """模拟滚动时域更新"""
    print(f"\n🔄 滚动时域更新模拟:")
    print("-" * 40)
    
    # 初始mean_sequence
    mean_sequence = np.array([1.0, 1.1, 1.2, 1.1, 1.0])
    print(f"初始mean_sequence: {mean_sequence}")
    
    # 模拟5次MPPI迭代
    for iteration in range(5):
        print(f"\n迭代 {iteration + 1}:")
        
        # 模拟MPPI求解结果（随机生成用于演示）
        np.random.seed(iteration)
        optimal_seq = mean_sequence + np.random.normal(0, 0.1, len(mean_sequence))
        optimal_seq = np.clip(optimal_seq, 0.5, 2.0)
        
        print(f"  求解得到optimal_seq: {optimal_seq}")
        
        # 🔥 滚动时域更新
        if len(optimal_seq) > 1:
            new_mean_sequence = np.concatenate([optimal_seq[1:], [optimal_seq[-1]]])
        else:
            new_mean_sequence = optimal_seq
        
        print(f"  更新后mean_sequence: {new_mean_sequence}")
        
        # 分析变化
        changes = new_mean_sequence - mean_sequence
        print(f"  变化量: {changes}")
        
        mean_sequence = new_mean_sequence

def create_mppi_simulation_example():
    """创建完整的MPPI仿真示例"""
    print(f"\n🎯 完整MPPI仿真示例:")
    print("-" * 40)
    
    # 这里可以添加完整的MPPI仿真代码
    # 类似于 mppi-power.py 中的 LEDMPPISimulation 类
    
    print("""
    # 完整的MPPI仿真流程：
    
    1. 初始化:
       - LEDPlant: 热力学模型 + 功率模型
       - LEDMPPIController: MPPI参数设置
       - mean_sequence: 初始参考序列
    
    2. 控制循环:
       for each time_step:
           # MPPI求解
           optimal_u, optimal_seq = controller.solve(
               current_temp, mean_sequence
           )
           
           # 执行控制
           solar_vol, temp, power, photo = plant.step(optimal_u)
           
           # 🔥 滚动时域更新
           mean_sequence = roll_horizon(optimal_seq)
           
           # 更新状态
           current_temp = temp
    
    3. 关键特性:
       - mean_sequence动态更新
       - 基于控制量变化判断升温/降温
       - 滚动时域优化
    """)

def main():
    """主函数"""
    print("🔬 MPPI动态mean_sequence完整演示")
    print("=" * 60)
    
    # 检查模型文件
    model_dir = Path("Thermal/exported_models")
    required_files = [
        "heating_mlp_model.pkl",
        "cooling_mlp_model.pkl", 
        "heating_thermal_model.json",
        "cooling_thermal_model.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not (model_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少模型文件: {missing_files}")
        print("请确保Thermal/exported_models目录中有所有必需的模型文件")
        return
    
    # 运行演示
    demonstrate_dynamic_mean_sequence()
    analyze_control_change_patterns()
    simulate_rolling_horizon()
    create_mppi_simulation_example()
    
    print("\n✅ 演示完成！")
    print("\n📋 关键要点:")
    print("1. 🔄 mean_sequence是动态的，每步都会滚动更新")
    print("2. 🎯 基于u0-u1变化判断升温/降温阶段")
    print("3. 🔥 热力学模型自动选择正确的模型")
    print("4. 📊 MPPI围绕动态mean_sequence进行优化")

if __name__ == "__main__":
    main()
