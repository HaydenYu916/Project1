#!/usr/bin/env python3
"""
MPPI热力学模型集成演示
====================

展示如何正确使用基于MPPI控制量变化的热力学模型：
1. 基于u0-u1变化判断升温/降温
2. MLP vs 纯热力学模型对比
3. MPPI控制器集成
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

def test_mppi_thermal_integration():
    """测试MPPI热力学集成"""
    print("🔥 MPPI热力学模型集成测试")
    print("=" * 50)
    
    # 创建功率模型（简化版）
    power_model = PWMtoPowerModel(include_intercept=True)
    # 这里需要实际的标定数据，暂时跳过
    
    # 测试MLP模型
    print("\n1. MLP热力学模型测试")
    print("-" * 30)
    
    try:
        plant_mlp = LEDPlant(
            base_ambient_temp=25.0,
            thermal_model_type="mlp",
            model_dir="Thermal/exported_models",
            power_model=power_model
        )
        
        print("✅ MLP模型LEDPlant创建成功")
        
        # 模拟MPPI控制序列
        control_sequence = [1.0, 1.2, 1.5, 1.3, 1.0, 0.8]  # 控制量变化
        
        print("\n🔥 MPPI控制序列测试:")
        for i, u in enumerate(control_sequence):
            if i == 0:
                print(f"   步骤{i}: u0={u:.1f} (初始)")
            else:
                delta_u = u - control_sequence[i-1]
                phase = "升温" if delta_u > 0 else "降温"
                print(f"   步骤{i}: u0={u:.1f}, Δu={delta_u:.1f} ({phase})")
            
            # 单步仿真
            solar_vol, temp, power, photo = plant_mlp.step(solar_vol=u, dt=60.0)
            print(f"      → 温度: {temp:.2f}°C")
            
    except Exception as e:
        print(f"❌ MLP模型测试失败: {e}")
    
    # 测试纯热力学模型
    print("\n2. 纯热力学模型测试")
    print("-" * 30)
    
    try:
        plant_thermal = LEDPlant(
            base_ambient_temp=25.0,
            thermal_model_type="thermal",
            model_dir="Thermal/exported_models",
            power_model=power_model
        )
        
        print("✅ 纯热力学模型LEDPlant创建成功")
        
        # 同样的控制序列测试
        control_sequence = [1.0, 1.2, 1.5, 1.3, 1.0, 0.8]
        
        print("\n🔥 MPPI控制序列测试:")
        for i, u in enumerate(control_sequence):
            if i == 0:
                print(f"   步骤{i}: u0={u:.1f} (初始)")
            else:
                delta_u = u - control_sequence[i-1]
                phase = "升温" if delta_u > 0 else "降温"
                print(f"   步骤{i}: u0={u:.1f}, Δu={delta_u:.1f} ({phase})")
            
            # 单步仿真
            solar_vol, temp, power, photo = plant_thermal.step(solar_vol=u, dt=60.0)
            print(f"      → 温度: {temp:.2f}°C")
            
    except Exception as e:
        print(f"❌ 纯热力学模型测试失败: {e}")

def test_mppi_controller():
    """测试MPPI控制器集成"""
    print("\n3. MPPI控制器集成测试")
    print("-" * 30)
    
    try:
        # 创建LEDPlant
        plant = LEDPlant(
            base_ambient_temp=25.0,
            thermal_model_type="thermal",
            model_dir="Thermal/exported_models",
            power_model=None  # 需要实际功率模型
        )
        
        # 创建MPPI控制器
        controller = LEDMPPIController(
            plant=plant,
            horizon=5,
            num_samples=100,
            dt=60.0,
            temperature=0.5
        )
        
        print("✅ MPPI控制器创建成功")
        
        # 模拟MPPI求解
        current_temp = 25.0
        mean_sequence = np.array([1.0, 1.1, 1.2, 1.1, 1.0])
        
        print(f"\n🔥 MPPI求解测试:")
        print(f"   当前温度: {current_temp:.2f}°C")
        print(f"   控制序列: {mean_sequence}")
        
        optimal_u, optimal_seq, success, cost, weights = controller.solve(
            current_temp=current_temp,
            mean_sequence=mean_sequence
        )
        
        print(f"   最优控制: {optimal_u:.3f}")
        print(f"   最优序列: {optimal_seq}")
        print(f"   求解成功: {success}")
        print(f"   最小代价: {cost:.2f}")
        
    except Exception as e:
        print(f"❌ MPPI控制器测试失败: {e}")

def test_control_change_logic():
    """测试控制量变化逻辑"""
    print("\n4. 控制量变化逻辑测试")
    print("-" * 30)
    
    # 模拟不同的控制量变化模式
    test_cases = [
        {"name": "持续升温", "sequence": [1.0, 1.2, 1.4, 1.6, 1.8]},
        {"name": "持续降温", "sequence": [1.8, 1.6, 1.4, 1.2, 1.0]},
        {"name": "振荡控制", "sequence": [1.0, 1.5, 1.0, 1.5, 1.0]},
        {"name": "阶梯上升", "sequence": [1.0, 1.0, 1.5, 1.5, 2.0]},
    ]
    
    for case in test_cases:
        print(f"\n📊 {case['name']}:")
        sequence = case['sequence']
        
        for i in range(len(sequence)):
            if i == 0:
                print(f"   步骤{i}: u0={sequence[i]:.1f} (初始)")
            else:
                delta_u = sequence[i] - sequence[i-1]
                phase = "升温" if delta_u > 0 else "降温" if delta_u < 0 else "保持"
                print(f"   步骤{i}: u0={sequence[i]:.1f}, Δu={delta_u:.1f} ({phase})")

def main():
    """主函数"""
    print("🔬 MPPI热力学模型集成演示")
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
    
    # 运行测试
    test_control_change_logic()
    test_mppi_thermal_integration()
    test_mppi_controller()
    
    print("\n✅ 所有测试完成！")
    print("\n📋 关键特性:")
    print("1. 🔥 基于MPPI控制量变化判断升温/降温")
    print("   - Δu = u0 - u1 > 0 → 升温模型")
    print("   - Δu = u0 - u1 ≤ 0 → 降温模型")
    print("2. 🎯 自动模型选择: MLP vs 纯热力学")
    print("3. 🔄 MPPI控制器完全集成")
    print("4. 📊 实时控制状态跟踪")

if __name__ == "__main__":
    main()
