#!/usr/bin/env python3
"""
MPPI单条路径流程演示（温度修正版）
================================

修复温度计算问题，使用更合理的时间步长和参数
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 直接导入led模块
from led import (
    LedThermalParams,
    ThermalModelManager,
    PWMtoPowerModel
)

class MPPIPathDemoFixed:
    """MPPI单条路径演示类（温度修正版）"""
    
    def __init__(self):
        self.setup_models()
    
    def setup_models(self):
        """设置模型"""
        print("🔧 设置MPPI演示模型...")
        
        # 创建简化的功率模型
        self.power_model = PWMtoPowerModel(include_intercept=True)
        
        # 创建热力学模型
        # 从examples目录向上查找Thermal目录
        current_dir = Path(__file__).parent
        model_dir = current_dir.parent / "Thermal" / "exported_models"
        
        self.thermal_params = LedThermalParams(
            base_ambient_temp=25.0,
            model_type="thermal",
            model_dir=str(model_dir)
        )
        
        self.thermal_model = ThermalModelManager(self.thermal_params)
        
        print("✅ 模型设置完成")
    
    def simulate_plant_step(self, solar_vol, current_temp, dt=300.0):  # 5分钟步长
        """模拟植物单步仿真"""
        # 简化的Solar Vol到PWM转换
        r_pwm = solar_vol * 50.0  # 简化转换
        b_pwm = solar_vol * 30.0
        total_pwm = r_pwm + b_pwm
        
        # 简化的功率计算
        power = total_pwm * 0.5  # 简化功率模型
        
        # 热力学步进 - 返回温度变化量
        delta_temp = self.thermal_model.step(
            power=power,
            dt=dt,
            solar_vol=solar_vol
        )
        
        # 计算绝对温度 = 当前温度 + 温度变化量
        new_temp = current_temp + delta_temp
        
        # 简化的光合作用计算
        photo = solar_vol * 10.0 * max(0, 1 - abs(new_temp - 25) * 0.01)
        
        return solar_vol, new_temp, power, photo
    
    def simulate_plant_predict(self, solar_vol_sequence, initial_temp, dt=300.0):
        """模拟植物预测"""
        temp = initial_temp
        solar_vols = []
        temps = []
        powers = []
        photos = []
        
        for solar_vol in solar_vol_sequence:
            sv, temp, power, photo = self.simulate_plant_step(solar_vol, temp, dt)
            solar_vols.append(sv)
            temps.append(temp)
            powers.append(power)
            photos.append(photo)
        
        return (
            np.array(solar_vols),
            np.array(temps),
            np.array(powers),
            np.array(photos)
        )
    
    def sample_control_sequences(self, mean_sequence, num_samples=6, u_std=0.1):
        """生成控制序列样本"""
        np.random.seed(42)  # 固定随机种子，便于演示
        noise = np.random.normal(0, u_std, (num_samples, len(mean_sequence)))
        samples = mean_sequence[np.newaxis, :] + noise
        return np.clip(samples, 0.8, 1.4)  # 限制在合理范围内
    
    def compute_cost(self, sample, current_temp, u_prev=1.0):
        """计算代价函数"""
        try:
            solar_vols, temps, powers, photos = self.simulate_plant_predict(
                sample, current_temp, dt=300.0
            )
            
            cost = 0.0
            
            # 1. 光合作用代价（负值，因为要最大化）
            cost -= 10.0 * np.sum(photos)
            
            # 2. 温度约束惩罚
            temp_min, temp_max = 20.0, 35.0
            temp_penalty = 1e3
            for temp in temps:
                if temp > temp_max:
                    violation = temp - temp_max
                    cost += temp_penalty * violation**2
                if temp < temp_min:
                    violation = temp_min - temp
                    cost += temp_penalty * violation**2
            
            # 3. 功率代价
            cost += 0.1 * np.sum(powers**2)
            
            # 4. 控制变化代价
            prev_u = u_prev
            for u in sample:
                du = u - prev_u
                cost += 0.1 * du**2
                prev_u = u
            
            return cost
            
        except Exception as e:
            return 1e10
    
    def demonstrate_single_path(self, mean_sequence, current_temp=25.0):
        """演示单条控制路径的完整流程"""
        print(f"\n🔥 MPPI单条路径演示（温度修正版）")
        print("=" * 60)
        print(f"当前温度: {current_temp:.2f}°C")
        print(f"参考序列: {mean_sequence}")
        print(f"预测时域: {len(mean_sequence)}")
        print(f"时间步长: 5分钟")
        
        # 步骤1: 生成控制序列样本
        print(f"\n📊 步骤1: 生成控制序列样本")
        print("-" * 40)
        
        samples = self.sample_control_sequences(mean_sequence, num_samples=6)
        print(f"生成了 {len(samples)} 个控制序列样本:")
        
        for i, sample in enumerate(samples):
            print(f"  样本{i+1}: {sample}")
        
        # 步骤2: 计算每个样本的代价
        print(f"\n💰 步骤2: 计算每个样本的代价")
        print("-" * 40)
        
        costs = []
        detailed_results = []
        
        for i, sample in enumerate(samples):
            cost = self.compute_cost(sample, current_temp)
            costs.append(cost)
            
            # 获取详细预测结果
            try:
                solar_vols, temps, powers, photos = self.simulate_plant_predict(
                    sample, current_temp, dt=300.0
                )
                detailed_results.append({
                    'sample': sample,
                    'cost': cost,
                    'solar_vols': solar_vols,
                    'temps': temps,
                    'powers': powers,
                    'photos': photos
                })
                
                print(f"  样本{i+1}: 代价={cost:.2f}")
                print(f"    Solar Vol: {solar_vols}")
                print(f"    温度: {temps}")
                print(f"    功率: {powers}")
                print(f"    光合作用: {photos}")
                
                # 分析控制量变化
                print(f"    控制量变化分析:")
                for j in range(len(sample)):
                    if j == 0:
                        delta_u = sample[j] - 1.0  # 假设前一个控制量为1.0
                        phase = "升温" if delta_u > 0 else "降温"
                        print(f"      步骤{j}: u0={sample[j]:.3f}, Δu={delta_u:.3f} ({phase})")
                    else:
                        delta_u = sample[j] - sample[j-1]
                        phase = "升温" if delta_u > 0 else "降温"
                        print(f"      步骤{j}: u0={sample[j]:.3f}, Δu={delta_u:.3f} ({phase})")
                print()
                
            except Exception as e:
                print(f"  样本{i+1}: 代价={cost:.2f} (预测失败: {e})")
        
        # 步骤3: 计算权重
        print(f"\n⚖️ 步骤3: 计算Softmax权重")
        print("-" * 40)
        
        costs = np.array(costs)
        costs = np.nan_to_num(costs, nan=1e10, posinf=1e10)
        
        temperature = 0.5
        min_cost = np.min(costs)
        exp_costs = np.exp(-(costs - min_cost) / temperature)
        weights = exp_costs / np.sum(exp_costs)
        
        print(f"最小代价: {min_cost:.2f}")
        print(f"温度参数: {temperature}")
        print(f"权重分布:")
        
        for i, (cost, weight) in enumerate(zip(costs, weights)):
            print(f"  样本{i+1}: 代价={cost:.2f}, 权重={weight:.4f}")
        
        # 步骤4: 计算最优控制序列
        print(f"\n🎯 步骤4: 计算最优控制序列")
        print("-" * 40)
        
        optimal_seq = np.sum(weights[:, np.newaxis] * samples, axis=0)
        optimal_seq = np.clip(optimal_seq, 0.8, 1.4)
        optimal_u = optimal_seq[0]
        
        print(f"加权平均序列: {optimal_seq}")
        print(f"最优控制量: {optimal_u:.3f}")
        
        # 步骤5: 执行最优控制
        print(f"\n🚀 步骤5: 执行最优控制")
        print("-" * 40)
        
        try:
            solar_vol, new_temp, power, photo = self.simulate_plant_step(
                solar_vol=optimal_u, 
                current_temp=current_temp,
                dt=300.0
            )
            
            print(f"执行结果:")
            print(f"  Solar Vol: {solar_vol:.3f}")
            print(f"  新温度: {new_temp:.2f}°C")
            print(f"  功率: {power:.2f}W")
            print(f"  光合作用: {photo:.2f}")
            
            # 分析控制量变化
            delta_u = optimal_u - 1.0  # 假设前一个控制量为1.0
            phase = "升温" if delta_u > 0 else "降温"
            print(f"  控制量变化: Δu={delta_u:.3f} ({phase})")
            
        except Exception as e:
            print(f"执行失败: {e}")
        
        return optimal_u, optimal_seq, costs, weights

def main():
    """主函数"""
    print("🔬 MPPI单条路径流程演示（温度修正版）")
    print("=" * 60)
    
    # 检查模型文件
    # 从examples目录向上查找Thermal目录
    current_dir = Path(__file__).parent
    model_dir = current_dir.parent / "Thermal" / "exported_models"
    
    required_files = [
        "heating_thermal_model.json",
        "cooling_thermal_model.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not (model_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少模型文件: {missing_files}")
        print(f"模型目录: {model_dir.absolute()}")
        print("请确保Thermal/exported_models目录中有必需的模型文件")
        return
    
    # 创建演示
    demo = MPPIPathDemoFixed()
    
    # 演示单条路径
    mean_sequence = np.array([1.0, 1.1, 1.2, 1.1, 1.0])
    optimal_u, optimal_seq, costs, weights = demo.demonstrate_single_path(mean_sequence)
    
    print(f"\n✅ 演示完成！")
    print(f"\n📋 关键要点:")
    print(f"1. 🔥 MPPI围绕mean_sequence生成随机样本")
    print(f"2. 🌡️ 热力学模型根据控制量变化选择升温/降温模型")
    print(f"3. 💰 代价函数包含多个组成部分")
    print(f"4. ⚖️ Softmax权重基于代价计算")
    print(f"5. 🎯 加权平均得到最优控制序列")
    print(f"\n🔧 温度修正:")
    print(f"- 使用5分钟时间步长（更符合模型参数）")
    print(f"- 正确计算绝对温度 = 当前温度 + 温度变化量")
    print(f"- 限制控制量在合理范围内")

if __name__ == "__main__":
    main()
