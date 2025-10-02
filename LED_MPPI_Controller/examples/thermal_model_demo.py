#!/usr/bin/env python3
"""
热力学模型演示脚本 - 15分钟步长

本脚本演示了LED热力学模型的使用，包括：
1. MLP模型和纯热力学模型的对比
2. 不同功率和Solar值下的温度响应
3. 15分钟步长的温度变化过程
4. 可视化温度变化曲线

作者: LED控制系统
日期: 2024
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from led import (
    LedThermalParams, 
    ThermalModelManager, 
    Led,
    create_model,
    create_default_params
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ThermalModelDemo:
    """热力学模型演示类"""
    
    def __init__(self):
        """初始化演示"""
        self.dt = 15 * 60  # 15分钟 = 900秒
        self.total_time_hours = 8  # 总演示时间8小时
        self.total_steps = int(self.total_time_hours * 3600 / self.dt)
        
        # 创建模型参数
        self.params_thermal = LedThermalParams(
            base_ambient_temp=25.0,
            thermal_resistance=0.05,
            time_constant_s=7.5,
            thermal_mass=150.0,
            model_type="thermal",
            solar_threshold=1.4
        )
        
        # 创建模型 - 只使用热力学模型
        self.thermal_model = ThermalModelManager(self.params_thermal)
        self.mlp_model = None  # 暂时禁用MLP模型
        
        print(f"热力学模型演示初始化完成")
        print(f"步长: {self.dt/60:.1f} 分钟")
        print(f"总时间: {self.total_time_hours} 小时")
        print(f"总步数: {self.total_steps}")
        print(f"模型类型: thermal (MLP模型暂时禁用)")
    
    def run_scenario(self, scenario_name: str, power_profile: List[float], 
                    solar_profile: List[float] = None) -> Dict:
        """运行特定场景的仿真"""
        print(f"\n=== 运行场景: {scenario_name} ===")
        
        # 重置模型
        self.thermal_model.reset()
        if self.mlp_model:
            self.mlp_model.reset()
        
        # 存储结果
        results = {
            'time_hours': [],
            'thermal_temps': [],
            'mlp_temps': [],
            'power_values': [],
            'solar_values': [],
            'scenario_name': scenario_name
        }
        
        # 运行仿真
        for step in range(self.total_steps):
            time_hour = step * self.dt / 3600
            
            # 获取当前功率和Solar值
            power = power_profile[min(step, len(power_profile)-1)]
            solar = solar_profile[min(step, len(solar_profile)-1)] if solar_profile else 1.4
            
            # 热力学模型步进
            thermal_temp = self.thermal_model.step(power=power, dt=self.dt, solar_vol=solar)
            
            # MLP模型步进（如果可用）
            if self.mlp_model:
                mlp_temp = self.mlp_model.step(power=power, dt=self.dt, solar_vol=solar)
            else:
                mlp_temp = thermal_temp  # 使用热力学模型结果
            
            # 记录结果
            results['time_hours'].append(time_hour)
            results['thermal_temps'].append(thermal_temp)
            results['mlp_temps'].append(mlp_temp)
            results['power_values'].append(power)
            results['solar_values'].append(solar)
            
            # 每2小时打印一次状态
            if step % 8 == 0:  # 8步 = 2小时
                print(f"时间: {time_hour:.1f}h, 功率: {power:.1f}W, Solar: {solar:.2f}")
                print(f"  热力学模型温度: {thermal_temp:.2f}°C")
                print(f"  MLP模型温度: {mlp_temp:.2f}°C")
        
        return results
    
    def create_power_profile(self, profile_type: str) -> List[float]:
        """创建功率配置文件"""
        if profile_type == "constant_high":
            return [100.0] * self.total_steps
        elif profile_type == "constant_medium":
            return [50.0] * self.total_steps
        elif profile_type == "constant_low":
            return [20.0] * self.total_steps
        elif profile_type == "step_up":
            # 前2小时低功率，后6小时高功率
            low_power_steps = int(2 * 3600 / self.dt)
            return [20.0] * low_power_steps + [80.0] * (self.total_steps - low_power_steps)
        elif profile_type == "step_down":
            # 前2小时高功率，后6小时低功率
            high_power_steps = int(2 * 3600 / self.dt)
            return [80.0] * high_power_steps + [20.0] * (self.total_steps - high_power_steps)
        elif profile_type == "sinusoidal":
            # 正弦波功率变化
            powers = []
            for step in range(self.total_steps):
                time_hour = step * self.dt / 3600
                power = 50.0 + 30.0 * np.sin(2 * np.pi * time_hour / 4)  # 4小时周期
                powers.append(max(10.0, min(100.0, power)))
            return powers
        else:
            return [50.0] * self.total_steps
    
    def create_solar_profile(self, profile_type: str) -> List[float]:
        """创建Solar值配置文件"""
        if profile_type == "constant_high":
            return [2.0] * self.total_steps
        elif profile_type == "constant_low":
            return [0.8] * self.total_steps
        elif profile_type == "day_night":
            # 模拟昼夜变化
            solar_values = []
            for step in range(self.total_steps):
                time_hour = step * self.dt / 3600
                if 6 <= time_hour <= 18:  # 白天
                    solar = 1.8 + 0.4 * np.sin(np.pi * (time_hour - 6) / 12)
                else:  # 夜晚
                    solar = 0.6
                solar_values.append(solar)
            return solar_values
        else:
            return [1.4] * self.total_steps
    
    def plot_results(self, results_list: List[Dict], save_path: str = None):
        """绘制结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('热力学模型演示 - 15分钟步长', fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, results in enumerate(results_list):
            color = colors[i % len(colors)]
            time_hours = results['time_hours']
            
            # 温度对比图
            axes[0, 0].plot(time_hours, results['thermal_temps'], 
                           color=color, linestyle='-', linewidth=2,
                           label=f"{results['scenario_name']} - 热力学模型")
            axes[0, 0].plot(time_hours, results['mlp_temps'], 
                           color=color, linestyle='--', linewidth=2,
                           label=f"{results['scenario_name']} - MLP模型")
            
            # 功率变化图
            axes[0, 1].plot(time_hours, results['power_values'], 
                           color=color, linewidth=2,
                           label=results['scenario_name'])
            
            # Solar值变化图
            axes[1, 0].plot(time_hours, results['solar_values'], 
                           color=color, linewidth=2,
                           label=results['scenario_name'])
            
            # 温度差图
            temp_diff = [mlp - thermal for mlp, thermal in 
                        zip(results['mlp_temps'], results['thermal_temps'])]
            axes[1, 1].plot(time_hours, temp_diff, 
                           color=color, linewidth=2,
                           label=results['scenario_name'])
        
        # 设置图表
        axes[0, 0].set_title('温度变化对比')
        axes[0, 0].set_xlabel('时间 (小时)')
        axes[0, 0].set_ylabel('温度 (°C)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        axes[0, 1].set_title('功率变化')
        axes[0, 1].set_xlabel('时间 (小时)')
        axes[0, 1].set_ylabel('功率 (W)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        axes[1, 0].set_title('Solar值变化')
        axes[1, 0].set_xlabel('时间 (小时)')
        axes[1, 0].set_ylabel('Solar值')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        axes[1, 1].set_title('MLP与热力学模型温度差')
        axes[1, 1].set_xlabel('时间 (小时)')
        axes[1, 1].set_ylabel('温度差 (°C)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def print_summary(self, results: Dict):
        """打印结果摘要"""
        print(f"\n=== {results['scenario_name']} 结果摘要 ===")
        
        thermal_temps = results['thermal_temps']
        mlp_temps = results['mlp_temps']
        
        print(f"热力学模型:")
        print(f"  初始温度: {thermal_temps[0]:.2f}°C")
        print(f"  最终温度: {thermal_temps[-1]:.2f}°C")
        print(f"  最高温度: {max(thermal_temps):.2f}°C")
        print(f"  最低温度: {min(thermal_temps):.2f}°C")
        print(f"  温度变化范围: {max(thermal_temps) - min(thermal_temps):.2f}°C")
        
        print(f"MLP模型:")
        print(f"  初始温度: {mlp_temps[0]:.2f}°C")
        print(f"  最终温度: {mlp_temps[-1]:.2f}°C")
        print(f"  最高温度: {max(mlp_temps):.2f}°C")
        print(f"  最低温度: {min(mlp_temps):.2f}°C")
        print(f"  温度变化范围: {max(mlp_temps) - min(mlp_temps):.2f}°C")
        
        # 计算平均温度差
        temp_diff = [mlp - thermal for mlp, thermal in zip(mlp_temps, thermal_temps)]
        avg_diff = np.mean(temp_diff)
        max_diff = max(temp_diff)
        min_diff = min(temp_diff)
        
        print(f"模型差异:")
        print(f"  平均温度差: {avg_diff:.3f}°C")
        print(f"  最大温度差: {max_diff:.3f}°C")
        print(f"  最小温度差: {min_diff:.3f}°C")

def main():
    """主函数"""
    print("热力学模型演示 - 15分钟步长")
    print("=" * 50)
    
    # 创建演示实例
    demo = ThermalModelDemo()
    
    # 定义测试场景
    scenarios = [
        {
            'name': '恒定高功率',
            'power_profile': demo.create_power_profile('constant_high'),
            'solar_profile': demo.create_solar_profile('constant_high')
        },
        {
            'name': '恒定低功率',
            'power_profile': demo.create_power_profile('constant_low'),
            'solar_profile': demo.create_solar_profile('constant_low')
        },
        {
            'name': '功率阶跃上升',
            'power_profile': demo.create_power_profile('step_up'),
            'solar_profile': demo.create_solar_profile('day_night')
        },
        {
            'name': '正弦波功率',
            'power_profile': demo.create_power_profile('sinusoidal'),
            'solar_profile': demo.create_solar_profile('day_night')
        }
    ]
    
    # 运行所有场景
    all_results = []
    for scenario in scenarios:
        results = demo.run_scenario(
            scenario['name'],
            scenario['power_profile'],
            scenario['solar_profile']
        )
        all_results.append(results)
        demo.print_summary(results)
    
    # 绘制结果
    save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'thermal_model_demo_15min.png')
    demo.plot_results(all_results, save_path)
    
    # 保存详细结果到JSON
    json_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'thermal_model_demo_15min.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存到: {json_path}")
    
    print("\n演示完成！")

if __name__ == "__main__":
    main()