#!/usr/bin/env python3
"""
LED Thermal Response Model Usage Example
======================================

This example demonstrates how to use the exported MLP and pure thermodynamic models for LED heating/cooling phases
"""

import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import importlib.util
import sys

# Dynamically import MLP class definitions to support pickle loading
def import_mlp_classes():
    """Dynamically import MLP class definitions"""
    # Import heating module
    heating_spec = importlib.util.spec_from_file_location(
        'heating_module', 
        '22-improved_thermal_constrained_mlp_heating.py'
    )
    heating_module = importlib.util.module_from_spec(heating_spec)
    heating_spec.loader.exec_module(heating_module)
    
    # Import cooling module
    cooling_spec = importlib.util.spec_from_file_location(
        'cooling_module', 
        '20-improved_thermal_constrained_mlp_cooling.py'
    )
    cooling_module = importlib.util.module_from_spec(cooling_spec)
    cooling_spec.loader.exec_module(cooling_module)
    
    return heating_module, cooling_module

# Import MLP classes
heating_module, cooling_module = import_mlp_classes()

# Set Chinese font support
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_models():
    """加载模型"""
    # 加载MLP模型
    with open('exported_models/heating_mlp_model.pkl', 'rb') as f:
        heating_mlp = pickle.load(f)
    
    with open('exported_models/cooling_mlp_model.pkl', 'rb') as f:
        cooling_mlp = pickle.load(f)
    
    # 加载纯热力学模型参数
    with open('exported_models/heating_thermal_model.json', 'r', encoding='utf-8') as f:
        heating_thermal = json.load(f)
    
    with open('exported_models/cooling_thermal_model.json', 'r', encoding='utf-8') as f:
        cooling_thermal = json.load(f)
    
    return heating_mlp, cooling_mlp, heating_thermal, cooling_thermal

def thermal_heating_response(t, a1_val, params):
    """开灯阶段纯热力学响应函数"""
    K1_base = params['parameters']['K1_base']
    tau1 = params['parameters']['tau1']
    K2_base = params['parameters']['K2_base']
    tau2 = params['parameters']['tau2']
    alpha_solar = params['parameters']['alpha_solar']
    a1_ref = params['a1_ref']
    
    solar_factor = 1 + alpha_solar * (a1_val - a1_ref)
    K1_solar = K1_base * solar_factor
    K2_solar = K2_base * solar_factor
    
    return K1_solar * (1 - np.exp(-t / tau1)) + K2_solar * (1 - np.exp(-t / tau2))

def thermal_cooling_response(t, a1_val, params):
    """关灯阶段纯热力学响应函数"""
    K1_base = params['parameters']['K1_base']
    tau1 = params['parameters']['tau1']
    K2_base = params['parameters']['K2_base']
    tau2 = params['parameters']['tau2']
    alpha_solar = params['parameters']['alpha_solar']
    a1_ref = params['a1_ref']
    
    solar_factor = 1 + alpha_solar * (a1_val - a1_ref)
    K1_solar = K1_base * solar_factor
    K2_solar = K2_base * solar_factor
    
    return K1_solar * np.exp(-t / tau1) + K2_solar * np.exp(-t / tau2)

def main():
    """主函数"""
    print("🔬 LED热响应模型使用示例")
    print("=" * 40)
    
    # 加载模型
    heating_mlp, cooling_mlp, heating_thermal, cooling_thermal = load_models()
    
    # 生成预测数据
    solar_values = [1.296, 1.418, 1.438, 1.541, 1.549]
    time_heating = np.linspace(0, 800, 200)
    time_cooling = np.linspace(0, 400, 200)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('LED热响应模型预测示例', fontsize=16, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(solar_values)))
    
    # 开灯MLP预测
    ax1 = axes[0, 0]
    ax1.set_title('开灯MLP模型预测', fontsize=14)
    ax1.set_xlabel('时间 (分钟)')
    ax1.set_ylabel('温差 (°C)')
    ax1.grid(True, alpha=0.3)
    
    for i, solar_val in enumerate(solar_values):
        pred = heating_mlp.predict(time_heating, np.full_like(time_heating, solar_val))
        ax1.plot(time_heating, pred, color=colors[i], linewidth=2, 
                label=f'Solar {solar_val:.3f}')
    
    ax1.legend()
    
    # 开灯纯热力学预测
    ax2 = axes[0, 1]
    ax2.set_title('开灯纯热力学模型预测', fontsize=14)
    ax2.set_xlabel('时间 (分钟)')
    ax2.set_ylabel('温差 (°C)')
    ax2.grid(True, alpha=0.3)
    
    for i, solar_val in enumerate(solar_values):
        pred = thermal_heating_response(time_heating, np.full_like(time_heating, solar_val), heating_thermal)
        ax2.plot(time_heating, pred, color=colors[i], linewidth=2, 
                label=f'Solar {solar_val:.3f}')
    
    ax2.legend()
    
    # 关灯MLP预测
    ax3 = axes[1, 0]
    ax3.set_title('关灯MLP模型预测', fontsize=14)
    ax3.set_xlabel('时间 (分钟)')
    ax3.set_ylabel('温差 (°C)')
    ax3.grid(True, alpha=0.3)
    
    for i, solar_val in enumerate(solar_values):
        pred = cooling_mlp.predict(time_cooling, np.full_like(time_cooling, solar_val))
        ax3.plot(time_cooling, pred, color=colors[i], linewidth=2, 
                label=f'Solar {solar_val:.3f}')
    
    ax3.legend()
    
    # 关灯纯热力学预测
    ax4 = axes[1, 1]
    ax4.set_title('关灯纯热力学模型预测', fontsize=14)
    ax4.set_xlabel('时间 (分钟)')
    ax4.set_ylabel('温差 (°C)')
    ax4.grid(True, alpha=0.3)
    
    for i, solar_val in enumerate(solar_values):
        pred = thermal_cooling_response(time_cooling, np.full_like(time_cooling, solar_val), cooling_thermal)
        ax4.plot(time_cooling, pred, color=colors[i], linewidth=2, 
                label=f'Solar {solar_val:.3f}')
    
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('LED热响应模型预测示例.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    print("\n✅ 模型使用示例完成！")
    print("   生成了预测示例图: LED热响应模型预测示例.png")

if __name__ == "__main__":
    main()
