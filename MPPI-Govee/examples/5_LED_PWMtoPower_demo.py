#!/usr/bin/env python3
"""PWM-功率转换系统验证演示

本脚本验证led.py中模块5的PWM-功率转换系统功能：
1. PWMtoPowerModel模型加载和拟合
2. 功率预测功能验证
3. MPPI成本函数中的功率计算
4. 不同PWM设置的功率对比
5. 功率优化效果展示
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from led import (
    DEFAULT_CALIB_CSV,
    PWMtoPowerModel,
    PowerInterpolator,
    forward_step,
    FirstOrderThermalModel,
    LedThermalParams
)


def load_and_fit_power_model(csv_path: str) -> PWMtoPowerModel:
    """加载标定数据并拟合功率模型"""
    print("=" * 60)
    print("1. 加载标定数据并拟合PWM-功率模型")
    print("=" * 60)
    
    # 创建模型并拟合
    model = PWMtoPowerModel(include_intercept=True).fit(csv_path)
    
    print(f"标定数据文件: {csv_path}")
    print(f"可用比例键: {list(model.by_key.keys())}")
    print()
    
    # 显示整体模型系数
    if model.overall:
        print("整体模型系数:")
        print(f"  功率(W) = {model.overall.a:.4f} × 总PWM(%) + {model.overall.c:.2f}")
        print()
    
    # 显示各比例键的模型系数
    print("各比例键模型系数:")
    for key, line in model.by_key.items():
        print(f"  {key}: 功率(W) = {line.a:.4f} × 总PWM(%) + {line.c:.2f}")
    print()
    
    return model


def test_power_prediction(model: PWMtoPowerModel):
    """测试功率预测功能"""
    print("=" * 60)
    print("2. 测试功率预测功能")
    print("=" * 60)
    
    # 测试不同PWM设置的功率预测
    test_cases = [
        (50, 10, "5:1"),    # 红光50%, 蓝光10%, 总60%
        (60, 12, "5:1"),    # 红光60%, 蓝光12%, 总72%
        (70, 14, "5:1"),    # 红光70%, 蓝光14%, 总84%
        (40, 8, "5:1"),     # 红光40%, 蓝光8%, 总48%
    ]
    
    print("PWM设置 → 功率预测:")
    print("红光%  蓝光%  总PWM%  功率(W)  比例键")
    print("-" * 45)
    
    for r_pwm, b_pwm, key in test_cases:
        total_pwm = r_pwm + b_pwm
        power = model.predict(total_pwm=total_pwm, key=key)
        print(f"{r_pwm:4d}   {b_pwm:4d}   {total_pwm:5d}   {power:6.2f}   {key}")
    
    print()


def test_power_interpolator(csv_path: str):
    """测试功率插值器功能"""
    print("=" * 60)
    print("3. 测试功率插值器功能")
    print("=" * 60)
    
    # 创建插值器
    interp = PowerInterpolator.from_csv(csv_path)
    
    print(f"可用比例键: {list(interp.by_key.keys())}")
    print()
    
    # 测试插值预测
    test_pwm = 75.0
    key = "5:1"
    
    try:
        power = interp.predict_power(total_pwm=test_pwm, key=key)
        print(f"插值预测: 总PWM={test_pwm}%, 比例键={key} → 功率={power:.2f}W")
    except KeyError as e:
        print(f"插值预测失败: {e}")
    
    print()


def test_mppi_power_cost():
    """测试MPPI成本函数中的功率计算"""
    print("=" * 60)
    print("4. 测试MPPI成本函数中的功率计算")
    print("=" * 60)
    
    # 创建功率模型
    power_model = PWMtoPowerModel(include_intercept=True).fit(DEFAULT_CALIB_CSV)
    
    # 模拟MPPI采样序列
    pwm_sequences = [
        np.array([[50, 10], [55, 11], [60, 12]]),  # 序列1: 总PWM 60%, 66%, 72%
        np.array([[70, 14], [75, 15], [80, 16]]),  # 序列2: 总PWM 84%, 90%, 96%
        np.array([[40, 8], [45, 9], [50, 10]]),    # 序列3: 总PWM 48%, 54%, 60%
    ]
    
    # MPPI权重设置
    R_power = 0.01  # 功率权重
    
    print("PWM序列 → 功率预测 → 功率成本:")
    print("序列    总PWM%    功率(W)    功率成本")
    print("-" * 45)
    
    for i, pwm_seq in enumerate(pwm_sequences, 1):
        total_pwms = pwm_seq[:, 0] + pwm_seq[:, 1]
        powers = []
        
        for total_pwm in total_pwms:
            power = power_model.predict(total_pwm=total_pwm, key="5:1")
            powers.append(power)
        
        powers = np.array(powers)
        power_cost = np.sum(powers**2) * R_power
        
        print(f"序列{i}   {total_pwms[0]:3d}-{total_pwms[-1]:3d}    {powers[0]:6.2f}-{powers[-1]:6.2f}    {power_cost:8.4f}")
    
    print()


def test_forward_step_power():
    """测试forward_step函数中的功率计算"""
    print("=" * 60)
    print("5. 测试forward_step函数中的功率计算")
    print("=" * 60)
    
    # 创建模型
    power_model = PWMtoPowerModel(include_intercept=True).fit(DEFAULT_CALIB_CSV)
    thermal_model = FirstOrderThermalModel(
        params=LedThermalParams(
            base_ambient_temp=25.0,
            thermal_resistance=0.05,
            time_constant_s=7.5
        ),
        initial_temp=25.0
    )
    
    # 测试不同PWM设置
    test_cases = [
        (60, 12, "5:1"),    # 红光60%, 蓝光12%
        (70, 14, "5:1"),    # 红光70%, 蓝光14%
        (50, 10, "5:1"),    # 红光50%, 蓝光10%
    ]
    
    print("PWM设置 → forward_step输出:")
    print("红光%  蓝光%  功率(W)  发热功率(W)  温度(°C)")
    print("-" * 50)
    
    for r_pwm, b_pwm, key in test_cases:
        output = forward_step(
            thermal_model=thermal_model,
            r_pwm=r_pwm,
            b_pwm=b_pwm,
            dt=0.1,
            power_model=power_model,
            model_key=key,
            heat_scale=1.0
        )
        
        print(f"{r_pwm:4d}   {b_pwm:4d}   {output.power:6.2f}   {output.heat_power:8.2f}   {output.temp:7.2f}")
    
    print()


def create_power_comparison_plot(model: PWMtoPowerModel):
    """创建功率对比图"""
    print("=" * 60)
    print("6. 创建功率对比图")
    print("=" * 60)
    
    # 创建结果目录
    result_dir = Path(__file__).parent / "result" / "5_PWMtoPower"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备数据
    total_pwm_range = np.linspace(0, 100, 101)
    
    # 获取可用的比例键
    available_keys = list(model.by_key.keys())
    if not available_keys:
        print("没有可用的比例键，跳过绘图")
        return
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 子图1: 不同比例键的功率对比
    colors = plt.cm.tab10(np.linspace(0, 1, len(available_keys)))
    
    for i, key in enumerate(available_keys):
        powers = []
        for total_pwm in total_pwm_range:
            power = model.predict(total_pwm=total_pwm, key=key)
            powers.append(power)
        
        ax1.plot(total_pwm_range, powers, color=colors[i], label=f'{key}', linewidth=2)
    
    ax1.set_xlabel('Total PWM (%)')
    ax1.set_ylabel('Power (W)')
    ax1.set_title('Power Comparison for Different Ratio Keys')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 功率成本对比（模拟MPPI成本函数）
    R_power = 0.01
    power_costs = []
    
    for total_pwm in total_pwm_range:
        power = model.predict(total_pwm=total_pwm, key="5:1")
        power_cost = power**2 * R_power
        power_costs.append(power_cost)
    
    ax2.plot(total_pwm_range, power_costs, 'r-', linewidth=2, label='Power Cost (R_power=0.01)')
    ax2.set_xlabel('Total PWM (%)')
    ax2.set_ylabel('Power Cost')
    ax2.set_title('Power Cost in MPPI Cost Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = result_dir / "pwm_power_demo.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"功率对比图已保存到: {output_path}")
    print()


def power_calculator_demo(model: PWMtoPowerModel):
    """功率计算器演示（使用宏定义）"""
    print("=" * 60)
    print("7. 功率计算器演示")
    print("=" * 60)
    
    # 宏定义：测试用例
    test_cases = [
        (12, 9, "5:1"),    # 红光60%, 蓝光12%, 5:1比例
        (30, 10, "5:1"),    # 红光70%, 蓝光14%, 5:1比例
        (43, 14, "5:1"),    # 红光50%, 蓝光10%, 5:1比例
        (59, 19, "5:1"),     # 红光40%, 蓝光8%, 5:1比例
        (73, 26, "5:1"),    # 红光30%, 蓝光10%, 3:1比例
        (20, 20, "1:1"),    # 红光20%, 蓝光20%, 1:1比例
        (80, 20, "7:1"),    # 红光80%, 蓝光20%, 7:1比例
    ]
    
    print("测试用例功率计算结果:")
    print("-" * 60)
    print("红光%  蓝光%  总PWM%  红光功率  蓝光功率  总功率   红蓝比  PWM比例")
    print("-" * 60)
    
    for r_pwm, b_pwm, ratio_key in test_cases:
        try:
            total_pwm = r_pwm + b_pwm
            
            # 计算总功率
            total_power = model.predict(total_pwm=total_pwm, key=ratio_key)
            
            # 计算红蓝功率分配（基于PWM比例）
            if total_pwm > 0:
                r_power_ratio = r_pwm / total_pwm
                b_power_ratio = b_pwm / total_pwm
                r_power = total_power * r_power_ratio
                b_power = total_power * b_power_ratio
            else:
                r_power = b_power = 0.0
            
            # 计算PWM比例
            pwm_ratio = f"{r_pwm/b_pwm:.1f}:1" if b_pwm > 0 else "N/A"
            
            print(f"{r_pwm:4.0f}   {b_pwm:4.0f}   {total_pwm:5.0f}   {r_power:7.2f}   {b_power:7.2f}   {total_power:7.2f}   {ratio_key:>4}   {pwm_ratio:>6}")
            
        except Exception as e:
            print(f"{r_pwm:4.0f}   {b_pwm:4.0f}   {total_pwm:5.0f}   ERROR: {e}")
    
    print()
    
    # 功率分析
    print("功率分析:")
    print("-" * 30)
    
    # 计算不同PWM设置下的功率效率
    efficiency_cases = [
        (60, 12, "5:1"),
        (70, 14, "5:1"),
        (50, 10, "5:1"),
    ]
    
    print("PWM效率分析 (5:1比例):")
    print("总PWM%  总功率  功率效率(W/%)")
    print("-" * 35)
    
    for r_pwm, b_pwm, ratio_key in efficiency_cases:
        total_pwm = r_pwm + b_pwm
        total_power = model.predict(total_pwm=total_pwm, key=ratio_key)
        efficiency = total_power / total_pwm if total_pwm > 0 else 0
        
        print(f"{total_pwm:6.0f}   {total_power:6.2f}   {efficiency:8.3f}")
    
    print()


def main():
    """主函数"""
    print("PWM-功率转换系统验证演示")
    print("=" * 60)
    
    try:
        # 1. 加载和拟合功率模型
        model = load_and_fit_power_model(DEFAULT_CALIB_CSV)
        
        # 2. 测试功率预测功能
        test_power_prediction(model)
        
        # 3. 测试功率插值器功能
        test_power_interpolator(DEFAULT_CALIB_CSV)
        
        # 4. 测试MPPI成本函数中的功率计算
        test_mppi_power_cost()
        
        # 5. 测试forward_step函数中的功率计算
        test_forward_step_power()
        
        # 6. 创建功率对比图
        create_power_comparison_plot(model)
        
        # 7. 功率计算器演示
        power_calculator_demo(model)
        
        print("=" * 60)
        print("✅ PWM-功率转换系统验证完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
