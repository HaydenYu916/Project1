#!/usr/bin/env python3
"""PWM-PPFD转换系统验证演示

本脚本验证led.py中模块4的PWM-PPFD转换系统功能：
1. PWMtoPPFDModel模型加载和拟合
2. PPFD到PWM的正向预测功能验证
3. PWM到PPFD的反向预测功能验证
4. 不同比例键的模型对比
5. solve_pwm_for_target_ppfd函数演示
6. 模型拟合质量评估
7. 可视化展示
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
    PWMtoPPFDModel,
    PpfdModelCoeffs,
    solve_pwm_for_target_ppfd
)


def load_and_fit_ppfd_model(csv_path: str) -> PWMtoPPFDModel:
    """加载标定数据并拟合PPFD模型"""
    print("=" * 60)
    print("1. 加载标定数据并拟合PWM-PPFD模型")
    print("=" * 60)
    
    # 创建模型并拟合
    model = PWMtoPPFDModel().fit(csv_path)
    
    print(f"标定数据文件: {csv_path}")
    print(f"可用比例键: {list(model.by_label.keys())}")
    print()
    
    # 显示模型拟合摘要
    print("模型拟合摘要:")
    print("-" * 80)
    print("比例键    α(红光斜率)  β(红光截距)  γ(蓝光斜率)  δ(蓝光截距)  R²(红光)  R²(蓝光)")
    print("-" * 80)
    
    fit_summary = model.get_fit_summary()
    for label, coeffs in fit_summary.items():
        print(f"{label:>6}   {coeffs['alpha']:8.4f}   {coeffs['beta']:8.2f}   {coeffs['gamma']:8.4f}   {coeffs['delta']:8.2f}   {coeffs['r_squared_r']:6.3f}   {coeffs['r_squared_b']:6.3f}")
    
    print()
    
    return model


def test_ppfd_to_pwm_prediction(model: PWMtoPPFDModel):
    """测试PPFD到PWM的正向预测功能"""
    print("=" * 60)
    print("2. 测试PPFD到PWM的正向预测功能")
    print("=" * 60)
    
    # 测试不同PPFD值的PWM预测
    test_ppfd_values = [100, 200, 300, 400, 500, 600]
    test_ratio = "5:1"
    
    print(f"PPFD值 → PWM预测 (比例键: {test_ratio}):")
    print("PPFD   红光PWM  蓝光PWM  总PWM")
    print("-" * 35)
    
    for ppfd in test_ppfd_values:
        try:
            r_pwm, b_pwm = model.predict_pwm(ppfd=ppfd, label=test_ratio)
            total_pwm = r_pwm + b_pwm
            print(f"{ppfd:4d}   {r_pwm:7.1f}   {b_pwm:7.1f}   {total_pwm:5.1f}")
        except KeyError as e:
            print(f"{ppfd:4d}   错误: {e}")
    
    print()
    
    # 测试不同比例键的预测
    print("不同比例键的PPFD预测对比 (PPFD=300):")
    print("比例键   红光PWM  蓝光PWM  总PWM   红蓝比")
    print("-" * 45)
    
    for ratio_key in model.list_labels():
        try:
            r_pwm, b_pwm = model.predict_pwm(ppfd=300, label=ratio_key)
            total_pwm = r_pwm + b_pwm
            ratio = r_pwm / b_pwm if b_pwm > 0 else float('inf')
            print(f"{ratio_key:>6}   {r_pwm:7.1f}   {b_pwm:7.1f}   {total_pwm:5.1f}   {ratio:6.1f}")
        except Exception as e:
            print(f"{ratio_key:>6}   错误: {e}")
    
    print()


def test_pwm_to_ppfd_prediction(model: PWMtoPPFDModel):
    """测试PWM到PPFD的反向预测功能"""
    print("=" * 60)
    print("3. 测试PWM到PPFD的反向预测功能")
    print("=" * 60)
    
    # 测试不同PWM设置的PPFD预测
    test_cases = [
        (12, 9, "5:1"),    # 红光60%, 蓝光12%, 5:1比例
        (30, 10, "5:1"),    # 红光70%, 蓝光14%, 5:1比例
        (43, 14, "5:1"),    # 红光50%, 蓝光10%, 5:1比例
        (59, 19, "5:1"),     # 红光40%, 蓝光8%, 5:1比例
        (73, 26, "5:1"),    # 红光30%, 蓝光10%, 3:1比例
        (20, 20, "1:1"),    # 红光20%, 蓝光20%, 1:1比例
        (80, 20, "7:1"),    # 红光80%, 蓝光20%, 7:1比例
    ]
    
    print("PWM设置 → PPFD预测:")
    print("红光%  蓝光%  总PWM%  预测PPFD")
    print("-" * 35)
    
    for r_pwm, b_pwm, ratio_key in test_cases:
        try:
            total_pwm = r_pwm + b_pwm
            predicted_ppfd = model.predict(r_pwm=r_pwm, b_pwm=b_pwm, key=ratio_key)
            print(f"{r_pwm:4d}   {b_pwm:4d}   {total_pwm:5d}   {predicted_ppfd:8.1f}")
        except Exception as e:
            print(f"{r_pwm:4d}   {b_pwm:4d}   {total_pwm:5d}   错误: {e}")
    
    print()
    
    # 验证正向和反向预测的一致性
    print("正向和反向预测一致性验证:")
    print("目标PPFD  预测PWM     反向预测PPFD   误差")
    print("-" * 50)
    
    test_ppfd = 350
    ratio_key = "5:1"
    
    try:
        # 正向预测：PPFD → PWM
        r_pwm, b_pwm = model.predict_pwm(ppfd=test_ppfd, label=ratio_key)
        
        # 反向预测：PWM → PPFD
        predicted_ppfd = model.predict(r_pwm=r_pwm, b_pwm=b_pwm, key=ratio_key)
        
        error = abs(predicted_ppfd - test_ppfd)
        print(f"{test_ppfd:7d}   {r_pwm:5.1f},{b_pwm:4.1f}   {predicted_ppfd:10.1f}   {error:6.2f}")
        
    except Exception as e:
        print(f"验证失败: {e}")
    
    print()


def test_solve_pwm_function(model: PWMtoPPFDModel):
    """测试solve_pwm_for_target_ppfd函数"""
    print("=" * 60)
    print("4. 测试solve_pwm_for_target_ppfd函数")
    print("=" * 60)
    
    # 测试不同目标PPFD的PWM求解
    target_ppfd_values = [150, 250, 350, 450, 550]
    ratio_key = "5:1"
    
    print(f"目标PPFD → PWM求解 (比例键: {ratio_key}):")
    print("目标PPFD  红光PWM  蓝光PWM  总PWM")
    print("-" * 40)
    
    for target_ppfd in target_ppfd_values:
        try:
            r_pwm, b_pwm, total_pwm = solve_pwm_for_target_ppfd(
                model=model,
                target_ppfd=target_ppfd,
                label=ratio_key,
                pwm_clip=(0.0, 100.0),
                integer_output=True
            )
            print(f"{target_ppfd:8d}   {r_pwm:7d}   {b_pwm:7d}   {total_pwm:5d}")
        except Exception as e:
            print(f"{target_ppfd:8d}   错误: {e}")
    
    print()
    
    # 测试不同比例键的求解
    print("不同比例键的PPFD求解对比 (目标PPFD=400):")
    print("比例键   红光PWM  蓝光PWM  总PWM")
    print("-" * 35)
    
    for ratio_key in model.list_labels():
        try:
            r_pwm, b_pwm, total_pwm = solve_pwm_for_target_ppfd(
                model=model,
                target_ppfd=400,
                label=ratio_key,
                pwm_clip=(0.0, 100.0),
                integer_output=True
            )
            print(f"{ratio_key:>6}   {r_pwm:7d}   {b_pwm:7d}   {total_pwm:5d}")
        except Exception as e:
            print(f"{ratio_key:>6}   错误: {e}")
    
    print()


def test_model_quality_assessment(model: PWMtoPPFDModel):
    """测试模型拟合质量评估"""
    print("=" * 60)
    print("5. 模型拟合质量评估")
    print("=" * 60)
    
    print("模型拟合质量分析:")
    print("-" * 60)
    
    fit_summary = model.get_fit_summary()
    
    for label, coeffs in fit_summary.items():
        print(f"\n比例键 {label}:")
        print(f"  红光模型: R_PWM = {coeffs['alpha']:.4f} × PPFD + {coeffs['beta']:.2f}")
        print(f"  红光模型R² = {coeffs['r_squared_r']:.3f} ({'优秀' if coeffs['r_squared_r'] > 0.95 else '良好' if coeffs['r_squared_r'] > 0.90 else '一般' if coeffs['r_squared_r'] > 0.80 else '较差'})")
        print(f"  蓝光模型: B_PWM = {coeffs['gamma']:.4f} × PPFD + {coeffs['delta']:.2f}")
        print(f"  蓝光模型R² = {coeffs['r_squared_b']:.3f} ({'优秀' if coeffs['r_squared_b'] > 0.95 else '良好' if coeffs['r_squared_b'] > 0.90 else '一般' if coeffs['r_squared_b'] > 0.80 else '较差'})")
        
        # 计算平均拟合质量
        avg_r_squared = (coeffs['r_squared_r'] + coeffs['r_squared_b']) / 2
        print(f"  平均拟合质量: {avg_r_squared:.3f}")
    
    print()


def test_different_ratio_comparison(model: PWMtoPPFDModel):
    """测试不同比例键的对比"""
    print("=" * 60)
    print("6. 不同比例键的对比分析")
    print("=" * 60)
    
    # 比较不同比例键在相同PPFD下的PWM需求
    test_ppfd = 400
    
    print(f"相同PPFD({test_ppfd})下不同比例键的PWM需求对比:")
    print("比例键   红光PWM  蓝光PWM  总PWM   红蓝比   PWM效率")
    print("-" * 55)
    
    for ratio_key in model.list_labels():
        try:
            r_pwm, b_pwm = model.predict_pwm(ppfd=test_ppfd, label=ratio_key)
            total_pwm = r_pwm + b_pwm
            ratio = r_pwm / b_pwm if b_pwm > 0 else float('inf')
            efficiency = test_ppfd / total_pwm if total_pwm > 0 else 0  # PPFD per total PWM
            
            print(f"{ratio_key:>6}   {r_pwm:7.1f}   {b_pwm:7.1f}   {total_pwm:5.1f}   {ratio:6.1f}   {efficiency:8.2f}")
            
        except Exception as e:
            print(f"{ratio_key:>6}   错误: {e}")
    
    print()
    
    # 分析PWM效率
    print("PWM效率分析 (PPFD per Total PWM):")
    print("比例键   效率排序  特点")
    print("-" * 35)
    
    efficiency_data = []
    for ratio_key in model.list_labels():
        try:
            r_pwm, b_pwm = model.predict_pwm(ppfd=test_ppfd, label=ratio_key)
            total_pwm = r_pwm + b_pwm
            efficiency = test_ppfd / total_pwm if total_pwm > 0 else 0
            efficiency_data.append((ratio_key, efficiency))
        except:
            continue
    
    # 按效率排序
    efficiency_data.sort(key=lambda x: x[1], reverse=True)
    
    for i, (ratio_key, efficiency) in enumerate(efficiency_data, 1):
        if efficiency > 6.0:
            characteristic = "高效"
        elif efficiency > 5.0:
            characteristic = "中等"
        else:
            characteristic = "较低"
        
        print(f"{ratio_key:>6}   {i:>6}   {characteristic}")
    
    print()


def create_ppfd_model_visualization(model: PWMtoPPFDModel):
    """创建PPFD模型可视化图表"""
    print("=" * 60)
    print("7. 创建PPFD模型可视化图表")
    print("=" * 60)
    
    # 创建结果目录
    result_dir = Path(__file__).parent / "result" / "4_PWMtoPPFD"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取可用的比例键
    available_keys = model.list_labels()
    if not available_keys:
        print("没有可用的比例键，跳过绘图")
        return
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 准备数据
    ppfd_range = np.linspace(50, 600, 100)
    colors = plt.cm.tab10(np.linspace(0, 1, len(available_keys)))
    
    # 子图1: PPFD到PWM的预测曲线
    for i, key in enumerate(available_keys):
        try:
            r_pwms = []
            b_pwms = []
            
            for ppfd in ppfd_range:
                r_pwm, b_pwm = model.predict_pwm(ppfd=ppfd, label=key)
                r_pwms.append(r_pwm)
                b_pwms.append(b_pwm)
            
            ax1.plot(ppfd_range, r_pwms, color=colors[i], linestyle='-', linewidth=2, label=f'{key} (Red)')
            ax1.plot(ppfd_range, b_pwms, color=colors[i], linestyle='--', linewidth=2, label=f'{key} (Blue)')
            
        except Exception as e:
            print(f"绘图失败 {key}: {e}")
    
    ax1.set_xlabel('PPFD (μmol/m²/s)')
    ax1.set_ylabel('PWM (%)')
    ax1.set_title('PPFD to PWM Prediction Curves')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 不同比例键的总PWM对比
    for i, key in enumerate(available_keys):
        try:
            total_pwms = []
            
            for ppfd in ppfd_range:
                r_pwm, b_pwm = model.predict_pwm(ppfd=ppfd, label=key)
                total_pwms.append(r_pwm + b_pwm)
            
            ax2.plot(ppfd_range, total_pwms, color=colors[i], linewidth=2, label=f'{key}')
            
        except Exception as e:
            print(f"绘图失败 {key}: {e}")
    
    ax2.set_xlabel('PPFD (μmol/m²/s)')
    ax2.set_ylabel('Total PWM (%)')
    ax2.set_title('Total PWM Comparison for Different Ratio Keys')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: PWM效率对比
    test_ppfd = 400
    efficiency_data = []
    
    for key in available_keys:
        try:
            r_pwm, b_pwm = model.predict_pwm(ppfd=test_ppfd, label=key)
            total_pwm = r_pwm + b_pwm
            efficiency = test_ppfd / total_pwm if total_pwm > 0 else 0
            efficiency_data.append((key, efficiency))
        except:
            continue
    
    if efficiency_data:
        keys, efficiencies = zip(*efficiency_data)
        bars = ax3.bar(range(len(keys)), efficiencies, color=colors[:len(keys)])
        ax3.set_xlabel('Ratio Key')
        ax3.set_ylabel('PWM Efficiency (PPFD/Total PWM)')
        ax3.set_title(f'PWM Efficiency Comparison (PPFD={test_ppfd})')
        ax3.set_xticks(range(len(keys)))
        ax3.set_xticklabels(keys)
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{eff:.2f}', ha='center', va='bottom')
    
    # 子图4: 模型拟合质量对比
    fit_summary = model.get_fit_summary()
    labels = list(fit_summary.keys())
    r_squared_r = [fit_summary[label]['r_squared_r'] for label in labels]
    r_squared_b = [fit_summary[label]['r_squared_b'] for label in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, r_squared_r, width, label='Red Model R²', alpha=0.8)
    bars2 = ax4.bar(x + width/2, r_squared_b, width, label='Blue Model R²', alpha=0.8)
    
    ax4.set_xlabel('Ratio Key')
    ax4.set_ylabel('R² Value')
    ax4.set_title('Model Fitting Quality Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = result_dir / "ppfd_model_demo.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PPFD模型可视化图表已保存到: {output_path}")
    print()


def ppfd_calculator_demo(model: PWMtoPPFDModel):
    """PPFD计算器演示"""
    print("=" * 60)
    print("8. PPFD计算器演示")
    print("=" * 60)
    
    # 测试用例：不同场景的PPFD需求
    test_scenarios = [
        (200, "5:1", "幼苗期"),
        (350, "5:1", "生长期"),
        (500, "5:1", "开花期"),
        (300, "3:1", "营养生长期"),
        (400, "7:1", "果实发育期"),
        (150, "1:1", "低光照期"),
    ]
    
    print("不同生长阶段的PPFD-PWM配置:")
    print("-" * 70)
    print("阶段      目标PPFD  比例键   红光PWM  蓝光PWM  总PWM   效率")
    print("-" * 70)
    
    for target_ppfd, ratio_key, stage in test_scenarios:
        try:
            r_pwm, b_pwm, total_pwm = solve_pwm_for_target_ppfd(
                model=model,
                target_ppfd=target_ppfd,
                label=ratio_key,
                pwm_clip=(0.0, 100.0),
                integer_output=True
            )
            
            efficiency = target_ppfd / total_pwm if total_pwm > 0 else 0
            
            print(f"{stage:>8}   {target_ppfd:6d}   {ratio_key:>6}   {r_pwm:7d}   {b_pwm:7d}   {total_pwm:5d}   {efficiency:6.2f}")
            
        except Exception as e:
            print(f"{stage:>8}   {target_ppfd:6d}   {ratio_key:>6}   错误: {e}")
    
    print()
    
    # PPFD范围分析
    print("PPFD范围分析:")
    print("-" * 50)
    
    ppfd_ranges = [
        (100, 200, "低光照"),
        (200, 400, "中等光照"),
        (400, 600, "高光照"),
        (600, 800, "超高光照"),
    ]
    
    ratio_key = "5:1"
    
    print(f"光照范围分析 (比例键: {ratio_key}):")
    print("范围       最低PPFD   最高PPFD   最低PWM   最高PWM   PWM范围")
    print("-" * 65)
    
    for min_ppfd, max_ppfd, description in ppfd_ranges:
        try:
            # 计算最低PPFD对应的PWM
            r_pwm_min, b_pwm_min, total_pwm_min = solve_pwm_for_target_ppfd(
                model=model,
                target_ppfd=min_ppfd,
                label=ratio_key,
                pwm_clip=(0.0, 100.0),
                integer_output=True
            )
            
            # 计算最高PPFD对应的PWM
            r_pwm_max, b_pwm_max, total_pwm_max = solve_pwm_for_target_ppfd(
                model=model,
                target_ppfd=max_ppfd,
                label=ratio_key,
                pwm_clip=(0.0, 100.0),
                integer_output=True
            )
            
            pwm_range = total_pwm_max - total_pwm_min
            
            print(f"{description:>8}   {min_ppfd:6d}     {max_ppfd:6d}     {total_pwm_min:6d}     {total_pwm_max:6d}   {pwm_range:6d}")
            
        except Exception as e:
            print(f"{description:>8}   {min_ppfd:6d}     {max_ppfd:6d}     错误: {e}")
    
    print()


def main():
    """主函数"""
    print("PWM-PPFD转换系统验证演示")
    print("=" * 60)
    
    try:
        # 1. 加载和拟合PPFD模型
        model = load_and_fit_ppfd_model(DEFAULT_CALIB_CSV)
        
        # 2. 测试PPFD到PWM的正向预测功能
        test_ppfd_to_pwm_prediction(model)
        
        # 3. 测试PWM到PPFD的反向预测功能
        test_pwm_to_ppfd_prediction(model)
        
        # 4. 测试solve_pwm_for_target_ppfd函数
        test_solve_pwm_function(model)
        
        # 5. 模型拟合质量评估
        test_model_quality_assessment(model)
        
        # 6. 不同比例键的对比分析
        test_different_ratio_comparison(model)
        
        # 7. 创建PPFD模型可视化图表
        create_ppfd_model_visualization(model)
        
        # 8. PPFD计算器演示
        ppfd_calculator_demo(model)
        
        print("=" * 60)
        print("✅ PWM-PPFD转换系统验证完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
