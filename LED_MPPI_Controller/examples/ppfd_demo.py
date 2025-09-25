#!/usr/bin/env python3
"""LED控制系统 - PWM-PPFD转换系统验证演示

本脚本验证led.py中模块4的PWM-PPFD转换系统功能：
1. 模型加载和拟合
2. 前向预测功能
3. 反向求解功能
4. 模型拟合质量可视化
5. 不同比例对比分析
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
    solve_pwm_for_target_ppfd
)


def load_and_fit_model(csv_path: str) -> PWMtoPPFDModel:
    """加载标定数据并拟合线性模型"""
    print("============================================================")
    print("1. 加载标定数据并拟合分开的线性模型")
    print("============================================================")
    
    # 创建模型并拟合
    model = PWMtoPPFDModel().fit(csv_path)
    
    print(f"标定数据文件: {csv_path}")
    print(f"可用标签: {model.list_labels()}")
    print()
    
    # 显示每个标签的拟合结果
    print("拟合摘要:")
    summary = model.get_fit_summary()
    for label, info in summary.items():
        print(f"  {label}:")
        print(f"    R_PWM = {info['alpha']:.4f} × PPFD + {info['beta']:.2f} (R²={info['r_squared_r']:.3f})")
        print(f"    B_PWM = {info['gamma']:.4f} × PPFD + {info['delta']:.2f} (R²={info['r_squared_b']:.3f})")
    print()
    
    return model


def demo_forward_prediction(model: PWMtoPPFDModel):
    """演示前向预测"""
    print("============================================================")
    print("2. 前向预测演示（根据PPFD预测PWM）")
    print("============================================================")
    
    # 测试不同PPFD下的PWM预测
    test_ppfds = [100, 200, 300, 400, 500]
    labels = model.list_labels()[:3]  # 测试前3个标签
    
    for label in labels:
        print(f"标签 {label}:")
        for ppfd in test_ppfds:
            try:
                r_pwm, b_pwm = model.predict_pwm(ppfd=ppfd, label=label)
                total_pwm = r_pwm + b_pwm
                print(f"  PPFD={ppfd:3d} μmol/m²/s → R_PWM={r_pwm:5.1f}%, B_PWM={b_pwm:5.1f}%, Total={total_pwm:5.1f}%")
            except Exception as e:
                print(f"  PPFD={ppfd:3d} μmol/m²/s → 预测失败: {e}")
        print()


def demo_reverse_solving(model: PWMtoPPFDModel):
    """演示反向求解"""
    print("============================================================")
    print("3. 反向求解演示（使用solve_pwm_for_target_ppfd函数）")
    print("============================================================")
    
    # 测试不同目标PPFD下的PWM求解
    target_ppfds = [100, 200, 300, 400, 500]
    labels = model.list_labels()[:3]  # 测试前3个标签
    
    for label in labels:
        print(f"标签 {label}:")
        for target_ppfd in target_ppfds:
            try:
                r_pwm, b_pwm, total_pwm = solve_pwm_for_target_ppfd(
                    model=model,
                    target_ppfd=target_ppfd,
                    label=label,
                    integer_output=False
                )
                print(f"  Target PPFD={target_ppfd:3d} → R_PWM={r_pwm:5.1f}%, B_PWM={b_pwm:5.1f}%, Total={total_pwm:5.1f}%")
            except Exception as e:
                print(f"  Target PPFD={target_ppfd:3d} → 求解失败: {e}")
        print()


def plot_model_fitting(model: PWMtoPPFDModel, save_path: str = None):
    """Plot separated fitting model quality: PPFD vs PWM"""
    print("=" * 60)
    print("Separated Fitting Model Quality Visualization")
    print("=" * 60)

    labels = model.list_labels()
    if not labels:
        print("No available label data")
        return

    fig, axes = plt.subplots(len(labels), 2, figsize=(15, 4 * len(labels)))
    if len(labels) == 1:
        axes = np.array([axes]).reshape(1, -1)

    import csv
    for i, label in enumerate(labels):
        coeffs = model.by_label[label]
        
        # 载入该标签的数据
        ppfds = []
        r_pwms = []
        b_pwms = []
        with open(model.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader((line for line in f if line.strip() and not line.lstrip().startswith("#")))
            for row in reader:
                row_label = row.get("Label") or row.get("LABEL") or row.get("KEY") or row.get("Key") or row.get("R:B") or row.get("ratio")
                if row_label and row_label.lower().replace(" ", "") == label.lower().replace(" ", ""):
                    ppfd = float(row.get("PPFD", 0))
                    r_pwm = float(row.get("R_PWM", 0))
                    b_pwm = float(row.get("B_PWM", 0))
                    ppfds.append(ppfd)
                    r_pwms.append(r_pwm)
                    b_pwms.append(b_pwm)

        # Left plot: PPFD vs R_PWM
        ax_r = axes[i, 0]
        if ppfds:
            ax_r.scatter(ppfds, r_pwms, c="#ff4444", s=50, alpha=0.7, label=f"Data ({len(ppfds)} points)")
            
            # Draw fitting line
            ppfd_line = np.linspace(min(ppfds), max(ppfds), 100)
            r_pwm_line = [coeffs.predict_r_pwm(p) for p in ppfd_line]
            ax_r.plot(ppfd_line, r_pwm_line, '-', color='#cc0000', linewidth=2, 
                     label=f'R_PWM = {coeffs.alpha:.4f}×PPFD + {coeffs.beta:.2f}')
        
        ax_r.set_title(f"Label {label} - Red Channel\nR² = {coeffs.r_squared_r:.3f}")
        ax_r.set_xlabel('PPFD (μmol/m²/s)')
        ax_r.set_ylabel('R_PWM (%)')
        ax_r.grid(True, alpha=0.3)
        ax_r.legend()

        # Right plot: PPFD vs B_PWM
        ax_b = axes[i, 1]
        if ppfds:
            ax_b.scatter(ppfds, b_pwms, c="#4444ff", s=50, alpha=0.7, label=f"Data ({len(ppfds)} points)")
            
            # Draw fitting line
            ppfd_line = np.linspace(min(ppfds), max(ppfds), 100)
            b_pwm_line = [coeffs.predict_b_pwm(p) for p in ppfd_line]
            ax_b.plot(ppfd_line, b_pwm_line, '-', color='#0000cc', linewidth=2,
                     label=f'B_PWM = {coeffs.gamma:.4f}×PPFD + {coeffs.delta:.2f}')
        
        ax_b.set_title(f"Label {label} - Blue Channel\nR² = {coeffs.r_squared_b:.3f}")
        ax_b.set_xlabel('PPFD (μmol/m²/s)')
        ax_b.set_ylabel('B_PWM (%)')
        ax_b.grid(True, alpha=0.3)
        ax_b.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    plt.show()


def plot_comparison(model: PWMtoPPFDModel, save_path: str = None):
    """Plot model comparison across different labels"""
    print("=" * 60)
    print("Model Comparison Across Different Labels")
    print("=" * 60)

    labels = model.list_labels()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Top left: R_PWM slope comparison across labels
    alpha_values = []
    r_squared_r_values = []
    for label in labels:
        coeffs = model.by_label[label]
        alpha_values.append(coeffs.alpha)
        r_squared_r_values.append(coeffs.r_squared_r)
    
    bars1 = ax1.bar(labels, alpha_values, alpha=0.7, color='#ff4444')
    ax1.set_xlabel('Label')
    ax1.set_ylabel('R_PWM Slope (α)')
    ax1.set_title('Red Channel Slope Comparison')
    ax1.grid(True, alpha=0.3)
    
    for bar, r2 in zip(bars1, r_squared_r_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'R²={r2:.3f}', ha='center', va='bottom', fontsize=9)

    # Top right: B_PWM slope comparison across labels
    gamma_values = []
    r_squared_b_values = []
    for label in labels:
        coeffs = model.by_label[label]
        gamma_values.append(coeffs.gamma)
        r_squared_b_values.append(coeffs.r_squared_b)
    
    bars2 = ax2.bar(labels, gamma_values, alpha=0.7, color='#4444ff')
    ax2.set_xlabel('Label')
    ax2.set_ylabel('B_PWM Slope (γ)')
    ax2.set_title('Blue Channel Slope Comparison')
    ax2.grid(True, alpha=0.3)
    
    for bar, r2 in zip(bars2, r_squared_b_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'R²={r2:.3f}', ha='center', va='bottom', fontsize=9)

    # Bottom left: PPFD vs predicted Total PWM comparison
    ppfd_range = np.linspace(0, 500, 100)
    for label in labels:
        coeffs = model.by_label[label]
        r_pwms = [coeffs.predict_r_pwm(p) for p in ppfd_range]
        b_pwms = [coeffs.predict_b_pwm(p) for p in ppfd_range]
        total_pwms = [r + b for r, b in zip(r_pwms, b_pwms)]
        ax3.plot(ppfd_range, total_pwms, 'o-', label=f'Label {label}', linewidth=2, markersize=3)
    
    ax3.set_xlabel('PPFD (μmol/m²/s)')
    ax3.set_ylabel('Predicted Total PWM (%)')
    ax3.set_title('PPFD vs Total PWM Prediction Comparison')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Bottom right: R² comparison
    x_pos = np.arange(len(labels))
    width = 0.35
    
    bars3 = ax4.bar(x_pos - width/2, r_squared_r_values, width, label='R_PWM R²', alpha=0.7, color='#ff4444')
    bars4 = ax4.bar(x_pos + width/2, r_squared_b_values, width, label='B_PWM R²', alpha=0.7, color='#4444ff')
    
    ax4.set_xlabel('Label')
    ax4.set_ylabel('R² Value')
    ax4.set_title('Model Fit Quality Comparison')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison chart saved to: {save_path}")
    plt.show()


def export_ppfd_pwm_table(model: PWMtoPPFDModel, label: str = "5:1"):
    """导出PPFD到PWM的对应表"""
    print("=" * 60)
    print(f"导出 {label} 标签的PPFD-PWM对应表")
    print("=" * 60)
    
    # 生成0-600 PPFD的对应PWM值
    ppfd_values = list(range(0, 601, 10))  # 0, 10, 20, ..., 600
    results = []
    
    for ppfd in ppfd_values:
        try:
            r_pwm, b_pwm, total_pwm = solve_pwm_for_target_ppfd(
                model=model,
                target_ppfd=ppfd,
                label=label,
                integer_output=False
            )
            results.append({
                'PPFD': ppfd,
                'R_PWM': round(r_pwm, 2),
                'B_PWM': round(b_pwm, 2),
                'Total_PWM': round(total_pwm, 2)
            })
        except Exception as e:
            print(f"PPFD={ppfd} 求解失败: {e}")
            results.append({
                'PPFD': ppfd,
                'R_PWM': 0.0,
                'B_PWM': 0.0,
                'Total_PWM': 0.0
            })
    
    # 保存到CSV文件
    import csv
    result_dir = Path(__file__).parent / "result"
    csv_path = result_dir / f"ppfd_pwm_table_{label.replace(':', '_')}.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['PPFD', 'R_PWM', 'B_PWM', 'Total_PWM'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"PPFD-PWM对应表已保存到: {csv_path}")
    print(f"共生成 {len(results)} 个数据点")
    
    # 显示前10个和后10个数据点
    print("\n前10个数据点:")
    for i, row in enumerate(results[:10]):
        print(f"  PPFD={row['PPFD']:3d} → R_PWM={row['R_PWM']:5.1f}%, B_PWM={row['B_PWM']:5.1f}%, Total={row['Total_PWM']:5.1f}%")
    
    print("\n后10个数据点:")
    for i, row in enumerate(results[-10:]):
        print(f"  PPFD={row['PPFD']:3d} → R_PWM={row['R_PWM']:5.1f}%, B_PWM={row['B_PWM']:5.1f}%, Total={row['Total_PWM']:5.1f}%")
    
    return csv_path


def demo_models():
    """演示分开拟合的线性模型"""
    print("LED控制系统 - PWM-PPFD转换系统演示（重构版本）")
    print("=" * 80)
    
    # 检查标定数据文件
    csv_path = DEFAULT_CALIB_CSV
    if not os.path.exists(csv_path):
        print(f"错误: 标定数据文件不存在: {csv_path}")
        return
    
    try:
        # 1. 加载数据并拟合模型
        model = load_and_fit_model(csv_path)
        
        # 2. 前向预测演示
        demo_forward_prediction(model)
        
        # 3. 反向求解演示
        demo_reverse_solving(model)
        
        # 4. 导出PPFD-PWM对应表
        result_dir = Path(__file__).parent / "result"
        labels = model.list_labels()
        if labels:
            # 为第一个标签生成表格
            export_ppfd_pwm_table(model, labels[0])
            # 如果"5:1"标签存在，也为它生成表格
            if "5:1" in labels:
                export_ppfd_pwm_table(model, "5:1")
        
        # 5. 绘制拟合质量图
        plot_model_fitting(model, str(result_dir / "ppfd_model_fitting.png"))
        
        # 6. 绘制对比图
        plot_comparison(model, str(result_dir / "ppfd_comparison.png"))
        
        print("=" * 80)
        print("演示完成！")
        print("生成的文件:")
        if labels:
            print(f"- result/ppfd_pwm_table_{labels[0].replace(':', '_')}.csv: PPFD-PWM对应表")
            if "5:1" in labels:
                print("- result/ppfd_pwm_table_5_1.csv: PPFD-PWM对应表")
        print("- result/ppfd_model_fitting.png: 模型拟合质量")
        print("- result/ppfd_comparison.png: 不同标签对比")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 确保result目录存在
    result_dir = Path(__file__).parent / "result"
    result_dir.mkdir(exist_ok=True)
    
    demo_models()
