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
import matplotlib.pyplot as plt
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from led import (
    DEFAULT_CALIB_CSV,
    PWMtoPPFDModel,
    solve_pwm_for_target_ppfd,
    _weights_from_key
)


def load_and_fit_model(csv_path: str) -> PWMtoPPFDModel:
    """加载标定数据并拟合线性模型"""
    print("============================================================")
    print("1. 加载标定数据并拟合线性模型")
    print("============================================================")
    
    # 创建模型并拟合
    model = PWMtoPPFDModel().fit(csv_path)
    
    print(f"标定数据文件: {csv_path}")
    print(f"可用比例键: {list(model.by_key.keys())}")
    print(f"整体系数: a_r={model.overall.a_r:.3f}, a_b={model.overall.a_b:.3f}, intercept={model.overall.intercept:.3f}")
    print()
    
    return model


def demo_forward_prediction(model: PWMtoPPFDModel):
    """演示前向预测"""
    print("============================================================")
    print("2. 前向预测演示")
    print("============================================================")
    
    # 测试不同总PWM下的PPFD预测
    test_pwms = [20, 40, 60, 80, 100]
    keys = list(model.by_key.keys())[:3]  # 测试前3个比例
    
    for key in keys:
        print(f"比例 {key}:")
        for total_pwm in test_pwms:
            # 使用权重分配计算R_PWM和B_PWM
            w_r, w_b = _weights_from_key(key)
            r_pwm = total_pwm * w_r
            b_pwm = total_pwm * w_b
            ppfd_pred = model.predict(r_pwm=r_pwm, b_pwm=b_pwm, key=key)
            print(f"  R_PWM={r_pwm:4.1f}%, B_PWM={b_pwm:4.1f}%, Total={total_pwm:3d}% → PPFD={ppfd_pred:6.1f} μmol/m²/s")
        print()


def demo_reverse_solving(model: PWMtoPPFDModel):
    """演示反向求解"""
    print("============================================================")
    print("3. 反向求解演示")
    print("============================================================")
    
    # 测试不同目标PPFD下的PWM求解
    target_ppfds = [100, 200, 300, 400, 500]
    keys = list(model.by_key.keys())[:3]  # 测试前3个比例
    
    for key in keys:
        print(f"比例 {key}:")
        for target_ppfd in target_ppfds:
            try:
                r_pwm, b_pwm, total_pwm = solve_pwm_for_target_ppfd(
                    model=model,
                    target_ppfd=target_ppfd,
                    key=key,
                    integer_output=True
                )
                print(f"  Target PPFD={target_ppfd:3d} → R_PWM={r_pwm:3d}%, B_PWM={b_pwm:3d}%, Total={total_pwm:3d}%")
            except Exception as e:
                print(f"  Target PPFD={target_ppfd:3d} → 求解失败: {e}")
        print()


def plot_model_fitting(model: PWMtoPPFDModel, save_path: str = None):
    """绘制线性模型拟合质量 - 分开显示红蓝LED"""
    print("=" * 60)
    print("线性模型拟合质量可视化")
    print("=" * 60)
    
    # 获取可用的比例键
    keys = list(model.by_key.keys())
    if not keys:
        print("没有可用的比例键数据")
        return
    
    # 创建子图 - 每个比例两个子图：红LED和蓝LED
    fig, axes = plt.subplots(len(keys), 2, figsize=(15, 4*len(keys)))
    if len(keys) == 1:
        axes = axes.reshape(1, -1)
    
    for i, key in enumerate(keys):
        coeffs = model.by_key[key]
        
        # 加载原始数据用于绘制散点
        import csv
        r_pwms = []
        b_pwms = []
        ppfds = []
        
        with open(model.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader((line for line in f if line.strip() and not line.lstrip().startswith("#")))
            for row in reader:
                row_key = row.get("R:B") or row.get("ratio") or row.get("Key") or row.get("KEY")
                if row_key == key:
                    r_pwm = float(row.get("R_PWM", 0))
                    b_pwm = float(row.get("B_PWM", 0))
                    ppfd = float(row.get("PPFD", 0))
                    r_pwms.append(r_pwm)
                    b_pwms.append(b_pwm)
                    ppfds.append(ppfd)
        
        # 左图：红LED
        ax_r = axes[i, 0]
        ax_r.scatter(r_pwms, ppfds, c='red', s=50, alpha=0.7, label=f'Red LED Data ({len(r_pwms)} points)')
        
        # 绘制红LED拟合线（B_PWM=0）
        if r_pwms:
            r_min, r_max = min(r_pwms), max(r_pwms)
            r_line = np.linspace(0, r_max, 100)
            y_r_line = [coeffs.predict(r_pwm=r, b_pwm=0) for r in r_line]
            ax_r.plot(r_line, y_r_line, 'r-', linewidth=2, label=f'Red LED Fit')
            
            # 计算红LED的R²
            y_r_pred = [coeffs.predict(r_pwm=r_pwms[j], b_pwm=b_pwms[j]) for j in range(len(r_pwms))]
            ss_res = sum((y - y_r_pred[j])**2 for j, y in enumerate(ppfds))
            ss_tot = sum((y - np.mean(ppfds))**2 for y in ppfds)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # 计算平均误差
            errors = [abs(y_r_pred[j] - ppfds[j]) for j in range(len(ppfds))]
            mean_error = sum(errors) / len(errors)
        else:
            r_squared = 0
            mean_error = 0
        
        ax_r.set_title(f'Ratio {key} - Red LED\nPPFD = {coeffs.a_r:.2f}×R_PWM + {coeffs.a_b:.2f}×B_PWM + {coeffs.intercept:.1f}\nR² = {r_squared:.3f}, Mean Error = {mean_error:.1f}')
        ax_r.set_xlabel('Red PWM (%)')
        ax_r.set_ylabel('PPFD (μmol/m²/s)')
        ax_r.grid(True, alpha=0.3)
        ax_r.legend()
        
        # 右图：蓝LED
        ax_b = axes[i, 1]
        ax_b.scatter(b_pwms, ppfds, c='blue', s=50, alpha=0.7, label=f'Blue LED Data ({len(b_pwms)} points)')
        
        # 绘制蓝LED拟合线（R_PWM=0）
        if b_pwms:
            b_min, b_max = min(b_pwms), max(b_pwms)
            b_line = np.linspace(0, b_max, 100)
            y_b_line = [coeffs.predict(r_pwm=0, b_pwm=b) for b in b_line]
            ax_b.plot(b_line, y_b_line, 'b-', linewidth=2, label=f'Blue LED Fit')
        
        ax_b.set_title(f'Ratio {key} - Blue LED\nPPFD = {coeffs.a_r:.2f}×R_PWM + {coeffs.a_b:.2f}×B_PWM + {coeffs.intercept:.1f}')
        ax_b.set_xlabel('Blue PWM (%)')
        ax_b.set_ylabel('PPFD (μmol/m²/s)')
        ax_b.grid(True, alpha=0.3)
        ax_b.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def plot_comparison(model: PWMtoPPFDModel, save_path: str = None):
    """绘制不同比例的对比图"""
    print("=" * 60)
    print("不同比例下的线性模型对比")
    print("=" * 60)
    
    # 测试不同总PWM下的PPFD输出
    total_pwms = np.linspace(0, 120, 25)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：总PWM vs PPFD
    for key, coeffs in model.by_key.items():
        w_r, w_b = _weights_from_key(key)
        ppfds = []
        for total_pwm in total_pwms:
            r_pwm = total_pwm * w_r
            b_pwm = total_pwm * w_b
            ppfd_pred = coeffs.predict(r_pwm, b_pwm)
            ppfds.append(ppfd_pred)
        ax1.plot(total_pwms, ppfds, 'o-', label=f'Ratio {key}', linewidth=2, markersize=4)
    
    ax1.set_xlabel('Total PWM (%)')
    ax1.set_ylabel('PPFD (μmol/m²/s)')
    ax1.set_title('Simple Linear Models: PPFD vs Total PWM')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 右图：系数对比
    keys = list(model.by_key.keys())
    # 使用等效斜率：a_r * w_r + a_b * w_b
    slopes = []
    for key in keys:
        coeffs = model.by_key[key]
        w_r, w_b = _weights_from_key(key)
        equiv_slope = coeffs.a_r * w_r + coeffs.a_b * w_b
        slopes.append(equiv_slope)
    
    # 计算R²值
    r_squareds = []
    for key in keys:
        coeffs = model.by_key[key]
        # 加载该比例的数据计算R²
        import csv
        total_pwms_data = []
        ppfds_data = []
        
        with open(model.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader((line for line in f if line.strip() and not line.lstrip().startswith("#")))
            for row in reader:
                row_key = row.get("R:B") or row.get("ratio") or row.get("Key") or row.get("KEY")
                if row_key == key:
                    r_pwm = float(row.get("R_PWM", 0))
                    b_pwm = float(row.get("B_PWM", 0))
                    ppfd = float(row.get("PPFD", 0))
                    total_pwms_data.append(r_pwm + b_pwm)
                    ppfds_data.append(ppfd)
        
        if len(total_pwms_data) > 1:
            w_r, w_b = _weights_from_key(key)
            y_pred = []
            for i, total_pwm in enumerate(total_pwms_data):
                r_pwm = total_pwm * w_r
                b_pwm = total_pwm * w_b
                ppfd_pred = coeffs.predict(r_pwm, b_pwm)
                y_pred.append(ppfd_pred)
            ss_res = sum((y - y_pred[i])**2 for i, y in enumerate(ppfds_data))
            ss_tot = sum((y - np.mean(ppfds_data))**2 for y in ppfds_data)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            r_squared = 0
        r_squareds.append(r_squared)
    
    bars = ax2.bar(keys, slopes, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax2.set_xlabel('Ratio')
    ax2.set_ylabel('Slope (PPFD per Total PWM %)')
    ax2.set_title('Model Slopes by Ratio')
    ax2.grid(True, alpha=0.3)
    
    # 在柱状图上添加R²值
    for i, (bar, r2) in enumerate(zip(bars, r_squareds)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'R²={r2:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
    
    plt.show()


def export_ppfd_pwm_table(model: PWMtoPPFDModel, ratio_key: str = "5:1"):
    """导出PPFD到PWM的对应表"""
    print("=" * 60)
    print(f"导出 {ratio_key} 比例的PPFD-PWM对应表")
    print("=" * 60)
    
    # 生成0-600 PPFD的对应PWM值
    ppfd_values = list(range(0, 601, 10))  # 0, 10, 20, ..., 600
    results = []
    
    for ppfd in ppfd_values:
        try:
            r_pwm, b_pwm, total_pwm = solve_pwm_for_target_ppfd(
                model=model,
                target_ppfd=ppfd,
                key=ratio_key,
                integer_output=True
            )
            results.append({
                'PPFD': ppfd,
                'R_PWM': r_pwm,
                'B_PWM': b_pwm,
                'Total_PWM': total_pwm
            })
        except Exception as e:
            print(f"PPFD={ppfd} 求解失败: {e}")
            results.append({
                'PPFD': ppfd,
                'R_PWM': 0,
                'B_PWM': 0,
                'Total_PWM': 0
            })
    
    # 保存到CSV文件
    import csv
    result_dir = Path(__file__).parent / "result"
    csv_path = result_dir / f"ppfd_pwm_table_{ratio_key.replace(':', '_')}.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['PPFD', 'R_PWM', 'B_PWM', 'Total_PWM'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"PPFD-PWM对应表已保存到: {csv_path}")
    print(f"共生成 {len(results)} 个数据点")
    
    # 显示前10个和后10个数据点
    print("\n前10个数据点:")
    for i, row in enumerate(results[:10]):
        print(f"  PPFD={row['PPFD']:3d} → R_PWM={row['R_PWM']:3d}%, B_PWM={row['B_PWM']:3d}%, Total={row['Total_PWM']:3d}%")
    
    print("\n后10个数据点:")
    for i, row in enumerate(results[-10:]):
        print(f"  PPFD={row['PPFD']:3d} → R_PWM={row['R_PWM']:3d}%, B_PWM={row['B_PWM']:3d}%, Total={row['Total_PWM']:3d}%")
    
    return csv_path


def demo_models():
    """演示线性模型"""
    print("LED控制系统 - PWM-PPFD转换系统演示")
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
        export_ppfd_pwm_table(model, "5:1")
        
        # 5. 绘制拟合质量图
        plot_model_fitting(model, str(result_dir / "ppfd_model_fitting.png"))
        
        # 6. 绘制对比图
        plot_comparison(model, str(result_dir / "ppfd_comparison.png"))
        
        print("=" * 80)
        print("演示完成！")
        print("生成的文件:")
        print("- result/ppfd_pwm_table_5_1.csv: PPFD-PWM对应表")
        print("- result/ppfd_model_fitting.png: 模型拟合质量")
        print("- result/ppfd_comparison.png: 不同比例对比")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 确保result目录存在
    result_dir = Path(__file__).parent / "result"
    result_dir.mkdir(exist_ok=True)
    
    demo_models()
