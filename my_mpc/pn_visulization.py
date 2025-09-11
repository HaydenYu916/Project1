#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光合速率预测热图生成器
生成不同PPFD和温度组合下的光合速率预测结果热图
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pn_prediction.predict import PhotosynthesisPredictor
import warnings

warnings.filterwarnings("ignore")

# 设置matplotlib中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def generate_parameter_combinations(ppfd_range, temp_range):
    """生成参数组合"""
    ppfd_values = np.arange(ppfd_range[0], ppfd_range[1] + 1, 50)  # 每50递增
    temp_values = np.arange(temp_range[0], temp_range[1] + 1, 2)  # 每2度递增

    print(f"PPFD范围: {ppfd_range[0]} - {ppfd_range[1]} umol/m2/s (步长: 50)")
    print(f"温度范围: {temp_range[0]} - {temp_range[1]} °C (步长: 2)")
    print(
        f"总共生成 {len(ppfd_values)} × {len(temp_values)} = {len(ppfd_values) * len(temp_values)} 个组合"
    )

    return ppfd_values, temp_values


def create_prediction_matrix(predictor, ppfd_values, temp_values):
    """创建预测结果矩阵"""
    prediction_matrix = np.zeros((len(temp_values), len(ppfd_values)))

    print("开始预测...")
    total_predictions = len(ppfd_values) * len(temp_values)
    current_prediction = 0

    for i, temp in enumerate(temp_values):
        for j, ppfd in enumerate(ppfd_values):
            try:
                prediction = predictor.predict(ppfd, temp)
                prediction_matrix[i, j] = prediction if prediction is not None else 0
                current_prediction += 1

                # 显示进度
                if current_prediction % 50 == 0:
                    progress = (current_prediction / total_predictions) * 100
                    print(
                        f"预测进度: {current_prediction}/{total_predictions} ({progress:.1f}%)"
                    )
            except Exception as e:
                print(f"预测错误 (PPFD={ppfd}, T={temp}): {e}")
                prediction_matrix[i, j] = 0

    print("预测完成!")
    return prediction_matrix


def plot_heatmap(prediction_matrix, ppfd_values, temp_values, save_path=None):
    """绘制热图"""
    plt.figure(figsize=(10, 8))

    # 创建热图
    ax = sns.heatmap(
        prediction_matrix,
        xticklabels=ppfd_values,
        yticklabels=temp_values,
        cmap="YlOrRd",
        annot=False,
        fmt=".2f",
        cbar_kws={"label": "Photosynthetic Rate (umol/m2/s)"},
    )

    # 设置标题和标签
    plt.title(
        "Basil Photosynthetic Rate Prediction Heatmap\n(CO2=400ppm, R:B=0.83)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("PPFD (umol/m2/s)", fontsize=14, fontweight="bold")
    plt.ylabel("Temperature (°C)", fontsize=14, fontweight="bold")

    # 调整刻度标签
    ax.set_xticklabels(ppfd_values, rotation=45)
    ax.set_yticklabels(temp_values, rotation=0)

    # 添加网格
    plt.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"热图已保存至: {save_path}")

    plt.show()


def analyze_results(prediction_matrix, ppfd_values, temp_values):
    """分析结果"""
    print("\n" + "=" * 50)
    print("预测结果分析")
    print("=" * 50)

    # 基本统计
    valid_predictions = prediction_matrix[prediction_matrix > 0]
    print(f"有效预测数量: {len(valid_predictions)}")
    print(f"最大光合速率: {np.max(valid_predictions):.3f} umol/m2/s")
    print(f"最小光合速率: {np.min(valid_predictions):.3f} umol/m2/s")
    print(f"平均光合速率: {np.mean(valid_predictions):.3f} umol/m2/s")
    print(f"标准差: {np.std(valid_predictions):.3f} umol/m2/s")

    # 找到最优条件
    max_idx = np.unravel_index(np.argmax(prediction_matrix), prediction_matrix.shape)
    optimal_temp = temp_values[max_idx[0]]
    optimal_ppfd = ppfd_values[max_idx[1]]
    max_rate = prediction_matrix[max_idx]

    print(f"\n最优条件:")
    print(f"  PPFD: {optimal_ppfd} umol/m2/s")
    print(f"  温度: {optimal_temp} °C")
    print(f"  预测光合速率: {max_rate:.3f} umol/m2/s")

    # 温度效应分析
    print(f"\n温度效应分析:")
    for i, temp in enumerate([15, 20, 25, 30, 35]):
        if temp in temp_values:
            temp_idx = list(temp_values).index(temp)
            temp_max = np.max(prediction_matrix[temp_idx, :])
            temp_avg = np.mean(prediction_matrix[temp_idx, :])
            print(f"  {temp}°C: 最大={temp_max:.3f}, 平均={temp_avg:.3f}")


def main():
    """主函数"""
    print("Basil光合速率预测热图生成器")
    print("=" * 50)

    # 初始化预测器
    predictor = PhotosynthesisPredictor()

    if not predictor.is_loaded:
        print("无法加载模型，程序终止")
        return

    # 设置参数范围
    ppfd_range = (100, 1000)  # PPFD范围
    temp_range = (15, 35)  # 温度范围

    # 生成参数组合
    ppfd_values, temp_values = generate_parameter_combinations(ppfd_range, temp_range)

    # 创建预测矩阵
    prediction_matrix = create_prediction_matrix(predictor, ppfd_values, temp_values)

    # 绘制热图
    plot_heatmap(
        prediction_matrix,
        ppfd_values,
        temp_values,
        save_path="photosynthesis_heatmap.png",
    )

    # 分析结果
    analyze_results(prediction_matrix, ppfd_values, temp_values)

    print("\n程序执行完成!")


if __name__ == "__main__":
    main()
