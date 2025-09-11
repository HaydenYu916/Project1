#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光合速率预测热图生成器 - 带数据点标记
Generate photosynthesis rate heatmap with experimental data points overlay
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
    temp_values = np.arange(temp_range[0], temp_range[1] + 1, 1)  # 每1度递增

    print(f"PPFD范围: {ppfd_range[0]} - {ppfd_range[1]} umol/m2/s (步长: 50)")
    print(f"温度范围: {temp_range[0]} - {temp_range[1]} °C (步长: 1)")
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


def plot_heatmap_with_points(
    prediction_matrix, ppfd_values, temp_values, data_points, save_path=None
):
    """绘制带数据点和箭头序列的热图"""
    plt.figure(figsize=(10, 8))

    # 创建热图
    ax = sns.heatmap(
        prediction_matrix,
        xticklabels=ppfd_values,
        yticklabels=temp_values,
        cmap="YlOrRd",
        annot=False,
        fmt=".2f",
        cbar_kws={"label": "Photosynthetic Rate (μmol/m²/s)"},
        alpha=0.8,
    )

    # 准备绘制箭头的坐标
    x_coords = []
    y_coords = []

    # 添加数据点和收集坐标
    point_positions = {}  # 跟踪每个网格位置的点数

    for i, point in enumerate(data_points):
        # 找到最接近的网格点索引
        ppfd_idx = np.argmin(np.abs(ppfd_values - point["ppfd"]))
        temp_idx = np.argmin(np.abs(temp_values - point["temp"]))

        # 创建位置键
        pos_key = (ppfd_idx, temp_idx)

        # 如果这个位置已经有点了，添加小的偏移
        if pos_key in point_positions:
            point_positions[pos_key] += 1
            offset_count = point_positions[pos_key]
            # 围绕原点创建圆形偏移
            angle = 2 * np.pi * offset_count / 8  # 最多8个点围成圆
            offset_x = 0.3 * np.cos(angle)
            offset_y = 0.3 * np.sin(angle)
        else:
            point_positions[pos_key] = 0
            offset_x = 0
            offset_y = 0

        # 应用偏移
        plot_x = ppfd_idx + offset_x
        plot_y = temp_idx + offset_y

        x_coords.append(plot_x)
        y_coords.append(plot_y)

        # 在热图上标记点
        ax.scatter(
            plot_x,
            plot_y,
            color="red",
            s=120,
            edgecolors="white",
            linewidth=2,
            marker="o",
            alpha=0.9,
            zorder=5,
        )

        # 添加时间标签
        ax.annotate(
            f'{point["time"]}s',
            xy=(plot_x, plot_y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
            zorder=6,
        )

    # 绘制箭头连接点
    for i in range(len(x_coords) - 1):
        # 计算箭头方向
        dx = x_coords[i + 1] - x_coords[i]
        dy = y_coords[i + 1] - y_coords[i]

        # 只有当两点不重合时才绘制箭头
        if dx != 0 or dy != 0:
            # 稍微缩短箭头，避免与点重叠
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                start_x = x_coords[i] + 0.2 * dx / length
                start_y = y_coords[i] + 0.2 * dy / length
                end_x = x_coords[i + 1] - 0.2 * dx / length
                end_y = y_coords[i + 1] - 0.2 * dy / length
            else:
                start_x, start_y = x_coords[i], y_coords[i]
                end_x, end_y = x_coords[i + 1], y_coords[i + 1]

            ax.annotate(
                "",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2, alpha=0.8),
                zorder=4,
            )

    # 添加起始点和结束点标记
    if len(x_coords) > 0:
        ax.scatter(
            x_coords[0],
            y_coords[0],
            color="green",
            s=150,
            marker="s",
            edgecolors="white",
            linewidth=2,
            alpha=0.9,
            zorder=7,
        )

        ax.scatter(
            x_coords[-1],
            y_coords[-1],
            color="purple",
            s=150,
            marker="^",
            edgecolors="white",
            linewidth=2,
            alpha=0.9,
            zorder=7,
        )

    # 设置标题和标签
    plt.title(
        "Basil Photosynthetic Rate Prediction Heatmap with Experimental Sequence\n(CO₂=400ppm, R:B=0.83)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("PPFD (μmol/m²/s)", fontsize=14, fontweight="bold")
    plt.ylabel("Temperature (°C)", fontsize=14, fontweight="bold")

    # 调整刻度标签
    ax.set_xticklabels(ppfd_values, rotation=45)
    ax.set_yticklabels(temp_values, rotation=0)

    # 添加图例
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="Data Points",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="green",
            markersize=10,
            label="Start (t=0s)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="purple",
            markersize=10,
            label="End (t=110s)",
        ),
        Line2D([0], [0], color="blue", lw=2, label="Sequence"),
    ]
    # ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.05, 1))

    # 添加网格
    plt.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"热图已保存至: {save_path}")

    plt.show()


def analyze_experimental_data(data_points):
    """分析实验数据"""
    print("\n" + "=" * 60)
    print("实验数据分析")
    print("=" * 60)

    ppfds = [point["ppfd"] for point in data_points]
    temps = [point["temp"] for point in data_points]
    photos = [point["photo"] for point in data_points]

    print("数据统计:")
    print(f"数据点数量: {len(data_points)}")
    print(f"PPFD范围: {min(ppfds):.0f} - {max(ppfds):.0f} μmol/m²/s")
    print(f"温度范围: {min(temps):.1f} - {max(temps):.1f} °C")
    print(f"光合速率范围: {min(photos):.1f} - {max(photos):.1f} μmol/m²/s")

    print(f"\n平均值:")
    print(f"PPFD: {np.mean(ppfds):.1f} μmol/m²/s")
    print(f"温度: {np.mean(temps):.1f} °C")
    print(f"光合速率: {np.mean(photos):.1f} μmol/m²/s")

    # 找到最高光合速率的条件
    max_idx = np.argmax(photos)
    max_point = data_points[max_idx]
    print(f"\n最高光合速率条件:")
    print(f"时间: {max_point['time']}s")
    print(f"PPFD: {max_point['ppfd']} μmol/m²/s")
    print(f"温度: {max_point['temp']} °C")
    print(f"光合速率: {max_point['photo']} μmol/m²/s")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光合速率预测热图生成器 - 带数据点标记
Generate photosynthesis rate heatmap with experimental data points overlay
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


def load_data_from_log(file_path):
    """从日志文件加载实验数据

    Args:
        file_path (str): 日志文件路径

    Returns:
        list: 包含实验数据点的列表

    文件格式: time,pwm,ppfd,temp,photo,cost
    例如: 0.0,20.6,103.1,22.0,4.5,-5.4e+02
    """
    data_points = []

    try:
        with open(file_path, "r") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line or line.startswith("#"):  # 跳过空行和注释行
                    continue

                try:
                    # 解析CSV行
                    parts = line.split(",")
                    if len(parts) >= 5:  # 至少需要5列数据
                        time = float(parts[0])
                        pwm = float(parts[1])
                        ppfd = float(parts[2])
                        temp = float(parts[3])
                        photo = float(parts[4])
                        # cost是可选的（第6列）
                        cost = float(parts[5]) if len(parts) > 5 else None

                        data_point = {
                            "time": time,
                            "pwm": pwm,
                            "ppfd": ppfd,
                            "temp": temp,
                            "photo": photo,
                        }
                        if cost is not None:
                            data_point["cost"] = cost

                        data_points.append(data_point)
                    else:
                        print(f"警告: 第{line_num}行数据不完整，跳过: {line}")

                except ValueError as e:
                    print(f"警告: 第{line_num}行数据格式错误，跳过: {line} (错误: {e})")
                    continue

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return []
    except Exception as e:
        print(f"错误: 读取文件时发生错误: {e}")
        return []

    print(f"成功加载 {len(data_points)} 个数据点")
    return data_points


def main():
    """主函数"""
    print("Basil光合速率预测热图生成器 - 带实验数据点")
    print("=" * 60)

    # 实验数据点
    # data_points = [
    #     {"time": 0, "pwm": 27.4, "ppfd": 137, "temp": 22.0, "photo": 6.0},
    #     {"time": 10, "pwm": 57.7, "ppfd": 288, "temp": 22.6, "photo": 12.1},
    #     {"time": 20, "pwm": 48.9, "ppfd": 245, "temp": 23.2, "photo": 10.9},
    #     {"time": 30, "pwm": 63.3, "ppfd": 317, "temp": 23.8, "photo": 13.5},
    #     {"time": 40, "pwm": 54.8, "ppfd": 274, "temp": 24.4, "photo": 12.3},
    #     {"time": 50, "pwm": 55.0, "ppfd": 275, "temp": 24.9, "photo": 12.4},
    #     {"time": 60, "pwm": 53.2, "ppfd": 266, "temp": 25.3, "photo": 12.1},
    #     {"time": 70, "pwm": 52.4, "ppfd": 262, "temp": 25.8, "photo": 11.9},
    #     {"time": 80, "pwm": 55.1, "ppfd": 275, "temp": 26.1, "photo": 12.3},
    #     {"time": 90, "pwm": 56.0, "ppfd": 280, "temp": 26.5, "photo": 12.4},
    #     {"time": 100, "pwm": 63.2, "ppfd": 316, "temp": 26.9, "photo": 13.5},
    #     {"time": 110, "pwm": 58.9, "ppfd": 294, "temp": 27.3, "photo": 12.6},
    #     ######
    #     # {"time": 0, "pwm": 38.0, "ppfd": 190, "temp": 22.0, "photo": 8.3},
    #     # {"time": 10, "pwm": 58.2, "ppfd": 291, "temp": 22.7, "photo": 12.2},
    #     # {"time": 20, "pwm": 59.0, "ppfd": 295, "temp": 23.3, "photo": 12.6},
    #     # {"time": 30, "pwm": 53.3, "ppfd": 266, "temp": 23.9, "photo": 11.9},
    #     # {"time": 40, "pwm": 52.2, "ppfd": 261, "temp": 24.4, "photo": 11.8},
    #     # {"time": 50, "pwm": 57.9, "ppfd": 289, "temp": 25.0, "photo": 12.9},
    #     # {"time": 60, "pwm": 58.2, "ppfd": 291, "temp": 25.4, "photo": 13.0},
    #     # {"time": 70, "pwm": 56.8, "ppfd": 284, "temp": 25.9, "photo": 12.7},
    #     # {"time": 80, "pwm": 58.1, "ppfd": 291, "temp": 26.4, "photo": 12.8},
    #     # {"time": 90, "pwm": 57.6, "ppfd": 288, "temp": 26.8, "photo": 12.6},
    #     # {"time": 100, "pwm": 57.8, "ppfd": 289, "temp": 27.1, "photo": 12.5},
    #     # {"time": 110, "pwm": 58.7, "ppfd": 293, "temp": 27.5, "photo": 12.5},
    #     # {"time": 120, "pwm": 57.6, "ppfd": 288, "temp": 27.9, "photo": 12.1},
    #     # {"time": 130, "pwm": 52.7, "ppfd": 264, "temp": 28.2, "photo": 11.1},
    #     # {"time": 140, "pwm": 52.9, "ppfd": 265, "temp": 28.4, "photo": 11.0},
    #     # {"time": 150, "pwm": 54.1, "ppfd": 271, "temp": 28.7, "photo": 11.1},
    #     # {"time": 160, "pwm": 44.0, "ppfd": 220, "temp": 28.8, "photo": 9.1},
    #     # {"time": 170, "pwm": 45.0, "ppfd": 225, "temp": 28.9, "photo": 9.3},
    #     # {"time": 180, "pwm": 44.7, "ppfd": 224, "temp": 29.0, "photo": 9.2},
    #     # {"time": 190, "pwm": 46.6, "ppfd": 233, "temp": 29.0, "photo": 9.6},
    #     # {"time": 200, "pwm": 44.3, "ppfd": 221, "temp": 29.0, "photo": 9.2},
    #     # {"time": 210, "pwm": 48.0, "ppfd": 240, "temp": 29.0, "photo": 9.9},
    #     # {"time": 220, "pwm": 44.3, "ppfd": 221, "temp": 29.0, "photo": 9.1},
    #     # {"time": 230, "pwm": 30.1, "ppfd": 150, "temp": 28.9, "photo": 6.3},
    # ]
    data_points = load_data_from_log("mppi_log.txt")
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

    # 绘制带数据点的热图
    plot_heatmap_with_points(
        prediction_matrix,
        ppfd_values,
        temp_values,
        data_points,
        save_path="photosynthesis_heatmap_with_points.png",
    )

    # 分析实验数据
    analyze_experimental_data(data_points)

    print("\n程序执行完成!")


if __name__ == "__main__":
    main()
