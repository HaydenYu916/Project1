#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPPI v2 (Solar Vol 控制) 最小可运行演示

演示内容：
- 使用 PWM→Power 标定文件创建功率模型
- 创建以 Solar Vol 为控制量的 LEDPlant
- 用 LEDMPPIController 求解最优 Solar Vol，并调用 plant.step 执行单步仿真

运行：
    python examples/mppi_v2_demo.py
"""

import os
import sys


def main():
    # 添加 src 到路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    src_dir = os.path.join(project_root, 'src')
    data_dir = os.path.join(project_root, 'data')
    sys.path.insert(0, src_dir)

    # 导入库
    try:
        from mppi_v2 import LEDPlant, LEDMPPIController
        from led import PWMtoPowerModel
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print(f"请确认路径存在: {src_dir}")
        sys.exit(1)

    # 加载功率标定
    calib_csv = os.path.join(data_dir, 'calib_data.csv')
    if not os.path.exists(calib_csv):
        print(f"❌ 找不到标定文件: {calib_csv}")
        print("请确认 data/calib_data.csv 存在")
        sys.exit(1)

    print("🔧 拟合功率模型 (PWM → Power)...")
    power_model = PWMtoPowerModel().fit(calib_csv)

    # 创建植物模型（优先启用 Solar Vol 光合模型；若缺模型则自动回退）
    print("🌿 初始化以 Solar Vol 控制的 LEDPlant...")
    try:
        plant = LEDPlant(
            base_ambient_temp=22.0,
            max_solar_vol=2.0,
            max_power=100.0,
            power_model=power_model,
            r_b_ratio=0.83,
            use_solar_vol_model=True,  # 若模型文件缺失，将在下面捕获并回退
        )
    except Exception as e:
        print(f"⚠️ Solar Vol 光合模型不可用，回退到禁用模型: {e}")
        plant = LEDPlant(
            base_ambient_temp=22.0,
            max_solar_vol=2.0,
            max_power=100.0,
            power_model=power_model,
            r_b_ratio=0.83,
            use_solar_vol_model=False,
        )

    # 创建控制器（dt=900 表示 15 分钟步长）
    print("🧠 初始化 MPPI 控制器 (Solar Vol 控制)...")
    ctrl = LEDMPPIController(
        plant=plant,
        horizon=5,
        num_samples=200,
        dt=900.0,
        temperature=0.8,
    )

    print("\n🚀 开始 4 步控制-仿真循环 (每步 900s)...\n")
    for k in range(4):
        # 由控制器给出本步的最优 Solar Vol 控制量
        u, u_seq, ok, cost, w = ctrl.solve(current_temp=plant.ambient_temp)
        # 将控制量传入 step，推进模型一步
        sv, temp, power, pn = plant.step(solar_vol=u, dt=900.0)
        print(
            f"step={k} | solar_vol={sv:.3f} V | temp={temp:.2f} °C | power={power:.1f} W | pn={pn:.2f}"
        )

    print("\n✅ demo 完成：已展示‘控制器输出的 solar_vol 传入 step’的完整流程。")


if __name__ == '__main__':
    main()

