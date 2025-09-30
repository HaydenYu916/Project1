#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPPI v2 Rollout Demo: 直接调用 plant.predict 对一条 Solar Vol 控制序列做 n 步前向仿真

运行：
    python examples/mppi_v2_rollout_demo.py
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
        from mppi_v2 import LEDPlant
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

    # 创建植物模型（若 Solar Vol 光合模型缺失，将在 except 中回退）
    print("🌿 初始化以 Solar Vol 控制的 LEDPlant...")
    try:
        plant = LEDPlant(
            base_ambient_temp=22.0,
            max_solar_vol=2.0,
            max_power=100.0,
            power_model=power_model,
            r_b_ratio=0.83,
            use_solar_vol_model=True,
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

    # 设定一条 Solar Vol 控制序列（单位 V）与初始温度
    solar_vol_seq = [1.00, 1.20, 1.40, 1.10, 1.50]
    initial_temp = 22.0
    dt = 900.0  # 每步 900s

    print("\n🚀 对单条控制序列做 n 步前向仿真 (plant.predict)...\n")
    (sv_in, t_pred, p_pred, pn_pred, r_pwm_pred, b_pwm_pred) = plant.predict(
        solar_vol_control_sequence=solar_vol_seq,
        initial_temp=initial_temp,
        dt=dt,
    )

    print("step  solar_vol(V)   R_PWM(%)  B_PWM(%)   temp(°C)   power(W)   pn")
    print("----  ------------   --------  --------   ---------   --------   --------")
    for i in range(len(sv_in)):
        print(
            f"{i+1:>4}  {sv_in[i]:>12.3f}   {r_pwm_pred[i]:>8.1f}  {b_pwm_pred[i]:>8.1f}   "
            f"{t_pred[i]:>9.2f}   {p_pred[i]:>8.1f}   {pn_pred[i]:>8.2f}"
        )

    print("\n✅ rollout demo 完成：这就是 MPPI 内部对每条样本序列所做的多步仿真。")


if __name__ == '__main__':
    main()

