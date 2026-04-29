#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPPI v2 (Growpod / PPFD 控制) 最小可运行演示

演示内容:
- 使用 PWM→Power 标定文件创建功率模型
- 创建以 PPFD 为控制量的 LEDPlant (Pn 模型: EnvtoPN, 输入 [T,CO2,R:B,PPFD])
- 用 LEDMPPIController 求解最优 PPFD,并调用 plant.step 执行单步仿真

运行:
    python examples/mppi_v2_demo.py
"""

import os
import sys


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    src_dir = os.path.join(project_root, 'src')
    data_dir = os.path.join(project_root, 'data')
    sys.path.insert(0, src_dir)

    try:
        from mppi_v2 import LEDPlant, LEDMPPIController
        from led import PWMtoPowerModel
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print(f"请确认路径存在: {src_dir}")
        sys.exit(1)

    calib_csv = os.path.join(data_dir, 'calib_data.csv')
    if not os.path.exists(calib_csv):
        print(f"❌ 找不到标定文件: {calib_csv}")
        sys.exit(1)

    print("🔧 拟合功率模型 (PWM → Power)...")
    power_model = PWMtoPowerModel().fit(calib_csv)

    print("🌿 初始化以 PPFD 控制的 LEDPlant (target=400)...")
    plant = LEDPlant(
        base_ambient_temp=22.0,
        max_ppfd=500.0,
        max_power=100.0,
        power_model=power_model,
        r_b_ratio=0.83,
        target_ppfd=400.0,
        use_pn_model=True,
        use_sp_to_ppfd=False,   # demo 不读光谱
    )

    print("🧠 初始化 MPPI 控制器 (PPFD 控制)...")
    ctrl = LEDMPPIController(
        plant=plant,
        horizon=5,
        num_samples=200,
        dt=900.0,
        temperature=0.8,
    )

    print("\n🚀 开始 4 步控制-仿真循环 (每步 900s)...\n")
    for k in range(4):
        u, u_seq, ok, cost, w = ctrl.solve(current_temp=plant.ambient_temp)
        ppfd, temp, power, pn = plant.step(ppfd=u, dt=900.0)
        print(
            f"step={k} | PPFD={ppfd:.1f} μmol | T={temp:.2f} °C | P={power:.1f} W | Pn={pn:.2f}"
        )

    print("\n✅ demo 完成: 控制器输出的 PPFD 传入 plant.step 完成单步推进。")


if __name__ == '__main__':
    main()
