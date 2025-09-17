"""
MPPI 演示脚本（双通道 PWM，闭环运行）

包含两个部分：
- quick_start: 演示一次求解并打印推荐 PWM
- run_closed_loop: 运行多步闭环，记录并绘图（仿照 mpc_farming_organized/core/mppi.py 的任务风格）
"""

from __future__ import annotations

import numpy as np
from mppi import LEDPlant, LEDMPPIController, RB_RATIO_KEY, get_ratio_weights
from led import LedThermalParams, PWMtoPPFDModel, DEFAULT_CALIB_CSV, solve_pwm_for_target_ppfd
import matplotlib.pyplot as plt
import os


def quick_start():
    print("🚀 开始 MPPI 控制器演示 - 详细步骤")
    print("=" * 60)
    
    # 步骤 1: 构建 Plant
    print("\n📋 步骤 1: 构建 LED 植物模型")
    params = LedThermalParams(base_ambient_temp=23.0, thermal_resistance=0.2, time_constant_s=1800.0)
    print(f"   热学参数: 环境温度={params.base_ambient_temp}°C, 热阻={params.thermal_resistance}, 时间常数={params.time_constant_s}s")
    
    plant = LEDPlant(params=params, model_key=RB_RATIO_KEY, use_efficiency=False, heat_scale=1.0)
    print(f"   LED 植物模型: 比例键={RB_RATIO_KEY}, 效率模式=关闭, 热量缩放={1.0}")
    print("   ✅ LED 植物模型创建完成")

    # 步骤 2: 构建控制器
    print("\n🎯 步骤 2: 构建 MPPI 控制器")
    ctrl = LEDMPPIController(plant, horizon=10, num_samples=300, dt=1.0, temperature=1.0, 
                           maintain_rb_ratio=True, rb_ratio_key=RB_RATIO_KEY)
    print(f"   控制参数: 预测步长={10}, 采样数={300}, 时间步长={1.0}s, 温度参数={1.0}")
    print(f"   🔒 R:B 比例约束: 启用 ({RB_RATIO_KEY})")
    
    ctrl.set_constraints(pwm_min=0.0, pwm_max=100.0, temp_min=15.0, temp_max=35.0)
    print(f"   约束条件: PWM范围=[0, 100]%, 温度范围=[15, 35]°C")
    
    ctrl.set_weights(Q_photo=10.0, R_pwm=1e-3, R_dpwm=5e-2, R_power=1e-2)
    print(f"   权重设置: 光合作用={10.0}, PWM惩罚={1e-3}, PWM变化惩罚={5e-2}, 功率惩罚={1e-2}")
    print("   ✅ MPPI 控制器配置完成")

    # 步骤 3: 初始化状态和序列
    print("\n📊 步骤 3: 初始化状态和控制序列")
    current_temp = params.base_ambient_temp
    base_total = 40.0
    w_r, w_b = get_ratio_weights(RB_RATIO_KEY)
    print(f"   当前温度: {current_temp}°C")
    print(f"   基础总 PWM: {base_total}%")
    print(f"   比例权重: R={w_r:.3f}, B={w_b:.3f}")
    
    mean_rb = np.array([base_total * w_r, base_total * w_b], dtype=float)
    mean_seq = np.tile(mean_rb, (ctrl.horizon, 1))
    print(f"   初始 PWM: R={mean_rb[0]:.1f}%, B={mean_rb[1]:.1f}%")
    print(f"   均值序列形状: {mean_seq.shape} (预测步长 × 2通道)")
    print("   ✅ 初始化完成")

    # 步骤 4: MPPI 求解
    print("\n🧠 步骤 4: MPPI 优化求解")
    print("   正在采样控制序列并评估成本...")
    action_rb, seq_rb, ok, min_cost, weights = ctrl.solve(current_temp, mean_sequence=mean_seq)
    r, b = action_rb
    print(f"   ✅ 求解完成!")
    print(f"   求解状态: {'成功' if ok else '失败'}")
    print(f"   最小成本: {min_cost:.4f}")
    print(f"   优化后的 PWM: R={r:.1f}%, B={b:.1f}%")
    
    # 步骤 5: 验证结果
    print("\n🔍 步骤 5: 验证控制结果")
    ppfd, temp_pred, power, pn_rate = plant.step(r, b, dt=1.0)
    print(f"   应用 PWM 后的预测:")
    print(f"   - PPFD: {ppfd:.1f} μmol m⁻² s⁻¹")
    print(f"   - 温度: {temp_pred:.1f}°C")
    print(f"   - 功率: {power:.1f}W")
    print(f"   - 光合速率: {pn_rate:.3f}")
    
    print("\n" + "=" * 60)
    print(f"🎉 MPPI 控制器推荐: R={r:.1f}%  B={b:.1f}%")
    print("=" * 60)


def run_closed_loop(steps: int = 20, dt: float = 900.0, base_total: float = 40.0, target_ppfd_baseline: float = 300.0):
    """按固定步长运行闭环（默认每步15min，共5h），记录并绘图保存。

    - steps: 步数（默认 5 小时 / 15 分钟 = 20 步）
    - dt: 每步秒数（默认 900s = 15 分钟）
    - base_total: MPPI 初始均值的总 PWM（用于初始化 mean 序列）
    - target_ppfd_baseline: 基线的固定 PPFD 目标（按 RB_RATIO_KEY 求解固定 R/B）
    """
    # 使用更慢的热动态与更高热阻以体现温度变化
    params = LedThermalParams(base_ambient_temp=23.0, thermal_resistance=0.2, time_constant_s=1800.0)
    plant = LEDPlant(params=params, model_key=RB_RATIO_KEY, use_efficiency=False, heat_scale=1.0)
    ctrl = LEDMPPIController(plant, horizon=10, num_samples=400, dt=dt, temperature=1.0,
                           maintain_rb_ratio=True, rb_ratio_key=RB_RATIO_KEY)
    ctrl.set_constraints(pwm_min=0.0, pwm_max=100.0, temp_min=15.0, temp_max=35.0)
    ctrl.set_weights(Q_photo=10.0, R_pwm=1e-3, R_dpwm=5e-2, R_power=1e-2)

    # 初始化均值序列
    w_r, w_b = get_ratio_weights(RB_RATIO_KEY)
    mean_rb = np.array([base_total * w_r, base_total * w_b], dtype=float)
    mean_seq = np.tile(mean_rb, (ctrl.horizon, 1))
    temp = params.base_ambient_temp

    # 记录容器（MPPI）
    t_axis = []
    r_list, b_list = [], []
    ppfd_list, temp_list, power_list, pn_list = [], [], [], []

    for k in range(steps):
        action_rb, seq_rb, ok, min_cost, weights = ctrl.solve(temp, mean_sequence=mean_seq)
        r, b = float(action_rb[0]), float(action_rb[1])
        ppfd, new_temp, power, _ = plant.step(r, b, dt)

        t_axis.append(k * dt)
        r_list.append(r)
        b_list.append(b)
        ppfd_list.append(ppfd)
        temp_list.append(new_temp)
        power_list.append(power)
        # PN 使用模型温度与当前 R:B 比例
        rb = r / max(1e-6, (r + b))
        pn_val = plant.get_photosynthesis_rate(ppfd, temperature=new_temp, rb_ratio=rb)
        pn_list.append(float(pn_val))

        temp = new_temp
        mean_seq = np.roll(mean_seq, shift=-1, axis=0)
        mean_seq[-1, :] = action_rb

    # 基线控制：固定 PPFD 与比例键，先用标定模型反解出整数 R/B，再全程恒定
    plant_bl = LEDPlant(params=params, model_key=RB_RATIO_KEY, use_efficiency=False, heat_scale=1.0)
    r_bl = []; b_bl = []; ppfd_bl = []; temp_bl = []; power_bl = []; pn_bl = []
    ppfd_model = PWMtoPPFDModel(include_intercept=True).fit(DEFAULT_CALIB_CSV)
    r_i, b_i, total_i = solve_pwm_for_target_ppfd(
        model=ppfd_model,
        target_ppfd=float(target_ppfd_baseline),
        key=RB_RATIO_KEY,
        ratio_strategy="label",
        integer_output=True,
    )
    action_bl = np.array([float(r_i), float(b_i)], dtype=float)
    for k in range(steps):
        ppfd0, t0, p0, _ = plant_bl.step(action_bl[0], action_bl[1], dt)
        r_bl.append(float(action_bl[0])); b_bl.append(float(action_bl[1]))
        ppfd_bl.append(float(ppfd0)); temp_bl.append(float(t0)); power_bl.append(float(p0))
        rb0 = action_bl[0] / max(1e-6, (action_bl[0] + action_bl[1]))
        pn_bl.append(float(plant_bl.get_photosynthesis_rate(ppfd0, temperature=t0, rb_ratio=rb0)))

    # 绘图
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    axs[0].plot(np.array(t_axis)/3600.0, r_list, label='R PWM (MPPI)')
    axs[0].plot(np.array(t_axis)/3600.0, b_list, label='B PWM (MPPI)')
    axs[0].plot(np.arange(steps)*dt/3600.0, r_bl, '--', label='R PWM (baseline)')
    axs[0].plot(np.arange(steps)*dt/3600.0, b_bl, '--', label='B PWM (baseline)')
    axs[0].set_ylabel('PWM (%)')
    axs[0].legend(); axs[0].grid(True, alpha=0.3)

    axs[1].plot(np.array(t_axis)/3600.0, ppfd_list, 'g-', label='PPFD (MPPI)')
    axs[1].plot(np.arange(steps)*dt/3600.0, ppfd_bl, 'g--', label='PPFD (baseline)')
    axs[1].set_ylabel('PPFD (µmol m⁻² s⁻¹)')
    axs[1].legend(); axs[1].grid(True, alpha=0.3)

    axs[2].plot(np.array(t_axis)/3600.0, temp_list, 'r-', label='Temp (MPPI)')
    axs[2].plot(np.arange(steps)*dt/3600.0, temp_bl, 'r--', label='Temp (baseline)')
    axs[2].axhline(35.0, color='r', linestyle='--', alpha=0.4)
    axs[2].set_ylabel('Temp (°C)')
    axs[2].legend(); axs[2].grid(True, alpha=0.3)

    axs[3].plot(np.array(t_axis)/3600.0, power_list, 'm-', label='Power (MPPI)')
    axs[3].plot(np.arange(steps)*dt/3600.0, power_bl, 'm--', label='Power (baseline)')
    axs[3].set_ylabel('Power (W)')
    axs[4].plot(np.array(t_axis)/3600.0, pn_list, 'c-', label='PN (MPPI)')
    axs[4].plot(np.arange(steps)*dt/3600.0, pn_bl, 'c--', label='PN (baseline)')
    axs[4].set_ylabel('PN')
    axs[4].set_xlabel('Time (h)')
    axs[3].legend(); axs[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, 'mppi_closed_loop.png')
    plt.savefig(out_png, dpi=150)
    print(f"已保存图像: {out_png}")
    print(f"[MPPI] 最终温度: {temp_list[-1]:.1f}°C, 最高温度: {max(temp_list):.1f}°C, 平均功率: {np.mean(power_list):.1f}W, 平均PN: {np.mean(pn_list):.2f}")
    print(f"[BASE] 最终温度: {temp_bl[-1]:.1f}°C, 最高温度: {max(temp_bl):.1f}°C, 平均功率: {np.mean(power_bl):.1f}W, 平均PN: {np.mean(pn_bl):.2f}")

    # 可选：打印每步日志（展示确实是多步）
    for k in range(steps):
        print(
            f"t={k*dt/60:.0f}min | MPPI R={r_list[k]:.1f} B={b_list[k]:.1f} PPFD={ppfd_list[k]:.0f} T={temp_list[k]:.1f} P={power_list[k]:.1f} PN={pn_list[k]:.2f} | "
            f"BASE R={r_bl[k]:.1f} B={b_bl[k]:.1f} PPFD={ppfd_bl[k]:.0f} T={temp_bl[k]:.1f} P={power_bl[k]:.1f} PN={pn_bl[k]:.2f}"
        )


if __name__ == "__main__":
    quick_start()
    # 按 15 分钟/步 共 5 小时运行闭环对比
    run_closed_loop(steps=20, dt=900.0, base_total=40.0, target_ppfd_baseline=300.0)
