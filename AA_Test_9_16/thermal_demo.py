"""
LED 热学库演示（仅热模型）

本演示展示如何使用 core/led.py 中的 LedThermalModel：
- 仅根据“发热功率 heat_power_w”更新环境温度
- PWM→热功率 的映射仅为演示目的，作为库外部逻辑示例
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, List

import matplotlib.pyplot as plt

from led import (
    LedThermalModel,
    LedThermalParams,
    DEFAULT_BASE_AMBIENT_TEMP,
)


def pwm_to_heat_power(pwm_percent: float, params: LedThermalParams) -> Tuple[float, float, float]:
    """将 PWM 映射为热功率（演示用）

    说明：此映射逻辑不属于热学库，由演示脚本实现，以便将来更灵活地替换。

    返回: (power_electrical_W, efficiency, heat_power_W)
    """
    pwm = max(0.0, min(100.0, float(pwm_percent)))
    frac = pwm / 100.0

    # 演示效率模型：随功率上升而轻微下降
    eff = params.led_efficiency + (1.0 - params.led_efficiency) * math.exp(-params.efficiency_decay * frac)
    eff = max(1e-3, min(1.0, eff))

    # 演示电功率模型：以效率折算
    power = (params.max_power * frac) / eff

    heat_power = power * (1.0 - eff)
    return float(power), float(eff), float(heat_power)


def run_on_off_sequence(model: LedThermalModel, params: LedThermalParams) -> None:
    """运行一个简单的开-关-中功率的演示序列，并绘图显示。"""
    dt = 1.0
    # 场景：75% 60s -> 0% 60s -> 50% 30s
    phases = [
        (75.0, 60),
        (0.0, 60),
        (50.0, 30),
    ]

    time: List[float] = []
    temp: List[float] = []
    pwm: List[float] = []
    heat: List[float] = []

    t = 0.0
    model.reset(params.base_ambient_temp)

    print("LED Thermal-Only Demo")
    print("=" * 50)
    print(f"Base ambient: {params.base_ambient_temp:.1f}°C\n")
    print("Time\tPWM%\tTemp(°C)\tHeat(W)\tTarget(°C)")
    print("-" * 50)

    for pwm_percent, duration in phases:
        for _ in range(duration):
            power, eff, heat_power = pwm_to_heat_power(pwm_percent, params)
            target = model.target_temperature(heat_power)
            new_T = model.step(heat_power, dt)

            # 记录
            time.append(t)
            temp.append(new_T)
            pwm.append(pwm_percent)
            heat.append(heat_power)

            # 打印每 10s
            if int(t) % 10 == 0:
                print(f"{int(t):3d}s\t{pwm_percent:4.0f}\t{new_T:8.2f}\t{heat_power:7.2f}\t{target:9.2f}")

            t += dt

    # 绘图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle("LED Thermal-Only Model Demo", fontsize=14)

    ax1.plot(time, temp, "r-", label="Ambient Temp")
    ax1.axhline(y=params.base_ambient_temp, color="k", linestyle="--", alpha=0.5, label="Base Ambient")
    ax1.set_ylabel("Temperature (°C)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(time, heat, "m-", label="Heat Power (W)")
    ax2.set_ylabel("Heat Power (W)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(time, pwm, "b-", label="PWM (%)")
    ax3.set_ylabel("PWM (%)")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.show()


def main() -> None:
    params = LedThermalParams(
        base_ambient_temp=DEFAULT_BASE_AMBIENT_TEMP,
        thermal_resistance=0.05,
        time_constant_s=7.5,
        thermal_mass=150.0,
        # 预留参数（演示中用于 PWM→热功率 的映射）
        max_ppfd=600.0,
        max_power=86.4,
        led_efficiency=0.8,
        efficiency_decay=2.0,
    )
    model = LedThermalModel(params, initial_temp=params.base_ambient_temp)
    run_on_off_sequence(model, params)


if __name__ == "__main__":
    main()

