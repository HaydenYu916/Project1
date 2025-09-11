from __future__ import annotations

import math
from typing import Tuple


def _clip(x: float, lo: float, hi: float) -> float:
    """将数值限制在指定范围内"""
    return max(lo, min(x, hi))


class LEDSystem:
    """
    LED 照明系统一阶热模型。

    封装LED系统的物理参数与动态状态，可用于时间步进仿真和稳态估算。
    """

    def __init__(
        self,
        base_ambient_temp: float = 23.0,
        max_ppfd: float = 600.0,
        max_power: float = 86.4,
        thermal_resistance: float = 0.05,
        thermal_mass: float = 150.0,
        initial_temp: float | None = None,
    ):
        self.base_ambient_temp = base_ambient_temp
        self.max_ppfd = max_ppfd
        self.max_power = max_power
        self.thermal_resistance = thermal_resistance
        self.thermal_mass = thermal_mass

        # 热时间常数(秒) = 热阻 × 热容
        self.time_constant_s = self.thermal_resistance * self.thermal_mass

        # 当前环境温度
        self.ambient_temp = (
            initial_temp if initial_temp is not None else self.base_ambient_temp
        )

    def step(self, pwm_percent: float, dt: float) -> Tuple[float, float, float, float]:
        """
        仿真一个时间步长，返回 (ppfd, new_ambient_temp, power, efficiency)
        """
        if not (math.isfinite(pwm_percent) and math.isfinite(dt)):
            raise ValueError("pwm_percent和dt必须是有限实数")
        if dt <= 0:
            raise ValueError("dt必须为正数")

        pwm_fraction = _clip(pwm_percent / 100.0, 0.0, 1.0)

        # PPFD 输出（与 PWM 线性）
        ppfd = self.max_ppfd * pwm_fraction

        # 光效（高占空比略降）
        efficiency = 0.8 + 0.2 * math.exp(-2.0 * pwm_fraction)
        efficiency = _clip(efficiency, 1e-3, 1.0)

        # 功率与发热
        power = (self.max_power * pwm_fraction) / efficiency
        heat_power = power * (1.0 - efficiency)

        # 目标温度（该 PWM 恒定时的稳态温度）
        target_temp = self.base_ambient_temp + heat_power * self.thermal_resistance

        # 一阶热动态离散更新
        tau = max(self.time_constant_s, 1e-6)
        alpha = 1.0 - math.exp(-dt / tau)
        self.ambient_temp += alpha * (target_temp - self.ambient_temp)

        return float(ppfd), float(self.ambient_temp), float(power), float(efficiency)

    def steady_state(self, pwm_percent: float) -> Tuple[float, float, float, float]:
        """返回给定 PWM 下的稳态 (ppfd, final_ambient_temp, power, efficiency)"""
        pwm_fraction = _clip(float(pwm_percent) / 100.0, 0.0, 1.0)

        ppfd = self.max_ppfd * pwm_fraction
        efficiency = 0.8 + 0.2 * math.exp(-2.0 * pwm_fraction)
        efficiency = _clip(efficiency, 1e-3, 1.0)
        power = (self.max_power * pwm_fraction) / efficiency
        heat_power = power * (1.0 - efficiency)
        final_ambient_temp = (
            self.base_ambient_temp + heat_power * self.thermal_resistance
        )

        return float(ppfd), float(final_ambient_temp), float(power), float(efficiency)


# -------------------------------------------------
# 面向 mppi.py 的函数式 API（无状态包装）
# -------------------------------------------------
def led_step(
    *,
    pwm_percent: float,
    ambient_temp: float,
    base_ambient_temp: float,
    dt: float,
    max_ppfd: float,
    max_power: float,
    thermal_resistance: float,
    thermal_mass: float,
):
    """
    与 mppi.py 期望的签名一致的单步仿真函数。
    根据传入参数即时构建 LEDSystem，初始温度使用 ambient_temp，进行一步更新。
    返回 (ppfd, new_ambient_temp, power, efficiency)
    """
    sys = LEDSystem(
        base_ambient_temp=base_ambient_temp,
        max_ppfd=max_ppfd,
        max_power=max_power,
        thermal_resistance=thermal_resistance,
        thermal_mass=thermal_mass,
        initial_temp=ambient_temp,
    )
    return sys.step(pwm_percent=float(pwm_percent), dt=float(dt))


def led_steady_state(
    *,
    pwm_percent: float,
    base_ambient_temp: float,
    max_ppfd: float,
    max_power: float,
    thermal_resistance: float,
    thermal_mass: float,
):
    """
    与 mppi.py 期望的签名一致的稳态计算函数。
    返回 (ppfd, final_ambient_temp, power, efficiency)
    """
    sys = LEDSystem(
        base_ambient_temp=base_ambient_temp,
        max_ppfd=max_ppfd,
        max_power=max_power,
        thermal_resistance=thermal_resistance,
        thermal_mass=thermal_mass,
        initial_temp=base_ambient_temp,
    )
    return sys.steady_state(pwm_percent=float(pwm_percent))

