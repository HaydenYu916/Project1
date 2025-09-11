from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple
import matplotlib.pyplot as plt


# ---------------------------
# 默认参数宏定义
# ---------------------------
# LED 物理参数
DEFAULT_BASE_AMBIENT_TEMP = 23.0  # 环境基准温度 (°C)
DEFAULT_MAX_PPFD = 600.0          # 最大PPFD (μmol/m²/s)
DEFAULT_MAX_POWER = 86.4          # 最大功率 (W)
DEFAULT_THERMAL_RESISTANCE = 0.05 # 热阻 (K/W)
DEFAULT_TIME_CONSTANT_S = 7.5     # 时间常数 (s)
DEFAULT_THERMAL_MASS = 150.0      # 热容 (J/°C)

# 仿真参数
DEFAULT_DT = 0.1                  # 默认时间步长 (s)


# ---------------------------
# 参数对象（集中管理默认值）
# ---------------------------
@dataclass
class LedParams:
    """LED 与环境模型参数

    - base_ambient_temp: 无 LED 加热时的环境基准温度(°C)LED 不工作时，环境温度
    - max_ppfd: 100% PWM 时的最大 PPFD(μmol/m²/s)
    - max_power: 100% PWM 时的最大功率(W)
    - thermal_resistance: 热阻(K/W)，单位功率引起的温升
    - time_constant_s: 一阶热动态时间常数(秒)
    """

    base_ambient_temp: float = DEFAULT_BASE_AMBIENT_TEMP  # 环境温度后面会调整成arduino的温度
    max_ppfd: float = DEFAULT_MAX_PPFD  # 5:1
    max_power: float = DEFAULT_MAX_POWER  # 62.8 + 23.6 = 86.4 
    thermal_resistance: float = DEFAULT_THERMAL_RESISTANCE
    time_constant_s: float = DEFAULT_TIME_CONSTANT_S  


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _led_step_core(
    pwm_percent: float,
    ambient_temp: float,
    params: LedParams,
    dt: float = 1.0,
) -> Tuple[float, float, float, float]:
    """核心计算（标量实现）

    返回: (ppfd, new_ambient_temp, power, efficiency)
    """
    if not (math.isfinite(pwm_percent) and math.isfinite(ambient_temp) and math.isfinite(dt)):
        raise ValueError("pwm_percent/ambient_temp/dt 必须为有限实数")
    if dt <= 0:
        raise ValueError("dt 必须为正数")

    pwm_fraction = _clip(pwm_percent / 100.0, 0.0, 1.0)

    # PPFD 输出（线性随 PWM）
    ppfd = params.max_ppfd * pwm_fraction

    # LED 光效（高功率时略有下降）
    efficiency = 0.8 + 0.2 * math.exp(-2.0 * pwm_fraction)
    # 防止极端数值
    efficiency = _clip(efficiency, 1e-3, 1.0)

    # 功率与发热
    power = (params.max_power * pwm_fraction) / efficiency
    heat_power = power * (1.0 - efficiency)

    # 目标温度（受发热影响）
    target_ambient_temp = params.base_ambient_temp + heat_power * params.thermal_resistance

    # 一阶热动态：alpha ∈ [0,1]
    tau = max(params.time_constant_s, 1e-6)
    alpha = _clip(dt / tau, 0.0, 1.0)
    new_ambient_temp = ambient_temp + alpha * (target_ambient_temp - ambient_temp)

    return float(ppfd), float(new_ambient_temp), float(power), float(efficiency)


# ---------------------------
# 兼容旧 API 的包装
# ---------------------------
def led_step(
    pwm_percent,
    ambient_temp,
    base_ambient_temp=DEFAULT_BASE_AMBIENT_TEMP,
    dt=DEFAULT_DT,
    max_ppfd=DEFAULT_MAX_PPFD,
    max_power=DEFAULT_MAX_POWER,
    thermal_resistance=DEFAULT_THERMAL_RESISTANCE,
    thermal_mass=DEFAULT_THERMAL_MASS,  # 改为真实热容 (J/°C)，默认值按五板系统
):
    """LED 仿真单步，保持旧参数兼容，同时物理意义更清晰"""
    # time_constant_s = Rth * Cth
    time_constant_s = float(thermal_resistance) * float(thermal_mass)

    params = LedParams(
        base_ambient_temp=float(base_ambient_temp),
        max_ppfd=float(max_ppfd),
        max_power=float(max_power),
        thermal_resistance=float(thermal_resistance),
        time_constant_s=time_constant_s,
    )
    return _led_step_core(float(pwm_percent), float(ambient_temp), params, float(dt))



def led_steady_state(
    pwm_percent,
    base_ambient_temp=DEFAULT_BASE_AMBIENT_TEMP,
    max_ppfd=DEFAULT_MAX_PPFD,
    max_power=DEFAULT_MAX_POWER,
    thermal_resistance=DEFAULT_THERMAL_RESISTANCE,
):
    """固定 PWM 下的稳态量（向后兼容的签名）

    返回: (ppfd, final_ambient_temp, power, efficiency)
    """
    params = LedParams(
        base_ambient_temp=float(base_ambient_temp),
        max_ppfd=float(max_ppfd),
        max_power=float(max_power),
        thermal_resistance=float(thermal_resistance),
        time_constant_s=240.0,  # 稳态与时间常数无关，此处仅占位
    )

    pwm_fraction = _clip(float(pwm_percent) / 100.0, 0.0, 1.0)
    ppfd = params.max_ppfd * pwm_fraction
    efficiency = 0.8 + 0.2 * math.exp(-2.0 * pwm_fraction)
    efficiency = _clip(efficiency, 1e-3, 1.0)
    power = (params.max_power * pwm_fraction) / efficiency
    heat_power = power * (1.0 - efficiency)
    final_ambient_temp = params.base_ambient_temp + heat_power * params.thermal_resistance
    return float(ppfd), float(final_ambient_temp), float(power), float(efficiency)


# ---------------------------
# 简单演示（仅打印，不保存 PNG/不依赖 GUI）
# ---------------------------
def run_led_on_off_example() -> None:
    """LED 开/关温度示例：打印 + 绘图（仅显示，不保存）"""

    base_ambient = DEFAULT_BASE_AMBIENT_TEMP
    ambient_temp = DEFAULT_BASE_AMBIENT_TEMP
    dt = 1.0

    # 采样数据用于绘图
    time_data = []
    temp_data = []
    ppfd_data = []
    pwm_data = []
    now_t = 0

    print("LED On/Off Temperature Example")
    print("=" * 50)
    print(f"Base ambient temperature: {base_ambient}°C\n")

    # Phase 1: 75% PWM 加热 60s
    print("Phase 1: LED ON (75% PWM) - Heating up")
    print("Time\tPWM\tPPFD\tTemp\tRise")
    print("-" * 40)
    for t in range(60):
        ppfd, ambient_temp, power, eff = led_step(75.0, ambient_temp, base_ambient, dt)
        time_data.append(now_t)
        temp_data.append(ambient_temp)
        ppfd_data.append(ppfd)
        pwm_data.append(75.0)
        now_t += dt
        if t % 10 == 0:
            temp_rise = ambient_temp - base_ambient
            print(f"{t}s\t75%\t{ppfd:.0f}\t{ambient_temp:.1f}°C\t+{temp_rise:.1f}°C")

    print()

    # Phase 2: 0% PWM 冷却 60s
    print("Phase 2: LED OFF (0% PWM) - Cooling down")
    print("Time\tPWM\tPPFD\tTemp\tDiff from base")
    print("-" * 45)
    for t in range(60):
        ppfd, ambient_temp, power, eff = led_step(0.0, ambient_temp, base_ambient, dt)
        time_data.append(now_t)
        temp_data.append(ambient_temp)
        ppfd_data.append(ppfd)
        pwm_data.append(0.0)
        now_t += dt
        if t % 10 == 0:
            temp_diff = ambient_temp - base_ambient
            print(f"{60+t}s\t0%\t{ppfd:.0f}\t{ambient_temp:.1f}°C\t{temp_diff:+.1f}°C")

    print()

    # Phase 3: 50% PWM 再加热 30s
    print("Phase 3: LED ON again (50% PWM) - Moderate heating")
    print("Time\tPWM\tPPFD\tTemp\tRise")
    print("-" * 40)
    for t in range(30):
        ppfd, ambient_temp, power, eff = led_step(50.0, ambient_temp, base_ambient, dt)
        time_data.append(now_t)
        temp_data.append(ambient_temp)
        ppfd_data.append(ppfd)
        pwm_data.append(50.0)
        now_t += dt
        if t % 10 == 0:
            temp_rise = ambient_temp - base_ambient
            print(f"{120+t}s\t50%\t{ppfd:.0f}\t{ambient_temp:.1f}°C\t+{temp_rise:.1f}°C")

    # 绘图（只显示，不保存）
    plot_results(time_data, temp_data, ppfd_data, pwm_data, base_ambient)


def plot_results(time_data, temp_data, ppfd_data, pwm_data, base_ambient):
    """绘制仿真结果（只显示，不保存）"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle("LED On/Off Heating and Cooling Example", fontsize=14)

    # 温度曲线
    ax1.plot(time_data, temp_data, "r-", linewidth=2, label="Ambient Temperature")
    ax1.axhline(y=base_ambient, color="k", linestyle="--", alpha=0.5, label="Base Ambient")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Temperature Response")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axvspan(0, 60, alpha=0.15, color="red", label="LED ON (75%)")
    ax1.axvspan(60, 120, alpha=0.15, color="blue", label="LED OFF (0%)")
    ax1.axvspan(120, 150, alpha=0.15, color="orange", label="LED ON (50%)")

    # PPFD 曲线
    ax2.plot(time_data, ppfd_data, "g-", linewidth=2)
    ax2.set_ylabel("PPFD (μmol/m²/s)")
    ax2.set_title("Light Output (PPFD)")
    ax2.grid(True, alpha=0.3)

    # PWM 曲线
    ax3.plot(time_data, pwm_data, "b-", linewidth=2)
    ax3.set_ylabel("PWM (%)")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_title("PWM Control Signal")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_led_on_off_example()
