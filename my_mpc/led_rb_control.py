#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红蓝双通道 PWM 支持（基于 led.py 的最新模型）

本模块在不改变外部接口的前提下，按 led.py 的最新建模方式
对红蓝控制进行了重构与对齐：

- 同步 led.py 的默认物理参数与记号（环境温度、热阻、热容→时间常数等）。
- 统一功率/效率/热模型的计算公式与数值稳定性处理。
- 保留原有公开类与方法：RedBlueDataParser, PpfdLinearInterpolator,
  RedBlueLEDModel，以兼容 mppi_rb_control.py 与 create_rb_led_system.py。
- 新增与 led.py 风格一致的便捷函数：rb_led_step 与 rb_led_steady_state。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import optimize as _opt

# 从 led.py 导入默认常量与记号，确保参数一致性
from led import (
    DEFAULT_BASE_AMBIENT_TEMP,    # 默认基础环境温度
    DEFAULT_THERMAL_RESISTANCE,   # 默认热阻
    DEFAULT_THERMAL_MASS,         # 默认热容
    DEFAULT_MAX_POWER,            # 默认最大功率
    DEFAULT_DT,                   # 默认时间步长
)


# ---------------------------
# 数据解析与插值器
# ---------------------------
class PpfdLinearInterpolator:
    """PPFD线性插值器：根据红蓝PWM值预测光合光子通量密度
    
    使用线性模型：PPFD ≈ a_r * red_pwm + a_b * blue_pwm + bias
    其中 a_r, a_b 为红蓝通道的系数，bias 为偏置项
    """

    def __init__(self, a_red: float, a_blue: float, bias: float = 0.0):
        """初始化线性插值器
        
        Args:
            a_red: 红色通道PWM系数
            a_blue: 蓝色通道PWM系数  
            bias: 线性模型偏置项
        """
        self.a_red = float(a_red)
        self.a_blue = float(a_blue)
        self.bias = float(bias)

    def predict_ppfd(self, red_pwm: float, blue_pwm: float) -> float:
        """根据红蓝PWM值预测PPFD
        
        Args:
            red_pwm: 红色PWM百分比 (0-100)
            blue_pwm: 蓝色PWM百分比 (0-100)
            
        Returns:
            预测的PPFD值 (μmol/m²/s)
        """
        # 限制PWM值在有效范围内
        red_pwm = float(np.clip(red_pwm, 0.0, 100.0))
        blue_pwm = float(np.clip(blue_pwm, 0.0, 100.0))
        # 计算线性预测值
        ppfd = self.bias + self.a_red * red_pwm + self.a_blue * blue_pwm
        # 确保PPFD非负
        return float(max(0.0, ppfd))

    def __call__(self, red_pwm: float, blue_pwm: float) -> float:
        """使对象可调用，直接调用predict_ppfd方法"""
        return self.predict_ppfd(red_pwm, blue_pwm)


class RedBlueDataParser:
    """红蓝数据解析器：解析用户输入数据并拟合线性PPFD模型
    
    用于处理格式为 "红蓝比例-目标PPFD-红蓝PWM" 的数据行，
    并自动拟合线性插值器用于PPFD预测
    """

    def __init__(self) -> None:
        """初始化数据解析器"""
        self.df: Optional[pd.DataFrame] = None  # 存储解析后的数据
        self._interpolator: Optional[PpfdLinearInterpolator] = None  # 拟合的线性插值器

    @staticmethod
    def _parse_line(line: str) -> Tuple[float, float, float, float, float]:
        """解析单行数据字符串
        
        数据格式: "红比例:蓝比例-目标PPFD-红PWM:蓝PWM"
        例如: "1:2-300-39:78" 表示红蓝比例1:2，目标PPFD 300，红PWM 39%，蓝PWM 78%
        
        Args:
            line: 待解析的数据行字符串
            
        Returns:
            (ratio_r, ratio_b, target_ppfd, red_pwm, blue_pwm) 元组
            
        Raises:
            ValueError: 当数据格式不正确时
        """
        s = line.strip()
        if not s:
            raise ValueError("空行无法解析")
        try:
            # 按"-"分割：红蓝比例-目标PPFD-红蓝PWM
            rb_part, ppfd_part, pwm_part = s.split("-")
            # 解析红蓝比例
            r_s, b_s = rb_part.split(":")
            # 解析红蓝PWM
            red_s, blue_s = pwm_part.split(":")
            # 转换为浮点数
            ratio_r = float(r_s)
            ratio_b = float(b_s)
            target_ppfd = float(ppfd_part)
            red_pwm = float(red_s)
            blue_pwm = float(blue_s)
            return ratio_r, ratio_b, target_ppfd, red_pwm, blue_pwm
        except Exception as e:
            raise ValueError(f"无法解析数据行: {line}") from e

    def load_data_from_list(self, lines: Iterable[str]) -> pd.DataFrame:
        """从字符串列表加载数据并返回DataFrame
        
        解析每行数据并计算红蓝比例，然后拟合线性PPFD模型
        
        Args:
            lines: 包含数据行的可迭代对象
            
        Returns:
            包含解析数据的DataFrame，列包括：
            ['ratio_r', 'ratio_b', 'rb_ratio', 'ppfd', 'red_pwm', 'blue_pwm']
            
        Raises:
            ValueError: 当没有有效数据时
        """
        rows = []
        for line in lines:
            if not str(line).strip():
                continue
            # 解析单行数据
            ratio_r, ratio_b, target_ppfd, red_pwm, blue_pwm = self._parse_line(str(line))
            # 计算红蓝比例（避免除零）
            rb_ratio = float(ratio_r) / max(float(ratio_b), 1e-6)
            # 添加到数据行
            rows.append(
                {
                    "ratio_r": float(ratio_r),
                    "ratio_b": float(ratio_b),
                    "rb_ratio": rb_ratio,
                    "ppfd": float(target_ppfd),
                    "red_pwm": float(red_pwm),
                    "blue_pwm": float(blue_pwm),
                }
            )
        if not rows:
            raise ValueError("未提供有效数据")
        # 创建DataFrame
        self.df = pd.DataFrame(rows)

        # 自动拟合线性模型
        self._fit_linear_model()

        return self.df

    def _fit_linear_model(self) -> None:
        """使用最小二乘法拟合线性PPFD模型
        
        拟合模型：PPFD = bias + a_red * red_pwm + a_blue * blue_pwm
        
        Raises:
            ValueError: 当没有数据可用于拟合时
        """
        if self.df is None or self.df.empty:
            raise ValueError("无数据可用于拟合")

        # 准备输入特征（红蓝PWM值）和目标值（PPFD）
        X = self.df[["red_pwm", "blue_pwm"]].to_numpy(dtype=float)
        y = self.df["ppfd"].to_numpy(dtype=float)

        # 添加偏置项进行线性最小二乘拟合
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])  # [bias, red, blue]
        coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        bias, a_red, a_blue = coeffs

        # 确保系数非负（物理合理性）
        a_red = float(max(0.0, a_red))
        a_blue = float(max(0.0, a_blue))

        # 创建线性插值器
        self._interpolator = PpfdLinearInterpolator(a_red, a_blue, bias)

    def get_interpolator(self) -> PpfdLinearInterpolator:
        """获取拟合的线性插值器
        
        Returns:
            用于PPFD预测的线性插值器
        """
        if self._interpolator is None:
            self._fit_linear_model()
        assert self._interpolator is not None
        return self._interpolator


# ---------------------------
# 红/蓝 LED 物理与热模型（与 led.py 对齐）
# ---------------------------
@dataclass
class RBLedParams:
    """红蓝LED物理参数配置类
    
    包含LED系统的热学参数和功率限制，与led.py中的参数定义保持一致
    """
    base_ambient_temp: float = DEFAULT_BASE_AMBIENT_TEMP  # 基础环境温度 (°C)
    thermal_resistance: float = DEFAULT_THERMAL_RESISTANCE   # 热阻 (K/W)，单位功率温升
    time_constant_s: float = DEFAULT_THERMAL_RESISTANCE * DEFAULT_THERMAL_MASS  # 一阶热惯性时间常数 τ = Rth*Cth (s)
    max_power: float = DEFAULT_MAX_POWER  # 两通道合计在100%+100%时的标称最大功率 (W)

    @staticmethod
    def from_physical_params(
        base_ambient_temp: float = DEFAULT_BASE_AMBIENT_TEMP,
        thermal_resistance: float = DEFAULT_THERMAL_RESISTANCE,
        thermal_mass: float = DEFAULT_THERMAL_MASS,
        max_power: float = DEFAULT_MAX_POWER,
    ) -> "RBLedParams":
        """通过热阻与热容计算时间常数，构建参数对象
        
        与led.py的行为保持一致，通过物理参数计算热时间常数
        
        Args:
            base_ambient_temp: 基础环境温度 (°C)
            thermal_resistance: 热阻 (K/W)
            thermal_mass: 热容 (J/K)
            max_power: 最大功率 (W)
            
        Returns:
            配置好的RBLedParams对象
        """
        # 计算热时间常数：τ = Rth * Cth
        tau = float(thermal_resistance) * float(thermal_mass)
        return RBLedParams(
            base_ambient_temp=float(base_ambient_temp),
            thermal_resistance=float(thermal_resistance),
            time_constant_s=float(tau),
            max_power=float(max_power),
        )


def _clip(x: float, lo: float, hi: float) -> float:
    """数值裁剪函数：将值限制在指定范围内
    
    Args:
        x: 待裁剪的值
        lo: 下界
        hi: 上界
        
    Returns:
        裁剪后的值
    """
    return lo if x < lo else hi if x > hi else x


class RedBlueLEDModel:
    """红蓝双通道LED模型
    
    集成PPFD预测、功率计算和热学建模功能：
    - 使用插值器估计PPFD输出
    - 使用简化效率模型估计功耗
    - 使用一阶热惯性模型估计环境温度变化
    """

    def __init__(
        self,
        led_data_interpolator=None,
        max_power: float = DEFAULT_MAX_POWER,
        params: Optional[RBLedParams] = None,
    ):
        """初始化红蓝LED模型
        
        Args:
            led_data_interpolator: PPFD插值器，支持可调用对象或带predict_ppfd方法的对象
            max_power: 合成通道在100%+100%时的标称最大功率 (W)
            params: RBLedParams对象，若提供则覆盖max_power与默认参数
        """
        self.interp = led_data_interpolator  # PPFD插值器
        self.params = params if params is not None else RBLedParams(max_power=max_power)  # 物理参数

    def _estimate_ppfd(self, red_pwm: float, blue_pwm: float) -> float:
        """估计PPFD输出值
        
        Args:
            red_pwm: 红色PWM百分比 (0-100)
            blue_pwm: 蓝色PWM百分比 (0-100)
            
        Returns:
            估计的PPFD值 (μmol/m²/s)
        """
        if self.interp is None:
            # 兜底方案：使用默认线性权重（无数据时）
            a_r, a_b = 5.0, 5.0  # 默认系数，避免0输出
            return max(0.0, a_r * red_pwm + a_b * blue_pwm)

        # 支持可调用对象形式
        if callable(self.interp):
            try:
                return float(self.interp(red_pwm, blue_pwm))
            except TypeError:
                pass
        # 支持带predict_ppfd方法的对象
        if hasattr(self.interp, "predict_ppfd"):
            return float(self.interp.predict_ppfd(red_pwm, blue_pwm))
        # 最后兜底：简单相加
        return float(red_pwm + blue_pwm)

    def step(
        self,
        red_pwm: float,
        blue_pwm: float,
        ambient_temp: float,
        base_ambient_temp: float = DEFAULT_BASE_AMBIENT_TEMP,
        dt: float = DEFAULT_DT,
    ) -> Tuple[float, float, float, float]:
        """执行单步时间演化
        
        根据当前PWM设置和环境温度，计算下一时刻的状态
        
        Args:
            red_pwm: 红色PWM百分比 (0-100)
            blue_pwm: 蓝色PWM百分比 (0-100)
            ambient_temp: 当前环境温度 (°C)
            base_ambient_temp: 基础环境温度 (°C)
            dt: 时间步长 (s)
            
        Returns:
            (ppfd, new_ambient_temp, power, rb_ratio) 元组：
            - ppfd: 预测的PPFD值 (μmol/m²/s)
            - new_ambient_temp: 新的环境温度 (°C)
            - power: 总功耗 (W)
            - rb_ratio: 红蓝比例
            
        Raises:
            ValueError: 当输入参数无效时
        """
        # 输入参数有效性检查
        if not (np.isfinite(red_pwm) and np.isfinite(blue_pwm) and np.isfinite(ambient_temp) and np.isfinite(dt)):
            raise ValueError("red_pwm/blue_pwm/ambient_temp/dt 必须为有限实数")
        if dt <= 0:
            raise ValueError("dt 必须为正数")

        # PWM值裁剪到有效范围
        r = _clip(float(red_pwm), 0.0, 100.0)
        b = _clip(float(blue_pwm), 0.0, 100.0)

        # 估计PPFD输出
        ppfd = self._estimate_ppfd(r, b)

        # 计算合成PWM分数用于效率/功耗模型（0~1）
        pwm_fraction = _clip((r + b) / 200.0, 0.0, 1.0)

        # 简化效率模型：功率高时效率降低
        efficiency = 0.8 + 0.2 * math.exp(-2.0 * pwm_fraction)
        efficiency = _clip(efficiency, 1e-3, 1.0)

        # 计算功率与发热（合并两通道）
        power = (self.params.max_power * pwm_fraction) / efficiency
        heat_power = power * (1.0 - efficiency)

        # 一阶热惯性模型（与led.py对齐）
        target_ambient_temp = float(base_ambient_temp) + heat_power * self.params.thermal_resistance
        tau = max(float(self.params.time_constant_s), 1e-6)  # 避免除零
        alpha = _clip(float(dt) / tau, 0.0, 1.0)  # 时间常数归一化
        new_ambient_temp = float(ambient_temp) + alpha * (target_ambient_temp - float(ambient_temp))

        # 计算红蓝比例（PWM比例近似代替光谱比）
        rb_ratio = (r / max(b, 1e-6)) if (r > 0.0 and b > 0.0) else (float("inf") if b == 0.0 and r > 0.0 else 0.0)
        if not np.isfinite(rb_ratio):
            rb_ratio = 0.0 if b > 0.0 else 1.0

        return float(ppfd), float(new_ambient_temp), float(power), float(rb_ratio)


# ---------------------------
# 便捷函数（与 led.py 旧 API 风格一致）
# ---------------------------
def rb_led_step(
    red_pwm: float,
    blue_pwm: float,
    ambient_temp: float,
    base_ambient_temp: float = DEFAULT_BASE_AMBIENT_TEMP,
    dt: float = DEFAULT_DT,
    max_power: float = DEFAULT_MAX_POWER,
    thermal_resistance: float = DEFAULT_THERMAL_RESISTANCE,
    thermal_mass: float = DEFAULT_THERMAL_MASS,
    led_data_interpolator=None,
) -> Tuple[float, float, float, float]:
    """红蓝通道LED仿真单步（保持旧参数兼容，物理意义与led.py对齐）
    
    便捷函数，用于快速进行单步仿真，保持与led.py相同的参数接口
    
    Args:
        red_pwm: 红色PWM百分比 (0-100)
        blue_pwm: 蓝色PWM百分比 (0-100)
        ambient_temp: 当前环境温度 (°C)
        base_ambient_temp: 基础环境温度 (°C)
        dt: 时间步长 (s)
        max_power: 最大功率 (W)
        thermal_resistance: 热阻 (K/W)
        thermal_mass: 热容 (J/K)
        led_data_interpolator: PPFD插值器
        
    Returns:
        (ppfd, new_ambient_temp, power, rb_ratio) 元组
    """
    # 从物理参数构建配置对象
    params = RBLedParams.from_physical_params(
        base_ambient_temp=base_ambient_temp,
        thermal_resistance=thermal_resistance,
        thermal_mass=thermal_mass,
        max_power=max_power,
    )
    # 创建模型并执行单步
    model = RedBlueLEDModel(led_data_interpolator=led_data_interpolator, params=params)
    return model.step(red_pwm, blue_pwm, ambient_temp, base_ambient_temp, dt)


def rb_led_steady_state(
    red_pwm: float,
    blue_pwm: float,
    base_ambient_temp: float = DEFAULT_BASE_AMBIENT_TEMP,
    max_power: float = DEFAULT_MAX_POWER,
    thermal_resistance: float = DEFAULT_THERMAL_RESISTANCE,
    led_data_interpolator=None,
) -> Tuple[float, float, float, float]:
    """计算固定红蓝PWM下的稳态量（向后兼容签名）
    
    计算在给定PWM设置下达到热平衡时的稳态值，不涉及时间演化
    
    Args:
        red_pwm: 红色PWM百分比 (0-100)
        blue_pwm: 蓝色PWM百分比 (0-100)
        base_ambient_temp: 基础环境温度 (°C)
        max_power: 最大功率 (W)
        thermal_resistance: 热阻 (K/W)
        led_data_interpolator: PPFD插值器
        
    Returns:
        (ppfd, final_ambient_temp, power, rb_ratio) 元组
    """
    # 构建模型参数（时间常数对稳态无影响，设为占位值）
    params = RBLedParams(
        base_ambient_temp=float(base_ambient_temp),
        thermal_resistance=float(thermal_resistance),
        time_constant_s=1.0,  # 占位：稳态无需时间常数
        max_power=float(max_power),
    )
    model = RedBlueLEDModel(led_data_interpolator=led_data_interpolator, params=params)

    # 计算稳态量（与step方法相同的效率/功率模型）
    r = _clip(float(red_pwm), 0.0, 100.0)
    b = _clip(float(blue_pwm), 0.0, 100.0)
    ppfd = model._estimate_ppfd(r, b)
    pwm_fraction = _clip((r + b) / 200.0, 0.0, 1.0)
    efficiency = _clip(0.8 + 0.2 * math.exp(-2.0 * pwm_fraction), 1e-3, 1.0)
    power = (params.max_power * pwm_fraction) / efficiency
    heat_power = power * (1.0 - efficiency)
    final_ambient_temp = float(base_ambient_temp) + heat_power * params.thermal_resistance

    # 计算红蓝比例
    rb_ratio = (r / max(b, 1e-6)) if (r > 0.0 and b > 0.0) else (float("inf") if b == 0.0 and r > 0.0 else 0.0)
    if not np.isfinite(rb_ratio):
        rb_ratio = 0.0 if b > 0.0 else 1.0

    return float(ppfd), float(final_ambient_temp), float(power), float(rb_ratio)


# ---------------------------
# 基于实测数据的热参数辨识
# ---------------------------
def _simulate_step_first_order(Tk: float, Uk: float, base: float, K: float, tau: float, dt: float) -> float:
    """连续一阶系统的离散精确解：

    T_{k+1} = T_k * exp(-dt/τ) + (1-exp(-dt/τ)) * (base + K * U_k)
    """
    dt = float(max(dt, 1e-9))
    tau = float(max(tau, 1e-9))
    a = math.exp(-dt / tau)
    return float(Tk) * a + (1.0 - a) * (float(base) + float(K) * float(Uk))


def fit_rb_thermal_from_pwm(
    time_s: Iterable[float],
    red_pwm: Iterable[float],
    blue_pwm: Iterable[float],
    temp_c: Iterable[float],
    base_ambient_temp: float = DEFAULT_BASE_AMBIENT_TEMP,
    max_power: float = DEFAULT_MAX_POWER,
) -> RBLedParams:
    """根据实测的 PWM 与温度序列拟合热参数（R_th 与 τ）。

    - 输入：按等间隔或变间隔时间序列 time_s（秒）、对应的 red/blue PWM（%）与温度（°C）。
    - 模型：与 RedBlueLEDModel 相同，先将 PWM→热功率 U_k（包含效率折减），
            再拟合 T 动力学的一阶参数 K=R_th 与 τ。

    返回：RBLedParams，其中 thermal_resistance=拟合的 R_th，time_constant_s=τ。
    """
    t = np.asarray(list(time_s), dtype=float)
    r = np.asarray(list(red_pwm), dtype=float)
    b = np.asarray(list(blue_pwm), dtype=float)
    T = np.asarray(list(temp_c), dtype=float)
    if not (len(t) == len(r) == len(b) == len(T)):
        raise ValueError("time_s/red_pwm/blue_pwm/temp_c 长度必须一致")
    if len(t) < 3:
        raise ValueError("数据点过少，至少需要 3 个时间点")

    # 计算每步 dt
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("time_s 必须严格递增")

    # 计算每步输入 Uk = heat_power_k（与模型一致）
    r_clipped = np.clip(r, 0.0, 100.0)
    b_clipped = np.clip(b, 0.0, 100.0)
    pwm_frac = np.clip((r_clipped + b_clipped) / 200.0, 0.0, 1.0)
    efficiency = 0.8 + 0.2 * np.exp(-2.0 * pwm_frac)
    efficiency = np.clip(efficiency, 1e-3, 1.0)
    power = (float(max_power) * pwm_frac) / efficiency
    U = power * (1.0 - efficiency)  # 热功率

    # 最小二乘拟合：未知 [K=R_th, tau]
    def residuals(theta: np.ndarray) -> np.ndarray:
        K, tau = float(theta[0]), float(theta[1])
        if K <= 0 or tau <= 0:
            return np.full(len(dt), 1e6)
        errs = []
        for k in range(len(dt)):
            T_pred_next = _simulate_step_first_order(T[k], U[k], base_ambient_temp, K, tau, dt[k])
            errs.append(T[k + 1] - T_pred_next)
        return np.asarray(errs, dtype=float)

    # 初值：使用当前默认
    K0 = DEFAULT_THERMAL_RESISTANCE
    tau0 = DEFAULT_THERMAL_RESISTANCE * DEFAULT_THERMAL_MASS
    res = _opt.least_squares(
        residuals,
        x0=np.array([K0, tau0], dtype=float),
        bounds=(np.array([1e-6, 1e-3]), np.array([10.0, 1e6])),
        max_nfev=2000,
    )
    K_fit, tau_fit = float(res.x[0]), float(res.x[1])

    return RBLedParams(
        base_ambient_temp=float(base_ambient_temp),
        thermal_resistance=K_fit,
        time_constant_s=tau_fit,
        max_power=float(max_power),
    )


@dataclass
class EffectivePpfdThermalModel:
    """仅基于 PPFD 的等效热模型：target = base + K_ppfd * PPFD，τ 为一阶时间常数"""
    base_ambient_temp: float = DEFAULT_BASE_AMBIENT_TEMP
    k_ppfd_to_temp: float = 0.02  # °C 每 (μmol/m²/s) 的等效温升增益（示例初值）
    time_constant_s: float = DEFAULT_THERMAL_RESISTANCE * DEFAULT_THERMAL_MASS

    def step(self, ppfd: float, ambient_temp: float, dt: float = DEFAULT_DT) -> float:
        T_target = float(self.base_ambient_temp) + float(self.k_ppfd_to_temp) * float(ppfd)
        tau = max(float(self.time_constant_s), 1e-9)
        a = math.exp(-float(dt) / tau)
        return float(ambient_temp) * a + (1.0 - a) * T_target


def fit_rb_thermal_from_ppfd(
    time_s: Iterable[float],
    ppfd: Iterable[float],
    temp_c: Iterable[float],
    base_ambient_temp: float = DEFAULT_BASE_AMBIENT_TEMP,
) -> EffectivePpfdThermalModel:
    """仅根据 PPFD 与温度序列辨识等效热模型（K_ppfd 与 τ）。

    适用于仅记录了 PPFD 与温度、没有 PWM/功率数据的场景。
    返回一个 EffectivePpfdThermalModel，可直接用于温度预测。
    """
    t = np.asarray(list(time_s), dtype=float)
    P = np.asarray(list(ppfd), dtype=float)
    T = np.asarray(list(temp_c), dtype=float)
    if not (len(t) == len(P) == len(T)):
        raise ValueError("time_s/ppfd/temp_c 长度必须一致")
    if len(t) < 3:
        raise ValueError("数据点过少，至少需要 3 个时间点")

    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("time_s 必须严格递增")

    def residuals(theta: np.ndarray) -> np.ndarray:
        Kppfd, tau = float(theta[0]), float(theta[1])
        if tau <= 0:
            return np.full(len(dt), 1e6)
        errs = []
        for k in range(len(dt)):
            T_pred_next = _simulate_step_first_order(T[k], P[k], base_ambient_temp, Kppfd, tau, dt[k])
            errs.append(T[k + 1] - T_pred_next)
        return np.asarray(errs, dtype=float)

    # 初值：Kppfd 从温升/ppfd 简单估计，tau 沿用默认
    dT = float(np.nanmax(T) - np.nanmin(T) + 1e-6)
    dP = float(np.nanmax(P) - np.nanmin(P) + 1e-6)
    K0 = dT / dP
    tau0 = DEFAULT_THERMAL_RESISTANCE * DEFAULT_THERMAL_MASS
    res = _opt.least_squares(
        residuals,
        x0=np.array([K0, tau0], dtype=float),
        bounds=(np.array([0.0, 1e-3]), np.array([10.0, 1e6])),
        max_nfev=2000,
    )
    K_fit, tau_fit = float(res.x[0]), float(res.x[1])

    return EffectivePpfdThermalModel(
        base_ambient_temp=float(base_ambient_temp),
        k_ppfd_to_temp=K_fit,
        time_constant_s=tau_fit,
    )


# ---------------------------
# 兼容单步/稳态函数式 API
# ---------------------------
def _build_interp_from_coeffs(
    a_red: Optional[float], a_blue: Optional[float],
    max_ppfd_red: float, max_ppfd_blue: float, bias: float
) -> PpfdLinearInterpolator:
    """从系数构建线性插值器
    
    如果系数未提供，则使用最大PPFD值计算默认系数
    
    Args:
        a_red: 红色通道系数（可选）
        a_blue: 蓝色通道系数（可选）
        max_ppfd_red: 红色通道最大PPFD值
        max_ppfd_blue: 蓝色通道最大PPFD值
        bias: 偏置项
        
    Returns:
        配置好的线性插值器
    """
    if a_red is None:
        a_red = float(max_ppfd_red) / 100.0  # 默认：线性比例
    if a_blue is None:
        a_blue = float(max_ppfd_blue) / 100.0  # 默认：线性比例
    return PpfdLinearInterpolator(a_red, a_blue, float(bias))


def led_rb_step(
    red_pwm_percent: float,
    blue_pwm_percent: float,
    ambient_temp: float,
    base_ambient_temp: float = 25.0,
    dt: float = 0.1,
    *,
    max_power: float = 100.0,
    thermal_resistance: float = 2.5,
    thermal_mass: float = 0.5,
    # PPFD 线性模型参数（可选）
    max_ppfd_red: float = 500.0,
    max_ppfd_blue: float = 500.0,
    a_red: Optional[float] = None,
    a_blue: Optional[float] = None,
    bias: float = 0.0,
    interpolator: Optional[PpfdLinearInterpolator] = None,
) -> Tuple[float, float, float, float]:
    """红/蓝双PWM LED仿真单步
    
    提供更灵活的PPFD模型配置选项，支持自定义线性系数或使用插值器
    
    Args:
        red_pwm_percent: 红色PWM百分比 (0-100)
        blue_pwm_percent: 蓝色PWM百分比 (0-100)
        ambient_temp: 当前环境温度 (°C)
        base_ambient_temp: 基础环境温度 (°C)
        dt: 时间步长 (s)
        max_power: 最大功率 (W)
        thermal_resistance: 热阻 (K/W)
        thermal_mass: 热容 (J/K)
        max_ppfd_red: 红色通道最大PPFD值 (μmol/m²/s)
        max_ppfd_blue: 蓝色通道最大PPFD值 (μmol/m²/s)
        a_red: 红色通道线性系数（可选）
        a_blue: 蓝色通道线性系数（可选）
        bias: 线性模型偏置项
        interpolator: 自定义PPFD插值器（可选）
        
    Returns:
        (ppfd, new_ambient_temp, power, rb_ratio) 元组
        
    Note:
        - 若未提供interpolator，则使用线性模型 PPFD ≈ a_r*R + a_b*B + bias
        - 默认 a_r=max_ppfd_red/100, a_b=max_ppfd_blue/100
        - 热惯性时间常数 = thermal_mass * 30（与led.py对齐）
    """
    # 构建插值器（如果未提供）
    if interpolator is None:
        interpolator = _build_interp_from_coeffs(a_red, a_blue, max_ppfd_red, max_ppfd_blue, bias)

    # 构建物理参数（时间常数与led.py对齐）
    params = RBLedParams(
        base_ambient_temp=float(base_ambient_temp),
        thermal_resistance=float(thermal_resistance),
        time_constant_s=float(thermal_mass) * 30.0,  # 与led.py对齐的时间常数计算
        max_power=float(max_power),
    )

    # 创建模型并执行单步
    model = RedBlueLEDModel(interpolator, max_power=float(max_power), params=params)
    return model.step(
        float(red_pwm_percent), float(blue_pwm_percent), float(ambient_temp), float(base_ambient_temp), float(dt)
    )


def led_rb_steady_state(
    red_pwm_percent: float,
    blue_pwm_percent: float,
    base_ambient_temp: float = 25.0,
    *,
    max_power: float = 100.0,
    thermal_resistance: float = 2.5,
    # PPFD 线性模型参数（可选）
    max_ppfd_red: float = 500.0,
    max_ppfd_blue: float = 500.0,
    a_red: Optional[float] = None,
    a_blue: Optional[float] = None,
    bias: float = 0.0,
    interpolator: Optional[PpfdLinearInterpolator] = None,
) -> Tuple[float, float, float, float]:
    """计算固定红/蓝PWM下的稳态量
    
    计算在给定PWM设置下达到热平衡时的稳态值，提供灵活的PPFD模型配置
    
    Args:
        red_pwm_percent: 红色PWM百分比 (0-100)
        blue_pwm_percent: 蓝色PWM百分比 (0-100)
        base_ambient_temp: 基础环境温度 (°C)
        max_power: 最大功率 (W)
        thermal_resistance: 热阻 (K/W)
        max_ppfd_red: 红色通道最大PPFD值 (μmol/m²/s)
        max_ppfd_blue: 蓝色通道最大PPFD值 (μmol/m²/s)
        a_red: 红色通道线性系数（可选）
        a_blue: 蓝色通道线性系数（可选）
        bias: 线性模型偏置项
        interpolator: 自定义PPFD插值器（可选）
        
    Returns:
        (ppfd, final_ambient_temp, power, rb_ratio) 元组
    """
    # 构建插值器（如果未提供）
    if interpolator is None:
        interpolator = _build_interp_from_coeffs(a_red, a_blue, max_ppfd_red, max_ppfd_blue, bias)

    # PWM值裁剪
    r = _clip(float(red_pwm_percent), 0.0, 100.0)
    b = _clip(float(blue_pwm_percent), 0.0, 100.0)
    
    # 估计PPFD（支持不同接口）
    if hasattr(interpolator, "predict_ppfd"):
        ppfd = float(interpolator.predict_ppfd(r, b))
    else:
        ppfd = float(interpolator(r, b))

    # 计算合成PWM分数
    pwm_fraction = _clip((r + b) / 200.0, 0.0, 1.0)
    # 简化效率模型
    efficiency = 0.8 + 0.2 * math.exp(-2.0 * pwm_fraction)
    efficiency = _clip(efficiency, 1e-3, 1.0)
    # 计算功率与发热
    power = (float(max_power) * pwm_fraction) / efficiency
    heat_power = power * (1.0 - efficiency)
    # 计算稳态环境温度
    final_ambient_temp = float(base_ambient_temp) + heat_power * float(thermal_resistance)

    # 计算红蓝比例
    rb_ratio = (r / max(b, 1e-6)) if (r > 0.0 and b > 0.0) else (float("inf") if b == 0.0 and r > 0.0 else 0.0)
    if not np.isfinite(rb_ratio):
        rb_ratio = 0.0 if b > 0.0 else 1.0

    return float(ppfd), float(final_ambient_temp), float(power), float(rb_ratio)


def run_rb_on_off_example() -> None:
    """红/蓝PWM示例：演示LED加热和冷却过程
    
    运行一个三阶段的仿真示例：
    1. 红蓝各40%加热60秒
    2. 关灯冷却60秒  
    3. 红50%蓝20%再加热30秒
    
    打印仿真结果并绘制图表（只显示，不保存）
    """
    # 初始化仿真参数
    base_ambient = 25.0  # 基础环境温度
    ambient_temp = 25.0  # 当前环境温度
    dt = 1.0  # 时间步长

    # 数据记录列表
    time_data: list[float] = []
    temp_data: list[float] = []
    ppfd_data: list[float] = []
    red_pwm_data: list[float] = []
    blue_pwm_data: list[float] = []
    now_t = 0.0

    print("Red/Blue LED On/Off Example")
    print("=" * 48)
    print(f"Base ambient temperature: {base_ambient}°C\n")

    # 阶段1：R=40%, B=40% 加热60秒
    print("Phase 1: R=40%, B=40% - Heating")
    print("Time  R%  B%  PPFD  Temp  R:B")
    print("-" * 40)
    for t in range(60):
        ppfd, ambient_temp, power, rb = led_rb_step(40, 40, ambient_temp, base_ambient, dt)
        time_data.append(now_t)
        temp_data.append(ambient_temp)
        ppfd_data.append(ppfd)
        red_pwm_data.append(40.0)
        blue_pwm_data.append(40.0)
        if t % 10 == 0:  # 每10秒打印一次
            print(f"{t:3d}s {40:3d} {40:3d} {ppfd:5.0f} {ambient_temp:5.2f} {rb:4.2f}")
        now_t += dt

    # 阶段2：关灯冷却60秒
    print("\nPhase 2: OFF - Cooling")
    print("Time  R%  B%  PPFD  Temp  R:B")
    print("-" * 40)
    for t in range(60):
        ppfd, ambient_temp, power, rb = led_rb_step(0, 0, ambient_temp, base_ambient, dt)
        time_data.append(now_t)
        temp_data.append(ambient_temp)
        ppfd_data.append(ppfd)
        red_pwm_data.append(0.0)
        blue_pwm_data.append(0.0)
        if t % 10 == 0:  # 每10秒打印一次
            print(f"{60+t:3d}s {0:3d} {0:3d} {ppfd:5.0f} {ambient_temp:5.2f} {rb:4.2f}")
        now_t += dt

    # 阶段3：R=50%, B=20% 再加热30秒
    print("\nPhase 3: R=50%, B=20% - Heating")
    print("Time  R%  B%  PPFD  Temp  R:B")
    print("-" * 40)
    for t in range(30):
        ppfd, ambient_temp, power, rb = led_rb_step(50, 20, ambient_temp, base_ambient, dt)
        time_data.append(now_t)
        temp_data.append(ambient_temp)
        ppfd_data.append(ppfd)
        red_pwm_data.append(50.0)
        blue_pwm_data.append(20.0)
        if t % 10 == 0:  # 每10秒打印一次
            print(f"{120+t:3d}s {50:3d} {20:3d} {ppfd:5.0f} {ambient_temp:5.2f} {rb:4.2f}")
        now_t += dt

    # 绘制结果图表
    plot_rb_results(time_data, temp_data, ppfd_data, red_pwm_data, blue_pwm_data, base_ambient)


def plot_rb_results(
    time_data, temp_data, ppfd_data, red_pwm_data, blue_pwm_data, base_ambient
):
    """绘制红/蓝双通道仿真结果（只显示，不保存）
    
    创建包含温度响应、PPFD输出和PWM控制信号的三子图
    
    Args:
        time_data: 时间序列数据
        temp_data: 温度序列数据
        ppfd_data: PPFD序列数据
        red_pwm_data: 红色PWM序列数据
        blue_pwm_data: 蓝色PWM序列数据
        base_ambient: 基础环境温度
    """
    # 创建3个子图的图形
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle("Red/Blue LED Heating and Cooling Example", fontsize=14)

    # 子图1：温度响应曲线
    ax1.plot(time_data, temp_data, "r-", linewidth=2, label="Ambient Temperature")
    ax1.axhline(y=base_ambient, color="k", linestyle="--", alpha=0.5, label="Base Ambient")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Temperature Response")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # 添加阶段背景色
    ax1.axvspan(0, 60, alpha=0.15, color="red", label="R40 B40")
    ax1.axvspan(60, 120, alpha=0.15, color="blue", label="OFF")
    ax1.axvspan(120, 150, alpha=0.15, color="orange", label="R50 B20")

    # 子图2：PPFD输出曲线
    ax2.plot(time_data, ppfd_data, "g-", linewidth=2)
    ax2.set_ylabel("PPFD (μmol/m²/s)")
    ax2.set_title("Light Output (PPFD)")
    ax2.grid(True, alpha=0.3)

    # 子图3：PWM控制信号（红/蓝）
    ax3.plot(time_data, red_pwm_data, color="red", linewidth=2, label="Red PWM")
    ax3.plot(time_data, blue_pwm_data, color="blue", linewidth=2, label="Blue PWM")
    ax3.set_ylabel("PWM (%)")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_title("PWM Control Signals")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 调整布局并显示
    plt.tight_layout()
    plt.show()
