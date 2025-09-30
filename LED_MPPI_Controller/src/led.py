from __future__ import annotations

"""LED 控制系统核心库

本模块提供 LED 控制系统的核心功能，包括：
1. 热力学模型 - LED 温度动态建模
2. PWM-PPFD 转换 - 控制信号到光输出的映射
3. PWM-功率转换 - 控制信号到功耗的映射
4. 前向步进接口 - MPPI 控制器使用的前向仿真
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
import os
import csv
import re
from typing import Iterable, Optional, Tuple, Dict, Callable, Sequence, List
import numpy as np


# =============================================================================
# 模块 1: 默认参数与配置
# =============================================================================
DEFAULT_BASE_AMBIENT_TEMP = 23.0     # 环境基准温度 (°C)
DEFAULT_THERMAL_RESISTANCE = 0.05    # 热阻 (K/W)
DEFAULT_TIME_CONSTANT_S = 7.5        # 一阶时间常数 (s)
DEFAULT_THERMAL_MASS = 150.0         # 热容/热惯量占位 (J/°C)

# 预留：光学/功率相关参数（当前热模型未使用，后续扩展）
DEFAULT_MAX_PPFD = 600.0             # 最大PPFD (μmol/m²/s)
DEFAULT_MAX_POWER = 140             # 最大功率 (W)
DEFAULT_LED_EFFICIENCY = 0.8         # 基础光效 (0..1)
DEFAULT_EFFICIENCY_DECAY = 2.0       # 效率衰减系数（随PWM上升衰减）


# =============================================================================
# 模块 2: 热力学模型
# =============================================================================
@dataclass
class LedThermalParams:
    """LED 物理参数集合

    当前类用于热力学模型，但同时“保留”光学/功率相关参数，
    便于后续在同一处集中管理参数与扩展模型。
    """

    # 热学参数
    base_ambient_temp: float = 25.0
    thermal_resistance: float = DEFAULT_THERMAL_RESISTANCE
    time_constant_s: float = DEFAULT_TIME_CONSTANT_S
    thermal_mass: float = DEFAULT_THERMAL_MASS

    # 预留：光学/功率参数（当前热模型未使用）
    max_ppfd: float = DEFAULT_MAX_PPFD
    max_power: float = DEFAULT_MAX_POWER
    led_efficiency: float = DEFAULT_LED_EFFICIENCY
    efficiency_decay: float = DEFAULT_EFFICIENCY_DECAY


class BaseThermalModel(ABC):
    """热力学模型抽象基类（仅热学，不含PWM/功耗/PPFD）。"""

    params: LedThermalParams
    ambient_temp: float

    @abstractmethod
    def reset(self, ambient_temp: float | None = None) -> None:  # pragma: no cover - interface
        pass

    @abstractmethod
    def target_temperature(self, heat_power_w: float) -> float:  # pragma: no cover - interface
        pass

    @abstractmethod
    def step(self, heat_power_w: float, dt: float) -> float:  # pragma: no cover - interface
        pass


class FirstOrderThermalModel(BaseThermalModel):
    """一阶 LED 热力学模型

    该模型只负责温度动态：在给定发热功率 heat_power_w 时，
    环境温度向目标温度收敛：

        target = base_ambient + heat_power_w * thermal_resistance
        new_T  = T + alpha * (target - T),  alpha = clip(dt / tau, 0..1)
    """

    def __init__(self, params: LedThermalParams | None = None, *, initial_temp: float | None = None) -> None:
        self.params = params or LedThermalParams()
        self.ambient_temp: float = (
            float(initial_temp)
            if initial_temp is not None
            else float(self.params.base_ambient_temp)
        )

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    def reset(self, ambient_temp: float | None = None) -> None:
        """重置当前环境温度状态。"""
        if ambient_temp is None:
            self.ambient_temp = float(self.params.base_ambient_temp)
        else:
            self.ambient_temp = float(ambient_temp)

    def target_temperature(self, power: float) -> float:
        """在给定发热功率下的目标温度（稳态）。"""
        return float(self.params.base_ambient_temp + float(power) * self.params.thermal_resistance)

    def step(self, power: float, dt: float) -> float:
        """前进一步，返回新的环境温度。

        仅进行热学计算，不涉及 PWM、功耗、PPFD 等。
        """
        if not (math.isfinite(power) and math.isfinite(dt)):
            raise ValueError("power/dt 必须为有限实数")
        if dt <= 0:
            raise ValueError("dt 必须为正数")

        tau = max(float(self.params.time_constant_s), 1e-6)
        # 指数离散化：alpha = 1 - exp(-dt/tau)，更符合连续一阶惯性离散化
        alpha = 1.0 - math.exp(-float(dt) / tau)

        target = self.target_temperature(float(power))
        self.ambient_temp = float(self.ambient_temp + alpha * (target - self.ambient_temp))
        return self.ambient_temp


class SecondOrderThermalModel(BaseThermalModel):
    """二阶热力学模型示例：引入内部温度状态。"""

    def __init__(self, params: LedThermalParams | None = None, *, initial_temp: float | None = None) -> None:
        self.params = params or LedThermalParams()
        base = float(initial_temp) if initial_temp is not None else float(self.params.base_ambient_temp)
        self.ambient_temp: float = base
        self._internal_temp: float = base

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    def reset(self, ambient_temp: float | None = None) -> None:
        base = float(self.params.base_ambient_temp) if ambient_temp is None else float(ambient_temp)
        self.ambient_temp = base
        self._internal_temp = base

    def target_temperature(self, power: float) -> float:
        return float(self.params.base_ambient_temp + float(power) * self.params.thermal_resistance)

    def step(self, power: float, dt: float) -> float:
        if not (math.isfinite(power) and math.isfinite(dt)):
            raise ValueError("power/dt 必须为有限实数")
        if dt <= 0:
            raise ValueError("dt 必须为正数")

        # 目标内部温度由发热决定
        target_internal = self.target_temperature(float(power))

        # 定义两个时间常数（可调比例)
        tau_internal = max(float(self.params.time_constant_s) * 0.5, 1e-6)
        tau_ambient = max(float(self.params.time_constant_s) * 2.0, 1e-6)

        a_int = self._clip(float(dt) / tau_internal, 0.0, 1.0)
        a_amb = self._clip(float(dt) / tau_ambient, 0.0, 1.0)

        # 先更新内部温度，再驱动环境温度
        self._internal_temp = float(self._internal_temp + a_int * (target_internal - self._internal_temp))
        self.ambient_temp = float(self.ambient_temp + a_amb * (self._internal_temp - self.ambient_temp))
        return self.ambient_temp


@dataclass(frozen=True)
class UnifiedPPFDParams:
    k1_a: float = 0.013198      # K1(u) = k1_a * u + k1_b
    k1_b: float = 0.493192
    t1_a: float = 0.009952      # tau1(u) = t1_a * u + t1_b
    t1_b: float = 0.991167
    k2_a: float = 0.013766      # K2(u) = k2_a * u**k2_p
    k2_p: float = 0.988656
    t2_a: float = 0.796845      # tau2(u) = t2_a * u**t2_p
    t2_p: float = -0.144439


def unified_temp_diff_model(t: float, input_value: float, p: UnifiedPPFDParams) -> float:
    """纯函数：给定时间 t 与输入量 u，返回温度差 ΔT。"""
    u = float(max(0.0, input_value))
    if u <= 0.0:
        return 0.0
    k1 = p.k1_a * u + p.k1_b
    tau1 = max(p.t1_a * u + p.t1_b, 1e-6)
    k2 = p.k2_a * (u ** p.k2_p)
    tau2 = max(p.t2_a * (u ** p.t2_p), 1e-6)
    return float(k1 * (1.0 - math.exp(-float(t) / tau1)) + k2 * (1.0 - math.exp(-float(t) / tau2)))


def _unified_steady_delta(input_value: float, p: UnifiedPPFDParams) -> float:
    """稳态温差 ΔT∞(u)。

    给定输入量 u（如 PPFD 或 Solar_Vol），返回长时间后达到的温差：
        ΔT∞(u) = K1(u) + K2(u)
    """
    u = float(max(0.0, input_value))
    if u <= 0.0:
        return 0.0
    k1 = p.k1_a * u + p.k1_b
    k2 = p.k2_a * (u ** p.k2_p)
    return float(k1 + k2)


class UnifiedPPFDThermalModel(BaseThermalModel):
    """统一PPFD热力学模型（函数式核心 + 面向对象外壳）

    温度差统一模型（函数式）：
        ΔT(t, u) = K1(u) * (1 - exp(-t/tau1(u))) + K2(u) * (1 - exp(-t/tau2(u)))
    其中 u 为输入量（默认按 PPFD；5:1 情况可用 Solar_Vol 数值代入）。

    质量：R²≈0.9254, MAE≈0.4654°C, RMSE≈0.6077°C。
    """

    def __init__(self, params: LedThermalParams | None = None, *, initial_temp: float | None = None, model_params: UnifiedPPFDParams | None = None) -> None:
        self.params = params or LedThermalParams()
        self.ambient_temp: float = (
            float(initial_temp)
            if initial_temp is not None
            else float(self.params.base_ambient_temp)
        )
        self._time_elapsed: float = 0.0
        self._current_ppfd: float = 0.0
        self._eps_input: float = 1e-9  # 输入变化阈值；超过则视为新阶跃
        self._mp: UnifiedPPFDParams = model_params or UnifiedPPFDParams()

    def reset(self, ambient_temp: float | None = None) -> None:
        """重置温度状态和时间累计"""
        if ambient_temp is None:
            self.ambient_temp = float(self.params.base_ambient_temp)
        else:
            self.ambient_temp = float(ambient_temp)
        self._time_elapsed = 0.0
        self._current_ppfd = 0.0

    def target_temperature(self, ppfd: float) -> float:
        """稳态温度：环境基准温度 + ΔT∞(u)。"""
        ppfd_val = float(ppfd)
        delta_t_steady = _unified_steady_delta(ppfd_val, self._mp)
        return float(self.params.base_ambient_temp + delta_t_steady)

    def step(self, ppfd: float, dt: float) -> float:
        """前进一步，返回新的环境温度
        
        参数:
            ppfd: 当前输入量（常规为PPFD；在5:1场景可直接传入Solar_Vol）
            dt: 时间步长 (s)
        """
        if not (math.isfinite(ppfd) and math.isfinite(dt)):
            raise ValueError("ppfd/dt 必须为有限实数")
        if dt <= 0:
            raise ValueError("dt 必须为正数")

        ppfd_val = float(ppfd)
        dt_val = float(dt)
        
        # 输入变化检测：若变化超过阈值，视为新的阶跃，时间归零
        if abs(ppfd_val - self._current_ppfd) > self._eps_input:
            self._time_elapsed = 0.0
            self._current_ppfd = ppfd_val
        else:
            self._time_elapsed += dt_val
        
        if ppfd_val <= 0:
            # PPFD为0时，温度向环境温度衰减
            tau_decay = 10.0  # 衰减时间常数
            alpha = 1.0 - math.exp(-dt_val / tau_decay)
            self.ambient_temp = float(self.ambient_temp + alpha * (self.params.base_ambient_temp - self.ambient_temp))
            return self.ambient_temp
        
        # 计算当前时刻的温度差（调用纯函数）
        delta_t = unified_temp_diff_model(self._time_elapsed, ppfd_val, self._mp)
        
        # 更新环境温度
        self.ambient_temp = float(self.params.base_ambient_temp + delta_t)
        
        return self.ambient_temp

    def get_model_info(self) -> Dict[str, float]:
        """获取当前模型状态信息"""
        return {
            'time_elapsed': self._time_elapsed,
            'current_ppfd': self._current_ppfd,        # 兼容字段
            'current_solar_vol': self._current_ppfd,   # 电压输入时更直观的命名
            'ambient_temp': self.ambient_temp,
            'base_ambient_temp': self.params.base_ambient_temp
        }

    # --- 便捷别名（当输入为 Solar_Vol 时可直接调用） ---
    def step_with_solar_vol(self, solar_vol: float, dt: float) -> float:
        return self.step(solar_vol, dt)

    def target_temperature_solar_vol(self, solar_vol: float) -> float:
        return self.target_temperature(solar_vol)


# 兼容命名：保持 LedThermalModel 为一阶模型别名，便于外部直接使用
LedThermalModel = FirstOrderThermalModel


# =============================================================================
# 模块 3: LED 外观封装
# =============================================================================
class Led:
    """LED 外观封装（支持热学和PPFD模型）。

    - 统一管理 `params` 与 `model` 的组合
    - 面向业务的最小接口：reset / step_with_heat / step_with_ppfd / target_temperature / get_temperature
    - 支持传统热学模型和新的统一PPFD模型
    """

    def __init__(
        self,
        model_type: str = "first_order",
        params: LedThermalParams | None = None,
        *,
        initial_temp: float | None = None,
    ) -> None:
        self.params = params or LedThermalParams()
        self.model: BaseThermalModel = create_model(model_type, self.params, initial_temp=initial_temp)
        self._is_ppfd_model = isinstance(self.model, UnifiedPPFDThermalModel)

    def reset(self, ambient_temp: float | None = None) -> None:
        self.model.reset(ambient_temp)

    def step_with_heat(self, power: float, dt: float) -> float:
        """传统热学步进：使用发热功率"""
        if self._is_ppfd_model:
            raise ValueError("统一PPFD模型不支持step_with_heat，请使用step_with_ppfd")
        return self.model.step(power, dt)

    def step_with_ppfd(self, ppfd: float, dt: float) -> float:
        """PPFD步进：使用PPFD值（仅统一PPFD模型支持）"""
        if not self._is_ppfd_model:
            raise ValueError("非PPFD模型不支持step_with_ppfd，请使用step_with_heat")
        return self.model.step(ppfd, dt)

    def target_temperature(self, input_val: float) -> float:
        """计算目标温度
        
        参数:
            input_val: 对于传统模型为发热功率(W)，对于PPFD模型为PPFD值(μmol/m²/s)
        """
        return self.model.target_temperature(input_val)

    @property
    def temperature(self) -> float:
        """当前模型维护的环境温度（°C）。"""
        return self.model.ambient_temp

    @property
    def is_ppfd_model(self) -> bool:
        """是否为PPFD模型"""
        return self._is_ppfd_model

    def get_model_info(self) -> Dict[str, float]:
        """获取模型状态信息（仅PPFD模型支持）"""
        if hasattr(self.model, 'get_model_info'):
            return self.model.get_model_info()
        else:
            return {
                'ambient_temp': self.model.ambient_temp,
                'base_ambient_temp': self.params.base_ambient_temp
            }


def create_model(
    model_type: str = "first_order",
    params: LedThermalParams | None = None,
    *,
    initial_temp: float | None = None,
) -> BaseThermalModel:
    """模型工厂：创建指定类型的热学模型。"""
    params = params or LedThermalParams()
    mt = model_type.lower().strip()
    if mt in {"first_order", "first", "1", "fo"}:
        return FirstOrderThermalModel(params, initial_temp=initial_temp)
    if mt in {"second_order", "second", "2", "so"}:
        return SecondOrderThermalModel(params, initial_temp=initial_temp)
    if mt in {"unified_ppfd", "ppfd", "unified", "3", "up"}:
        return UnifiedPPFDThermalModel(params, initial_temp=initial_temp)
    raise ValueError(f"不支持的模型类型: {model_type}")
def create_default_params() -> LedThermalParams:
    """便捷函数：创建默认热学参数。"""
    return LedThermalParams()


# 为了向后使用者方便，可以导出一个通用别名（可选）
LedParams = LedThermalParams


# =============================================================================
# 模块 4: PWM-PPFD 转换系统（重构版本）
# =============================================================================
_DEFAULT_DIR = os.path.dirname(__file__)
DEFAULT_CALIB_CSV = os.path.join(_DEFAULT_DIR, "..", "data", "calib_data.csv")


@dataclass(frozen=True)
class PpfdModelCoeffs:
    """PPFD线性模型系数：R_PWM = α * PPFD + β, B_PWM = γ * PPFD + δ"""

    alpha: float  # R_PWM = alpha * PPFD + beta
    beta: float
    gamma: float  # B_PWM = gamma * PPFD + delta  
    delta: float
    r_squared_r: float = 0.0  # R_PWM模型的R²
    r_squared_b: float = 0.0  # B_PWM模型的R²

    def predict_r_pwm(self, ppfd: float) -> float:
        """预测R_PWM值"""
        return float(self.alpha * float(ppfd) + self.beta)

    def predict_b_pwm(self, ppfd: float) -> float:
        """预测B_PWM值"""
        return float(self.gamma * float(ppfd) + self.delta)

    def predict_pwm(self, ppfd: float) -> Tuple[float, float]:
        """同时预测R_PWM和B_PWM值"""
        return self.predict_r_pwm(ppfd), self.predict_b_pwm(ppfd)


def _normalize_key(key: str) -> str:
    """标准化标签键"""
    s = str(key).strip().lower()
    # 处理r1, r2等格式
    m = re.fullmatch(r"r\s*(\d+)", s)
    if m:
        return f"r{m.group(1)}"
    # 处理其他格式，移除空格
    s = re.sub(r"\s+", "", s)
    return s


def _load_calib_data(csv_path: str) -> Dict[str, List[Tuple[float, float, float]]]:
    """加载标定数据并按Label分组
    
    返回格式：{label: [(ppfd, r_pwm, b_pwm), ...]}
    """
    by_label: Dict[str, List[Tuple[float, float, float]]] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.lstrip().startswith("#"))
        )

        def _get(row: dict, *names: str) -> Optional[str]:
            """从行数据中获取指定字段的值"""
            for name in names:
                if name in row:
                    return row[name]
            for name in names:
                for k, v in row.items():
                    if k.replace(" ", "").lower() == name.replace(" ", "").lower():
                        return v
            return None

        for row in reader:
            # 提取必要字段
            label = _get(row, "Label", "LABEL", "KEY", "Key", "R:B", "ratio")
            ppfd_s = _get(row, "PPFD", "ppfd")
            r_pwm_s = _get(row, "R_PWM", "r_pwm", "Red_PWM", "R PWM")
            b_pwm_s = _get(row, "B_PWM", "b_pwm", "Blue_PWM", "B PWM")

            if label is None or ppfd_s is None or r_pwm_s is None or b_pwm_s is None:
                continue

            try:
                ppfd = float(ppfd_s)
                r_pwm = float(r_pwm_s)
                b_pwm = float(b_pwm_s)
            except (TypeError, ValueError):
                continue

            # 标准化标签并存储数据
            label_norm = _normalize_key(label)
            by_label.setdefault(label_norm, []).append((ppfd, r_pwm, b_pwm))

    return by_label


def _fit_separate_models(data_points: List[Tuple[float, float, float]]) -> PpfdModelCoeffs:
    """分开拟合：R_PWM = α * PPFD + β, B_PWM = γ * PPFD + δ
    
    参数:
        data_points: [(ppfd, r_pwm, b_pwm), ...] 格式的数据点列表
    
    返回:
        PpfdModelCoeffs: 包含四个系数和R²值的模型系数
    """
    if len(data_points) < 2:
        raise ValueError("需要至少2个数据点进行拟合")
    
    # 提取数据
    ppfd_vals = [float(point[0]) for point in data_points]
    r_pwm_vals = [float(point[1]) for point in data_points]
    b_pwm_vals = [float(point[2]) for point in data_points]
    
    # 拟合R_PWM对PPFD的线性模型: R_PWM = alpha * PPFD + beta
    alpha, beta, r_squared_r = _fit_linear_regression(ppfd_vals, r_pwm_vals)
    
    # 拟合B_PWM对PPFD的线性模型: B_PWM = gamma * PPFD + delta  
    gamma, delta, r_squared_b = _fit_linear_regression(ppfd_vals, b_pwm_vals)
    
    return PpfdModelCoeffs(
        alpha=alpha,
        beta=beta, 
        gamma=gamma,
        delta=delta,
        r_squared_r=r_squared_r,
        r_squared_b=r_squared_b
    )


def _fit_linear_regression(x_vals: List[float], y_vals: List[float]) -> Tuple[float, float, float]:
    """使用最小二乘法拟合一元线性回归模型: y = slope * x + intercept
    
    参数:
        x_vals: 自变量值列表
        y_vals: 因变量值列表
    
    返回:
        (slope, intercept, r_squared): 斜率、截距和决定系数
    """
    if len(x_vals) != len(y_vals):
        raise ValueError("x_vals和y_vals长度必须相同")
    
    n = len(x_vals)
    if n < 2:
        return 0.0, 0.0, 0.0
    
    # 转换为numpy数组进行计算
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    
    # 计算均值
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # 计算斜率和截距
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if abs(denominator) < 1e-12:
        slope = 0.0
        intercept = y_mean
    else:
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
    
    # 计算R²
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)  # 残差平方和
    ss_tot = np.sum((y - y_mean) ** 2)  # 总平方和
    
    if abs(ss_tot) < 1e-12:
        r_squared = 0.0
    else:
        r_squared = 1.0 - (ss_res / ss_tot)
    
    return float(slope), float(intercept), float(r_squared)


class PWMtoPPFDModel:
    """基于分开拟合的PWM→PPFD模型集合
    
    对每个Label分别拟合：
    - R_PWM = α * PPFD + β
    - B_PWM = γ * PPFD + δ
    
    然后可以通过PPFD值预测所需的PWM设定
    """

    def __init__(
        self,
        exclude_labels: Optional[Iterable[str]] = None,
    ):
        self.exclude_labels: set[str] = set(_normalize_key(k) for k in (exclude_labels or []))
        self.by_label: Dict[str, PpfdModelCoeffs] = {}
        self.csv_path: Optional[str] = None

    def fit(self, csv_path: str) -> "PWMtoPPFDModel":
        """从CSV文件拟合模型
        
        参数:
            csv_path: 标定数据CSV文件路径
        
        返回:
            self: 支持链式调用
        """
        self.csv_path = csv_path
        by_label_data = _load_calib_data(self.csv_path)

        # 排除指定的标签
        if self.exclude_labels:
            by_label_data = {
                k: v for k, v in by_label_data.items() 
                if _normalize_key(k) not in self.exclude_labels
            }

        # 对每个标签分别拟合模型
        for label, data_points in by_label_data.items():
            try:
                coeffs = _fit_separate_models(data_points)
                self.by_label[label] = coeffs
            except ValueError as e:
                print(f"警告：标签 '{label}' 拟合失败: {e}")
                continue

        if not self.by_label:
            raise ValueError("没有成功拟合任何模型")

        return self

    def predict_pwm(self, *, ppfd: float, label: str) -> Tuple[float, float]:
        """根据PPFD值预测所需的PWM设定
        
        参数:
            ppfd: 目标PPFD值
            label: 标签（如"5:1", "r1"等）
        
        返回:
            (r_pwm, b_pwm): 预测的红光和蓝光PWM值
        """
        label_norm = _normalize_key(label)
        coeffs = self.by_label.get(label_norm)
        if coeffs is None:
            raise KeyError(f"标签 '{label}' 的模型未找到")
        
        return coeffs.predict_pwm(ppfd)

    def predict(self, *, r_pwm: float, b_pwm: float, key: Optional[str] = None) -> float:
        """根据PWM值预测PPFD（反向预测）
        
        参数:
            r_pwm: 红光PWM值
            b_pwm: 蓝光PWM值
            key: 标签（如"5:1", "r1"等），None使用第一个可用标签
        
        返回:
            ppfd: 预测的PPFD值
        """
        if key is None:
            # 使用第一个可用标签
            if not self.by_label:
                raise ValueError("没有可用的模型")
            key = list(self.by_label.keys())[0]
        
        label_norm = _normalize_key(key)
        coeffs = self.by_label.get(label_norm)
        if coeffs is None:
            raise KeyError(f"标签 '{key}' 的模型未找到")
        
        # 使用R_PWM模型进行反向预测：PPFD = (R_PWM - beta) / alpha
        if abs(coeffs.alpha) < 1e-12:
            raise ValueError(f"标签 '{key}' 的R_PWM模型斜率为0，无法进行反向预测")
        
        ppfd = (float(r_pwm) - coeffs.beta) / coeffs.alpha
        return max(0.0, ppfd)  # 确保PPFD非负

    def get_model_info(self, label: str) -> Dict[str, float]:
        """获取指定标签的模型信息
        
        参数:
            label: 标签
        
        返回:
            包含模型系数和R²值的字典
        """
        label_norm = _normalize_key(label)
        coeffs = self.by_label.get(label_norm)
        if coeffs is None:
            raise KeyError(f"标签 '{label}' 的模型未找到")
        
        return {
            'alpha': coeffs.alpha,
            'beta': coeffs.beta,
            'gamma': coeffs.gamma,
            'delta': coeffs.delta,
            'r_squared_r': coeffs.r_squared_r,
            'r_squared_b': coeffs.r_squared_b
        }

    def list_labels(self) -> List[str]:
        """列出所有可用的标签"""
        return list(self.by_label.keys())

    def get_fit_summary(self) -> Dict[str, Dict[str, float]]:
        """获取所有模型的拟合摘要"""
        summary = {}
        for label, coeffs in self.by_label.items():
            summary[label] = {
                'alpha': coeffs.alpha,
                'beta': coeffs.beta, 
                'gamma': coeffs.gamma,
                'delta': coeffs.delta,
                'r_squared_r': coeffs.r_squared_r,
                'r_squared_b': coeffs.r_squared_b
            }
        return summary








def solve_pwm_for_target_ppfd(
    *,
    model: PWMtoPPFDModel,
    target_ppfd: float,
    label: str,
    pwm_clip: Tuple[float, float] = (0.0, 100.0),
    integer_output: bool = True,
) -> Tuple[int | float, int | float, int | float]:
    """求解目标PPFD对应的PWM值
    
    参数:
        model: PWM到PPFD的线性模型
        target_ppfd: 目标PPFD值
        label: 标签（如"5:1", "r1"等）
        pwm_clip: PWM范围限制
        integer_output: 是否输出整数PWM值
    
    返回:
        (r_pwm, b_pwm, total_pwm)
    """
    # 直接使用新的预测方法
    r_pwm_f, b_pwm_f = model.predict_pwm(ppfd=target_ppfd, label=label)
    total_pwm_f = r_pwm_f + b_pwm_f
    
    # 应用PWM范围限制
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(x, hi))
    
    r_pwm_clipped = _clip(r_pwm_f, pwm_clip[0], pwm_clip[1])
    b_pwm_clipped = _clip(b_pwm_f, pwm_clip[0], pwm_clip[1])
    total_pwm_clipped = r_pwm_clipped + b_pwm_clipped
    
    if not integer_output:
        return float(r_pwm_clipped), float(b_pwm_clipped), float(total_pwm_clipped)
    
    # 整数量化
    r_pwm_int = int(round(r_pwm_clipped))
    b_pwm_int = int(round(b_pwm_clipped))
    total_pwm_int = r_pwm_int + b_pwm_int
    
    # 确保整数结果也在范围内
    r_pwm_int = max(0, min(100, r_pwm_int))
    b_pwm_int = max(0, min(100, b_pwm_int))
    total_pwm_int = r_pwm_int + b_pwm_int
    
    return r_pwm_int, b_pwm_int, total_pwm_int


# =============================================================================
# 模块 5: PWM-功率转换系统
# =============================================================================
class PowerInterpolator:
    """按比例键对 Total PWM→Total Power(W) 做线性插值。

    用法：
        interp = PowerInterpolator.from_csv(DEFAULT_CALIB_CSV)
        p_w = interp.predict_power(total_pwm=90.0, key="5:1")
    """

    def __init__(self) -> None:
        self.by_key: Dict[str, Tuple[list[float], list[float]]] = {}

    @staticmethod
    def _normalize_key(key: str) -> str:
        """标准化标签键 - 与模块4保持一致"""
        s = str(key).strip().lower()
        # 处理r1, r2等格式
        m = re.fullmatch(r"r\s*(\d+)", s)
        if m:
            return f"r{m.group(1)}"
        # 处理其他格式，移除空格
        s = re.sub(r"\s+", "", s)
        return s

    @classmethod
    def from_csv(cls, csv_path: str) -> "PowerInterpolator":
        """从标定CSV构建插值器（按比例键聚合 total PWM→total Power 样本）。"""
        inst = cls()
        by_key_pairs: Dict[str, list[Tuple[float, float]]] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader((line for line in f if line.strip() and not line.lstrip().startswith("#")))
            
            def _get(row: dict, *names: str) -> Optional[str]:
                """从行数据中获取指定字段的值 - 与模块4保持一致"""
                for name in names:
                    if name in row:
                        return row[name]
                for name in names:
                    for k, v in row.items():
                        if k.replace(" ", "").lower() == name.replace(" ", "").lower():
                            return v
                return None
            
            for row in reader:
                key = _get(row, "Label", "LABEL", "KEY", "Key", "R:B", "ratio")
                r_pwm = _get(row, "R_PWM", "r_pwm", "Red_PWM", "R PWM")
                b_pwm = _get(row, "B_PWM", "b_pwm", "Blue_PWM", "B PWM")
                r_pow = _get(row, "R_POWER", "r_power")
                b_pow = _get(row, "B_POWER", "b_power")
                if key is None or r_pwm is None or b_pwm is None or r_pow is None or b_pow is None:
                    continue
                try:
                    r_pwm_f = float(r_pwm); b_pwm_f = float(b_pwm)
                    r_pow_f = float(r_pow); b_pow_f = float(b_pow)
                except ValueError:
                    continue
                total_pwm = r_pwm_f + b_pwm_f
                total_pow = r_pow_f + b_pow_f
                k = cls._normalize_key(key)
                by_key_pairs.setdefault(k, []).append((total_pwm, total_pow))

        for k, pairs in by_key_pairs.items():
            pairs.sort(key=lambda t: t[0])
            xs: list[float] = []
            ys: list[float] = []
            last_x: Optional[float] = None
            for x, y in pairs:
                if last_x is not None and abs(x - last_x) < 1e-9:
                    ys[-1] = y
                else:
                    xs.append(x); ys.append(y)
                    last_x = x
            if len(xs) >= 2:
                inst.by_key[k] = (xs, ys)
        return inst

    def predict_power(self, *, total_pwm: float, key: str, clamp: bool = True) -> float:
        """预测在给定比例键下，总PWM对应的总功率(W)。

        参数:
            total_pwm: R+B 的总PWM百分比
            key: 比例键（如 "5:1"、"r1" 等）
            clamp: 是否对区间外进行端点截断
        """
        import bisect
        k = self._normalize_key(key)
        if k not in self.by_key:
            raise KeyError(f"calib中不存在比例键: {key}")
        xs, ys = self.by_key[k]
        x = float(total_pwm)
        if clamp:
            if x <= xs[0]:
                return float(ys[0])
            if x >= xs[-1]:
                return float(ys[-1])
        i = bisect.bisect_left(xs, x)
        i = max(1, min(i, len(xs) - 1))
        x0, x1 = xs[i - 1], xs[i]
        y0, y1 = ys[i - 1], ys[i]
        t = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
        return float(y0 + t * (y1 - y0))


# 说明：能耗工具函数未在主链路中使用，移出核心库以精简体积。


@dataclass(frozen=True)
class PowerLine:
    """简单线性模型 y = a*x + c（用于功率/电压等标定线）。

    - a: 斜率
    - c: 截距
    """
    a: float  # slope (W per % total PWM)
    c: float  # intercept (W)

    def predict(self, total_pwm: float) -> float:
        return float(self.a * float(total_pwm) + self.c)


class SolarVolModel:
    """按比例键（主要用于5:1）对 Total PWM→Solar_Vol 做线性拟合。

    数据来源：ALL_Data.csv，字段名包含 'R_PWM','B_PWM','Solar_Vol','R:B' 或 'R:B' 等。
    默认将相同 total_pwm 的多行做去重保留最后一个值，至少需要2个点来拟合。
    """

    def __init__(self) -> None:
        # 已弃用：该模型未在主控制链路中使用，建议改用 SolarVolToPPFDModel
        self.by_key: Dict[str, PowerLine] = {}
        self.overall: Optional[PowerLine] = None

    @staticmethod
    def _normalize_key(key: str) -> str:
        """标准化比例键表示（支持如"r1"、去除空白等）。"""
        s = str(key).strip().lower()
        m = re.fullmatch(r"r\s*(\d+)", s)
        if m:
            return f"r{m.group(1)}"
        return re.sub(r"\s+", "", s)

    @classmethod
    def from_all_data_csv(cls, csv_path: str, *, focus_key: Optional[str] = None) -> "SolarVolModel":
        """从 ALL_Data.csv 拟合 total PWM → Solar_Vol 的直线模型。

        focus_key: 若指定（如 "5:1"），则仅拟合该比例键并同时给出overall。
        """
        # 保留兼容接口，但标记为不推荐使用
        inst = cls()
        rows_by_key: Dict[str, list[Tuple[float, float]]] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader((line for line in f if line.strip() and not line.lstrip().startswith("#")))

            def _get(row: dict, *names: str) -> Optional[str]:
                for name in names:
                    if name in row:
                        return row[name]
                for name in names:
                    for k, v in row.items():
                        if k.replace(" ", "").lower() == name.replace(" ", "").lower():
                            return v
                return None

            for row in reader:
                key = _get(row, "R:B", "ratio", "Label", "KEY", "Key")
                r_pwm = _get(row, "R_PWM", "r_pwm", "Red_PWM", "R PWM")
                b_pwm = _get(row, "B_PWM", "b_pwm", "Blue_PWM", "B PWM")
                solar_vol = _get(row, "Solar_Vol", "solar_vol")
                if key is None or r_pwm is None or b_pwm is None or solar_vol is None:
                    continue
                if focus_key is not None and cls._normalize_key(key) != cls._normalize_key(focus_key):
                    continue
                try:
                    r_pwm_f = float(r_pwm); b_pwm_f = float(b_pwm)
                    sv_f = float(solar_vol)
                except ValueError:
                    continue
                total_pwm = r_pwm_f + b_pwm_f
                k = cls._normalize_key(key)
                rows_by_key.setdefault(k, []).append((total_pwm, sv_f))

        def _fit(pairs: list[Tuple[float, float]]) -> PowerLine:
            pairs.sort(key=lambda t: t[0])
            xs: list[float] = []
            ys: list[float] = []
            last_x: Optional[float] = None
            for x, y in pairs:
                if last_x is not None and abs(x - last_x) < 1e-9:
                    ys[-1] = y
                else:
                    xs.append(x); ys.append(y); last_x = x
            return _fit_line(list(zip(xs, ys)))

        all_pairs: list[Tuple[float, float]] = []
        for k, pairs in rows_by_key.items():
            line = _fit(pairs)
            inst.by_key[k] = line
            all_pairs.extend(pairs)
        inst.overall = _fit(all_pairs) if all_pairs else PowerLine(0.0, 0.0)
        return inst

    def predict(self, *, total_pwm: float, key: Optional[str] = None) -> float:
        line = self.overall
        if key is not None:
            line = self.by_key.get(self._normalize_key(key), line)
        if line is None:
            raise RuntimeError("SolarVolModel 未拟合")
        return line.predict(total_pwm)


class SolarVolToPPFDModel:
    """按比例键对 Solar_Vol ↔ PPFD 做线性拟合（带截距）。

    - 主要用于 5:1（R:B≈0.83）场景，但也支持从文件中为其它比例键分别拟合。
    - 提供双向预测：给定 Solar_Vol 预测 PPFD，或给定 PPFD 反解 Solar_Vol。
    - 从 `Solar_Vol_clean.csv` 读取列："Solar_Vol", "PPFD", 以及键列（优先 "R:B"，其次 "ratio"/"Label"/"KEY"/"Key"）。
    """

    def __init__(self) -> None:
        self.by_key: Dict[str, PowerLine] = {}
        self.overall: Optional[PowerLine] = None

    @staticmethod
    def _normalize_key(key: str) -> str:
        # 复用模块级规范；若失败则简单去空白
        try:
            return _normalize_key(key)
        except Exception:
            return re.sub(r"\s+", "", str(key).strip().lower())

    @classmethod
    def from_csv(cls, csv_path: str, *, focus_key: Optional[str] = None) -> "SolarVolToPPFDModel":
        """从 Solar_Vol_clean.csv 拟合 Solar_Vol→PPFD 直线模型。

        focus_key: 若指定（如 "5:1" 或 "0.83"），则仅拟合该键并同时给出 overall。
        """
        inst = cls()

        def _map_ratio_to_key(ratio_val: float) -> str:
            # 将 0.83 视作 5:1；其余返回原数值字符串
            if math.isfinite(ratio_val) and abs(ratio_val - 0.83) < 0.02:
                return "5:1"
            return f"{ratio_val:.2f}"

        rows_by_key: Dict[str, list[Tuple[float, float]]] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader((line for line in f if line.strip() and not line.lstrip().startswith("#")))

            def _get(row: dict, *names: str) -> Optional[str]:
                for name in names:
                    if name in row:
                        return row[name]
                for name in names:
                    for k, v in row.items():
                        if k.replace(" ", "").lower() == name.replace(" ", "").lower():
                            return v
                return None

            for row in reader:
                sv = _get(row, "Solar_Vol", "solar_vol")
                ppfd = _get(row, "PPFD", "ppfd")
                key_raw = _get(row, "R:B", "ratio", "Label", "KEY", "Key")
                if sv is None or ppfd is None:
                    continue
                try:
                    sv_f = float(sv); ppfd_f = float(ppfd)
                except ValueError:
                    continue

                # 跳过零点样本，避免 (0,0) 影响拟合（不强制经过零）
                if abs(sv_f) < 1e-9 and abs(ppfd_f) < 1e-9:
                    continue

                if key_raw is None:
                    key_norm = "overall"
                else:
                    # 允许数值比例或字符串比例
                    try:
                        ratio_val = float(key_raw)
                        key_norm = cls._normalize_key(_map_ratio_to_key(ratio_val))
                    except ValueError:
                        key_norm = cls._normalize_key(str(key_raw))

                if focus_key is not None and cls._normalize_key(focus_key) != key_norm:
                    continue

                rows_by_key.setdefault(key_norm, []).append((sv_f, ppfd_f))

        def _fit(pairs: list[Tuple[float, float]]) -> PowerLine:
            # 去除相同 x 的重复，仅保留最后值
            pairs.sort(key=lambda t: t[0])
            xs: list[float] = []
            ys: list[float] = []
            last_x: Optional[float] = None
            for x, y in pairs:
                if last_x is not None and abs(x - last_x) < 1e-9:
                    ys[-1] = y
                else:
                    xs.append(x); ys.append(y); last_x = x
            return _fit_line(list(zip(xs, ys)))

        all_pairs: list[Tuple[float, float]] = []
        for k, pairs in rows_by_key.items():
            line = _fit(pairs)
            inst.by_key[k] = line
            all_pairs.extend(pairs)
        inst.overall = _fit(all_pairs) if all_pairs else PowerLine(0.0, 0.0)
        return inst

    def _get_line(self, key: Optional[str]) -> PowerLine:
        line = self.overall
        if key is not None:
            line = self.by_key.get(self._normalize_key(key), line)
        if line is None:
            raise RuntimeError("SolarVolToPPFDModel 未拟合")
        return line

    def predict_ppfd(self, *, solar_vol: float, key: Optional[str] = None) -> float:
        """给定 Solar_Vol，预测 PPFD。"""
        line = self._get_line(key)
        return float(line.predict(float(solar_vol)))

    def predict_solar_vol(self, *, ppfd: float, key: Optional[str] = None) -> float:
        """给定 PPFD，反解 Solar_Vol（使用 y = a*x + c 的解析逆）。"""
        line = self._get_line(key)
        a = float(line.a); c = float(line.c)
        if abs(a) < 1e-12:
            return 0.0
        return float((float(ppfd) - c) / a)

    def list_keys(self) -> List[str]:
        return list(self.by_key.keys())

    def get_line_info(self, key: Optional[str] = None) -> Dict[str, float]:
        line = self._get_line(key)
        return {"a": float(line.a), "c": float(line.c)}

def _load_power_rows(csv_path: str) -> Dict[str, list[Tuple[float, float]]]:
    """加载功率数据并按标签分组 - 与模块4保持一致"""
    rows_by_key: Dict[str, list[Tuple[float, float]]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader((line for line in f if line.strip() and not line.lstrip().startswith("#")))
        
        def _get(row: dict, *names: str) -> Optional[str]:
            """从行数据中获取指定字段的值 - 与模块4保持一致"""
            for name in names:
                if name in row:
                    return row[name]
            for name in names:
                for k, v in row.items():
                    if k.replace(" ", "").lower() == name.replace(" ", "").lower():
                        return v
            return None
        
        for row in reader:
            key = _get(row, "Label", "LABEL", "KEY", "Key", "R:B", "ratio")
            r_pwm = _get(row, "R_PWM", "r_pwm", "Red_PWM", "R PWM")
            b_pwm = _get(row, "B_PWM", "b_pwm", "Blue_PWM", "B PWM")
            r_pow = _get(row, "R_POWER", "r_power")
            b_pow = _get(row, "B_POWER", "b_power")
            if key is None or r_pwm is None or b_pwm is None or r_pow is None or b_pow is None:
                continue
            try:
                r_pwm_f = float(r_pwm); b_pwm_f = float(b_pwm)
                r_pow_f = float(r_pow); b_pow_f = float(b_pow)
            except ValueError:
                continue
            total_pwm = r_pwm_f + b_pwm_f
            total_pow = r_pow_f + b_pow_f
            rows_by_key.setdefault(_normalize_key(key), []).append((total_pwm, total_pow))
    return rows_by_key


def _fit_line(rows: Iterable[Tuple[float, float]], *, with_intercept: bool = True) -> PowerLine:
    sx = sx2 = sy = sxy = 0.0
    n = 0
    for x, y in rows:
        x = float(x); y = float(y)
        sx += x; sy += y
        sx2 += x * x
        sxy += x * y
        n += 1
    if n == 0:
        return PowerLine(0.0, 0.0)

    if with_intercept:
        denom = n * sx2 - sx * sx
        if abs(denom) < 1e-12:
            a = sxy / sx2 if sx2 > 0 else 0.0
            c = 0.0
        else:
            a = (n * sxy - sx * sy) / denom
            c = (sy * sx2 - sx * sxy) / denom
    else:
        a = sxy / sx2 if sx2 > 0 else 0.0
        c = 0.0
    return PowerLine(float(a), float(c))


class PWMtoPowerModel:
    """总 PWM→总功率(W) 的直线模型（按比例键 & overall）。"""

    def __init__(self, include_intercept: bool = True):
        self.include_intercept = bool(include_intercept)
        self.by_key: Dict[str, PowerLine] = {}
        self.overall: Optional[PowerLine] = None
        self.csv_path: Optional[str] = None

    def fit(self, csv_path: str) -> "PWMtoPowerModel":
        self.csv_path = csv_path
        rows_by_key = _load_power_rows(csv_path)
        all_rows: list[Tuple[float, float]] = []
        for key, rows in rows_by_key.items():
            self.by_key[key] = _fit_line(rows, with_intercept=self.include_intercept)
            all_rows.extend(rows)
        if all_rows:
            self.overall = _fit_line(all_rows, with_intercept=self.include_intercept)
        else:
            self.overall = PowerLine(0.0, 0.0)
        return self

    def predict(self, *, total_pwm: float, key: Optional[str] = None) -> float:
        line = self.overall
        if key is not None:
            line = self.by_key.get(_normalize_key(key), line)
        if line is None:
            raise RuntimeError("Power model not fitted")
        return line.predict(total_pwm)


# =============================================================================
# 模块 6: MPPI 前向步进接口
# =============================================================================
@dataclass
class LedForwardOutput:
    """前向步进的输出集合，便于 MPPI 代价函数使用。"""

    temp: float                 # 新的环境温度（或受控温度）
    ppfd: Optional[float]       # 预测 PPFD（若提供了 ppfd_model）
    power: float                # 电功率 (W)
    heat_power: float           # 用于热学模型的发热功率 (W)
    efficiency: Optional[float] # 若启用效率模型则给出，否则为 None
    r_pwm: float                # 红通道 PWM（裁剪后）
    b_pwm: float                # 蓝通道 PWM（裁剪后）
    total_pwm: float            # 总 PWM（r+b）


def forward_step(
    *,
    thermal_model: BaseThermalModel,
    r_pwm: float,
    b_pwm: float,
    dt: float,
    power_model: PWMtoPowerModel,
    ppfd_model: Optional[PWMtoPPFDModel] = None,
    model_key: Optional[str] = None,  # None/"overall" 使用整体系数，其它如 "5:1"
    use_efficiency: bool = False,
    eta_model: Optional[Callable[[float, float, float, float, LedThermalParams], float]] = None,
    heat_scale: float = 1.0,
    use_unified_ppfd: bool = False,  # 是否使用统一PPFD模型
    use_solar_vol_for_5_1: bool = False,  # 在5:1场景下以Solar_Vol替代PPFD
) -> LedForwardOutput:
    """统一的前向一步接口：PWM → 功率/PPFD → 热功率 → 温度。

    - 不依赖效率模型即可使用：默认 p_heat = heat_scale * p_elec（建议 heat_scale=1 或 0.6~0.8）
    - 若 use_efficiency=True，需提供 eta_model(r,b,total, temp, params) → η，热功率 p_heat = p_elec*(1-η)
    - model_key: None 或 "overall" 走整体模型；传 "5:1" 等则使用对应比例键
    - use_unified_ppfd: 是否使用统一PPFD模型（需要ppfd_model支持）
    """
    # 1) 裁剪 PWM 到 0..100
    r = max(0.0, min(100.0, float(r_pwm)))
    b = max(0.0, min(100.0, float(b_pwm)))
    total = r + b

    # 2) 预测电功率（W）：使用总 PWM 的直线模型
    key_arg = None if (model_key is None or str(model_key).lower() == "overall") else model_key
    p_elec = float(power_model.predict(total_pwm=total, key=key_arg))

    # 3) 预测 PPFD 或 Solar_Vol（可选）
    ppfd_val: Optional[float] = None
    if ppfd_model is not None:
        if use_solar_vol_for_5_1 and (key_arg is None or str(key_arg) == "5:1"):
            # 当指定5:1并要求使用Solar_Vol时，仍复用ppfd字段承载该数值
            ppfd_val = float(ppfd_model.predict(r_pwm=r, b_pwm=b, key="5:1"))
        else:
            ppfd_val = float(ppfd_model.predict(r_pwm=r, b_pwm=b, key=key_arg))

    # 4) 热学步进 - 根据模型类型选择不同方式
    if use_unified_ppfd and isinstance(thermal_model, UnifiedPPFDThermalModel):
        if ppfd_val is None:
            raise ValueError("使用统一PPFD模型需要提供ppfd_model")
        new_temp = float(thermal_model.step(ppfd=ppfd_val, dt=float(dt)))
        # 统一PPFD模型不计算热功率，设为0
        p_heat = 0.0
        eff_val = None
    else:
        # 传统热学模型：计算热功率
        eff_val: Optional[float] = None
        if use_efficiency:
            if eta_model is None:
                raise ValueError("use_efficiency=True 需要提供 eta_model 回调")
            eff_val = float(max(0.0, min(1.0, eta_model(r, b, total, thermal_model.ambient_temp, thermal_model.params))))
            p_heat = p_elec * (1.0 - eff_val)
        else:
            p_heat = p_elec * float(heat_scale)
        
        new_temp = float(thermal_model.step(power=p_heat, dt=float(dt)))

    return LedForwardOutput(
        temp=new_temp,
        ppfd=ppfd_val,
        power=p_elec,
        heat_power=p_heat,
        efficiency=eff_val,
        r_pwm=r,
        b_pwm=b,
        total_pwm=total,
    )


def forward_step_batch(
    *,
    thermal_models: Sequence[BaseThermalModel],
    r_pwms: Sequence[float],
    b_pwms: Sequence[float],
    dt: float,
    power_model: PWMtoPowerModel,
    ppfd_model: Optional[PWMtoPPFDModel] = None,
    model_key: Optional[str] = None,
    use_efficiency: bool = False,
    eta_model: Optional[Callable[[float, float, float, float, LedThermalParams], float]] = None,
    heat_scale: float = 1.0,
    use_unified_ppfd: bool = False,  # 是否使用统一PPFD模型
    use_solar_vol_for_5_1: bool = False,  # 在5:1场景下以Solar_Vol替代PPFD
) -> List[LedForwardOutput]:
    """批量前向步进：NumPy 向量化电功率/PPFD/热功率，热学更新逐实例写回。"""
    n = len(thermal_models)
    r = np.clip(np.asarray(r_pwms, dtype=float), 0.0, 100.0)
    b = np.clip(np.asarray(b_pwms, dtype=float), 0.0, 100.0)
    if r.shape != b.shape or r.ndim != 1 or r.size != n:
        raise ValueError("r_pwms/b_pwms 形状必须为 (N,) 且与 thermal_models 数量一致")

    total = r + b

    # 选择模型键（整体或指定比例）
    key_arg = None if (model_key is None or str(model_key).lower() == "overall") else model_key

    # 向量化电功率：P = a*total + c
    line = power_model.overall if key_arg is None else power_model.by_key.get(_normalize_key(key_arg), power_model.overall)
    if line is None:
        raise RuntimeError("power_model 未拟合或缺少系数")
    p_elec = line.a * total + line.c

    # 向量化 PPFD（如提供）
    if ppfd_model is not None:
        coeffs = ppfd_model.overall if key_arg is None else ppfd_model.by_key.get(_normalize_key(key_arg), ppfd_model.overall)
        if coeffs is None:
            raise RuntimeError("ppfd_model 未拟合或缺少系数")
        ppfd_vals = coeffs.a_r * r + coeffs.a_b * b + coeffs.intercept
        if use_solar_vol_for_5_1 and (key_arg is None or str(key_arg) == "5:1"):
            # 5:1 时使用相同线性形式承载 Solar_Vol 值
            pass
    else:
        ppfd_vals = np.full(n, np.nan)

    # 检查是否有统一PPFD模型
    has_unified_ppfd = any(isinstance(m, UnifiedPPFDThermalModel) for m in thermal_models)
    
    if use_unified_ppfd and has_unified_ppfd:
        # 统一PPFD模型：直接使用PPFD进行热学更新
        if ppfd_model is None:
            raise ValueError("使用统一PPFD模型需要提供ppfd_model")
        
        new_temps = np.empty(n, dtype=float)
        p_heat = np.zeros(n, dtype=float)  # 统一PPFD模型不计算热功率
        eff_list = np.full(n, np.nan)
        
        # 逐实例更新（因为每个模型可能有不同的累计时间）
        for i, m in enumerate(thermal_models):
            if isinstance(m, UnifiedPPFDThermalModel):
                new_temps[i] = float(m.step(ppfd=float(ppfd_vals[i]), dt=float(dt)))
            else:
                # 混合模型：传统模型仍使用热功率
                if use_efficiency and eta_model is not None:
                    eff_val = float(max(0.0, min(1.0, eta_model(float(r[i]), float(b[i]), float(total[i]), float(m.ambient_temp), m.params))))
                    p_heat[i] = p_elec[i] * (1.0 - eff_val)
                    eff_list[i] = eff_val
                else:
                    p_heat[i] = p_elec[i] * float(heat_scale)
                new_temps[i] = float(m.step(power=p_heat[i], dt=float(dt)))
    else:
        # 传统热学模型：计算热功率
        if use_efficiency:
            if eta_model is None:
                raise ValueError("use_efficiency=True 需要提供 eta_model 回调")
            # 逐实例计算效率（依赖温度和参数），此处循环
            eff_list = np.empty(n, dtype=float)
            for i in range(n):
                eff_list[i] = float(max(0.0, min(1.0, eta_model(float(r[i]), float(b[i]), float(total[i]), float(thermal_models[i].ambient_temp), thermal_models[i].params))))
            p_heat = p_elec * (1.0 - eff_list)
        else:
            eff_list = np.full(n, np.nan)
            p_heat = p_elec * float(heat_scale)

        # 向量化热学更新（读取参数与状态，批量计算，再写回）
        temps = np.array([m.ambient_temp for m in thermal_models], dtype=float)
        base = np.array([m.params.base_ambient_temp for m in thermal_models], dtype=float)
        rth = np.array([m.params.thermal_resistance for m in thermal_models], dtype=float)
        tau = np.maximum(np.array([m.params.time_constant_s for m in thermal_models], dtype=float), 1e-6)
        # 指数离散化 alpha = 1 - exp(-dt/tau)
        alpha = 1.0 - np.exp(-float(dt) / tau)
        target = base + p_heat * rth
        new_temps = temps + alpha * (target - temps)

        # 写回模型状态
        for i, m in enumerate(thermal_models):
            m.ambient_temp = float(new_temps[i])

    # 组装输出列表
    outputs: List[LedForwardOutput] = []
    for i in range(n):
        outputs.append(
            LedForwardOutput(
                temp=float(new_temps[i]),
                ppfd=None if np.isnan(ppfd_vals[i]) else float(ppfd_vals[i]),
                power=float(p_elec[i]),
                heat_power=float(p_heat[i]),
                efficiency=None if np.isnan(eff_list[i]) else float(eff_list[i]),
                r_pwm=float(r[i]),
                b_pwm=float(b[i]),
                total_pwm=float(total[i]),
            )
        )
    return outputs


# =============================================================================
# 模块 7: 导出接口
# =============================================================================

__all__ = [
    "LedThermalParams",
    "BaseThermalModel",
    "FirstOrderThermalModel",
    "SecondOrderThermalModel",
    "UnifiedPPFDThermalModel",  # 新增统一PPFD模型
    "LedThermalModel",
    "Led",
    "create_model",
    "create_default_params",
    "LedParams",
    "DEFAULT_BASE_AMBIENT_TEMP",
    "DEFAULT_THERMAL_RESISTANCE",
    "DEFAULT_TIME_CONSTANT_S",
    "DEFAULT_THERMAL_MASS",
    "DEFAULT_MAX_PPFD",
    "DEFAULT_MAX_POWER",
    "DEFAULT_LED_EFFICIENCY",
    "DEFAULT_EFFICIENCY_DECAY",
    # PWM/PPFD exports (重构版本)
    "DEFAULT_CALIB_CSV",
    "PpfdModelCoeffs",
    "PWMtoPPFDModel",
    "solve_pwm_for_target_ppfd",
    # Power interpolation
    "PowerInterpolator",
    # Power linear fit
    "PowerLine",
    "PWMtoPowerModel",
    "SolarVolModel",
    "SolarVolToPPFDModel",
    # Forward stepping for MPPI
    "LedForwardOutput",
    "forward_step",
    "forward_step_batch",
]
