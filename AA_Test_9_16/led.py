from __future__ import annotations

"""LED 热力学库（仅热模型，去除仿真/绘图/兼容封装）

本模块提供一个最小可用的 LED 热力学模型类，用于描述在给定
发热功率下环境温度的动态变化。后续如 PWM、功耗、光学输出
等功能将作为上层模块在其他文件中扩展。
"""

from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import math
import os
import json
import csv
import re
from typing import Iterable, Optional, Tuple, Dict, Callable, Sequence, List
import numpy as np


# ---------------------------
# 默认参数（包含热学与占位的光学/功率参数）
# ---------------------------
DEFAULT_BASE_AMBIENT_TEMP = 23.0     # 环境基准温度 (°C)
DEFAULT_THERMAL_RESISTANCE = 0.05    # 热阻 (K/W)
DEFAULT_TIME_CONSTANT_S = 7.5        # 一阶时间常数 (s)
DEFAULT_THERMAL_MASS = 150.0         # 热容/热惯量占位 (J/°C)

# 预留：光学/功率相关参数（当前热模型未使用，后续扩展）
DEFAULT_MAX_PPFD = 600.0             # 最大PPFD (μmol/m²/s)
DEFAULT_MAX_POWER = 86.4             # 最大功率 (W)
DEFAULT_LED_EFFICIENCY = 0.8         # 基础光效 (0..1)
DEFAULT_EFFICIENCY_DECAY = 2.0       # 效率衰减系数（随PWM上升衰减）


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


# 兼容命名：保持 LedThermalModel 为一阶模型别名，便于外部直接使用
LedThermalModel = FirstOrderThermalModel


class Led:
    """LED 外观封装（仅热学）。

    - 统一管理 `params` 与 `model` 的组合
    - 面向业务的最小接口：reset / step_with_heat / target_temperature / get_temperature
    - 不提供 PWM/功耗/PPFD 逻辑（保持热学解耦）。
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

    def reset(self, ambient_temp: float | None = None) -> None:
        self.model.reset(ambient_temp)

    def step_with_heat(self, power: float, dt: float) -> float:
        return self.model.step(power, dt)

    def target_temperature(self, power: float) -> float:
        return self.model.target_temperature(power)

    @property
    def temperature(self) -> float:
        return self.model.ambient_temp


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
    raise ValueError(f"不支持的模型类型: {model_type}")
def create_default_params() -> LedThermalParams:
    """便捷函数：创建默认热学参数。"""
    return LedThermalParams()


# 为了向后使用者方便，可以导出一个通用别名（可选）
LedParams = LedThermalParams


# ---------------------------
# PWM→PPFD 拟合与求解（从 MPC/LED/PWM 精简移植）
# ---------------------------
_DEFAULT_DIR = os.path.dirname(__file__)
DEFAULT_CALIB_CSV = os.path.join(_DEFAULT_DIR, "data", "calib_data.csv")


@dataclass(frozen=True)
class PpfdModelCoeffs:
    """PPFD 线性模型：ppfd ≈ a_r * R_PWM + a_b * B_PWM + intercept"""

    a_r: float
    a_b: float
    intercept: float = 0.0

    def predict(self, r_pwm: float, b_pwm: float) -> float:
        return float(self.a_r * float(r_pwm) + self.a_b * float(b_pwm) + self.intercept)


def _normalize_key(key: str) -> str:
    s = str(key).strip().lower()
    m = re.fullmatch(r"r\s*(\d+)", s)
    if m:
        return f"r:{m.group(1)}"
    s = re.sub(r"\s+", "", s)
    return s


def _load_calib_rows(csv_path: str) -> Tuple[Dict[str, list[Tuple[float, float, float]]], list[Tuple[float, float, float]]]:
    by_key: Dict[str, list[Tuple[float, float, float]]] = {}
    all_rows: list[Tuple[float, float, float]] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.lstrip().startswith("#"))
        )

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
            key = _get(row, "KEY", "Key", "R:B", "ratio")
            ppfd_s = _get(row, "PPFD", "ppfd")
            r_pwm_s = _get(row, "R_PWM", "r_pwm", "Red_PWM", "R PWM")
            b_pwm_s = _get(row, "B_PWM", "b_pwm", "Blue_PWM", "B PWM")

            if key is None or ppfd_s is None or r_pwm_s is None or b_pwm_s is None:
                continue

            try:
                ppfd = float(ppfd_s)
                r_pwm = float(r_pwm_s)
                b_pwm = float(b_pwm_s)
            except (TypeError, ValueError):
                continue

            triple = (r_pwm, b_pwm, ppfd)
            by_key.setdefault(_normalize_key(key), []).append(triple)
            all_rows.append(triple)

    return by_key, all_rows


def _fit_linear_zero_intercept(rows: Iterable[Tuple[float, float, float]]) -> PpfdModelCoeffs:
    s_rr = s_bb = s_rb = s_ry = s_by = 0.0
    n = 0
    for r_pwm, b_pwm, y in rows:
        r = float(r_pwm)
        b = float(b_pwm)
        y = float(y)
        s_rr += r * r
        s_bb += b * b
        s_rb += r * b
        s_ry += r * y
        s_by += b * y
        n += 1

    if n == 0:
        raise ValueError("没有可用于拟合的数据行")

    det = s_rr * s_bb - s_rb * s_rb
    if abs(det) < 1e-12:
        if s_rr > s_bb and s_rr > 0:
            a_r = s_ry / s_rr
            a_b = 0.0
        elif s_bb > 0:
            a_r = 0.0
            a_b = s_by / s_bb
        else:
            a_r = a_b = 0.0
    else:
        a_r = (s_ry * s_bb - s_by * s_rb) / det
        a_b = (s_by * s_rr - s_ry * s_rb) / det

    return PpfdModelCoeffs(a_r=float(a_r), a_b=float(a_b), intercept=0.0)


def _fit_linear_with_intercept(rows: Iterable[Tuple[float, float, float]]) -> PpfdModelCoeffs:
    s_rr = s_bb = s_rb = s_r1 = s_b1 = s_11 = 0.0
    s_ry = s_by = s_1y = 0.0
    n = 0
    for r_pwm, b_pwm, y in rows:
        r = float(r_pwm)
        b = float(b_pwm)
        y = float(y)
        s_rr += r * r
        s_bb += b * b
        s_rb += r * b
        s_r1 += r
        s_b1 += b
        s_11 += 1.0
        s_ry += r * y
        s_by += b * y
        s_1y += y
        n += 1

    if n == 0:
        raise ValueError("没有可用于拟合的数据行")

    def det3(a,b,c,d,e,f,g,h,i):
        return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

    A11, A12, A13 = s_rr, s_rb, s_r1
    A21, A22, A23 = s_rb, s_bb, s_b1
    A31, A32, A33 = s_r1, s_b1, s_11
    B1, B2, B3 = s_ry, s_by, s_1y

    D = det3(A11,A12,A13, A21,A22,A23, A31,A32,A33)
    if abs(D) < 1e-10:
        try:
            return _fit_linear_zero_intercept(rows)
        except Exception:
            if s_rr >= s_bb and s_rr > 0:
                a_r = s_ry / s_rr
                return PpfdModelCoeffs(a_r=float(a_r), a_b=0.0, intercept=0.0)
            elif s_bb > 0:
                a_b = s_by / s_bb
                return PpfdModelCoeffs(a_r=0.0, a_b=float(a_b), intercept=0.0)
            else:
                return PpfdModelCoeffs(0.0, 0.0, 0.0)

    D_r = det3(B1,A12,A13, B2,A22,A23, B3,A32,A33)
    D_b = det3(A11,B1,A13, A21,B2,A23, A31,B3,A33)
    D_c = det3(A11,A12,B1, A21,A22,B2, A31,A32,B3)
    a_r = D_r / D
    a_b = D_b / D
    c = D_c / D
    return PpfdModelCoeffs(a_r=float(a_r), a_b=float(a_b), intercept=float(c))


class PWMtoPPFDModel:
    """基于 CSV 标定数据的 PWM→PPFD 线性模型集合。"""

    def __init__(
        self,
        include_intercept: bool = True,
        exclude_keys: Optional[Iterable[str]] = None,
    ):
        self.include_intercept = bool(include_intercept)
        self.exclude_keys: set[str] = set(_normalize_key(k) for k in (exclude_keys or []))
        self.by_key: Dict[str, PpfdModelCoeffs] = {}
        self.overall: Optional[PpfdModelCoeffs] = None
        self.csv_path: Optional[str] = None

    def fit(self, csv_path: str) -> "PWMtoPPFDModel":
        self.csv_path = csv_path
        by_key_rows, _all_rows = _load_calib_rows(self.csv_path)

        if self.exclude_keys:
            by_key_rows = {
                k: v for k, v in by_key_rows.items() if _normalize_key(k) not in self.exclude_keys
            }

        all_rows = [t for rows in by_key_rows.values() for t in rows]

        for key, rows in by_key_rows.items():
            try:
                coeffs = (
                    _fit_linear_with_intercept(rows)
                    if self.include_intercept
                    else _fit_linear_zero_intercept(rows)
                )
                self.by_key[key] = coeffs
            except ValueError:
                continue

        if all_rows:
            self.overall = (
                _fit_linear_with_intercept(all_rows)
                if self.include_intercept
                else _fit_linear_zero_intercept(all_rows)
            )
        else:
            self.overall = PpfdModelCoeffs(0.0, 0.0, 0.0)
        return self

    def predict(self, *, r_pwm: float, b_pwm: float, key: Optional[str] = None) -> float:
        key_norm = _normalize_key(key) if key else None
        coeffs = self.by_key.get(key_norm) if key_norm else self.overall
        coeffs = coeffs or self.overall
        if coeffs is None:
            raise RuntimeError("Model not fitted.")
        return coeffs.predict(r_pwm, b_pwm)

    def save_json(self, path: str):
        payload = {
            "config": {
                "include_intercept": self.include_intercept,
                "exclude_keys": list(self.exclude_keys),
                "csv_path": self.csv_path,
            },
            "coeffs": {
                "by_key": {k: asdict(v) for k, v in self.by_key.items()},
                "overall": asdict(self.overall) if self.overall else None,
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "PWMtoPPFDModel":
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        config = payload.get("config", {})
        model = cls(
            include_intercept=config.get("include_intercept", True),
            exclude_keys=config.get("exclude_keys", []),
        )
        model.csv_path = config.get("csv_path")
        coeffs = payload.get("coeffs", {})
        model.by_key = {k: PpfdModelCoeffs(**v) for k, v in coeffs.get("by_key", {}).items()}
        if coeffs.get("overall"):
            model.overall = PpfdModelCoeffs(**coeffs["overall"])
        return model


def _weights_from_key(key: str) -> Tuple[float, float]:
    k = _normalize_key(key)
    if k == "r:1":
        return 1.0, 0.0
    m = re.fullmatch(r"(\d+)\s*:\s*(\d+)", k)
    if m:
        r = float(m.group(1))
        b = float(m.group(2))
        s = r + b if (r + b) > 0 else 1.0
        return r / s, b / s
    return 0.5, 0.5


def _empirical_weights_for_key(
    *,
    key: str,
    csv_path: str,
    exclude_keys: Optional[Iterable[str]] = None,
    target_ppfd: Optional[float] = None,
    window_ppfd: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    by_key_rows, _ = _load_calib_rows(csv_path)
    k = _normalize_key(key)
    if exclude_keys:
        ex = { _normalize_key(x) for x in exclude_keys }
        by_key_rows = { kk: vv for kk, vv in by_key_rows.items() if kk not in ex }

    rows = by_key_rows.get(k)
    if not rows:
        return None

    if target_ppfd is not None and window_ppfd is not None and window_ppfd > 0:
        lo, hi = target_ppfd - window_ppfd, target_ppfd + window_ppfd
        rows = [t for t in rows if lo <= t[2] <= hi]
        if not rows:
            rows = by_key_rows.get(k, [])

    num = den = 0.0
    for r_pwm, b_pwm, _ppfd in rows:
        s = float(r_pwm) + float(b_pwm)
        if s <= 0:
            continue
        num += float(r_pwm) / s
        den += 1.0
    if den <= 0:
        return None
    w_r = num / den
    w_b = 1.0 - w_r
    return float(w_r), float(w_b)


def solve_pwm_for_target_ppfd(
    *,
    model: PWMtoPPFDModel,
    target_ppfd: float,
    key: str,
    pwm_clip: Tuple[float, float] = (0.0, 100.0),
    ratio_strategy: str = "label",
    ratio_window_ppfd: float = 10.0,
    integer_output: bool = True,
) -> Tuple[int | float, int | float, int | float]:
    coeffs = model.by_key.get(_normalize_key(key), model.overall)
    if coeffs is None:
        raise RuntimeError("Model has no coefficients.")
    if not model.csv_path:
        raise RuntimeError("Model is missing csv_path, needed for empirical ratio.")

    if ratio_strategy == "empirical_global":
        w_r, w_b = _empirical_weights_for_key(
            key=key,
            csv_path=model.csv_path,
            exclude_keys=model.exclude_keys,
            target_ppfd=None,
            window_ppfd=None,
        ) or _weights_from_key(key)
    elif ratio_strategy == "empirical_local":
        w_r, w_b = _empirical_weights_for_key(
            key=key,
            csv_path=model.csv_path,
            exclude_keys=model.exclude_keys,
            target_ppfd=float(target_ppfd),
            window_ppfd=float(ratio_window_ppfd),
        ) or _weights_from_key(key)
    else:
        w_r, w_b = _weights_from_key(key)

    denom = coeffs.a_r * w_r + coeffs.a_b * w_b
    if abs(denom) < 1e-9:
        if w_r > w_b and abs(coeffs.a_r) > 1e-9:
            denom = coeffs.a_r * w_r
            w_b = 0.0
        elif abs(coeffs.a_b) > 1e-9:
            denom = coeffs.a_b * w_b
            w_r = 0.0
        else:
            return 0.0, 0.0, 0.0

    # 先按线性模型求解标量 s，使 r = s*w_r, b = s*w_b
    s = (float(target_ppfd) - coeffs.intercept) / denom

    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(x, hi))

    # 约束：每个通道的 PWM 分别在 [0,100]，因此 s 受限于 100/w_r 和 100/w_b
    s_max = float('inf')
    if w_r > 0:
        s_max = min(s_max, 100.0 / w_r)
    if w_b > 0:
        s_max = min(s_max, 100.0 / w_b)
    # 允许 s 超过 100（当某通道权重小于1时），但不超过各通道上限约束 s_max
    s = _clip(s, pwm_clip[0], s_max)

    r_pwm_f = s * w_r
    b_pwm_f = s * w_b

    if not integer_output:
        return float(r_pwm_f), float(b_pwm_f), float(s)

    # 整数量化，分别裁剪各通道到 [0,100]；total 显示为四舍五入的 s
    r_i = int(round(r_pwm_f))
    b_i = int(round(b_pwm_f))
    r_i = max(0, min(100, r_i))
    b_i = max(0, min(100, b_i))
    total_i = int(round(s))
    return r_i, b_i, total_i

# ---------------------------
# PWM→功率 一维插值（按比例键）与能耗计算
# ---------------------------
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
        return str(key).strip().lower().replace(" ", "")

    @classmethod
    def from_csv(cls, csv_path: str) -> "PowerInterpolator":
        inst = cls()
        by_key_pairs: Dict[str, list[Tuple[float, float]]] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader((line for line in f if line.strip() and not line.lstrip().startswith("#")))
            for row in reader:
                key = row.get("R:B") or row.get("ratio") or row.get("Key") or row.get("KEY")
                r_pwm = row.get("R_PWM") or row.get("r_pwm") or row.get("R PWM")
                b_pwm = row.get("B_PWM") or row.get("b_pwm") or row.get("B PWM")
                r_pow = row.get("R_POWER") or row.get("r_power")
                b_pow = row.get("B_POWER") or row.get("b_power")
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


def energy_kwh(power_w: float, hours: float) -> float:
    """单点恒功率能耗：P(W) × t(h) → kWh。"""
    return float(power_w) / 1000.0 * float(hours)


def energy_series_kwh(powers_w: Iterable[float], dt_s: float) -> float:
    """时间序列能耗：Σ P_i(W) × dt(s) / 3,600,000 → kWh。"""
    total_ws = 0.0
    dt = float(dt_s)
    for p in powers_w:
        total_ws += float(p) * dt
    return total_ws / 3_600_000.0


# ---------------------------
# PWM→功率 线性拟合模型（直线）
# ---------------------------
@dataclass(frozen=True)
class PowerLine:
    a: float  # slope (W per % total PWM)
    c: float  # intercept (W)

    def predict(self, total_pwm: float) -> float:
        return float(self.a * float(total_pwm) + self.c)


def _load_power_rows(csv_path: str) -> Dict[str, list[Tuple[float, float]]]:
    rows_by_key: Dict[str, list[Tuple[float, float]]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader((line for line in f if line.strip() and not line.lstrip().startswith("#")))
        for row in reader:
            key = row.get("R:B") or row.get("ratio") or row.get("Key") or row.get("KEY")
            r_pwm = row.get("R_PWM") or row.get("r_pwm") or row.get("R PWM")
            b_pwm = row.get("B_PWM") or row.get("b_pwm") or row.get("B PWM")
            r_pow = row.get("R_POWER") or row.get("r_power")
            b_pow = row.get("B_POWER") or row.get("b_power")
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


# ---------------------------
# MPPI 前向步进（统一接口）
# ---------------------------
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
) -> LedForwardOutput:
    """统一的前向一步接口：PWM → 功率/PPFD → 热功率 → 温度。

    - 不依赖效率模型即可使用：默认 p_heat = heat_scale * p_elec（建议 heat_scale=1 或 0.6~0.8）
    - 若 use_efficiency=True，需提供 eta_model(r,b,total, temp, params) → η，热功率 p_heat = p_elec*(1-η)
    - model_key: None 或 "overall" 走整体模型；传 "5:1" 等则使用对应比例键
    """
    # 1) 裁剪 PWM 到 0..100
    r = max(0.0, min(100.0, float(r_pwm)))
    b = max(0.0, min(100.0, float(b_pwm)))
    total = r + b

    # 2) 预测电功率（W）：使用总 PWM 的直线模型
    key_arg = None if (model_key is None or str(model_key).lower() == "overall") else model_key
    p_elec = float(power_model.predict(total_pwm=total, key=key_arg))

    # 3) 预测 PPFD（可选）
    ppfd_val: Optional[float] = None
    if ppfd_model is not None:
        ppfd_val = float(ppfd_model.predict(r_pwm=r, b_pwm=b, key=key_arg))

    # 4) 计算热功率
    eff_val: Optional[float] = None
    if use_efficiency:
        if eta_model is None:
            raise ValueError("use_efficiency=True 需要提供 eta_model 回调")
        eff_val = float(max(0.0, min(1.0, eta_model(r, b, total, thermal_model.ambient_temp, thermal_model.params))))
        p_heat = p_elec * (1.0 - eff_val)
    else:
        p_heat = p_elec * float(heat_scale)

    # 5) 热学步进
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
    else:
        ppfd_vals = np.full(n, np.nan)

    # 热功率（无效率模型时使用 heat_scale）
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

__all__ = [
    "LedThermalParams",
    "BaseThermalModel",
    "FirstOrderThermalModel",
    "SecondOrderThermalModel",
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
    # PWM/PPFD exports
    "DEFAULT_CALIB_CSV",
    "PpfdModelCoeffs",
    "PWMtoPPFDModel",
    "solve_pwm_for_target_ppfd",
    # Power interpolation & energy
    "PowerInterpolator",
    "energy_kwh",
    "energy_series_kwh",
    # Power linear fit
    "PowerLine",
    "PWMtoPowerModel",
    # Forward stepping for MPPI
    "LedForwardOutput",
    "forward_step",
    "forward_step_batch",
]
