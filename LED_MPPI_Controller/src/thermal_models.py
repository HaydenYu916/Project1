from __future__ import annotations

"""LED 热力学模型集合（基于 Thermal/exported_models 导出的最新版本）。"""

import json
import math
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# =============================================================================
# 默认常量（向后兼容 led.py 中的定义）
# =============================================================================
DEFAULT_BASE_AMBIENT_TEMP = 23.0     # 环境基准温度 (°C)
DEFAULT_THERMAL_RESISTANCE = 0.05    # 热阻 (K/W) — 旧一阶 RC 模型兼容字段
DEFAULT_TIME_CONSTANT_S = 7.5        # 时间常数 (s)  — 旧一阶 RC 模型兼容字段
DEFAULT_THERMAL_MASS = 150.0         # 热容占位 (J/°C)

DEFAULT_MAX_PPFD = 600.0             # 最大 PPFD (μmol/m²/s)
DEFAULT_MAX_POWER = 140              # 最大功率 (W)
DEFAULT_LED_EFFICIENCY = 0.8         # 基础光效 (0..1)
DEFAULT_EFFICIENCY_DECAY = 2.0       # 效率衰减系数

DEFAULT_MODEL_VARIANT = "pure"       # 可选："pure" / "mlp"
_DEFAULT_THERMAL_EXPORT_DIR = (
    Path(__file__).resolve().parent / ".." / "Thermal" / "exported_models"
).resolve()


# =============================================================================
# 参数定义
# =============================================================================
@dataclass
class LedThermalParams:
    """LED 热模型参数集合。

    旧版 RC 模型的字段仍然保留（thermal_resistance/time_constant_s 等），
    以便向后兼容既有代码；新版基于 Solar 的模型主要使用 Solar 相关配置。
    """

    base_ambient_temp: float = DEFAULT_BASE_AMBIENT_TEMP
    thermal_resistance: float = DEFAULT_THERMAL_RESISTANCE
    time_constant_s: float = DEFAULT_TIME_CONSTANT_S
    thermal_mass: float = DEFAULT_THERMAL_MASS

    max_ppfd: float = DEFAULT_MAX_PPFD
    max_power: float = DEFAULT_MAX_POWER
    led_efficiency: float = DEFAULT_LED_EFFICIENCY
    efficiency_decay: float = DEFAULT_EFFICIENCY_DECAY

    model_variant: str = DEFAULT_MODEL_VARIANT
    solar_on_threshold: float = 0.05
    solar_change_tolerance: float = 0.02
    solar_min: float = 1.296
    solar_max: float = 1.549
    model_dir: Optional[str] = None

    def clamp_solar(self, value: float) -> float:
        lo = min(self.solar_min, self.solar_max)
        hi = max(self.solar_min, self.solar_max)
        return float(max(lo, min(hi, value)))


class BaseThermalModel:
    """热模型抽象基类（为兼容 led.forward_step 提供统一接口）。"""

    params: LedThermalParams

    def reset(self, ambient_temp: float | None = None) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def step(  # pragma: no cover - interface
        self,
        power: float | None = None,
        dt: float | None = None,
        *,
        solar_vol: float | None = None,
        dt_seconds: float | None = None,
    ) -> float:
        raise NotImplementedError

    def target_temperature(self, power: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError


# =============================================================================
# 导出 MLP 模型所需的占位类（与训练脚本保持兼容）
# =============================================================================
class _ImprovedThermodynamicConstrainedMLPBase:
    """兼容 pickle 反序列化的基类，复现训练脚本中的 predict 行为。"""

    a1_ref: float = 1.4

    def __init__(self) -> None:
        self.thermal_params: dict[str, float] | None = None
        self.mlp_model = None
        self.scaler = None
        self.fitted: bool = False

    # --- 工具函数 ---------------------------------------------------------
    def _ensure_parameters(self) -> None:
        if not self.thermal_params:
            raise RuntimeError("missing thermal_params in pickled MLP model")

    @staticmethod
    def _safe_time_array(t) -> np.ndarray:
        return np.asarray(np.maximum(np.asarray(t, dtype=float), 0.0), dtype=float)

    @staticmethod
    def _safe_solar_array(a1) -> np.ndarray:
        return np.asarray(a1, dtype=float)

    def _build_features(
        self,
        t: np.ndarray,
        a1: np.ndarray,
        thermal_pred: np.ndarray,
    ) -> np.ndarray:
        return np.column_stack(
            [
                t / 100.0,
                a1,
                thermal_pred / 10.0,
                (t * a1) / 100.0,
                np.sqrt(np.maximum(t, 0.0)),
                np.log1p(np.maximum(t, 0.0)),
            ]
        )

    # --- 子类需实现 -------------------------------------------------------
    def _thermal_response(self, t: np.ndarray, a1: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # --- 推理接口 ---------------------------------------------------------
    def predict(self, t, a1):
        t_arr = self._safe_time_array(t)
        a_arr = self._safe_solar_array(a1)
        if t_arr.shape != a_arr.shape:
            raise ValueError("time and solar arrays must share the same shape")

        self._ensure_parameters()
        thermal_pred = self._thermal_response(t_arr, a_arr)

        if self.mlp_model is None or self.scaler is None:
            return thermal_pred

        features = self._build_features(t_arr, a_arr, thermal_pred)
        residual = self.mlp_model.predict(self.scaler.transform(features))
        return thermal_pred + residual

    def predict_thermal_only(self, t, a1):
        t_arr = self._safe_time_array(t)
        a_arr = self._safe_solar_array(a1)
        if t_arr.shape != a_arr.shape:
            raise ValueError("time and solar arrays must share the same shape")
        self._ensure_parameters()
        return self._thermal_response(t_arr, a_arr)


class ImprovedThermodynamicConstrainedMLPHeating(_ImprovedThermodynamicConstrainedMLPBase):
    """开灯阶段热模型（用于反序列化）。"""

    def _thermal_response(self, t: np.ndarray, a1: np.ndarray) -> np.ndarray:
        self._ensure_parameters()
        params = self.thermal_params or {}
        k1_base = float(params.get("K1_base", 0.0))
        tau1 = max(float(params.get("tau1", 1.0)), 1e-6)
        k2_base = float(params.get("K2_base", 0.0))
        tau2 = max(float(params.get("tau2", 1.0)), 1e-6)
        alpha = float(params.get("alpha_solar", 0.0))

        a1 = self._safe_solar_array(a1)
        solar_factor = np.maximum(0.0, 1.0 + alpha * (a1 - self.a1_ref))
        t = self._safe_time_array(t)

        k1 = k1_base * solar_factor
        k2 = k2_base * solar_factor
        return k1 * (1.0 - np.exp(-t / tau1)) + k2 * (1.0 - np.exp(-t / tau2))


class ImprovedThermodynamicConstrainedMLPCooling(_ImprovedThermodynamicConstrainedMLPBase):
    """关灯阶段热模型（用于反序列化）。"""

    def _thermal_response(self, t: np.ndarray, a1: np.ndarray) -> np.ndarray:
        self._ensure_parameters()
        params = self.thermal_params or {}
        k1_base = float(params.get("K1_base", 0.0))
        tau1 = max(float(params.get("tau1", 1.0)), 1e-6)
        k2_base = float(params.get("K2_base", 0.0))
        tau2 = max(float(params.get("tau2", 1.0)), 1e-6)
        alpha = float(params.get("alpha_solar", 0.0))

        a1 = self._safe_solar_array(a1)
        solar_factor = np.maximum(0.0, 1.0 + alpha * (a1 - self.a1_ref))
        t = self._safe_time_array(t)

        k1 = k1_base * solar_factor
        k2 = k2_base * solar_factor
        return k1 * np.exp(-t / tau1) + k2 * np.exp(-t / tau2)


# =============================================================================
# 响应函数包装
# =============================================================================
class ThermalResponse:
    """Phase-aware ΔT 响应基类。"""

    def __init__(self, phase: str) -> None:
        self.phase = phase

    def delta(self, time_minutes: float, solar_val: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def steady_state_delta(self, solar_val: float) -> float:
        return self.delta(1.0e6, solar_val)


class PureThermalResponse(ThermalResponse):
    """纯热力学模型响应（基于 JSON 参数）。"""

    def __init__(self, params: dict[str, object], phase: str) -> None:
        super().__init__(phase)
        self._params = params
        self._coeffs = params.get("parameters", {}) if isinstance(params, dict) else {}
        self._a1_ref = float(params.get("a1_ref", 1.4)) if isinstance(params, dict) else 1.4

    def _solar_factor(self, solar_val: float) -> float:
        alpha = float(self._coeffs.get("alpha_solar", 0.0))
        factor = 1.0 + alpha * (float(solar_val) - self._a1_ref)
        return max(0.0, factor)

    def delta(self, time_minutes: float, solar_val: float) -> float:
        coeffs = self._coeffs
        k1 = float(coeffs.get("K1_base", 0.0))
        tau1 = max(float(coeffs.get("tau1", 1.0)), 1e-6)
        k2 = float(coeffs.get("K2_base", 0.0))
        tau2 = max(float(coeffs.get("tau2", 1.0)), 1e-6)
        factor = self._solar_factor(solar_val)
        k1 *= factor
        k2 *= factor
        t = max(0.0, float(time_minutes))
        if self.phase == "heating":
            return float(k1 * (1.0 - math.exp(-t / tau1)) + k2 * (1.0 - math.exp(-t / tau2)))
        return float(k1 * math.exp(-t / tau1) + k2 * math.exp(-t / tau2))

    def steady_state_delta(self, solar_val: float) -> float:
        coeffs = self._coeffs
        factor = self._solar_factor(solar_val)
        k1 = float(coeffs.get("K1_base", 0.0)) * factor
        k2 = float(coeffs.get("K2_base", 0.0)) * factor
        if self.phase == "heating":
            return k1 + k2
        return 0.0


class MLPThermalResponse(ThermalResponse):
    """MLP+热力学混合响应（若MLP加载失败会退化为纯热模）。"""

    def __init__(self, *, phase: str, pickle_path: Path, fallback: PureThermalResponse) -> None:
        super().__init__(phase)
        self._fallback = fallback
        self._model = self._safe_load_model(pickle_path)

    @staticmethod
    def _safe_load_model(path: Path):
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            warnings.warn(f"未找到热力学 MLP 模型文件: {path}", RuntimeWarning, stacklevel=2)
        except Exception as exc:
            warnings.warn(f"加载热力学 MLP 模型失败 ({path.name}): {exc}", RuntimeWarning, stacklevel=2)
        return None

    def delta(self, time_minutes: float, solar_val: float) -> float:
        if self._model is None:
            return self._fallback.delta(time_minutes, solar_val)

        t_arr = np.asarray([max(0.0, float(time_minutes))], dtype=float)
        s_arr = np.asarray([float(solar_val)], dtype=float)
        try:
            pred = self._model.predict(t_arr, s_arr)
        except Exception as exc:
            warnings.warn(
                f"MLP 热模型推理失败（{self.phase}）: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._fallback.delta(time_minutes, solar_val)

        value = np.asarray(pred).reshape(-1)[0]
        if not np.isfinite(value):
            return self._fallback.delta(time_minutes, solar_val)
        return float(value)

    def steady_state_delta(self, solar_val: float) -> float:
        # MLP 版本使用纯热模的稳态作为基准，避免长时间推理。
        return self._fallback.steady_state_delta(solar_val)


# =============================================================================
# 一阶模型（阶段感知 + 兼容旧 RC 模型）
# =============================================================================
class FirstOrderThermalModel(BaseThermalModel):
    """阶段感知 LED 热模型。

    - Solar > threshold 时进入加热阶段，使用最新导出的 Thermal 模型；
    - Solar <= threshold 时进入冷却阶段；
    - 若无法提供 Solar，则回退到旧的一阶 RC 模型（power → 温度）。
    """

    def __init__(
        self,
        params: LedThermalParams | None = None,
        *,
        initial_temp: float | None = None,
        variant: Optional[str] = None,
    ) -> None:
        self.params = params or LedThermalParams()
        if variant is not None:
            self.params.model_variant = variant

        self._variant = (self.params.model_variant or DEFAULT_MODEL_VARIANT).lower().strip()
        self._model_dir = self._resolve_model_dir(self.params.model_dir)

        self._heating_response: Optional[ThermalResponse] = None
        self._cooling_response: Optional[ThermalResponse] = None

        start_temp = (
            float(initial_temp)
            if initial_temp is not None
            else float(self.params.base_ambient_temp)
        )

        self._temperature = start_temp
        self._legacy_temp = start_temp
        self._phase = "idle"
        self._phase_baseline = start_temp
        self._time_in_phase = 0.0
        self._active_solar = self.params.clamp_solar(self.params.solar_min)
        self._active_solar_raw = 0.0
        self._last_heating_solar = self._active_solar

        self._load_phase_models()
        # 确保起始温度生效
        self.reset(ambient_temp=start_temp)

    # --- 公共 API ---------------------------------------------------------
    @property
    def ambient_temp(self) -> float:
        return self._temperature

    @ambient_temp.setter
    def ambient_temp(self, value: float) -> None:
        self._temperature = float(value)
        self._legacy_temp = self._temperature
        self._phase = "idle"
        self._time_in_phase = 0.0
        self._phase_baseline = self._temperature

    @property
    def supports_solar_input(self) -> bool:
        return self._heating_response is not None and self._cooling_response is not None

    def reset(self, ambient_temp: float | None = None) -> None:
        base = (
            float(self.params.base_ambient_temp)
            if ambient_temp is None
            else float(ambient_temp)
        )
        self._temperature = base
        self._legacy_temp = base
        self._phase = "idle"
        self._time_in_phase = 0.0
        self._phase_baseline = base
        self._active_solar = self.params.clamp_solar(self.params.solar_min)
        self._active_solar_raw = 0.0
        self._last_heating_solar = self._active_solar

    def step(
        self,
        power: float | None = None,
        dt: float | None = None,
        *,
        solar_vol: float | None = None,
        dt_seconds: float | None = None,
    ) -> float:
        duration = self._resolve_dt(dt, dt_seconds)
        if duration <= 0:
            raise ValueError("dt must be positive")

        if solar_vol is not None and self.supports_solar_input:
            return self._step_with_solar(float(solar_vol), duration)

        if power is None:
            raise ValueError("power must be provided when solar_vol is absent")
        return self._legacy_step(float(power), duration)

    def step_with_solar(self, solar_vol: float, dt: float) -> float:
        return self.step(dt=dt, solar_vol=solar_vol)

    def target_temperature(self, power: float) -> float:
        return float(self.params.base_ambient_temp + float(power) * self.params.thermal_resistance)

    def target_temperature_solar(self, solar_vol: float) -> float:
        if not self.supports_solar_input or self._heating_response is None:
            return self.target_temperature(0.0)
        delta = self._heating_response.steady_state_delta(self.params.clamp_solar(solar_vol))
        return float(self.params.base_ambient_temp + delta)

    def get_model_info(self) -> dict[str, float]:
        return {
            "phase": self._phase,
            "time_in_phase_min": self._time_in_phase,
            "active_solar": self._active_solar,
            "baseline_temp": self._phase_baseline,
            "ambient_temp": self._temperature,
        }

    # --- 私有：Solar 驱动 --------------------------------------------------
    def _step_with_solar(self, solar_vol: float, dt_seconds: float) -> float:
        dt_minutes = dt_seconds / 60.0
        solar_input = float(solar_vol)
        phase = "heating" if solar_input > self.params.solar_on_threshold else "cooling"

        if phase == "heating":
            active_solar = self.params.clamp_solar(solar_input)
        else:
            active_solar = self.params.clamp_solar(self._last_heating_solar)
            if active_solar <= 0.0:
                active_solar = self.params.clamp_solar(self.params.solar_min)

        phase_changed = phase != self._phase
        if not phase_changed and phase == "heating":
            phase_changed = abs(solar_input - self._active_solar_raw) > self.params.solar_change_tolerance

        if phase == "heating":
            self._last_heating_solar = active_solar

        if phase_changed:
            self._start_new_phase(phase, active_solar, solar_input)

        self._time_in_phase += dt_minutes
        response = self._heating_response if self._phase == "heating" else self._cooling_response
        if response is None:
            return self._legacy_step(0.0, dt_seconds)

        delta = response.delta(self._time_in_phase, active_solar)
        self._temperature = self._phase_baseline + delta
        self._legacy_temp = self._temperature
        return self._temperature

    def _start_new_phase(self, phase: str, active_solar: float, raw_solar: float) -> None:
        self._phase = phase
        self._time_in_phase = 0.0
        self._active_solar = active_solar
        self._active_solar_raw = raw_solar

        if phase == "heating":
            self._phase_baseline = self._temperature
        else:
            response = self._cooling_response
            if response is None:
                self._phase_baseline = self.params.base_ambient_temp
            else:
                delta0 = response.delta(0.0, active_solar)
                baseline = self._temperature - delta0
                min_base = self.params.base_ambient_temp
                self._phase_baseline = float(max(min_base, min(self._temperature, baseline)))

    # --- 私有：旧 RC 模型 --------------------------------------------------
    def _legacy_step(self, power: float, dt_seconds: float) -> float:
        tau = max(float(self.params.time_constant_s), 1e-6)
        alpha = 1.0 - math.exp(-float(dt_seconds) / tau)
        target = self.target_temperature(power)
        self._legacy_temp = float(self._legacy_temp + alpha * (target - self._legacy_temp))
        self._temperature = self._legacy_temp
        return self._temperature

    # --- 私有：工具 --------------------------------------------------------
    @staticmethod
    def _resolve_dt(dt: float | None, dt_seconds: float | None) -> float:
        if dt is not None and dt_seconds is not None:
            raise ValueError("dt and dt_seconds cannot be provided simultaneously")
        value = dt_seconds if dt_seconds is not None else dt
        if value is None:
            raise ValueError("dt must be provided")
        return float(value)

    @staticmethod
    def _resolve_model_dir(model_dir: Optional[str]) -> Path:
        if model_dir:
            path = Path(model_dir).expanduser()
            if path.is_dir():
                return path
            warnings.warn(f"指定的热模型目录不存在: {path}", RuntimeWarning, stacklevel=2)
        return _DEFAULT_THERMAL_EXPORT_DIR

    def _load_phase_models(self) -> None:
        heating_params = self._load_json("heating_thermal_model.json")
        cooling_params = self._load_json("cooling_thermal_model.json")

        if heating_params:
            heating_pure = PureThermalResponse(heating_params, "heating")
            self._heating_response = self._wrap_variant("heating_mlp_model.pkl", heating_pure)
        if cooling_params:
            cooling_pure = PureThermalResponse(cooling_params, "cooling")
            self._cooling_response = self._wrap_variant("cooling_mlp_model.pkl", cooling_pure)

    def _wrap_variant(self, pickle_name: str, pure: PureThermalResponse) -> ThermalResponse:
        if self._variant != "mlp":
            return pure
        path = self._model_dir / pickle_name
        response = MLPThermalResponse(phase=pure.phase, pickle_path=path, fallback=pure)
        return response

    def _load_json(self, filename: str) -> Optional[dict]:
        path = self._model_dir / filename
        if not path.exists():
            warnings.warn(f"未找到热模型参数文件: {path}", RuntimeWarning, stacklevel=2)
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            warnings.warn(f"读取热模型参数失败 ({filename}): {exc}", RuntimeWarning, stacklevel=2)
            return None


# =============================================================================
# 便捷封装
# =============================================================================
LedThermalModel = FirstOrderThermalModel


class Led:
    """使用新版热模型的 LED 封装。"""

    def __init__(
        self,
        model_type: str = "first_order",
        params: LedThermalParams | None = None,
        *,
        initial_temp: float | None = None,
    ) -> None:
        params_obj = params or LedThermalParams()
        variant = None
        mt = model_type.lower().strip()
        if mt in {"mlp", "mlp_phase"}:
            variant = "mlp"
        elif mt in {"pure", "first_order", "first", "fo", "1", "solar"}:
            variant = "pure"
        else:
            raise ValueError(f"Unsupported thermal model type: {model_type}")

        params_obj.model_variant = variant
        self.params = params_obj
        self.model = FirstOrderThermalModel(self.params, initial_temp=initial_temp)

    def reset(self, ambient_temp: float | None = None) -> None:
        self.model.reset(ambient_temp)

    def step_with_heat(self, power: float, dt: float) -> float:
        return self.model.step(power=float(power), dt=float(dt))

    def step_with_solar(self, solar_vol: float, dt: float) -> float:
        return self.model.step(dt=float(dt), solar_vol=float(solar_vol))

    def target_temperature(self, power: float) -> float:
        return self.model.target_temperature(power)

    def target_temperature_solar(self, solar_vol: float) -> float:
        return self.model.target_temperature_solar(solar_vol)

    @property
    def temperature(self) -> float:
        return self.model.ambient_temp

    @property
    def supports_solar_input(self) -> bool:
        return self.model.supports_solar_input

    def get_model_info(self) -> dict[str, float]:
        return self.model.get_model_info()


def create_model(
    model_type: str = "first_order",
    params: LedThermalParams | None = None,
    *,
    initial_temp: float | None = None,
) -> FirstOrderThermalModel:
    params_obj = params or LedThermalParams()
    mt = model_type.lower().strip()
    if mt in {"mlp", "mlp_phase"}:
        params_obj.model_variant = "mlp"
    elif mt in {"pure", "first_order", "first", "1", "fo", "solar"}:
        params_obj.model_variant = "pure"
    else:
        raise ValueError(f"Unsupported thermal model type: {model_type}")
    return FirstOrderThermalModel(params_obj, initial_temp=initial_temp)


def create_default_params() -> LedThermalParams:
    return LedThermalParams()


LedParams = LedThermalParams


__all__ = [
    "DEFAULT_BASE_AMBIENT_TEMP",
    "DEFAULT_THERMAL_RESISTANCE",
    "DEFAULT_TIME_CONSTANT_S",
    "DEFAULT_THERMAL_MASS",
    "DEFAULT_MAX_PPFD",
    "DEFAULT_MAX_POWER",
    "DEFAULT_LED_EFFICIENCY",
    "DEFAULT_EFFICIENCY_DECAY",
    "DEFAULT_MODEL_VARIANT",
    "LedThermalParams",
    "BaseThermalModel",
    "FirstOrderThermalModel",
    "LedThermalModel",
    "Led",
    "create_model",
    "create_default_params",
    "LedParams",
]

