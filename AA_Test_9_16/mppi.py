"""
MPPI (Model Predictive Path Integral) 控制器核心模块（API）

- 提供 LEDPlant（双通道PWM）与 LEDMPPIController（核心算法）
- 不包含演示/绘图代码；示例请使用 mppi_demo.py
"""
import numpy as np
import warnings
from typing import Callable
warnings.filterwarnings("ignore")

# --- 新 LED 模型接口（解耦）---
import os
import sys
try:
    from .led import (
        LedThermalParams,
        create_model,
        PWMtoPowerModel,
        PWMtoPPFDModel,
        forward_step,
        DEFAULT_CALIB_CSV,
    )
except ImportError:
    from led import (
        LedThermalParams,
        create_model,
        PWMtoPowerModel,
        PWMtoPPFDModel,
        forward_step,
        DEFAULT_CALIB_CSV,
    )

# --- 全局宏：默认红蓝比例键（修改此处即可切换默认比例） ---
RB_RATIO_KEY = "5:1"


def get_ratio_weights(key: str) -> tuple[float, float]:
    s = str(key).strip().lower().replace(" ", "")
    if s in ("r1", "r:1"):
        return 1.0, 0.0
    if ":" in s:
        a, b = s.split(":", 1)
        try:
            ra = float(a); rb = float(b)
        except Exception:
            return 0.5, 0.5
        ssum = ra + rb if (ra + rb) > 0 else 1.0
        return ra / ssum, rb / ssum
    return 0.5, 0.5

# 可选光合作用模型（如不可用则退化）
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
try:
    # 优先使用基于 PPFD 的模型（加载路径指向 models/MODEL/PPFD）
    from pn_prediction.predict import PhotosynthesisPredictor
    PHOTOSYNTHESIS_AVAILABLE = True
except Exception:
    try:
        from pn_prediction.predict_corrected import CorrectedPhotosynthesisPredictor as PhotosynthesisPredictor
        PHOTOSYNTHESIS_AVAILABLE = True
    except Exception:
        PHOTOSYNTHESIS_AVAILABLE = False


class LEDPlant:
    def __init__(
        self,
        *,
        params: LedThermalParams | None = None,
        model_key: str | None = RB_RATIO_KEY,
        use_efficiency: bool = False,
        heat_scale: float = 1.0,
        power_model: PWMtoPowerModel | None = None,
        ppfd_model: PWMtoPPFDModel | None = None,
        model_type: str = "first_order",
        pn_predictor: "PhotosynthesisPredictor | None" = None,
        eta_model: "Callable[[float, float, float, float, LedThermalParams], float] | None" = None,
    ) -> None:
        self.params = params or LedThermalParams()
        self.thermal = create_model(model_type, self.params, initial_temp=self.params.base_ambient_temp)
        self.model_key = model_key
        self.use_efficiency = use_efficiency
        self.heat_scale = float(heat_scale)
        self.power_model = power_model or PWMtoPowerModel(include_intercept=True).fit(DEFAULT_CALIB_CSV)
        self.ppfd_model = ppfd_model or PWMtoPPFDModel(include_intercept=True).fit(DEFAULT_CALIB_CSV)
        self.eta_model = eta_model

        # 光合作用预测器（可选）
        self.photo_predictor = pn_predictor
        self.use_photo_model = False
        if self.photo_predictor is None and PHOTOSYNTHESIS_AVAILABLE:
            try:
                self.photo_predictor = PhotosynthesisPredictor()
                self.use_photo_model = getattr(self.photo_predictor, "is_loaded", True)
            except Exception:
                self.use_photo_model = False
        elif self.photo_predictor is not None:
            self.use_photo_model = getattr(self.photo_predictor, "is_loaded", True)

    def step(self, r_pwm: float, b_pwm: float, dt: float = 0.1):
        out = forward_step(
            thermal_model=self.thermal,
            r_pwm=float(r_pwm),
            b_pwm=float(b_pwm),
            dt=float(dt),
            power_model=self.power_model,
            ppfd_model=self.ppfd_model,
            model_key=self.model_key,
            use_efficiency=self.use_efficiency,
            heat_scale=self.heat_scale,
            eta_model=self.eta_model,
        )
        total = max(1e-6, float(r_pwm) + float(b_pwm))
        rb_ratio = float(r_pwm) / total
        photosynthesis_rate = self.get_photosynthesis_rate(out.ppfd or 0.0, out.temp, rb_ratio=rb_ratio)
        return (out.ppfd or 0.0), out.temp, out.power, photosynthesis_rate

    def get_photosynthesis_rate(self, ppfd, temperature, co2=400, rb_ratio=0.83):
        if self.use_photo_model and self.photo_predictor is not None:
            try:
                return float(self.photo_predictor.predict(ppfd, co2, temperature, rb_ratio))
            except Exception:
                pass
        # 退化的简化模型
        temp_factor = np.exp(-0.01 * (temperature - 25.0) ** 2)
        return float(max(0.0, (25.0 * ppfd / (300.0 + ppfd)) * temp_factor))

    def predict(self, pwm_sequence_rb: np.ndarray, initial_temp: float, dt: float = 0.1):
        self.thermal.reset(initial_temp)
        ppfd_pred, temp_pred, power_pred, photo_pred = [], [], [], []
        for r_pwm, b_pwm in np.asarray(pwm_sequence_rb, dtype=float):
            out = forward_step(
                thermal_model=self.thermal,
                r_pwm=float(r_pwm),
                b_pwm=float(b_pwm),
                dt=float(dt),
                power_model=self.power_model,
                ppfd_model=self.ppfd_model,
                model_key=self.model_key,
                use_efficiency=self.use_efficiency,
                heat_scale=self.heat_scale,
                eta_model=self.eta_model,
            )
            ppfd_pred.append(out.ppfd or 0.0)
            temp_pred.append(out.temp)
            power_pred.append(out.power)
            photo_pred.append(self.get_photosynthesis_rate(out.ppfd or 0.0, out.temp))
        return (
            np.asarray(ppfd_pred, dtype=float),
            np.asarray(temp_pred, dtype=float),
            np.asarray(power_pred, dtype=float),
            np.asarray(photo_pred, dtype=float),
        )


class LEDMPPIController:
    def __init__(self, plant, horizon=10, num_samples=1000, dt=0.1, temperature=1.0, maintain_rb_ratio=False, rb_ratio_key="5:1"):
        self.plant = plant
        self.horizon = horizon
        self.num_samples = num_samples
        self.dt = dt
        self.temperature = temperature
        self.u_prev = np.zeros(2, dtype=float)
        
        # R:B 比例约束相关参数
        self.maintain_rb_ratio = maintain_rb_ratio
        self.rb_ratio_key = rb_ratio_key
        if maintain_rb_ratio:
            self.w_r, self.w_b = get_ratio_weights(rb_ratio_key)
        else:
            self.w_r, self.w_b = 1.0, 1.0

        self.weights = {'Q_photo': 10.0, 'R_pwm': 0.001, 'R_dpwm': 0.05, 'R_power': 0.01}
        self.constraints = {'pwm_min': 0.0, 'pwm_max': 80.0, 'temp_min': 20.0, 'temp_max': 29.0}
        self.penalties = {'temp_penalty': 100000.0, 'pwm_penalty': 1000.0}
        self.pwm_std = np.array([15.0, 15.0], dtype=float)

    def set_weights(self, **kwargs):
        self.weights.update(kwargs)

    def set_constraints(self, **kwargs):
        self.constraints.update(kwargs)

    def set_mppi_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def solve(self, current_temp, mean_sequence=None):
        if mean_sequence is None:
            base = min(40.0, self.constraints['pwm_max'] * 0.5)
            w_r, w_b = get_ratio_weights(RB_RATIO_KEY)
            mean_rb = np.array([base * w_r, base * w_b], dtype=float)
            mean_sequence = np.tile(mean_rb, (self.horizon, 1))
        control_samples = self._sample_control_sequences(mean_sequence)

        costs = np.array([self._compute_total_cost(sample, current_temp) for sample in control_samples])
        costs = np.nan_to_num(costs, nan=1e10, posinf=1e10)

        min_cost = np.min(costs)
        exp_costs = np.exp(-(costs - min_cost) / self.temperature)
        weights = exp_costs / np.sum(exp_costs)

        optimal_sequence = np.sum(weights[:, np.newaxis, np.newaxis] * control_samples, axis=0)
        optimal_sequence = np.clip(optimal_sequence, self.constraints['pwm_min'], self.constraints['pwm_max'])

        optimal_action = optimal_sequence[0].copy()
        optimal_action = self._temperature_safety_check(optimal_action, current_temp)
        self.u_prev = optimal_action
        return optimal_action, optimal_sequence, True, np.min(costs), weights

    def _sample_control_sequences(self, mean_sequence):
        if self.maintain_rb_ratio:
            # 维持 R:B 比例约束的采样
            # 只对总功率进行采样，然后按比例分配
            total_pwm_mean = mean_sequence[:, 0] + mean_sequence[:, 1]  # 总功率均值
            total_pwm_std = np.sqrt(self.pwm_std[0]**2 + self.pwm_std[1]**2)  # 总功率标准差
            
            # 对总功率进行采样
            noise = np.random.normal(0.0, 1.0, (self.num_samples, self.horizon)) * total_pwm_std
            total_pwm_samples = total_pwm_mean[np.newaxis, :] + noise
            
            # 限制总功率范围
            max_total = self.constraints['pwm_max'] * 2  # 理论最大值
            min_total = self.constraints['pwm_min'] * 2  # 理论最小值
            total_pwm_samples = np.clip(total_pwm_samples, min_total, max_total)
            
            # 按比例分配到 R 和 B 通道
            samples = np.zeros((self.num_samples, self.horizon, 2))
            samples[:, :, 0] = total_pwm_samples * self.w_r  # R 通道
            samples[:, :, 1] = total_pwm_samples * self.w_b  # B 通道
            
            # 确保每个通道都在约束范围内
            samples = np.clip(samples, self.constraints['pwm_min'], self.constraints['pwm_max'])
            
            return samples
        else:
            # 原始的独立采样方法
            noise = np.random.normal(0.0, 1.0, (self.num_samples, self.horizon, 2)) * self.pwm_std
            samples = mean_sequence[np.newaxis, :, :] + noise
            return np.clip(samples, self.constraints['pwm_min'], self.constraints['pwm_max'])

    def _compute_total_cost(self, pwm_sequence_rb, current_temp):
        try:
            ppfd_pred, temp_pred, power_pred, photo_pred = self.plant.predict(
                pwm_sequence_rb, current_temp, self.dt
            )
            photo_cost = -np.sum(photo_pred) * self.weights['Q_photo']
            pwm_cost = np.sum(pwm_sequence_rb**2) * self.weights['R_pwm']
            du = np.diff(np.vstack([self.u_prev, pwm_sequence_rb]), axis=0)
            dpwm_cost = np.sum(du**2) * self.weights['R_dpwm']
            power_cost = np.sum(power_pred**2) * self.weights['R_power']

            temp_violation = (
                np.maximum(0, temp_pred - self.constraints['temp_max'])**2 +
                np.maximum(0, self.constraints['temp_min'] - temp_pred)**2
            )
            temp_penalty_cost = np.sum(temp_violation) * self.penalties['temp_penalty']
            return float(photo_cost + pwm_cost + dpwm_cost + power_cost + temp_penalty_cost)
        except Exception:
            return 1e10

    def _temperature_safety_check(self, pwm_action_rb, current_temp):
        seq = np.asarray([pwm_action_rb], dtype=float)
        _, temp_check, _, _ = self.plant.predict(seq, current_temp, self.dt)
        if temp_check[0] > self.constraints['temp_max']:
            reduced = np.clip(pwm_action_rb * 0.7, self.constraints['pwm_min'], self.constraints['pwm_max'])
            return reduced
        return pwm_action_rb
