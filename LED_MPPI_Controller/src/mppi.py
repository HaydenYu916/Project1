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

# 必需的光合作用模型（必须加载成功）
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# 多模型支持：加载所有可用的模型
try:
    import joblib
    import pickle
    import json
    from sklearn.preprocessing import StandardScaler
    
    # 模型配置
    MODEL_CONFIGS = {
        'solar_vol': {
            'feature_columns': ['Solar_Vol', 'CO2', 'T', 'R:B'],
            'model_path': os.path.join(os.path.dirname(__file__), '..', 'models', 'solar_vol', 'best_model.pkl'),
            'normalizer_path': os.path.join(os.path.dirname(__file__), '..', 'models', 'solar_vol', 'normalization_params.pkl'),
            'feature_info_path': os.path.join(os.path.dirname(__file__), '..', 'models', 'solar_vol', 'feature_info.pkl')
        },
        'ppfd': {
            'feature_columns': ['PPFD', 'CO2', 'T', 'R:B'],
            'model_path': os.path.join(os.path.dirname(__file__), '..', 'models', 'ppfd', 'best_model.pkl'),
            'normalizer_path': os.path.join(os.path.dirname(__file__), '..', 'models', 'ppfd', 'normalization_params.pkl'),
            'feature_info_path': os.path.join(os.path.dirname(__file__), '..', 'models', 'ppfd', 'feature_info.pkl')
        },
        'sp': {
            'feature_columns': ['sp_415', 'sp_445', 'sp_480', 'sp_515', 'sp_555', 'sp_590', 'sp_630', 'sp_680', 'CO2', 'T', 'R:B'],
            'model_path': os.path.join(os.path.dirname(__file__), '..', 'models', 'sp', 'best_model.pkl'),
            'normalizer_path': os.path.join(os.path.dirname(__file__), '..', 'models', 'sp', 'normalization_params.pkl'),
            'feature_info_path': os.path.join(os.path.dirname(__file__), '..', 'models', 'sp', 'feature_info.pkl')
        }
    }
    
    # 加载所有可用的模型
    AVAILABLE_MODELS = {}
    
    for model_name, config in MODEL_CONFIGS.items():
        try:
            # 检查文件是否存在
            if not all(os.path.exists(p) for p in [config['model_path'], config['normalizer_path'], config['feature_info_path']]):
                print(f"警告: {model_name} 模型文件不完整，跳过")
                continue
            
            # 加载模型和预处理器
            model = joblib.load(config['model_path'])
            normalizer = joblib.load(config['normalizer_path'])
            
            with open(config['feature_info_path'], 'rb') as f:
                feature_info = pickle.load(f)
            
            AVAILABLE_MODELS[model_name] = {
                'model': model,
                'normalizer': normalizer,
                'feature_info': feature_info,
                'config': config
            }
            print(f"✓ 成功加载 {model_name} 模型")
            
        except Exception as e:
            print(f"警告: 无法加载 {model_name} 模型: {e}")
            continue
    
    if not AVAILABLE_MODELS:
        raise ImportError("没有可用的模型")
    
    # 创建通用预测器类
    class ModelPredictor:
        def __init__(self, model_name='solar_vol'):
            if model_name not in AVAILABLE_MODELS:
                raise ValueError(f"模型 {model_name} 不可用。可用模型: {list(AVAILABLE_MODELS.keys())}")
            
            self.model_name = model_name
            model_data = AVAILABLE_MODELS[model_name]
            self.model = model_data['model']
            self.normalizer = model_data['normalizer']
            self.feature_info = model_data['feature_info']
            self.config = model_data['config']
            self.is_loaded = True
        
        def predict(self, ppfd, co2, temperature, rb_ratio):
            # 根据模型类型构造不同的输入特征
            if self.model_name == 'solar_vol':
                # solar_vol 模型：ppfd 对应 Solar_Vol
                features = np.array([[ppfd, co2, temperature, rb_ratio]], dtype=float)
            elif self.model_name == 'ppfd':
                # ppfd 模型：ppfd 对应 PPFD
                features = np.array([[ppfd, co2, temperature, rb_ratio]], dtype=float)
            elif self.model_name == 'sp':
                # sp 模型：需要光谱数据，这里用 ppfd 生成模拟光谱数据
                # 模拟光谱数据（基于 ppfd 和 rb_ratio）
                sp_415 = ppfd * 0.1 * rb_ratio  # 蓝光
                sp_445 = ppfd * 0.15 * rb_ratio
                sp_480 = ppfd * 0.2 * rb_ratio
                sp_515 = ppfd * 0.15 * (1 - rb_ratio)  # 绿光
                sp_555 = ppfd * 0.1 * (1 - rb_ratio)
                sp_590 = ppfd * 0.1 * (1 - rb_ratio)  # 橙光
                sp_630 = ppfd * 0.15 * (1 - rb_ratio)  # 红光
                sp_680 = ppfd * 0.05 * (1 - rb_ratio)
                
                features = np.array([[sp_415, sp_445, sp_480, sp_515, sp_555, sp_590, sp_630, sp_680, co2, temperature, rb_ratio]], dtype=float)
            else:
                raise ValueError(f"未知的模型类型: {self.model_name}")
            
            # 手动标准化输入
            feat_mean = self.normalizer['feat_mean']
            feat_std = self.normalizer['feat_std']
            features_normalized = (features - feat_mean) / feat_std
            
            # 预测光合作用速率
            photosynthesis_rate_normalized = self.model.predict(features_normalized)[0]
            
            # 反标准化输出
            target_mean = self.normalizer['target_mean']
            target_std = self.normalizer['target_std']
            photosynthesis_rate = photosynthesis_rate_normalized * target_std + target_mean
            
            return photosynthesis_rate
    
    # 默认使用 solar_vol 模型
    PhotosynthesisPredictor = lambda model_name='solar_vol': ModelPredictor(model_name)
    test_predictor = PhotosynthesisPredictor('solar_vol')
    PHOTOSYNTHESIS_AVAILABLE = True
    
    print(f"可用模型: {list(AVAILABLE_MODELS.keys())}")
    
except Exception as e:
    # 如果所有模型都加载失败，回退到原始的光合作用模型
    try:
        from pn_prediction.predict import PhotosynthesisPredictor
        # 测试模型是否能正常初始化
        test_predictor = PhotosynthesisPredictor()
        PHOTOSYNTHESIS_AVAILABLE = True
        print("使用原始 PPFD 模型")
    except Exception as e2:
        try:
            from pn_prediction.predict_corrected import CorrectedPhotosynthesisPredictor as PhotosynthesisPredictor
            # 测试模型是否能正常初始化
            test_predictor = PhotosynthesisPredictor()
            PHOTOSYNTHESIS_AVAILABLE = True
            print("使用修正的 PPFD 模型")
        except Exception as e3:
            raise ImportError(f"无法加载任何模型。错误: {e}") from e3


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
        model_name: str = "solar_vol",  # 新增：选择模型类型
    ) -> None:
        self.params = params or LedThermalParams()
        self.thermal = create_model(model_type, self.params, initial_temp=self.params.base_ambient_temp)
        self.model_key = model_key
        self.use_efficiency = use_efficiency
        self.heat_scale = float(heat_scale)
        self.power_model = power_model or PWMtoPowerModel(include_intercept=True).fit(DEFAULT_CALIB_CSV)
        self.ppfd_model = ppfd_model or PWMtoPPFDModel(include_intercept=True).fit(DEFAULT_CALIB_CSV)
        self.eta_model = eta_model
        self.model_name = model_name  # 保存模型名称

        # 光合作用预测器（必需）
        self.photo_predictor = pn_predictor
        if self.photo_predictor is None:
            if not PHOTOSYNTHESIS_AVAILABLE:
                raise ImportError("光合作用模型不可用，无法创建LEDPlant")
            try:
                # 使用指定的模型名称创建预测器
                self.photo_predictor = PhotosynthesisPredictor(model_name)
                self.use_photo_model = getattr(self.photo_predictor, "is_loaded", True)
                if not self.use_photo_model:
                    raise RuntimeError("光合作用模型加载失败")
                print(f"使用 {model_name} 模型进行光合作用预测")
            except Exception as e:
                raise RuntimeError(f"无法初始化光合作用预测器 ({model_name}): {e}") from e
        else:
            self.use_photo_model = getattr(self.photo_predictor, "is_loaded", True)
            if not self.use_photo_model:
                raise RuntimeError("提供的光合作用预测器未正确加载")

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
        if not self.use_photo_model or self.photo_predictor is None:
            raise RuntimeError("光合作用预测器未正确初始化")
        
        try:
            return float(self.photo_predictor.predict(ppfd, co2, temperature, rb_ratio))
        except Exception as e:
            raise RuntimeError(f"光合作用预测失败: {e}") from e

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
        self.constraints = {'pwm_min': 5.0, 'pwm_max': 95.0, 'temp_min': 20.0, 'temp_max': 30.0}
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
        except Exception as e:
            raise RuntimeError(f"成本计算失败: {e}") from e

    def _temperature_safety_check(self, pwm_action_rb, current_temp):
        seq = np.asarray([pwm_action_rb], dtype=float)
        _, temp_check, _, _ = self.plant.predict(seq, current_temp, self.dt)
        if temp_check[0] > self.constraints['temp_max']:
            reduced = np.clip(pwm_action_rb * 0.7, self.constraints['pwm_min'], self.constraints['pwm_max'])
            return reduced
        return pwm_action_rb
