import os
import warnings
import numpy as np
import joblib
import pandas as pd

warnings.filterwarnings("ignore")

# 底层LED库
try:
    from .led import (
        LedThermalParams,
        ThermalModelManager,
        FirstOrderThermalModel,  # 兼容性别名
        PWMtoPPFDModel,  # 备用（当前未直接使用）
        PWMtoPowerModel,
    )
except ImportError:
    # 直接运行本文件时，使用绝对导入（src 目录会在 sys.path 中）
    from led import (
        LedThermalParams,
        ThermalModelManager,
        FirstOrderThermalModel,  # 兼容性别名
        PWMtoPPFDModel,  # 备用（当前未直接使用）
        PWMtoPowerModel,
    )

# 传感器读取 - 简化版本，移除外部依赖
DEFAULT_DEVICE_ID = "T6ncwg=="
RIOTEE_DATA_PATH = "Sensor/riotee_sensor/data"
CO2_DATA_PATH = "Sensor/riotee_sensor/data"
DEFAULT_CO2_PPM = 400.0

class SensorReading:
    """简化的传感器读取类，用于MPPI演示"""
    def __init__(self, device_id=None, riotee_data_path=None, co2_data_path=None):
        self.device_id = device_id or DEFAULT_DEVICE_ID
        self.riotee_data_path = riotee_data_path or RIOTEE_DATA_PATH
        self.co2_data_path = co2_data_path or CO2_DATA_PATH
    
    def read_latest_riotee_data(self):
        """读取最新的Riotee数据 - 简化版本"""
        # 返回None表示使用默认温度
        return None, None, None, None
    
    def read_latest_co2_data(self):
        """读取最新的CO2数据 - 简化版本"""
        return DEFAULT_CO2_PPM


# ------------------------------
# LEDPlant (Solar Vol 控制)
# ------------------------------
class LEDPlant:
    """使用新版热力学模型的MPPI LED植物模型，支持Solar Vol控制"""

    def __init__(
        self,
        base_ambient_temp=25.0,
        max_solar_vol=2.0,  # 支持更广范围，实际数据范围1.202-1.912V
        max_power=130.0,
        thermal_resistance=0.05,
        time_constant_s=7.5,
        thermal_mass=150.0,
        solar_vol_model=None,
        power_model=None,
        co2_ppm=400.0,
        r_b_ratio=0.83,
        use_solar_vol_model=True,
        thermal_model_type: str = "thermal",  # "mlp" 或 "thermal"
        model_dir: str = "Thermal/exported_models",  # 模型文件目录
    ):
        self.base_ambient_temp = base_ambient_temp
        self.max_solar_vol = max_solar_vol
        self.max_power = max_power

        self.co2_ppm = co2_ppm
        self.r_b_ratio = r_b_ratio
        self.use_solar_vol_model = use_solar_vol_model
        self.thermal_model_type = thermal_model_type
        self.model_dir = model_dir
        
        # MPPI控制状态跟踪
        self.current_control = 0.0  # 当前控制量u0
        self.previous_control = 0.0  # 前一个控制量u1

        # 🔥 新版热力学参数与模型
        self.thermal_params = LedThermalParams(
            base_ambient_temp=base_ambient_temp,
            thermal_resistance=thermal_resistance,
            time_constant_s=time_constant_s,
            thermal_mass=thermal_mass,
            max_ppfd=max_solar_vol * 100,
            max_power=max_power,
            model_type=thermal_model_type,
            model_dir=model_dir,
            solar_threshold=1.4,  # 添加Solar阈值
        )
        
        # 🔥 使用新版热力学模型管理器
        self.thermal_model = ThermalModelManager(self.thermal_params)

        # 模型
        self.solar_vol_model = solar_vol_model
        self.power_model = power_model

        # 状态
        self.ambient_temp = base_ambient_temp
        self.time = 0.0
        self.current_solar_vol = 0.0

        # 传感器
        self.sensor_reader = SensorReading(
            device_id=DEFAULT_DEVICE_ID,
            riotee_data_path=RIOTEE_DATA_PATH,
            co2_data_path=CO2_DATA_PATH,
        )

        # 光合作用模型
        self._init_photosynthesis_models()
        
        print(f"🔥 LEDPlant初始化完成:")
        print(f"   热力学模型类型: {thermal_model_type}")
        print(f"   模型目录: {model_dir}")
        print(f"   MPPI控制跟踪: 当前u0={self.current_control}, 前一个u1={self.previous_control}")

    def _init_photosynthesis_models(self):#初始化光合作用预测模型 - 只使用Solar Vol模型
        """初始化光合作用预测模型 - 只使用Solar Vol模型"""
        self.solar_vol_model = None
        self.use_photo_model = False

        if self.use_solar_vol_model:
            try:
                self._load_solar_vol_model()
                self.use_photo_model = True
                print("成功加载Solar Vol光合作用模型")
                return
            except Exception as e:
                print(f"错误: Solar Vol模型加载失败: {e}")
                raise RuntimeError(f"Solar Vol模型加载失败: {e}")

        print("警告: 未启用Solar Vol模型")
        self.use_photo_model = False

    def _load_solar_vol_model(self):#加载Solar Vol模型 - 使用joblib
        SOLAR_VOL_MODEL_PATH = os.path.join(
            os.path.dirname(__file__), "..", "models", "solar_vol"
        )
        model_file = os.path.join(SOLAR_VOL_MODEL_PATH, "best_model.pkl")
        feature_file = os.path.join(SOLAR_VOL_MODEL_PATH, "feature_info.pkl")
        normalizer_file = os.path.join(
            SOLAR_VOL_MODEL_PATH, "normalization_params.pkl"
        )

        print(f"🔍 模型文件路径检查:")
        print(f"   模型目录: {SOLAR_VOL_MODEL_PATH}")
        print(f"   模型文件: {model_file}")
        print(f"   特征文件: {feature_file}")
        print(f"   归一化文件: {normalizer_file}")
        print(f"   模型目录存在: {os.path.exists(SOLAR_VOL_MODEL_PATH)}")
        print(f"   模型文件存在: {os.path.exists(model_file)}")

        self.feature_info = {
            "feature_columns": ["Solar_Vol", "CO2", "T", "R:B"],
            "pn_column": "Pn_avg",
            "model_name": "SVR (RBF)",
        }

        if os.path.exists(feature_file):
            try:
                import pickle

                with open(feature_file, "rb") as f:
                    feature_info_from_file = pickle.load(f)
                self.feature_info.update(feature_info_from_file)
                print(f"✅ 成功加载特征信息: {self.feature_info}")
            except Exception as e:
                print(f"警告: 无法加载特征信息文件，使用默认值: {e}")

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Solar Vol模型文件不存在: {model_file}")

        try:
            self.solar_vol_model = joblib.load(model_file)
            print(f"✅ 成功使用joblib加载Solar Vol模型: {type(self.solar_vol_model)}")
            print(f"Solar Vol模型信息: {self.feature_info}")

            if not os.path.exists(normalizer_file):
                raise FileNotFoundError(
                    f"归一化参数文件不存在: {normalizer_file}"
                )

            try:
                self.normalizer = joblib.load(normalizer_file)
                if not all(
                    k in self.normalizer
                    for k in ("feat_mean", "feat_std", "target_mean", "target_std")
                ):
                    raise ValueError(
                        "normalization_params.pkl 缺少必要键: feat_mean, feat_std, target_mean, target_std"
                    )
                print("✅ 已加载Solar Vol模型的归一化参数")
            except Exception as e:
                raise RuntimeError(f"加载归一化参数失败: {e}")

        except Exception as e:
            raise RuntimeError(f"joblib加载Solar Vol模型失败: {e}")
    #重点步骤
    def step(self, solar_vol, dt=900, device_id=DEFAULT_DEVICE_ID):
        """🔥 新版热力学模型单步仿真 - 基于MPPI控制量变化"""
        if device_id != DEFAULT_DEVICE_ID:
            self.sensor_reader.device_id = device_id
            
        # 读取实时温度（可选）
        try:
            current_temp, _csv_sv, _pn, timestamp = (
                self.sensor_reader.read_latest_riotee_data()
            )
            if current_temp is not None:
                self.ambient_temp = current_temp
                self.thermal_model.ambient_temp = current_temp
                print(f"🌡️ 使用最新温度: {current_temp:.2f}°C (时间戳: {timestamp})")
        except Exception:
            current_temp, timestamp = None, None

        # Solar Vol → (R_PWM, B_PWM)
        r_pwm, b_pwm = self._solar_vol_to_pwm(float(solar_vol), self.r_b_ratio)

        # 功率计算
        if self.power_model is None:
            raise RuntimeError("功率模型未提供，请使用 led.py 中的 PWMtoPowerModel")
        total_pwm_for_power = r_pwm + b_pwm
        power_key = self._get_power_model_key(self.r_b_ratio)
        power = self.power_model.predict(total_pwm=total_pwm_for_power, key=power_key)

        # 🔥 计算MPPI控制量变化: u0 - u1
        control_change = float(solar_vol) - self.previous_control
        
        # 🔥 新版热力学模型步进 - 基于控制量变化判断升温/降温
        new_ambient_temp = self.thermal_model.step(
            power=power,
            dt=dt,
            solar_vol=float(solar_vol),
            control_change=control_change,  # 传递控制量变化
        )

        # 状态更新
        self.current_solar_vol = float(solar_vol)
        self.ambient_temp = new_ambient_temp
        self.time += dt
        
        # 🔥 更新MPPI控制状态
        self.previous_control = self.current_control
        self.current_control = float(solar_vol)

        # CO2读取
        current_co2 = self.sensor_reader.read_latest_co2_data()

        # 光合作用预测
        photosynthesis_rate = self.get_photosynthesis_rate(
            self.current_solar_vol, new_ambient_temp, current_co2
        )

        # 输出调试信息
        phase = "升温" if control_change > 0 else "降温"
        print(f"🔥 MPPI热力学步进: u0={solar_vol:.3f}, Δu={control_change:.3f} ({phase}) → 温度: {new_ambient_temp:.2f}°C")

        return self.current_solar_vol, new_ambient_temp, power, photosynthesis_rate

    def get_photosynthesis_rate(
        self, solar_vol, temperature, co2_ppm=None, r_b_ratio=None):
        if not self.use_photo_model:
            raise RuntimeError("光合作用预测模型不可用，请确保模型正确安装")
        if self.solar_vol_model is None:
            raise RuntimeError("Solar Vol模型不可用，请确保模型正确加载")
        return self._predict_with_solar_vol_model(
            solar_vol,
            co2_ppm or self.co2_ppm,
            temperature,
            r_b_ratio or self.r_b_ratio,
        )

    def _predict_with_solar_vol_model(self, solar_vol, co2_ppm, temperature, r_b_ratio):
        if not hasattr(self, "normalizer") or self.normalizer is None:
            raise RuntimeError("归一化参数不可用，请确保 normalization_params.pkl 文件正确加载")

        features_raw = np.array([solar_vol, co2_ppm, temperature, r_b_ratio], dtype=float)
        feat_mean = np.asarray(self.normalizer.get("feat_mean"), dtype=float)
        feat_std = np.asarray(self.normalizer.get("feat_std"), dtype=float)
        target_mean = float(self.normalizer.get("target_mean"))
        target_std = float(self.normalizer.get("target_std"))
        feat_std = np.where(np.abs(feat_std) < 1e-9, 1.0, feat_std)
        features_norm = (features_raw - feat_mean) / feat_std
        y_norm = float(self.solar_vol_model.predict(features_norm.reshape(1, -1))[0])
        y = y_norm * target_std + target_mean
        raw_value = float(y)
        return max(0.0, raw_value)

    def _solar_vol_to_pwm(self, solar_vol, r_b_ratio=None):
        if r_b_ratio is None:
            r_b_ratio = self.r_b_ratio
        return self._lookup_solar_vol_to_pwm(solar_vol, r_b_ratio)

    def _lookup_solar_vol_to_pwm(self, solar_vol, r_b_ratio):
        try:
            # 懒加载查找表
            if not hasattr(self, "_solar_vol_data") or self._solar_vol_data is None:
                solar_vol_path = os.path.join(
                    os.path.dirname(__file__), "..", "data", "Solar_Vol_clean.csv"
                )
                calib_path = os.path.join(
                    os.path.dirname(__file__), "..", "data", "calib_data.csv"
                )
                self._solar_vol_data = pd.read_csv(solar_vol_path)
                self._calib_data = pd.read_csv(calib_path)
                self._rb_mapping = {
                    "1:1": 0.5,
                    "3:1": 0.75,
                    "5:1": 0.83,
                    "7:1": 0.88,
                    "r1": 1.0,
                }

            filtered_data = self._solar_vol_data[
                self._solar_vol_data["R:B"] == r_b_ratio
            ]
            if filtered_data.empty or filtered_data["Solar_Vol"].nunique() < 2:
                return self._fallback_linear_conversion(solar_vol)

            # 按 Solar Vol 进行线性插值得到目标 PPFD
            sorted_sv = filtered_data.sort_values("Solar_Vol")
            solar_vals = sorted_sv["Solar_Vol"].to_numpy(dtype=float)
            ppfd_vals = sorted_sv["PPFD"].to_numpy(dtype=float)
            target_ppfd = float(np.interp(float(solar_vol), solar_vals, ppfd_vals))

            calib_rb = None
            for calib_key, solar_val in self._rb_mapping.items():
                if abs(solar_val - r_b_ratio) < 0.01:
                    calib_rb = calib_key
                    break
            if calib_rb is None:
                return self._fallback_linear_conversion(solar_vol)

            calib_filtered = self._calib_data[self._calib_data["R:B"] == calib_rb]
            if calib_filtered.empty:
                return self._fallback_linear_conversion(solar_vol)

            calib_sorted = calib_filtered.sort_values("PPFD")
            ppfd_calib = calib_sorted["PPFD"].to_numpy(dtype=float)
            if len(ppfd_calib) < 2:
                return self._fallback_linear_conversion(solar_vol)

            r_pwm_interp = np.interp(target_ppfd, ppfd_calib, calib_sorted["R_PWM"].to_numpy(dtype=float))
            b_pwm_interp = np.interp(target_ppfd, ppfd_calib, calib_sorted["B_PWM"].to_numpy(dtype=float))

            r_pwm = min(100.0, max(0.0, float(r_pwm_interp)))
            b_pwm = min(100.0, max(0.0, float(b_pwm_interp)))
            return r_pwm, b_pwm
        except Exception:
            return self._fallback_linear_conversion(solar_vol)

    def _fallback_linear_conversion(self, solar_vol):
        total_pwm = (solar_vol / self.max_solar_vol) * 100.0
        total_pwm = max(0.0, min(100.0, total_pwm))
        r_pwm = total_pwm * self.r_b_ratio
        b_pwm = total_pwm * (1 - self.r_b_ratio)
        return r_pwm, b_pwm

    def _get_power_model_key(self, r_b_ratio):
        rb_to_key_mapping = {0.5: "1:1", 0.75: "3:1", 0.83: "5:1", 0.88: "7:1", 1.0: "r1"}
        closest_ratio = min(rb_to_key_mapping.keys(), key=lambda x: abs(x - r_b_ratio))
        if abs(closest_ratio - r_b_ratio) < 0.05:
            return rb_to_key_mapping[closest_ratio]
        return None

    def predict(
        self,
        solar_vol_control_sequence,
        initial_temp,
        dt=0.1,
        co2_sequence=None,
        r_b_sequence=None,
    ):
        """🔥 新版热力学模型预测整条控制序列 - 基于MPPI控制量变化"""
        # 创建独立的热力学模型实例用于预测
        temp_model = ThermalModelManager(self.thermal_params)
        temp_model.reset(initial_temp)

        temp = initial_temp
        solar_vol_inputs = []
        temp_predictions = []
        power_predictions = []
        photo_predictions = []
        r_pwm_predictions = []
        b_pwm_predictions = []
        
        # 🔥 MPPI控制状态跟踪
        prev_control = 0.0  # 初始控制量

        for i, solar_vol_control in enumerate(solar_vol_control_sequence):
            current_co2 = co2_sequence[i] if co2_sequence is not None else self.co2_ppm
            current_r_b = r_b_sequence[i] if r_b_sequence is not None else self.r_b_ratio

            r_pwm_control, b_pwm_control = self._solar_vol_to_pwm(solar_vol_control, current_r_b)

            if self.power_model is None:
                raise RuntimeError("功率模型未提供，请使用 led.py 中的 PWMtoPowerModel")

            total_pwm_for_power = r_pwm_control + b_pwm_control
            power_key = self._get_power_model_key(current_r_b)
            predicted_power = self.power_model.predict(total_pwm=total_pwm_for_power, key=power_key)

            # 🔥 计算控制量变化: u0 - u1
            control_change = float(solar_vol_control) - prev_control
            
            # 🔥 使用新版热力学模型预测 - 基于控制量变化
            predicted_temp = temp_model.step(
                power=predicted_power,
                dt=dt,
                solar_vol=float(solar_vol_control),
                control_change=control_change,  # 传递控制量变化
            )
            temp = predicted_temp
            
            # 🔥 更新控制状态
            prev_control = float(solar_vol_control)

            predicted_photosynthesis = self.get_photosynthesis_rate(
                solar_vol_control, predicted_temp, current_co2, current_r_b
            )

            solar_vol_inputs.append(solar_vol_control)
            temp_predictions.append(predicted_temp)
            power_predictions.append(predicted_power)
            photo_predictions.append(predicted_photosynthesis)
            r_pwm_predictions.append(r_pwm_control)
            b_pwm_predictions.append(b_pwm_control)

        return (
            np.array(solar_vol_inputs),
            np.array(temp_predictions),
            np.array(power_predictions),
            np.array(photo_predictions),
            np.array(r_pwm_predictions),
            np.array(b_pwm_predictions),
        )

    def get_thermal_model_info(self):
        """获取热力学模型信息"""
        return {
            "model_type": self.thermal_model_type,
            "model_dir": self.model_dir,
            "supports_solar_input": self.thermal_model.supports_solar_input,
            "current_temp": self.thermal_model.ambient_temp,
            "current_control": self.current_control,
            "previous_control": self.previous_control,
        }
    
    def reset_thermal_model(self, ambient_temp=None):
        """重置热力学模型状态"""
        self.thermal_model.reset(ambient_temp)
        if ambient_temp is not None:
            self.ambient_temp = ambient_temp
        # 重置MPPI控制状态
        self.current_control = 0.0
        self.previous_control = 0.0
    
    def set_thermal_model_type(self, model_type: str):
        """动态切换热力学模型类型"""
        if model_type not in ["mlp", "thermal"]:
            raise ValueError("model_type必须是'mlp'或'thermal'")
        
        self.thermal_model_type = model_type
        
        # 创建新的参数对象（因为LedThermalParams是frozen）
        self.thermal_params = LedThermalParams(
            base_ambient_temp=self.thermal_params.base_ambient_temp,
            thermal_resistance=self.thermal_params.thermal_resistance,
            time_constant_s=self.thermal_params.time_constant_s,
            thermal_mass=self.thermal_params.thermal_mass,
            max_ppfd=self.thermal_params.max_ppfd,
            max_power=self.thermal_params.max_power,
            led_efficiency=self.thermal_params.led_efficiency,
            efficiency_decay=self.thermal_params.efficiency_decay,
            model_type=model_type,
            model_dir=self.thermal_params.model_dir,
            solar_threshold=self.thermal_params.solar_threshold,
        )
        
        # 重新创建热力学模型
        self.thermal_model = ThermalModelManager(self.thermal_params)
        print(f"🔥 热力学模型已切换为: {model_type}")


# ------------------------------
# LEDMPPIController (Solar Vol 控制)
# ------------------------------
class LEDMPPIController:
    """基于Solar Vol输入的MPPI控制器"""

    def __init__(self, plant, horizon=10, num_samples=1000, dt=900.0, temperature=0.5):
        self.plant = plant
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.dt = float(dt)
        self.temperature = float(temperature)

        self.Q_photo = 10.0
        self.Q_ref = 0.0
        self.R_du = 0.05
        self.R_power = 0.05

        self.u_min = 1.0  # 实际数据最低约1.2V，设置1.0V作为安全边界
        self.u_max = float(getattr(self.plant, "max_solar_vol", 2.0))  # 支持到2.0V
        self.temp_min = 20.0
        self.temp_max = 32.0

        self.temp_penalty = 1e5
        self.u_penalty = 1e3

        self.u_std = 0.2
        self.u_prev = 0.0

    def set_weights(self, Q_photo=None, R_du=None, R_power=None, Q_ref=None):
        if Q_photo is not None:
            self.Q_photo = float(Q_photo)
        if R_du is not None:
            self.R_du = float(R_du)
        if R_power is not None:
            self.R_power = float(R_power)
        if Q_ref is not None:
            self.Q_ref = float(Q_ref)

    def set_constraints(self, u_min=None, u_max=None, temp_min=None, temp_max=None):
        if u_min is not None:
            self.u_min = float(u_min)
        if u_max is not None:
            self.u_max = float(u_max)
        if temp_min is not None:
            self.temp_min = float(temp_min)
        if temp_max is not None:
            self.temp_max = float(temp_max)

    def set_mppi_params(self, num_samples=None, temperature=None, u_std=None, horizon=None, dt=None):
        if num_samples is not None:
            self.num_samples = int(num_samples)
        if temperature is not None:
            self.temperature = float(temperature)
        if u_std is not None:
            self.u_std = float(u_std)
        if horizon is not None:
            self.horizon = int(horizon)
        if dt is not None:
            self.dt = float(dt)

    def _sample_control_sequences(self, mean_sequence: np.ndarray) -> np.ndarray:
        #生成噪声，用于MPPI函数的采样
        noise = np.random.normal(0.0, self.u_std, (self.num_samples, self.horizon))
        samples = mean_sequence[np.newaxis, :] + noise
        return np.clip(samples, self.u_min, self.u_max)
    #重要步骤整条控制序列（整段路径）的代价
    def _compute_cost(
        self,
        u_seq: np.ndarray,
        current_temp: float,
        solar_vol_ref_seq: np.ndarray | None = None,
    ) -> float:
        #计算单个PWM序列的代价 - 通过参考跟踪最大化光合作用
        try:
            (_u_in, temp_pred, power_pred, photo_pred, _r_pwm, _b_pwm) = self.plant.predict(
                u_seq, current_temp, dt=self.dt
            )

            cost = 0.0
            cost -= self.Q_photo * float(np.sum(photo_pred))

            # Solar Vol 参考跟踪代价
            if solar_vol_ref_seq is not None and len(solar_vol_ref_seq) > 0 and self.Q_ref != 0.0:
                ref = np.asarray(solar_vol_ref_seq, dtype=float)
                if ref.shape[0] < u_seq.shape[0]:
                    pad_len = u_seq.shape[0] - ref.shape[0]
                    ref = np.pad(ref, (0, pad_len), mode="edge")
                ref = ref[: u_seq.shape[0]]
                diff = u_seq - ref
                cost += self.Q_ref * float(np.sum(diff**2))

            over = np.maximum(0.0, temp_pred - self.temp_max)
            under = np.maximum(0.0, self.temp_min - temp_pred)
            cost += self.temp_penalty * float(np.sum(over**2 + under**2))

            cost += self.R_power * float(np.sum(power_pred**2))

            du = np.diff(np.concatenate([[self.u_prev], u_seq]))
            cost += self.R_du * float(np.sum(du**2))

            above = np.maximum(0.0, u_seq - self.u_max)
            below = np.maximum(0.0, self.u_min - u_seq)
            cost += self.u_penalty * float(np.sum(above**2 + below**2))

            return float(cost)
        except Exception:
            return 1e10
    #重要步骤整条控制序列（整段路径）的代价
    def solve(
        self,
        current_temp: float,
        mean_sequence: np.ndarray | None = None,
        solar_vol_ref_seq: np.ndarray | None = None,
    ):
        """🔥 MPPI求解 - 基于动态mean_sequence的滚动时域优化"""
        if mean_sequence is None:
            # 初始化参考控制序列
            base = 0.5 * self.u_max
            mean_sequence = np.full(self.horizon, base, dtype=float)

        # 🔥 生成控制序列样本（围绕动态mean_sequence）
        samples = self._sample_control_sequences(mean_sequence)
        
        # 🔥 计算所有样本的代价
        costs = np.array(
            [
                self._compute_cost(samples[i], current_temp, solar_vol_ref_seq)
                for i in range(self.num_samples)
            ],
            dtype=float,
        )
        costs = np.nan_to_num(costs, nan=1e10, posinf=1e10)

        # 🔥 Softmax权重计算
        min_cost = float(np.min(costs))
        weights = np.exp(-(costs - min_cost) / max(1e-6, self.temperature))
        weights /= float(np.sum(weights))

        # 🔥 计算最优控制序列
        optimal_seq = np.sum(weights[:, np.newaxis] * samples, axis=0)
        optimal_seq = np.clip(optimal_seq, self.u_min, self.u_max)
        optimal_u = float(optimal_seq[0])

        # 🔥 温度安全检查
        try:
            _sv, t_check, _pw, _pn, _r, _b = self.plant.predict([optimal_u], current_temp, dt=self.dt)
            if t_check[0] > self.temp_max:
                optimal_u = max(self.u_min, optimal_u * 0.7)
        except Exception:
            pass

        # 🔥 更新控制状态（用于下次MPPI迭代）
        self.u_prev = optimal_u
        
        return optimal_u, optimal_seq, True, min_cost, weights


__all__ = [
    "LEDPlant", 
    "LEDMPPIController", 
    "LedThermalParams", 
    "ThermalModelManager",  # 新增
    "FirstOrderThermalModel",  # 兼容性别名
    "PWMtoPowerModel", 
    "PWMtoPPFDModel"
]
