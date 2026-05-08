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
        FirstOrderThermalModel,
        PWMtoPPFDModel,
        PWMtoPowerModel,
    )
except ImportError:
    from led import (
        LedThermalParams,
        ThermalModelManager,
        FirstOrderThermalModel,
        PWMtoPPFDModel,
        PWMtoPowerModel,
    )

# 传感器读取 - 简化版本
DEFAULT_DEVICE_ID = "T6ncwg=="
RIOTEE_DATA_PATH = "../Tool/Sensor_riotee_server/logs/riotee_data_all.csv"
CO2_DATA_PATH = "/data/csv/co2_sensor.csv"
DEFAULT_CO2_PPM = 400.0

# Tool/Model 路径
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_TO_PN_PACKAGE = os.path.join(_PROJECT_ROOT, "Tool", "Model", "EnvtoPN", "model", "best_model_package.joblib")
# MPPI-Govee uses the new xyz-aware models trained on the 9-position grid.
# SPtoPPFD: 8 spectral channels + (x_cm, y_cm, z_cm) -> PPFD
# PWMtoPPFD: (pwm_r, pwm_b, x_cm, y_cm, z_cm) -> PPFD
_MPPI_GOVEE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP_TO_PPFD_PACKAGE = os.path.join(_MPPI_GOVEE_ROOT, "models", "sptoppfd", "best_model_package.joblib")
PWM_TO_PPFD_PACKAGE = os.path.join(_MPPI_GOVEE_ROOT, "models", "pwmtoppfd", "best_model_package.joblib")

# Growpod 默认参考 PPFD
DEFAULT_TARGET_PPFD = 400.0


class SensorReading:
    """简化的传感器读取类。read_latest_riotee_data 同时返回 spectrum 字典（如有）。"""
    def __init__(self, device_id=None, riotee_data_path=None, co2_data_path=None):
        self.device_id = device_id or DEFAULT_DEVICE_ID
        self.riotee_data_path = riotee_data_path or RIOTEE_DATA_PATH
        self.co2_data_path = co2_data_path or CO2_DATA_PATH

    def read_latest_riotee_data(self):
        """读取最新 Riotee 数据。返回 (temperature, solar_vol, pn, timestamp, spectrum_dict)。"""
        try:
            if not os.path.exists(self.riotee_data_path):
                print(f"警告: 数据文件不存在: {self.riotee_data_path}")
                return None, None, None, None, None

            df = pd.read_csv(self.riotee_data_path, comment="#")
            device_data = df[df["device_id"] == self.device_id].copy()
            if device_data.empty:
                print(f"警告: 未找到设备ID {self.device_id} 的数据")
                return None, None, None, None, None

            device_data["timestamp"] = pd.to_datetime(device_data["timestamp"])
            latest_time = device_data["timestamp"].max()
            window_start = latest_time - pd.Timedelta(minutes=10)
            recent = device_data[device_data["timestamp"] >= window_start].copy()
            if recent.empty:
                print(f"警告: 过去10分钟内没有数据")
                return None, None, None, None, None

            win = min(5, len(recent))
            recent["a1_raw_filtered"] = recent["a1_raw"].rolling(window=win, center=True).mean()

            temperature = float(device_data.iloc[-1]["temperature"])
            solar_vol = float(recent["a1_raw_filtered"].mean()) if "a1_raw" in recent.columns else None
            pn_avg = None

            spectrum = None
            sp_cols = ["sp_415", "sp_445", "sp_480", "sp_515", "sp_555", "sp_590", "sp_630", "sp_680"]
            if all(c in device_data.columns for c in sp_cols):
                latest_row = device_data.iloc[-1]
                spectrum = {c: float(latest_row[c]) for c in sp_cols}

            return temperature, solar_vol, pn_avg, latest_time, spectrum
        except Exception as e:
            print(f"错误: 读取Riotee数据失败: {e}")
            return None, None, None, None, None

    def read_latest_co2_data(self):
        try:
            if not self.co2_data_path or not os.path.exists(self.co2_data_path):
                print(f"警告: CO2数据文件不存在: {self.co2_data_path}，使用默认值 {DEFAULT_CO2_PPM} ppm")
                return DEFAULT_CO2_PPM
            df = pd.read_csv(self.co2_data_path, header=None, names=["timestamp", "co2"])
            if df.empty:
                return DEFAULT_CO2_PPM
            latest = df.iloc[-1]
            co2_value = latest.get("co2")
            if co2_value is None or pd.isna(co2_value):
                return DEFAULT_CO2_PPM
            return float(co2_value)
        except Exception as e:
            print(f"错误: 读取CO2数据失败: {e}，使用默认值 {DEFAULT_CO2_PPM} ppm")
            return DEFAULT_CO2_PPM


# ------------------------------
# LEDPlant (PPFD 控制 - Growpod 版)
# 链路: spectrum -> PPFD (观测), PPFD -> (R_PWM,B_PWM) (执行), PPFD -> Pn (评估)
# ------------------------------
class LEDPlant:
    """Growpod MPPI 被控对象：控制量 = 目标 PPFD。"""

    def __init__(
        self,
        base_ambient_temp=25.0,
        max_ppfd=500.0,
        max_power=130.0,
        thermal_resistance=0.05,
        time_constant_s=7.5,
        thermal_mass=150.0,
        power_model=None,
        co2_ppm=400.0,
        r_b_ratio=0.83,
        target_ppfd=DEFAULT_TARGET_PPFD,
        use_pn_model=True,
        use_sp_to_ppfd=True,
        thermal_model_type: str = "thermal",
        model_dir: str = "Thermal/exported_models",
        adaptive_thermal_enabled: bool = False,
        adaptive_thermal_params_path: str = "",
        x_cm: float = 0.0,
        y_cm: float = 0.0,
        z_cm: float = 10.0,
        use_govee_pwm_model: bool = True,
        demo_gain: float = 1.0,
        auto_target_ppfd: bool = False,
        auto_target_fraction: float = 0.7,
    ):
        self.base_ambient_temp = base_ambient_temp
        self.max_ppfd = max_ppfd
        self.max_power = max_power
        self.co2_ppm = co2_ppm
        self.r_b_ratio = r_b_ratio
        self.target_ppfd = float(target_ppfd)
        self.use_pn_model = use_pn_model
        self.use_sp_to_ppfd = use_sp_to_ppfd
        # Govee 位置（板心原点 cm；feeds the xyz-aware models）
        from types import SimpleNamespace
        self.current_position = SimpleNamespace(x_cm=float(x_cm),
                                                y_cm=float(y_cm),
                                                z_cm=float(z_cm))
        self.use_govee_pwm_model = use_govee_pwm_model
        # demo_gain: 在所有模型 PPFD 输出 / target 之间应用的虚拟放大倍数。
        # Govee H6056 在板上方 10 cm 时实际 PPFD 上限只有 ~35-50 µmol/m²/s
        # （真植物灯通常 200-500），不放大的话 MPPI 永远饱和到 100/100。
        # demo_gain=10 让 MPPI 控制器按"~250 PPFD"的目标空间运转，但底
        # 层 Govee 仍按真实的 ~25 PPFD 输出。仿真/演示用途；不要把它当
        # 真实物理。
        self.demo_gain = max(1e-6, float(demo_gain))
        self.auto_target_ppfd = bool(auto_target_ppfd)
        self.auto_target_fraction = float(auto_target_fraction)
        self.thermal_model_type = thermal_model_type
        self.model_dir = model_dir
        self.adaptive_thermal_enabled = adaptive_thermal_enabled
        self.adaptive_thermal_params_path = adaptive_thermal_params_path

        # 控制状态：在 PPFD 量纲
        self.current_control = 0.0
        self.previous_control = 0.0

        # 热力学参数与模型（仍按电压域 solar_vol 驱动相位判断）
        self.thermal_params = LedThermalParams(
            base_ambient_temp=base_ambient_temp,
            thermal_resistance=thermal_resistance,
            time_constant_s=time_constant_s,
            thermal_mass=thermal_mass,
            max_ppfd=max_ppfd,
            max_power=max_power,
            model_type=thermal_model_type,
            model_dir=model_dir,
            solar_threshold=1.4,
            adaptive_enabled=adaptive_thermal_enabled,
            adaptive_params_path=adaptive_thermal_params_path,
        )
        self.thermal_model = ThermalModelManager(self.thermal_params)

        self.power_model = power_model

        # 仿真状态
        self.ambient_temp = base_ambient_temp
        self.time = 0.0
        self.current_ppfd = 0.0

        # 传感器
        self.sensor_reader = SensorReading(
            device_id=DEFAULT_DEVICE_ID,
            riotee_data_path=RIOTEE_DATA_PATH,
            co2_data_path=CO2_DATA_PATH,
        )

        # PPFD→Pn 模型 (HistGBR pipeline)
        self.pn_pipeline = None
        self.pn_feature_columns = None
        self._init_pn_model()

        # SP→PPFD 模型 (PLSR pipeline)
        self.sp_pipeline = None
        self.sp_feature_columns = None
        if self.use_sp_to_ppfd:
            self._init_sp_to_ppfd_model()

        # Govee xyz-aware PWM→PPFD（GPR）— 取代 calib_data.csv 线性模型
        self.govee_pwm_model = None
        if self.use_govee_pwm_model:
            try:
                from govee_pwmtoppfd_model import GoveePWMtoPPFDModel, GoveePosition
                self.govee_pwm_model = GoveePWMtoPPFDModel(
                    model_path=PWM_TO_PPFD_PACKAGE,
                    position=GoveePosition(x_cm=self.current_position.x_cm,
                                           y_cm=self.current_position.y_cm,
                                           z_cm=self.current_position.z_cm),
                ).load()
                print(f"✅ 已加载 Govee PWMtoPPFD 模型: {PWM_TO_PPFD_PACKAGE}")
            except Exception as e:
                print(f"⚠️ Govee PWMtoPPFD 加载失败: {e}, 退回 calib_data.csv")
                self.govee_pwm_model = None

        # 测一下当前位置在 R:B 比例下的物理 max PPFD（用满 PWM 100/100），
        # 作为 demo banner 信息 + 可选 auto_target 基准。
        self._physical_max_ppfd = self._estimate_physical_max_ppfd()
        if self.auto_target_ppfd and self._physical_max_ppfd is not None:
            old_target = self.target_ppfd
            self.target_ppfd = float(self.auto_target_fraction
                                     * self._physical_max_ppfd
                                     * self.demo_gain)
            print(f"🎯 auto_target: target_ppfd {old_target} -> {self.target_ppfd:.1f}"
                  f" (physical_max={self._physical_max_ppfd:.2f} × demo_gain={self.demo_gain} × {self.auto_target_fraction})")

        eff_target_real = self.target_ppfd / self.demo_gain
        print(f"🌿 Govee LEDPlant 初始化完成 (控制量=PPFD, target={self.target_ppfd:.1f}, "
              f"pos=({self.current_position.x_cm},{self.current_position.y_cm},{self.current_position.z_cm}) cm)")
        if self._physical_max_ppfd is not None:
            print(f"   physical max PPFD @ pos = {self._physical_max_ppfd:.2f} µmol/m²/s "
                  f"(R:B={self.r_b_ratio})")
        if self.demo_gain != 1.0:
            print(f"   demo_gain={self.demo_gain}× → effective real PPFD target = {eff_target_real:.2f} µmol/m²/s")
            if eff_target_real > (self._physical_max_ppfd or 0):
                print(f"   ⚠️  effective target exceeds physical max ({self._physical_max_ppfd:.2f}); "
                      f"MPPI 会饱和到满 PWM。考虑 --auto-target 或更小 target。")

    def _estimate_physical_max_ppfd(self):
        if self.govee_pwm_model is None:
            return None
        try:
            rb = float(self.r_b_ratio)
            r_pwm = 100.0 * rb
            b_pwm = 100.0 * (1.0 - rb)
            return max(0.0, float(self.govee_pwm_model.predict(r_pwm=r_pwm, b_pwm=b_pwm)))
        except Exception:
            return None
        print(f"   热力学模型类型: {thermal_model_type}, 模型目录: {model_dir}")

    # ---------------- Pn 模型 (PPFD 输入) ----------------
    def _init_pn_model(self):
        if not self.use_pn_model:
            self.pn_pipeline = None
            return
        if not os.path.exists(ENV_TO_PN_PACKAGE):
            raise FileNotFoundError(f"EnvtoPN 模型不存在: {ENV_TO_PN_PACKAGE}")
        package = joblib.load(ENV_TO_PN_PACKAGE)
        self.pn_pipeline = package["pipeline"]
        self.pn_feature_columns = list(package["feature_columns"])  # [T, CO2, R:B, PPFD]
        # The Pn model was pickled with sklearn 1.3.x; running sklearn ≥ 1.4
        # added new internal attributes on HistGradientBoostingRegressor
        # (_preprocessor / _categorical_features / _known_categories) that
        # the old pickle doesn't carry. Backfill them with their "no
        # categorical features" defaults so .predict() works without any
        # functional fallback.
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            for _, est in self.pn_pipeline.steps:
                if isinstance(est, HistGradientBoostingRegressor):
                    for attr, default in (
                        ("_preprocessor", None),
                        ("_categorical_features", None),
                        ("_known_categories", None),
                        ("is_categorical_", None),
                    ):
                        if not hasattr(est, attr):
                            setattr(est, attr, default)
        except Exception:
            pass
        print(f"✅ 已加载 EnvtoPN 模型: {package.get('model_name')}, features={self.pn_feature_columns}")

    def get_photosynthesis_rate(self, ppfd, temperature, co2_ppm=None, r_b_ratio=None):
        if self.pn_pipeline is None:
            raise RuntimeError("Pn 模型不可用。")
        co2 = float(co2_ppm) if co2_ppm is not None else float(self.co2_ppm)
        rb = float(r_b_ratio) if r_b_ratio is not None else float(self.r_b_ratio)
        row = {"T": float(temperature), "CO2": co2, "R:B": rb, "PPFD": float(ppfd)}
        df = pd.DataFrame([row])[self.pn_feature_columns]
        y = float(self.pn_pipeline.predict(df)[0])
        return max(0.0, y)

    # ---------------- SP → PPFD 模型 (观测) ----------------
    def _init_sp_to_ppfd_model(self):
        if not os.path.exists(SP_TO_PPFD_PACKAGE):
            print(f"⚠️ SPtoPPFD 模型不存在: {SP_TO_PPFD_PACKAGE}, 关闭 spectrum 观测")
            self.sp_pipeline = None
            return
        try:
            package = joblib.load(SP_TO_PPFD_PACKAGE)
            self.sp_pipeline = package["pipeline"]
            self.sp_feature_columns = list(package["feature_columns"])
            print(f"✅ 已加载 SPtoPPFD 模型: {package.get('model_name')}, features={self.sp_feature_columns}")
        except Exception as e:
            print(f"⚠️ SPtoPPFD 加载失败: {e}, 关闭 spectrum 观测")
            self.sp_pipeline = None

    def spectrum_to_ppfd(self, spectrum_dict):
        """把 8 通道光谱字典转成 PPFD 观测。

        MPPI-Govee 使用 xyz-aware 的模型，feature_columns 含
        x_cm/y_cm/z_cm。如果调用方没在 spectrum_dict 里给位置，就
        自动从 self.current_position 注入。
        """
        if self.sp_pipeline is None:
            return None
        try:
            df = pd.DataFrame([spectrum_dict])
            if not all(c in df.columns for c in self.sp_feature_columns):
                rename = {}
                for col in self.sp_feature_columns:
                    short = col.replace("_mean", "")
                    if short in df.columns:
                        rename[short] = col
                if rename:
                    df = df.rename(columns=rename)
            for col in self.sp_feature_columns:
                if col not in df.columns and col in ("x_cm", "y_cm", "z_cm"):
                    df[col] = float(getattr(self.current_position, col))
            X = df[self.sp_feature_columns]
            real = float(self.sp_pipeline.predict(X)[0])
            return real * self.demo_gain
        except Exception as e:
            print(f"⚠️ spectrum_to_ppfd 失败: {e}")
            return None

    def set_position(self, x_cm: float, y_cm: float, z_cm: float):
        """更新当前传感器位置，会同步推到下游 Govee PWM 模型。"""
        self.current_position.x_cm = float(x_cm)
        self.current_position.y_cm = float(y_cm)
        self.current_position.z_cm = float(z_cm)
        if self.govee_pwm_model is not None:
            self.govee_pwm_model.set_position(x_cm, y_cm, z_cm)

    # ---------------- PPFD → PWM (执行) ----------------
    def _ppfd_to_pwm(self, ppfd, r_b_ratio=None):
        rb = float(r_b_ratio) if r_b_ratio is not None else float(self.r_b_ratio)
        if self.govee_pwm_model is not None:
            return self._govee_ppfd_to_pwm(float(ppfd), rb)
        return self._lookup_ppfd_to_pwm(float(ppfd), rb)

    def _govee_ppfd_to_pwm(self, ppfd: float, rb_fraction: float):
        """逆向求 (r_pwm, b_pwm)：
        固定红光占比 rb_fraction（red/(red+blue)），扫描 total_pwm 0..100，
        用 Govee GPR 取每个候选的预测 PPFD，再线性插值找最匹配 target 的总功率。

        `ppfd` 入参在 demo_gain 缩放空间（即 MPPI/控制器看到的目标）。
        转回真实物理 PPFD = ppfd / demo_gain，再用 GPR 找对应 PWM。
        """
        rb_fraction = max(0.0, min(1.0, float(rb_fraction)))
        totals = np.linspace(0.0, 100.0, 51)  # 2-step 粒度足够 inversion
        r_arr = totals * rb_fraction
        b_arr = totals * (1.0 - rb_fraction)
        ppfds = self.govee_pwm_model.predict_batch(r_arr, b_arr)
        ppfds = np.maximum(ppfds, 0.0)  # GPR 在弱光段会有微小负值
        target_real = max(0.0, float(ppfd) / self.demo_gain)
        if target_real <= float(ppfds[0]):
            return float(r_arr[0]), float(b_arr[0])
        if target_real >= float(ppfds[-1]):
            return float(r_arr[-1]), float(b_arr[-1])
        # 单调插值（PPFD 应随 total_pwm 单调上升）
        order = np.argsort(ppfds)
        ppfds_sorted = ppfds[order]
        totals_sorted = totals[order]
        total_hat = float(np.interp(target_real, ppfds_sorted, totals_sorted))
        return total_hat * rb_fraction, total_hat * (1.0 - rb_fraction)

    def _lookup_ppfd_to_pwm(self, ppfd, r_b_ratio):
        if not hasattr(self, "_calib_data") or self._calib_data is None:
            calib_path = os.path.join(os.path.dirname(__file__), "..", "data", "calib_data.csv")
            self._calib_data = pd.read_csv(calib_path)
            self._rb_mapping = {"1:1": 0.5, "3:1": 0.75, "5:1": 0.83, "7:1": 0.88, "r1": 1.0}

        calib_rb = None
        for k, v in self._rb_mapping.items():
            if abs(v - r_b_ratio) < 0.01:
                calib_rb = k
                break
        if calib_rb is None:
            raise ValueError(f"无法找到 R:B={r_b_ratio} 对应的标定键，支持: {self._rb_mapping}")

        calib_filtered = self._calib_data[self._calib_data["R:B"] == calib_rb]
        if calib_filtered.empty:
            raise ValueError(f"calib_data.csv 中找不到 R:B={calib_rb} 的数据")
        if len(calib_filtered) < 2:
            raise ValueError(f"calib_data.csv 中 R:B={calib_rb} 标定点不足")

        calib_sorted = calib_filtered.sort_values("PPFD")
        ppfd_arr = calib_sorted["PPFD"].to_numpy(dtype=float)
        r_pwm = float(np.interp(float(ppfd), ppfd_arr, calib_sorted["R_PWM"].to_numpy(dtype=float)))
        b_pwm = float(np.interp(float(ppfd), ppfd_arr, calib_sorted["B_PWM"].to_numpy(dtype=float)))
        return min(100.0, max(0.0, r_pwm)), min(100.0, max(0.0, b_pwm))

    # ---------------- PPFD → Solar_Vol (仅给热模型用) ----------------
    def _ppfd_to_solar_vol(self, ppfd, r_b_ratio=None):
        if not hasattr(self, "_solar_vol_data") or self._solar_vol_data is None:
            sv_path = os.path.join(os.path.dirname(__file__), "..", "data", "Solar_Vol_clean.csv")
            self._solar_vol_data = pd.read_csv(sv_path)
        rb = float(r_b_ratio) if r_b_ratio is not None else float(self.r_b_ratio)
        sub = self._solar_vol_data[self._solar_vol_data["R:B"] == rb]
        if sub.empty or sub["PPFD"].nunique() < 2:
            # fallback: 整表（最近邻 R:B）
            sub = self._solar_vol_data
        sub = sub.sort_values("PPFD")
        ppfd_arr = sub["PPFD"].to_numpy(dtype=float)
        sv_arr = sub["Solar_Vol"].to_numpy(dtype=float)
        return float(np.interp(float(ppfd), ppfd_arr, sv_arr))

    def _get_power_model_key(self, r_b_ratio):
        rb_to_key = {0.5: "1:1", 0.75: "3:1", 0.83: "5:1", 0.88: "7:1", 1.0: "r1"}
        closest = min(rb_to_key.keys(), key=lambda x: abs(x - r_b_ratio))
        if abs(closest - r_b_ratio) < 0.05:
            return rb_to_key[closest]
        return None

    # ---------------- 仿真步进 ----------------
    def step(self, ppfd, dt=900, device_id=DEFAULT_DEVICE_ID):
        """单步仿真：控制量 = ppfd（μmol/m²/s）。"""
        if device_id != DEFAULT_DEVICE_ID:
            self.sensor_reader.device_id = device_id

        # 读取实时温度（可选）
        try:
            current_temp, _csv_sv, _pn, timestamp, spectrum = self.sensor_reader.read_latest_riotee_data()
            if current_temp is not None:
                self.ambient_temp = current_temp
                self.thermal_model.ambient_temp = current_temp
                print(f"🌡️ 使用最新温度: {current_temp:.2f}°C (时间戳: {timestamp})")
            # 可选：用 spectrum 反演 PPFD 观测
            if spectrum is not None and self.sp_pipeline is not None:
                ppfd_obs = self.spectrum_to_ppfd(spectrum)
                if ppfd_obs is not None:
                    print(f"📡 SP→PPFD 观测: {ppfd_obs:.1f} μmol/m²/s")
        except Exception:
            pass

        ppfd = float(ppfd)

        # PPFD → PWM
        r_pwm, b_pwm = self._ppfd_to_pwm(ppfd, self.r_b_ratio)

        # 功率
        if self.power_model is None:
            raise RuntimeError("功率模型未提供")
        power_key = self._get_power_model_key(self.r_b_ratio)
        power = self.power_model.predict(total_pwm=r_pwm + b_pwm, key=power_key)

        # 控制量变化（PPFD 量纲）
        control_change = ppfd - self.current_control

        # 热模型：仍以电压域 solar_vol 驱动
        solar_vol_for_thermal = self._ppfd_to_solar_vol(ppfd, self.r_b_ratio)
        new_ambient_temp = self.thermal_model.step(
            power=power,
            dt=dt,
            solar_vol=solar_vol_for_thermal,
            control_change=control_change,
        )

        # 状态更新
        self.current_ppfd = ppfd
        self.ambient_temp = new_ambient_temp
        self.time += dt
        self.previous_control = self.current_control
        self.current_control = ppfd

        # CO2
        current_co2 = self.sensor_reader.read_latest_co2_data()

        # Pn
        pn = self.get_photosynthesis_rate(ppfd, new_ambient_temp, current_co2, self.r_b_ratio)

        phase = "升温" if control_change > 0 else "降温"
        print(f"🌱 Growpod step: PPFD={ppfd:.1f}, ΔPPFD={control_change:+.1f} ({phase}) → T={new_ambient_temp:.2f}°C, Pn={pn:.2f}")

        return self.current_ppfd, new_ambient_temp, power, pn

    # ---------------- 滚动多步预测 ----------------
    def predict(
        self,
        ppfd_control_sequence,
        initial_temp,
        dt=900.0,
        co2_sequence=None,
        r_b_sequence=None,
        prev_control=None,
    ):
        """对一条 PPFD 控制序列做 horizon 步预测。"""
        temp_model = ThermalModelManager(self.thermal_params)
        temp_model.reset(initial_temp)

        ppfd_inputs = []
        temp_predictions = []
        power_predictions = []
        photo_predictions = []
        r_pwm_predictions = []
        b_pwm_predictions = []

        if prev_control is None:
            prev_control = float(self.current_control)
        else:
            prev_control = float(prev_control)

        for i, ppfd_ctrl in enumerate(ppfd_control_sequence):
            ppfd_ctrl = float(ppfd_ctrl)
            current_co2 = co2_sequence[i] if co2_sequence is not None else self.co2_ppm
            current_rb = r_b_sequence[i] if r_b_sequence is not None else self.r_b_ratio

            r_pwm, b_pwm = self._ppfd_to_pwm(ppfd_ctrl, current_rb)

            if self.power_model is None:
                raise RuntimeError("功率模型未提供")
            total_pwm = r_pwm + b_pwm
            power_key = self._get_power_model_key(current_rb)
            predicted_power = self.power_model.predict(total_pwm=total_pwm, key=power_key)

            control_change = ppfd_ctrl - prev_control
            sv_for_thermal = self._ppfd_to_solar_vol(ppfd_ctrl, current_rb)

            predicted_temp = temp_model.step(
                power=predicted_power,
                dt=dt,
                solar_vol=sv_for_thermal,
                control_change=control_change,
            )

            prev_control = ppfd_ctrl

            predicted_pn = self.get_photosynthesis_rate(ppfd_ctrl, predicted_temp, current_co2, current_rb)

            ppfd_inputs.append(ppfd_ctrl)
            temp_predictions.append(predicted_temp)
            power_predictions.append(predicted_power)
            photo_predictions.append(predicted_pn)
            r_pwm_predictions.append(r_pwm)
            b_pwm_predictions.append(b_pwm)

        return (
            np.array(ppfd_inputs),
            np.array(temp_predictions),
            np.array(power_predictions),
            np.array(photo_predictions),
            np.array(r_pwm_predictions),
            np.array(b_pwm_predictions),
        )

    def get_thermal_model_info(self):
        runtime_info = {}
        if hasattr(self.thermal_model, "get_model_info"):
            runtime_info = self.thermal_model.get_model_info()
        return {
            "model_type": self.thermal_model_type,
            "model_dir": self.model_dir,
            "supports_solar_input": self.thermal_model.supports_solar_input,
            "current_temp": self.thermal_model.ambient_temp,
            "current_control": self.current_control,
            "previous_control": self.previous_control,
            "target_ppfd": self.target_ppfd,
            **runtime_info,
        }

    def refresh_thermal_model_runtime_parameters(self) -> bool:
        if hasattr(self.thermal_model, "refresh_runtime_parameters"):
            return bool(self.thermal_model.refresh_runtime_parameters(force=True))
        return False

    def reset_thermal_model(self, ambient_temp=None):
        self.thermal_model.reset(ambient_temp)
        if ambient_temp is not None:
            self.ambient_temp = ambient_temp
        self.current_control = 0.0
        self.previous_control = 0.0
        self.current_ppfd = 0.0

    def set_thermal_model_type(self, model_type: str):
        if model_type not in ["mlp", "thermal"]:
            raise ValueError("model_type必须是'mlp'或'thermal'")
        self.thermal_model_type = model_type
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
            solar_change_tolerance=self.thermal_params.solar_change_tolerance,
            adaptive_enabled=self.thermal_params.adaptive_enabled,
            adaptive_params_path=self.thermal_params.adaptive_params_path,
        )
        self.thermal_model = ThermalModelManager(self.thermal_params)
        print(f"🔥 热力学模型已切换为: {model_type}")


# ------------------------------
# LEDMPPIController (PPFD 控制 - Growpod 版)
# ------------------------------
class LEDMPPIController:
    """基于 PPFD 控制量的 MPPI 控制器，跟踪目标 PPFD（默认 400）。"""

    def __init__(self, plant, horizon=10, num_samples=1000, dt=900.0, temperature=0.5):
        self.plant = plant
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.dt = float(dt)
        self.temperature = float(temperature)

        # 经过批内 z-score 归一化后,各项权重都是 O(1)
        self.Q_photo = 1.0
        self.Q_ref = 0.0
        self.R_du = 0.3
        self.R_power = 0.3                      # 节能很弱,几乎纯追光合 (默认)
        self.target_mean_power = None
        self.power_budget_weight = 0.0

        # PPFD 量纲约束
        self.u_min = 80.0
        self.u_max = float(getattr(self.plant, "max_ppfd", 500.0))
        self.target_ppfd = float(getattr(self.plant, "target_ppfd", DEFAULT_TARGET_PPFD))

        # 训练分布软约束 (Pn 模型可靠区间; calib_data 与 EnvtoPN 训练范围都在 100~500)
        self.ppfd_train_min = 100.0
        self.ppfd_train_max = 500.0
        self.R_oob = 1e3                        # 越界软罚 (绝对量级)

        self.temp_min = 20.0
        self.temp_max = 32.0

        self.temp_penalty = 1e5  # 硬约束保留绝对量级,不做 z-score

        self.u_std = 20.0             # PPFD 量纲噪声
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

    def set_power_budget(self, target_mean_power=None, power_budget_weight=None):
        """启用功率预算后,自动关闭 R_power (二者语义重复)。"""
        if target_mean_power is None:
            self.target_mean_power = None
        else:
            self.target_mean_power = float(target_mean_power)
        if power_budget_weight is not None:
            self.power_budget_weight = float(power_budget_weight)
        # 互斥:只要启用了功率预算,就把"压低绝对功率"的项关掉
        if self.target_mean_power is not None and self.power_budget_weight > 0.0:
            if self.R_power != 0.0:
                print(f"⚙️ 启用 power_budget(target={self.target_mean_power:.1f}W) → 自动将 R_power 从 {self.R_power} 置 0")
            self.R_power = 0.0

    def set_ppfd_train_range(self, ppfd_min=None, ppfd_max=None, R_oob=None):
        """设置 PPFD 训练分布范围与越界软罚权重。"""
        if ppfd_min is not None:
            self.ppfd_train_min = float(ppfd_min)
        if ppfd_max is not None:
            self.ppfd_train_max = float(ppfd_max)
        if R_oob is not None:
            self.R_oob = float(R_oob)

    def set_constraints(self, u_min=None, u_max=None, temp_min=None, temp_max=None):
        if u_min is not None:
            self.u_min = float(u_min)
        if u_max is not None:
            self.u_max = float(u_max)
        if temp_min is not None:
            self.temp_min = float(temp_min)
        if temp_max is not None:
            self.temp_max = float(temp_max)

    def set_target_ppfd(self, target_ppfd):
        self.target_ppfd = float(target_ppfd)

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

    def set_penalties(self, temp_penalty=None, u_penalty=None):
        if temp_penalty is not None:
            self.temp_penalty = float(temp_penalty)
        # u_penalty 已废弃 (samples 在采样时就被 clip,不可能越界);保留参数以维持兼容
        _ = u_penalty

    def _sample_control_sequences(self, mean_sequence: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0.0, self.u_std, (self.num_samples, self.horizon))
        samples = mean_sequence[np.newaxis, :] + noise
        return np.clip(samples, self.u_min, self.u_max)

    def _compute_components(
        self,
        u_seq: np.ndarray,
        current_temp: float,
        ppfd_ref_seq: np.ndarray | None = None,
    ) -> dict:
        """对单条候选轨迹返回各 cost 分项的原始值;solve() 里再做批内归一化与合成。"""
        try:
            (_u_in, temp_pred, power_pred, photo_pred, _r_pwm, _b_pwm) = self.plant.predict(
                u_seq, current_temp, dt=self.dt, prev_control=self.u_prev
            )

            # 参考序列(若未指定则用标量 target_ppfd 广播)
            if ppfd_ref_seq is None:
                ref = np.full(u_seq.shape[0], float(self.target_ppfd)) if self.target_ppfd is not None else None
            else:
                ref = np.asarray(ppfd_ref_seq, dtype=float)
                if ref.shape[0] < u_seq.shape[0]:
                    ref = np.pad(ref, (0, u_seq.shape[0] - ref.shape[0]), mode="edge")
                ref = ref[: u_seq.shape[0]]

            ref_sq_sum = float(np.sum((u_seq - ref) ** 2)) if ref is not None else 0.0

            over = np.maximum(0.0, temp_pred - self.temp_max)
            under = np.maximum(0.0, self.temp_min - temp_pred)
            temp_violation = float(np.sum(over ** 2 + under ** 2))

            power_budget_dev = 0.0
            if self.target_mean_power is not None and self.power_budget_weight > 0.0:
                power_budget_dev = float((float(np.mean(power_pred)) - float(self.target_mean_power)) ** 2)

            du = np.diff(np.concatenate([[self.u_prev], u_seq]))

            # PPFD 越出训练分布的软罚 (Pn 模型外推不可靠区域)
            oob_low = np.maximum(0.0, self.ppfd_train_min - u_seq)
            oob_high = np.maximum(0.0, u_seq - self.ppfd_train_max)
            oob_violation = float(np.sum(oob_low ** 2 + oob_high ** 2))

            return {
                "valid": True,
                "photo_sum": float(np.sum(photo_pred)),       # 越大越好(后面取负)
                "power_sq_sum": float(np.sum(power_pred ** 2)),
                "du_sq_sum": float(np.sum(du ** 2)),
                "ref_sq_sum": ref_sq_sum,
                "temp_violation": temp_violation,             # 硬约束,不归一化
                "power_budget_dev": power_budget_dev,         # 同上
                "oob_violation": oob_violation,               # 训练分布软约束,绝对量级
            }
        except Exception:
            return {
                "valid": False,
                "photo_sum": 0.0,
                "power_sq_sum": 0.0,
                "du_sq_sum": 0.0,
                "ref_sq_sum": 0.0,
                "temp_violation": 0.0,
                "power_budget_dev": 0.0,
                "oob_violation": 0.0,
            }

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        """批内 z-score; 退化情形(std≈0)返回全零。"""
        std = float(np.std(x))
        if std < 1e-9:
            return np.zeros_like(x)
        return (x - float(np.mean(x))) / std

    def solve(
        self,
        current_temp: float,
        mean_sequence: np.ndarray | None = None,
        ppfd_ref_seq: np.ndarray | None = None,
    ):
        """MPPI 求解 - PPFD 控制量。批内 z-score + 自适应 softmax。"""
        if mean_sequence is None:
            base = float(self.target_ppfd) if self.target_ppfd is not None else 0.5 * self.u_max
            mean_sequence = np.full(self.horizon, base, dtype=float)

        # 默认参考 = target_ppfd 常量序列
        if ppfd_ref_seq is None and self.target_ppfd is not None and self.Q_ref != 0.0:
            ppfd_ref_seq = np.full(self.horizon, float(self.target_ppfd), dtype=float)

        samples = self._sample_control_sequences(mean_sequence)

        # 1. 收集每条样本的原始分项
        comps = [
            self._compute_components(samples[i], current_temp, ppfd_ref_seq)
            for i in range(self.num_samples)
        ]
        valid = np.array([c["valid"] for c in comps])
        photo_arr   = np.array([c["photo_sum"]        for c in comps], dtype=float)
        power_arr   = np.array([c["power_sq_sum"]     for c in comps], dtype=float)
        du_arr      = np.array([c["du_sq_sum"]        for c in comps], dtype=float)
        ref_arr     = np.array([c["ref_sq_sum"]       for c in comps], dtype=float)
        temp_arr    = np.array([c["temp_violation"]   for c in comps], dtype=float)
        budget_arr  = np.array([c["power_budget_dev"] for c in comps], dtype=float)
        oob_arr     = np.array([c["oob_violation"]    for c in comps], dtype=float)

        # 2. 软目标做批内 z-score:Pn / Power / Δu / Ref-tracking
        photo_z = self._zscore(photo_arr)
        power_z = self._zscore(power_arr)
        du_z    = self._zscore(du_arr)
        ref_z   = self._zscore(ref_arr) if self.Q_ref != 0.0 else np.zeros_like(ref_arr)

        # 3. 合成 cost:photo 越大越好(取负);其他越小越好
        cost_soft = (
            -self.Q_photo  * photo_z
            +  self.R_power * power_z
            +  self.R_du    * du_z
            +  self.Q_ref   * ref_z
        )
        # 硬约束/软约束:绝对量级,不参与归一化
        cost_hard = (
            self.temp_penalty       * temp_arr
            + self.power_budget_weight * budget_arr
            + self.R_oob              * oob_arr
        )
        costs = cost_soft + cost_hard

        # 4. 失败样本一律打高分
        costs = np.where(valid, costs, 1e10)
        costs = np.nan_to_num(costs, nan=1e10, posinf=1e10)

        # 5. 自适应 softmax 温度:用 cost std 作为温度尺度,避免权重退化为单点
        min_cost = float(np.min(costs))
        centered = costs - min_cost
        adaptive_temp = max(1e-6, float(np.std(centered)))
        weights = np.exp(-centered / adaptive_temp)
        weights /= float(np.sum(weights))

        optimal_seq = np.sum(weights[:, np.newaxis] * samples, axis=0)
        optimal_seq = np.clip(optimal_seq, self.u_min, self.u_max)
        optimal_u = float(optimal_seq[0])

        # 6. 温度安全检查 - 迭代缩放
        try:
            for _ in range(5):
                _sv, t_check, _pw, _pn, _r, _b = self.plant.predict(
                    [optimal_u], current_temp, dt=self.dt, prev_control=self.u_prev
                )
                if t_check[0] <= self.temp_max:
                    break
                scaled = max(self.u_min, optimal_u * 0.7)
                if scaled >= optimal_u:
                    break
                optimal_u = scaled
            optimal_seq[0] = optimal_u
        except Exception:
            pass

        self.u_prev = optimal_u

        # 失败率统计(可选调试信息)
        n_fail = int(np.sum(~valid))
        if n_fail > self.num_samples * 0.1:
            print(f"⚠️ MPPI: {n_fail}/{self.num_samples} 样本预测失败 (>10%),建议检查模型外推或约束设置")

        return optimal_u, optimal_seq, True, min_cost, weights


__all__ = [
    "LEDPlant",
    "LEDMPPIController",
    "LedThermalParams",
    "ThermalModelManager",
    "FirstOrderThermalModel",
    "PWMtoPowerModel",
    "PWMtoPPFDModel",
    "DEFAULT_TARGET_PPFD",
]
