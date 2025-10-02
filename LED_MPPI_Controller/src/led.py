from __future__ import annotations

"""LED 控制系统核心库

本模块提供 LED 控制系统的核心功能，包括：
1. 热力学模型 - LED 温度动态建模
2. PWM-PPFD 转换 - 控制信号到光输出的映射
3. PWM-功率转换 - 控制信号到功耗的映射
4. 前向步进接口 - MPPI 控制器使用的前向仿真
"""

import os
import csv
import math
import re
import sys
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Callable, Sequence, List
import numpy as np

# 热力学模型相关导入
import pickle
import json
import numpy as np
from typing import Optional, Literal

# =============================================================================
# 模块 1: 新版热力学模型系统（基于Thermal目录）
# =============================================================================

# 默认参数
DEFAULT_BASE_AMBIENT_TEMP = 25.0
DEFAULT_THERMAL_RESISTANCE = 0.05
DEFAULT_TIME_CONSTANT_S = 7.5
DEFAULT_THERMAL_MASS = 150.0
DEFAULT_MAX_PPFD = 200.0
DEFAULT_MAX_POWER = 130.0
DEFAULT_LED_EFFICIENCY = 0.8
DEFAULT_EFFICIENCY_DECAY = 0.1

@dataclass(frozen=True)
class LedThermalParams:
    """LED热力学参数"""
    base_ambient_temp: float = DEFAULT_BASE_AMBIENT_TEMP
    thermal_resistance: float = DEFAULT_THERMAL_RESISTANCE
    time_constant_s: float = DEFAULT_TIME_CONSTANT_S
    thermal_mass: float = DEFAULT_THERMAL_MASS
    max_ppfd: float = DEFAULT_MAX_PPFD
    max_power: float = DEFAULT_MAX_POWER
    led_efficiency: float = DEFAULT_LED_EFFICIENCY
    efficiency_decay: float = DEFAULT_EFFICIENCY_DECAY
    model_type: Literal["mlp", "thermal"] = "thermal"
    model_dir: str = "../Thermal/exported_models"
    solar_threshold: float = 1.4  # Solar值阈值，用于判断升温/降温

class ThermalModelManager:
    """热力学模型管理器 - 管理MLP和纯热力学模型"""
    
    def __init__(self, params: LedThermalParams):
        self.params = params
        self.model_dir = params.model_dir
        self.model_type = params.model_type
        self.solar_threshold = params.solar_threshold
        
        # 模型缓存
        self._heating_mlp_model = None
        self._cooling_mlp_model = None
        self._heating_thermal_params = None
        self._cooling_thermal_params = None
        
        # 当前状态
        self.current_temp = params.base_ambient_temp
        self.current_solar = params.solar_threshold
        
        # 时间累积状态
        self.elapsed_time_minutes = 0.0
        self.last_phase = None  # 'heating' or 'cooling'
        
        # 加载模型
        self._load_models()
    
    def _load_models(self):
        """加载所有模型"""
        try:
            if self.model_type == "mlp":
                # 尝试加载MLP模型
                try:
                    # 添加Thermal目录到路径以支持MLP类导入
                    thermal_dir = os.path.join(os.path.dirname(__file__), '..', 'Thermal')
                    if thermal_dir not in sys.path:
                        sys.path.insert(0, thermal_dir)
                    
                    # 动态导入MLP类并注册到全局命名空间
                    import importlib.util
                    
                    # 导入heating模块
                    heating_spec = importlib.util.spec_from_file_location(
                        'heating_module', 
                        os.path.join(thermal_dir, '22-improved_thermal_constrained_mlp_heating.py')
                    )
                    heating_module = importlib.util.module_from_spec(heating_spec)
                    heating_spec.loader.exec_module(heating_module)
                    
                    # 导入cooling模块
                    cooling_spec = importlib.util.spec_from_file_location(
                        'cooling_module', 
                        os.path.join(thermal_dir, '20-improved_thermal_constrained_mlp_cooling.py')
                    )
                    cooling_module = importlib.util.module_from_spec(cooling_spec)
                    cooling_spec.loader.exec_module(cooling_module)
                    
                    # 创建自定义unpickler来处理类定义问题
                    class CustomUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            if name == 'ImprovedThermodynamicConstrainedMLPHeating':
                                return heating_module.ImprovedThermodynamicConstrainedMLPHeating
                            elif name == 'ImprovedThermodynamicConstrainedMLPCooling':
                                return cooling_module.ImprovedThermodynamicConstrainedMLPCooling
                            return super().find_class(module, name)
                    
                    # 加载MLP模型
                    heating_path = os.path.join(self.model_dir, "heating_mlp_model.pkl")
                    cooling_path = os.path.join(self.model_dir, "cooling_mlp_model.pkl")
                    
                    with open(heating_path, 'rb') as f:
                        self._heating_mlp_model = CustomUnpickler(f).load()
                    with open(cooling_path, 'rb') as f:
                        self._cooling_mlp_model = CustomUnpickler(f).load()
                    
                    print("✅ MLP模型加载成功")
                        
                except Exception as mlp_error:
                    print(f"警告: MLP模型加载失败 ({mlp_error})，回退到纯热力学模型")
                    self.model_type = "thermal"
                    
            # 加载纯热力学模型参数
            heating_thermal_path = os.path.join(self.model_dir, "heating_thermal_model.json")
            cooling_thermal_path = os.path.join(self.model_dir, "cooling_thermal_model.json")
            
            with open(heating_thermal_path, 'r', encoding='utf-8') as f:
                self._heating_thermal_params = json.load(f)
            with open(cooling_thermal_path, 'r', encoding='utf-8') as f:
                self._cooling_thermal_params = json.load(f)
                
        except Exception as e:
            raise RuntimeError(f"加载热力学模型失败: {e}")
    
    def _is_heating_phase(self, solar_val: float) -> bool:
        """判断是否为升温阶段"""
        return solar_val > self.solar_threshold
    
    def _predict_mlp(self, time_minutes: float, solar_val: float, is_heating: bool) -> float:
        """使用MLP模型预测温度差"""
        model = self._heating_mlp_model if is_heating else self._cooling_mlp_model
        if model is None:
            raise RuntimeError(f"{'升温' if is_heating else '降温'}MLP模型未加载")
        
        # MLP模型需要时间数组和Solar值数组
        time_array = np.array([time_minutes])
        solar_array = np.array([solar_val])
        
        delta_temp = model.predict(time_array, solar_array)[0]
        return float(delta_temp)
    
    def _predict_thermal(self, time_minutes: float, solar_val: float, is_heating: bool) -> float:
        """使用纯热力学模型预测温度差"""
        params = self._heating_thermal_params if is_heating else self._cooling_thermal_params
        if params is None:
            raise RuntimeError(f"{'升温' if is_heating else '降温'}热力学模型参数未加载")
        
        # 提取参数
        K1_base = params['parameters']['K1_base']
        tau1 = params['parameters']['tau1']
        K2_base = params['parameters']['K2_base']
        tau2 = params['parameters']['tau2']
        alpha_solar = params['parameters']['alpha_solar']
        a1_ref = params['a1_ref']
        
        # Solar修正因子
        solar_factor = 1 + alpha_solar * (solar_val - a1_ref)
        K1_solar = K1_base * solar_factor
        K2_solar = K2_base * solar_factor
        
        # 计算温度差
        t = time_minutes
        if is_heating:
            # 升温公式: ΔT(t) = K1 × (1 - exp(-t/τ1)) + K2 × (1 - exp(-t/τ2))
            delta_temp = K1_solar * (1 - np.exp(-t / tau1)) + K2_solar * (1 - np.exp(-t / tau2))
        else:
            # 降温公式: ΔT(t) = K1 × exp(-t/τ1) + K2 × exp(-t/τ2)
            delta_temp = K1_solar * np.exp(-t / tau1) + K2_solar * np.exp(-t / tau2)
        
        return float(delta_temp)
    
    def step(self, power: float, dt: float, solar_vol: Optional[float] = None, control_change: Optional[float] = None) -> float:
        """热力学步进 - 支持基于控制量变化判断升温/降温"""
        # 使用Solar Vol或默认值
        solar_val = solar_vol if solar_vol is not None else self.current_solar
        
        # 🔥 基于控制量变化判断升温/降温阶段
        if control_change is not None:
            # MPPI控制量变化: u0 - u1
            is_heating = control_change > 0  # 控制量增加 → 升温
        else:
            # 回退到Solar值判断
            is_heating = self._is_heating_phase(solar_val)
        
        # 转换时间单位（秒转分钟）
        dt_minutes = dt / 60.0
        
        # 检查Solar电压或阶段是否改变
        current_phase = 'heating' if is_heating else 'cooling'
        solar_changed = abs(solar_val - self.current_solar) > 1e-6
        phase_changed = self.last_phase != current_phase
        
        if solar_changed or phase_changed:
            # Solar电压或阶段改变，重置时间累积
            self.elapsed_time_minutes = 0.0
            self.last_phase = current_phase
        
        # 计算累积时间
        self.elapsed_time_minutes += dt_minutes
        
        # 预测温度差（基于累积时间）
        if self.model_type == "mlp":
            delta_temp = self._predict_mlp(self.elapsed_time_minutes, solar_val, is_heating)
        else:
            delta_temp = self._predict_thermal(self.elapsed_time_minutes, solar_val, is_heating)
        
        # 更新温度（热力学模型预测的是相对于环境温度的温差）
        ambient_temp = self.params.base_ambient_temp
        new_temp = ambient_temp + delta_temp
        
        # 更新状态
        self.current_temp = new_temp
        self.current_solar = solar_val
        
        return new_temp
    
    def reset(self, ambient_temp: Optional[float] = None):
        """重置模型状态"""
        self.current_temp = ambient_temp if ambient_temp is not None else self.params.base_ambient_temp
        self.current_solar = self.solar_threshold
        # 重置时间累积状态
        self.elapsed_time_minutes = 0.0
        self.last_phase = None
    
    @property
    def ambient_temp(self) -> float:
        return self.current_temp
    
    @ambient_temp.setter
    def ambient_temp(self, value: float):
        self.current_temp = float(value)
    
    @property
    def supports_solar_input(self) -> bool:
        return True
    
    def target_temperature(self, power: float) -> float:
        """目标温度（简化计算）"""
        return self.params.base_ambient_temp + power * self.params.thermal_resistance
    
    def target_temperature_solar(self, solar_vol: float) -> float:
        """基于Solar Vol的目标温度"""
        is_heating = self._is_heating_phase(solar_vol)
        # 使用稳态温度差作为目标
        if self.model_type == "mlp":
            delta_temp = self._predict_mlp(1000.0, solar_vol, is_heating)  # 长时间预测
        else:
            delta_temp = self._predict_thermal(1000.0, solar_vol, is_heating)
        
        if is_heating:
            return self.params.base_ambient_temp + delta_temp
        else:
            return self.params.base_ambient_temp - delta_temp

# 兼容性别名
BaseThermalModel = ThermalModelManager
FirstOrderThermalModel = ThermalModelManager
LedThermalModel = ThermalModelManager

class Led:
    """LED封装类"""
    
    def __init__(self, model_type: str = "thermal", params: Optional[LedThermalParams] = None):
        if params is None:
            params_obj = LedThermalParams(model_type=model_type)
        else:
            # 创建新的参数对象以避免冻结问题
            params_obj = LedThermalParams(
                base_ambient_temp=params.base_ambient_temp,
                thermal_resistance=params.thermal_resistance,
                time_constant_s=params.time_constant_s,
                thermal_mass=params.thermal_mass,
                max_ppfd=params.max_ppfd,
                max_power=params.max_power,
                led_efficiency=params.led_efficiency,
                efficiency_decay=params.efficiency_decay,
                model_type=model_type,
                model_dir=params.model_dir,
                solar_threshold=params.solar_threshold
            )
        self.params = params_obj
        self.model = ThermalModelManager(params_obj)
    
    def reset(self, ambient_temp: Optional[float] = None):
        self.model.reset(ambient_temp)
    
    def step_with_heat(self, power: float, dt: float) -> float:
        return self.model.step(power=power, dt=dt)
    
    def step_with_solar(self, solar_vol: float, dt: float) -> float:
        return self.model.step(power=0.0, dt=dt, solar_vol=solar_vol)
    
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
    
    def get_model_info(self) -> dict:
        return {
            "model_type": self.params.model_type,
            "solar_threshold": self.params.solar_threshold,
            "base_ambient_temp": self.params.base_ambient_temp
        }

def create_model(model_type: str = "thermal", params: Optional[LedThermalParams] = None) -> ThermalModelManager:
    """创建热力学模型"""
    params_obj = params or LedThermalParams()
    params_obj.model_type = model_type
    return ThermalModelManager(params_obj)

def create_default_params() -> LedThermalParams:
    """创建默认参数"""
    return LedThermalParams()

# 兼容性别名
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
    use_solar_vol_for_5_1: bool = False,  # 在5:1场景下以Solar_Vol替代PPFD并驱动热模型
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

    # 3) 预测 PPFD 或 Solar_Vol（可选）
    ppfd_val: Optional[float] = None
    if ppfd_model is not None:
        if use_solar_vol_for_5_1 and (key_arg is None or str(key_arg) == "5:1"):
            # 当指定5:1并要求使用Solar_Vol时，仍复用ppfd字段承载该数值
            ppfd_val = float(ppfd_model.predict(r_pwm=r, b_pwm=b, key="5:1"))
        else:
            ppfd_val = float(ppfd_model.predict(r_pwm=r, b_pwm=b, key=key_arg))

    # 4) 热学步进 - 根据模型类型选择不同方式
    eff_val: Optional[float] = None
    if use_efficiency:
        if eta_model is None:
            raise ValueError("use_efficiency=True 需要提供 eta_model 回调")
        eff_val = float(
            max(
                0.0,
                min(1.0, eta_model(r, b, total, thermal_model.ambient_temp, thermal_model.params)),
            )
        )
        p_heat = p_elec * (1.0 - eff_val)
    else:
        p_heat = p_elec * float(heat_scale)

    solar_for_thermal: Optional[float] = None
    if (
        getattr(thermal_model, "supports_solar_input", False)
        and use_solar_vol_for_5_1
        and ppfd_val is not None
        and (key_arg is None or str(key_arg) == "5:1")
    ):
        solar_for_thermal = float(ppfd_val)

    if solar_for_thermal is not None:
        new_temp = float(thermal_model.step(power=p_heat, dt=float(dt), solar_vol=solar_for_thermal))
    else:
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
    use_solar_vol_for_5_1: bool = False,  # 在5:1场景下以Solar_Vol替代PPFD
) -> List[LedForwardOutput]:
    """批量前向步进（逐实例调用 forward_step，便于复用最新热模型逻辑）。"""

    if len(r_pwms) != len(b_pwms) or len(r_pwms) != len(thermal_models):
        raise ValueError("thermal_models、r_pwms、b_pwms 长度必须一致")

    outputs: List[LedForwardOutput] = []
    for model, r_pwm, b_pwm in zip(thermal_models, r_pwms, b_pwms):
        outputs.append(
            forward_step(
                thermal_model=model,
                r_pwm=r_pwm,
                b_pwm=b_pwm,
                dt=dt,
                power_model=power_model,
                ppfd_model=ppfd_model,
                model_key=model_key,
                use_efficiency=use_efficiency,
                eta_model=eta_model,
                heat_scale=heat_scale,
                use_solar_vol_for_5_1=use_solar_vol_for_5_1,
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
