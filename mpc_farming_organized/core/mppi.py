"""
MPPI (Model Predictive Path Integral) 控制器模块

该模块包含 MPPI 控制器的核心实现，用于优化 LED 光照系统以最大化光合作用。

主要组件:
- LEDPlant: 结合了 LED 的物理行为 (来自 led.py) 和植物的光合作用响应。
- LEDMPPIController: MPPI 控制器，通过随机采样优化控制序列。
- LEDMPPISimulation: 用于测试和可视化 MPPI 控制器性能的仿真环境。
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

# --- 依赖导入 ---

# 导入LED物理模型
try:
    from .led import led_step, led_steady_state
except ImportError:
    # 允许作为脚本直接运行时进行测试
    from led import led_step, led_steady_state

# 导入光合作用预测模型
# 最佳实践是将此模型注册为包的一部分，但暂时保持现有逻辑
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

try:
    from pn_prediction.predict_corrected import CorrectedPhotosynthesisPredictor as PhotosynthesisPredictor
    PHOTOSYNTHESIS_AVAILABLE = True
    print("✅ MPPI模块: 使用修正的光合作用预测器")
except ImportError:
    try:
        from pn_prediction.predict import PhotosynthesisPredictor
        PHOTOSYNTHESIS_AVAILABLE = True
        print("✅ MPPI模块: 使用标准光合作用预测器")
    except ImportError:
        print("⚠️ 警告: PhotosynthesisPredictor不可用。MPPI将使用简易模型。")
        PHOTOSYNTHESIS_AVAILABLE = False


# --- 系统模型 ---

class LEDPlant:
    """
    LED-植物系统模型

    该模型封装了 LED 的物理行为 (来自 led.py) 和植物的光合作用响应。
    它是 MPPI 控制器进行未来状态预测的基础。
    """

    def __init__(
        self,
        base_ambient_temp=25.0,
        max_ppfd=500.0,
        max_power=100.0,
        thermal_resistance=2.5,
        thermal_mass=0.5,
    ):
        """
        初始化LED植物模型
        
        参数:
        - base_ambient_temp: 环境基准温度(°C)
        - max_ppfd: 最大光合光子通量密度(μmol/m²/s)
        - max_power: 最大功率(W)
        - thermal_resistance: 热阻(K/W)
        - thermal_mass: 热容(J/°C)
        """
        self.base_ambient_temp = base_ambient_temp
        self.max_ppfd = max_ppfd
        self.max_power = max_power
        self.thermal_resistance = thermal_resistance
        self.thermal_mass = thermal_mass
        self.ambient_temp = base_ambient_temp
        self.time = 0.0

        # 初始化光合作用预测器
        self.photo_predictor = None
        if PHOTOSYNTHESIS_AVAILABLE:
            try:
                self.photo_predictor = PhotosynthesisPredictor()
                self.use_photo_model = self.photo_predictor.is_loaded
                if self.use_photo_model:
                    print("✅ 光合作用模型成功加载。")
                else:
                    print("⚠️ 光合作用模型未能加载，将使用简易模型。")
            except Exception as e:
                self.use_photo_model = False
                print(f"⚠️ 光合作用模型加载失败: {e}。将使用简易模型。")
        else:
            self.use_photo_model = False

    def step(self, pwm_percent, dt=0.1):
        """
        使用导入的led_step函数进行LED植物的单步仿真。
        
        返回:
        - ppfd, new_ambient_temp, power, photosynthesis_rate
        """
        ppfd, new_ambient_temp, power, _ = led_step(
            pwm_percent=pwm_percent,
            ambient_temp=self.ambient_temp,
            base_ambient_temp=self.base_ambient_temp,
            dt=dt,
            max_ppfd=self.max_ppfd,
            max_power=self.max_power,
            thermal_resistance=self.thermal_resistance,
            thermal_mass=self.thermal_mass,
        )

        self.ambient_temp = new_ambient_temp
        self.time += dt
        photosynthesis_rate = self.get_photosynthesis_rate(ppfd, new_ambient_temp)

        return ppfd, new_ambient_temp, power, photosynthesis_rate

    def get_photosynthesis_rate(self, ppfd, temperature, co2=400, rb_ratio=0.83):
        """获取光合作用速率，优先使用模型，否则回退到简易版。"""
        if self.use_photo_model:
            try:
                return self.photo_predictor.predict(ppfd, co2, temperature, rb_ratio)
            except Exception as e:
                print(f"⚠️ 光合作用预测失败: {e}。回退到简易模型。")
                return self._simple_photosynthesis_model(ppfd, temperature)
        else:
            return self._simple_photosynthesis_model(ppfd, temperature)

    def _simple_photosynthesis_model(self, ppfd, temperature):
        """一个备用的、简化的光合作用模型。"""
        temp_factor = np.exp(-0.01 * (temperature - 25) ** 2)  # 25°C时最优
        pn = (25 * ppfd / (300 + ppfd)) * temp_factor
        return max(0, pn)

    def predict(self, pwm_sequence, initial_temp, dt=0.1):
        """根据输入的PWM控制序列，预测未来的系统状态。"""
        temp = initial_temp
        ppfd_pred, temp_pred, power_pred, photo_pred = [], [], [], []

        for pwm in pwm_sequence:
            ppfd, new_temp, power, _ = led_step(
                pwm_percent=pwm,
                ambient_temp=temp,
                base_ambient_temp=self.base_ambient_temp,
                dt=dt,
                max_ppfd=self.max_ppfd,
                max_power=self.max_power,
                thermal_resistance=self.thermal_resistance,
                thermal_mass=self.thermal_mass,
            )
            temp = new_temp
            photosynthesis_rate = self.get_photosynthesis_rate(ppfd, temp)

            ppfd_pred.append(ppfd)
            temp_pred.append(temp)
            power_pred.append(power)
            photo_pred.append(photosynthesis_rate)

        return (
            np.array(ppfd_pred),
            np.array(temp_pred),
            np.array(power_pred),
            np.array(photo_pred),
        )

# --- MPPI 控制器 ---

class LEDMPPIController:
    """
    LED MPPI 控制器

    实现 MPPI 算法，用于计算最优的 PWM 控制序列，以在满足约束的同时最大化光合作用。
    """

    def __init__(self, plant, horizon=10, num_samples=1000, dt=0.1, temperature=1.0):
        """
        初始化控制器
        
        参数:
        - plant (LEDPlant): 被控对象的模型。
        - horizon (int): 预测时域的步数。
        - num_samples (int): 每次优化时采样的控制序列数量。
        - dt (float): 控制时间步长。
        - temperature (float): MPPI中的“温度”参数(lambda)，用于调节权重分布的平滑度。
        """
        self.plant = plant
        self.horizon = horizon
        self.num_samples = num_samples
        self.dt = dt
        self.temperature = temperature
        self.pwm_prev = 0.0

        # --- 默认参数 ---
        self.weights = {'Q_photo': 10.0, 'R_pwm': 0.001, 'R_dpwm': 0.05, 'R_power': 0.01}
        self.constraints = {'pwm_min': 0.0, 'pwm_max': 80.0, 'temp_min': 20.0, 'temp_max': 29.0}
        self.penalties = {'temp_penalty': 100000.0, 'pwm_penalty': 1000.0}
        self.pwm_std = 15.0

    def set_weights(self, **kwargs):
        """设置成本函数权重。"""
        self.weights.update(kwargs)

    def set_constraints(self, **kwargs):
        """设置PWM和温度的约束。"""
        self.constraints.update(kwargs)

    def set_mppi_params(self, **kwargs):
        """设置MPPI算法参数 (num_samples, temperature, pwm_std)。"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def solve(self, current_temp, mean_sequence=None):
        """
        求解 MPPI 优化问题，得到最优控制动作。

        算法步骤:
        1. 采样: 围绕上一时刻的最优序列（或初始值）生成大量随机控制序列。
        2. 评估: 对每个采样序列，使用 plant 模型预测未来的状态，并计算其总成本。
        3. 加权: 根据成本使用 softmax 函数计算每个采样序列的权重（成本越低，权重越高）。
        4. 更新: 计算所有采样序列的加权平均，得到新的最优控制序列。
        5. 应用: 将新序列的第一个控制动作作为当前时刻的输出，并进行安全检查。
        """
        # 1. 采样
        if mean_sequence is None:
            mean_sequence = np.ones(self.horizon) * min(40.0, self.constraints['pwm_max'] * 0.5)
        control_samples = self._sample_control_sequences(mean_sequence)

        # 2. 评估成本
        costs = np.array([self._compute_total_cost(sample, current_temp) for sample in control_samples])
        costs = np.nan_to_num(costs, nan=1e10, posinf=1e10)

        # 3. 计算权重
        min_cost = np.min(costs)
        exp_costs = np.exp(-(costs - min_cost) / self.temperature)
        weights = exp_costs / np.sum(exp_costs)

        # 4. 更新最优序列
        optimal_sequence = np.sum(weights[:, np.newaxis] * control_samples, axis=0)
        optimal_sequence = np.clip(optimal_sequence, self.constraints['pwm_min'], self.constraints['pwm_max'])

        # 5. 应用第一个控制并做安全检查
        optimal_pwm = optimal_sequence[0]
        optimal_pwm = self._temperature_safety_check(optimal_pwm, current_temp)

        self.pwm_prev = optimal_pwm
        
        return optimal_pwm, optimal_sequence, True, np.min(costs), weights

    def _sample_control_sequences(self, mean_sequence):
        """围绕均值序列进行高斯噪声采样，生成控制序列候选项。"""
        noise = np.random.normal(0, self.pwm_std, (self.num_samples, self.horizon))
        samples = mean_sequence[np.newaxis, :] + noise
        return np.clip(samples, self.constraints['pwm_min'], self.constraints['pwm_max'])

    def _compute_total_cost(self, pwm_sequence, current_temp):
        """
        为单个控制序列计算总成本。
        总成本 = 光合作用目标成本 + 控制成本 + 约束惩罚
        """
        try:
            ppfd_pred, temp_pred, power_pred, photo_pred = self.plant.predict(
                pwm_sequence, current_temp, self.dt
            )
            
            # 1. 光合作用成本 (目标是最大化，所以成本为负值)
            photo_cost = -np.sum(photo_pred) * self.weights['Q_photo']
            
            # 2. 控制成本 (控制量大小、变化率、功耗)
            pwm_cost = np.sum(pwm_sequence**2) * self.weights['R_pwm']
            dpwm_cost = np.sum(np.diff(np.insert(pwm_sequence, 0, self.pwm_prev))**2) * self.weights['R_dpwm']
            power_cost = np.sum(power_pred**2) * self.weights['R_power']
            
            # 3. 约束惩罚 (软约束)
            temp_violation = (
                np.maximum(0, temp_pred - self.constraints['temp_max'])**2 +
                np.maximum(0, self.constraints['temp_min'] - temp_pred)**2
            )
            temp_penalty_cost = np.sum(temp_violation) * self.penalties['temp_penalty']
            
            return photo_cost + pwm_cost + dpwm_cost + power_cost + temp_penalty_cost

        except Exception:
            return 1e10  # 对于无效序列返回高成本

    def _temperature_safety_check(self, pwm_action, current_temp):
        """对即将应用的PWM值进行一步安全预测，防止温度超限。"""
        _, temp_check, _, _ = self.plant.predict(np.array([pwm_action]), current_temp, self.dt)
        if temp_check[0] > self.constraints['temp_max']:
            reduced_pwm = max(self.constraints['pwm_min'], pwm_action * 0.7)
            print(f"🌡️ MPPI安全警告: 预测温度超限，紧急将PWM从{pwm_action:.1f}%降至{reduced_pwm:.1f}%")
            return reduced_pwm
        return pwm_action

# --- 仿真与可视化 (建议移至example脚本) ---
