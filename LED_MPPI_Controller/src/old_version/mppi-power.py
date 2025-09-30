import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

# 导入LED仿真函数
from led import led_step, led_steady_state

# 导入光合作用预测器
try:
    from pn_prediction.predict import PhotosynthesisPredictor

    PHOTOSYNTHESIS_AVAILABLE = True
except ImportError:
    print("警告: PhotosynthesisPredictor不可用，使用简单模型。")
    PHOTOSYNTHESIS_AVAILABLE = False


class LEDPlant:
    """使用导入LED函数的MPPI LED植物模型"""

    def __init__(
        self,
        base_ambient_temp=25.0,
        max_ppfd=500.0,
        max_power=100.0,
        thermal_resistance=2.5,
        thermal_mass=0.5,
    ):
        self.base_ambient_temp = base_ambient_temp
        self.max_ppfd = max_ppfd
        self.max_power = max_power
        self.thermal_resistance = thermal_resistance
        self.thermal_mass = thermal_mass

        # 当前状态
        self.ambient_temp = base_ambient_temp
        self.time = 0.0

        # 如果可用，初始化光合作用预测器
        if PHOTOSYNTHESIS_AVAILABLE:
            try:
                self.photo_predictor = PhotosynthesisPredictor()
                self.use_photo_model = self.photo_predictor.is_loaded
            except:
                self.use_photo_model = False
        else:
            self.use_photo_model = False

    def step(self, pwm_percent, dt=0.1):
        """使用导入led_step函数的LED植物单步仿真"""
        ppfd, new_ambient_temp, power, efficiency = led_step(
            pwm_percent=pwm_percent,
            ambient_temp=self.ambient_temp,
            base_ambient_temp=self.base_ambient_temp,
            dt=dt,
            max_ppfd=self.max_ppfd,
            max_power=self.max_power,
            thermal_resistance=self.thermal_resistance,
            thermal_mass=self.thermal_mass,
        )

        # 更新状态
        self.ambient_temp = new_ambient_temp
        self.time += dt

        # 计算光合作用速率
        photosynthesis_rate = self.get_photosynthesis_rate(ppfd, new_ambient_temp)

        return ppfd, new_ambient_temp, power, photosynthesis_rate

    def get_photosynthesis_rate(self, ppfd, temperature):
        """始终使用光合作用预测模型"""
        if self.use_photo_model:
            try:
                return self.photo_predictor.predict(ppfd, temperature)
            except Exception as e:
                print(f"警告: 光合作用预测失败: {e}")
                return self.simple_photosynthesis_model(ppfd, temperature)
        else:
            print(
                "警告: 使用简单光合作用模型 - 预测模型不可用"
            )
            return self.simple_photosynthesis_model(ppfd, temperature)

    def simple_photosynthesis_model(self, ppfd, temperature):
        """作为备选的简单光合作用模型"""
        ppfd_max = 1000  # μmol/m²/s
        pn_max = 25  # μmol/m²/s
        km = 300  # μmol/m²/s

        # 温度效应（25°C左右最优）
        temp_factor = np.exp(-0.01 * (temperature - 25) ** 2)

        # 光响应
        pn = (pn_max * ppfd / (km + ppfd)) * temp_factor

        return max(0, pn)

    def predict(self, pwm_sequence, initial_temp, dt=0.1):
        """给定PWM序列预测未来状态"""
        temp = initial_temp
        ppfd_pred = []
        temp_pred = []
        power_pred = []
        photo_pred = []

        for pwm in pwm_sequence:
            ppfd, new_temp, power, efficiency = led_step(
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


class LEDMPPIController:
    """LED系统的模型预测路径积分控制器 - 光合作用最大化"""

    def __init__(self, plant, horizon=10, num_samples=1000, dt=0.1, temperature=1.0):
        self.plant = plant
        self.horizon = horizon
        self.num_samples = num_samples
        self.dt = dt
        self.temperature = temperature  # MPPI温度参数

        # 代价函数权重 - 专注于光合作用最大化
        self.Q_photo = 10.0  # 光合作用最大化的高权重
        self.Q_ref = 5.0  # 参考轨迹跟踪权重
        self.R_pwm = 0.001  # 低控制努力惩罚
        self.R_dpwm = 0.05  # 平滑控制变化
        self.R_power = 0.1  # 效率的适度功耗惩罚

        # 约束条件
        self.pwm_min = 0.0
        self.pwm_max = 80.0
        self.temp_min = 20.0
        self.temp_max = 29.0

        # 控制参数
        self.pwm_std = 15.0  # PWM采样的标准差
        self.pwm_prev = 0.0

        # 约束惩罚
        self.temp_penalty = 100000.0  # 温度违规的极高惩罚
        self.pwm_penalty = 1000.0  # PWM约束违规惩罚

    def set_weights(
        self, Q_photo=10.0, Q_ref=5.0, R_pwm=0.001, R_dpwm=0.05, R_power=0.1
    ):
        """设置MPPI光合作用最大化的代价权重"""
        self.Q_photo = Q_photo
        self.Q_ref = Q_ref
        self.R_pwm = R_pwm
        self.R_dpwm = R_dpwm
        self.R_power = R_power

    def set_constraints(self, pwm_min=0.0, pwm_max=80.0, temp_min=20.0, temp_max=29.0):
        """设置MPPI约束条件"""
        self.pwm_min = pwm_min
        self.pwm_max = pwm_max
        self.temp_min = temp_min
        self.temp_max = temp_max

    def set_mppi_params(self, num_samples=1000, temperature=1.0, pwm_std=15.0):
        """设置MPPI算法参数"""
        self.num_samples = num_samples
        self.temperature = temperature
        self.pwm_std = pwm_std

    def sample_control_sequences(self, mean_sequence):
        """在均值周围采样控制序列"""
        # 创建采样噪声
        noise = np.random.normal(0, self.pwm_std, (self.num_samples, self.horizon))

        # 将噪声添加到均值序列
        samples = mean_sequence[np.newaxis, :] + noise

        # 通过裁剪应用约束
        samples = np.clip(samples, self.pwm_min, self.pwm_max)

        return samples

    def compute_cost(self, pwm_sequence, current_temp, photo_ref=None):
        """计算单个PWM序列的代价 - 通过参考跟踪最大化光合作用"""
        try:
            # 预测未来状态
            ppfd_pred, temp_pred, power_pred, photo_pred = self.plant.predict(
                pwm_sequence, current_temp, self.dt
            )

            cost = 0.0

            # 主要目标：在时间范围内最大化光合作用
            for k in range(self.horizon):
                # 负光合作用以最大化它（最小化负值）
                cost -= self.Q_photo * photo_pred[k]

                # 参考轨迹跟踪（如果提供）
                if photo_ref is not None and k < len(photo_ref):
                    ref_error = photo_pred[k] - photo_ref[k]
                    cost += self.Q_ref * ref_error**2

                # 温度的硬约束惩罚
                if temp_pred[k] > self.temp_max:
                    violation = temp_pred[k] - self.temp_max
                    cost += self.temp_penalty * violation**2
                if temp_pred[k] < self.temp_min:
                    violation = self.temp_min - temp_pred[k]
                    cost += self.temp_penalty * violation**2

            # 控制努力代价
            for k in range(self.horizon):
                cost += self.R_pwm * pwm_sequence[k] ** 2
                cost += self.R_power * power_pred[k] ** 2

            # 控制平滑性
            prev_pwm = self.pwm_prev
            for k in range(self.horizon):
                dpwm = pwm_sequence[k] - prev_pwm
                cost += self.R_dpwm * dpwm**2
                prev_pwm = pwm_sequence[k]

            # PWM约束惩罚
            for k in range(self.horizon):
                if pwm_sequence[k] > self.pwm_max:
                    violation = pwm_sequence[k] - self.pwm_max
                    cost += self.pwm_penalty * violation**2
                if pwm_sequence[k] < self.pwm_min:
                    violation = self.pwm_min - pwm_sequence[k]
                    cost += self.pwm_penalty * violation**2

            return cost

        except Exception:
            # 对无效序列返回非常高的代价
            return 1e10

    def solve(self, current_temp, mean_sequence=None, photo_ref=None):
        """求解MPPI优化以最大化光合作用，可选参考跟踪"""

        # 如果未提供，初始化均值序列
        if mean_sequence is None:
            # 以适中的PWM值开始
            mean_sequence = np.ones(self.horizon) * min(40.0, self.pwm_max * 0.5)

        # 采样控制序列
        control_samples = self.sample_control_sequences(mean_sequence)

        # 计算所有样本的代价
        costs = np.zeros(self.num_samples)

        for i in range(self.num_samples):
            costs[i] = self.compute_cost(control_samples[i], current_temp, photo_ref)

        # 处理无限或NaN代价
        costs = np.nan_to_num(costs, nan=1e10, posinf=1e10)

        # 使用softmax计算权重
        min_cost = np.min(costs)
        exp_costs = np.exp(-(costs - min_cost) / self.temperature)
        weights = exp_costs / np.sum(exp_costs)

        # 计算控制序列的加权平均
        optimal_sequence = np.sum(weights[:, np.newaxis] * control_samples, axis=0)

        # 应用最终约束
        optimal_sequence = np.clip(optimal_sequence, self.pwm_min, self.pwm_max)

        # 获取第一个控制动作
        optimal_pwm = optimal_sequence[0]

        # 温度安全检查
        _, temp_check, _, _ = self.plant.predict([optimal_pwm], current_temp, self.dt)
        if temp_check[0] > self.temp_max:
            # 紧急降低
            optimal_pwm = max(self.pwm_min, optimal_pwm * 0.7)
            print(
                f"MPPI: 因温度风险紧急降低PWM至{optimal_pwm:.1f}%"
            )

        self.pwm_prev = optimal_pwm

        # 返回额外信息
        success = True
        best_cost = np.min(costs)

        return optimal_pwm, optimal_sequence, success, best_cost, weights


class LEDMPPISimulation:
    """用于光合作用最大化的MPPI仿真环境，包含参考比较"""

    def __init__(self, plant, controller):
        self.plant = plant
        self.controller = controller

        # 数据存储
        self.time_data = []
        self.ppfd_data = []
        self.temp_data = []
        self.power_data = []
        self.pwm_data = []
        self.photo_data = []
        self.ppfd_ref_data = []  # 保留用于比较
        self.temp_ref_data = []  # 保留用于比较
        self.photo_ref_data = []  # 保留用于比较
        self.power_ref_data = []  # 参考功耗
        self.cost_data = []

        # 初始化用于评估的光合作用预测器
        if PHOTOSYNTHESIS_AVAILABLE:
            try:
                self.pn = PhotosynthesisPredictor()
            except:
                self.pn = None
        else:
            self.pn = None

    def create_reference_trajectory(self, duration, dt):
        """创建用于比较的参考轨迹"""
        time_points = np.arange(0, duration, dt)

        ppfd_ref = []
        temp_ref = []
        photo_ref = []
        power_ref = []

        current_temp = self.plant.base_ambient_temp
        for t in time_points:
            # PPFD参考：日光照周期
            ppfd_target = 700

            # 温度参考：使用LED步进与变化的环境温度
            _, temp_target, _, _ = led_step(
                pwm_percent=(ppfd_target / self.plant.max_ppfd) * 100,
                ambient_temp=current_temp,
                base_ambient_temp=self.plant.base_ambient_temp,
                dt=dt,
                max_ppfd=self.plant.max_ppfd,
                max_power=self.plant.max_power,
                thermal_resistance=self.plant.thermal_resistance,
                thermal_mass=self.plant.thermal_mass,
            )
            current_temp = temp_target

            # 光合作用参考（仅用于比较）
            if self.pn is not None:
                photo_target = self.pn.predict(ppfd_target, temp_target)
            else:
                # 使用简单模型作为参考
                ppfd_max = 1000
                pn_max = 25
                km = 300
                temp_factor = np.exp(-0.01 * (temp_target - 25) ** 2)
                photo_target = max(
                    0, (pn_max * ppfd_target / (km + ppfd_target)) * temp_factor
                )

            # 计算参考PPFD所需的PWM
            # 简化计算 - 实际上会更复杂
            # pwm_target = min(80.0, (ppfd_target / self.plant.max_ppfd) * 100)
            # pwm_target = ppfd_target / self.plant.max_ppfd * 100

            # 计算参考功耗
            # power_target = (pwm_target / 100) * self.plant.max_power
            # PPFD输出
            pwm_fraction = ppfd_target / self.plant.max_ppfd

            # LED效率
            efficiency = 0.8 + 0.2 * np.exp(-pwm_fraction * 2.0)

            # 功耗
            power_target = (self.plant.max_power * pwm_fraction) / efficiency

            ppfd_ref.append(ppfd_target)
            temp_ref.append(temp_target)
            photo_ref.append(photo_target)
            power_ref.append(power_target)

        return (
            np.array(ppfd_ref),
            np.array(temp_ref),
            np.array(photo_ref),
            np.array(power_ref),
        )

    def run_simulation(self, duration=120, dt=1.0):
        """运行MPPI仿真以最大化光合作用，包含参考比较"""

        print("开始LED MPPI仿真 - 光合作用最大化")
        print("=" * 60)
        print("注意：参考仅用于比较，不用于跟踪")
        print(
            f"MPPI参数: {self.controller.num_samples} 样本, 温度={self.controller.temperature}"
        )
        print(
            f"温度约束: ({self.controller.temp_min}, {self.controller.temp_max})°C"
        )
        print(
            f"PWM约束: ({self.controller.pwm_min}, {self.controller.pwm_max})%"
        )

        if self.plant.use_photo_model:
            print("使用训练的光合作用模型")
        else:
            print("使用简单光合作用模型")

        # 创建用于比较的参考轨迹
        ppfd_ref_full, temp_ref_full, photo_ref_full, power_ref_full = (
            self.create_reference_trajectory(duration, dt)
        )

        # 重置植物
        self.plant.ambient_temp = self.plant.base_ambient_temp
        self.plant.time = 0.0

        # 重置控制器
        self.controller.pwm_prev = 0.0

        # 清除数据
        self.clear_data()

        # 为MPPI初始化均值序列
        mean_sequence = np.ones(self.controller.horizon) * 30.0
        log_file = "mppi_log.txt"
        if os.path.exists(log_file):
            os.remove(log_file)
        # 仿真循环
        steps = int(duration / dt)
        for k in range(steps):
            current_time = k * dt

            # 为MPPI提取时间范围内的参考轨迹
            end_idx = min(k + self.controller.horizon, len(photo_ref_full))
            photo_ref_horizon = photo_ref_full[k:end_idx]

            # 求解MPPI以通过参考跟踪最大化光合作用
            pwm_optimal, optimal_sequence, success, cost, weights = (
                self.controller.solve(
                    self.plant.ambient_temp, mean_sequence, photo_ref_horizon
                )
            )

            # 更新下次迭代的均值序列（滚动时域）
            if len(optimal_sequence) > 1:
                mean_sequence = np.concatenate(
                    [optimal_sequence[1:], [optimal_sequence[-1]]]
                )
            else:
                mean_sequence = optimal_sequence

            # 将控制应用于植物
            ppfd, temp, power, photo_rate = self.plant.step(pwm_optimal, dt)

            # 存储数据（包括用于比较的参考）
            self.time_data.append(current_time)
            self.ppfd_data.append(ppfd)
            self.temp_data.append(temp)
            self.power_data.append(power)
            self.pwm_data.append(pwm_optimal)
            self.photo_data.append(photo_rate)
            self.ppfd_ref_data.append(ppfd_ref_full[k])
            self.temp_ref_data.append(temp_ref_full[k])
            self.photo_ref_data.append(photo_ref_full[k])
            self.power_ref_data.append(power_ref_full[k])
            self.cost_data.append(cost)

            # 打印进度
            if k % 10 == 0:
                temp_status = (
                    "✓"
                    if self.controller.temp_min <= temp <= self.controller.temp_max
                    else "✗"
                )
                print(
                    f"t={current_time:3.0f}s: PWM={pwm_optimal:5.1f}%, "
                    f"PPFD={ppfd:3.0f}, 温度={temp:4.1f}°C {temp_status}, "
                    f"光合作用={photo_rate:4.1f}, 代价={cost:.1e}"
                )
                with open("mppi_log.txt", "a+") as log_file:
                    log_file.write(
                        f"{current_time:.1f},{pwm_optimal:.1f},{ppfd:.1f},{temp:.1f},{photo_rate:.1f},{cost:.1e}\n"
                    )

        print("\n仿真完成！")

        # 分析约束满足情况
        temp_violations = np.sum(
            (np.array(self.temp_data) < self.controller.temp_min)
            | (np.array(self.temp_data) > self.controller.temp_max)
        )
        temp_satisfaction = 100 * (1 - temp_violations / len(self.temp_data))

        print(f"温度约束满足度: {temp_satisfaction:.1f}%")
        print(
            f"实现的温度范围: {np.min(self.temp_data):.1f} 至 {np.max(self.temp_data):.1f}°C"
        )
        print(f"平均光合作用速率: {np.mean(self.photo_data):.2f} μmol/m²/s")
        print(f"总光合作用: {np.sum(self.photo_data):.1f} μmol/m²/s·s")

        return self.get_results()

    def clear_data(self):
        """清除所有数据数组"""
        self.time_data = []
        self.ppfd_data = []
        self.temp_data = []
        self.power_data = []
        self.pwm_data = []
        self.photo_data = []
        self.ppfd_ref_data = []
        self.temp_ref_data = []
        self.photo_ref_data = []
        self.power_ref_data = []
        self.cost_data = []

    def get_results(self):
        """获取仿真结果"""
        return {
            "time": np.array(self.time_data),
            "ppfd": np.array(self.ppfd_data),
            "temp": np.array(self.temp_data),
            "power": np.array(self.power_data),
            "pwm": np.array(self.pwm_data),
            "photosynthesis": np.array(self.photo_data),
            "ppfd_ref": np.array(self.ppfd_ref_data),
            "temp_ref": np.array(self.temp_ref_data),
            "photo_ref": np.array(self.photo_ref_data),
            "power_ref": np.array(self.power_ref_data),
            "cost": np.array(self.cost_data),
        }

    def plot_results(self):
        """绘制MPPI仿真结果，比较实际与参考以及累积指标"""
        results = self.get_results()

        # 计算累积值
        dt = (
            results["time"][1] - results["time"][0] if len(results["time"]) > 1 else 1.0
        )

        # 累积和
        cumulative_pn_mppi = np.cumsum(results["photosynthesis"]) * dt
        cumulative_pn_ref = np.cumsum(results["photo_ref"]) * dt
        cumulative_power_mppi = np.cumsum(results["power"]) * dt
        cumulative_power_ref = np.cumsum(results["power_ref"]) * dt

        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(
            "LED MPPI控制 - 光合作用最大化与参考对比（含累积指标）",
            fontsize=16,
        )

        # PPFD比较
        axes[0, 0].plot(
            results["time"], results["ppfd"], "g-", linewidth=2, label="MPPI (实际)"
        )
        axes[0, 0].plot(
            results["time"],
            results["ppfd_ref"],
            "g--",
            linewidth=2,
            alpha=0.7,
            label="参考",
        )
        axes[0, 0].set_ylabel("PPFD (μmol/m²/s)")
        axes[0, 0].set_title("PPFD: MPPI与参考对比")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 带约束的温度比较
        axes[0, 1].plot(
            results["time"], results["temp"], "r-", linewidth=2, label="MPPI (实际)"
        )
        axes[0, 1].plot(
            results["time"],
            results["temp_ref"],
            "r--",
            linewidth=2,
            alpha=0.7,
            label="参考",
        )
        axes[0, 1].axhline(
            y=self.controller.temp_min,
            color="k",
            linestyle=":",
            alpha=0.7,
            label=f"最小值 ({self.controller.temp_min}°C)",
        )
        axes[0, 1].axhline(
            y=self.controller.temp_max,
            color="k",
            linestyle=":",
            alpha=0.7,
            label=f"最大值 ({self.controller.temp_max}°C)",
        )
        axes[0, 1].set_ylabel("温度 (°C)")
        axes[0, 1].set_title("温度: MPPI与参考对比")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 光合作用比较（关键性能指标）
        axes[0, 2].plot(
            results["time"],
            results["photosynthesis"],
            "orange",
            linewidth=3,
            label="MPPI (最大化)",
        )
        axes[0, 2].plot(
            results["time"],
            results["photo_ref"],
            "orange",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="参考",
        )
        axes[0, 2].set_ylabel("光合作用 (μmol/m²/s)")
        axes[0, 2].set_title("光合作用: MPPI与参考对比")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # PWM控制
        axes[1, 0].plot(results["time"], results["pwm"], "b-", linewidth=2)
        axes[1, 0].axhline(
            y=self.controller.pwm_max,
            color="k",
            linestyle=":",
            alpha=0.7,
            label=f"最大值 ({self.controller.pwm_max}%)",
        )
        axes[1, 0].set_ylabel("PWM (%)")
        axes[1, 0].set_title("控制信号")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 功耗比较
        axes[1, 1].plot(
            results["time"], results["power"], "m-", linewidth=2, label="MPPI"
        )
        axes[1, 1].plot(
            results["time"],
            results["power_ref"],
            "m--",
            linewidth=2,
            alpha=0.7,
            label="参考",
        )
        axes[1, 1].set_ylabel("功率 (W)")
        axes[1, 1].set_title("功耗: MPPI与参考对比")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 代价演化（由于光合作用最大化应为负值）
        axes[1, 2].plot(results["time"], results["cost"], "purple", linewidth=2)
        axes[1, 2].set_ylabel("代价")
        axes[1, 2].set_title("MPPI代价演化")
        axes[1, 2].grid(True, alpha=0.3)

        # 新增：累积光合作用比较
        axes[2, 0].plot(
            results["time"],
            cumulative_pn_mppi,
            "orange",
            linewidth=3,
            label="MPPI (累积)",
        )
        axes[2, 0].plot(
            results["time"],
            cumulative_pn_ref,
            "orange",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="参考 (累积)",
        )
        axes[2, 0].set_ylabel("累积Pn (μmol/m²)")
        axes[2, 0].set_xlabel("时间 (s)")
        axes[2, 0].set_title("累积光合作用: MPPI与参考对比")
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # 新增：累积功耗比较
        axes[2, 1].plot(
            results["time"],
            cumulative_power_mppi,
            "m-",
            linewidth=2,
            label="MPPI (累积)",
        )
        axes[2, 1].plot(
            results["time"],
            cumulative_power_ref,
            "m--",
            linewidth=2,
            alpha=0.7,
            label="参考 (累积)",
        )
        axes[2, 1].set_ylabel("累积功率 (W·s)")
        axes[2, 1].set_xlabel("时间 (s)")
        axes[2, 1].set_title("累积功耗: MPPI与参考对比")
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        # 新增：效率比较（Pn/功率比随时间变化）
        efficiency_mppi = results["photosynthesis"] / np.maximum(
            results["power"], 0.1
        )  # 避免除零
        efficiency_ref = results["photo_ref"] / np.maximum(results["power_ref"], 0.1)

        axes[2, 2].plot(
            results["time"], efficiency_mppi, "c-", linewidth=2, label="MPPI效率"
        )
        axes[2, 2].plot(
            results["time"],
            efficiency_ref,
            "c--",
            linewidth=2,
            alpha=0.7,
            label="参考效率",
        )
        axes[2, 2].set_ylabel("效率 (Pn/功率)")
        axes[2, 2].set_xlabel("时间 (s)")
        axes[2, 2].set_title("能效: MPPI与参考对比")
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 性能指标
        self.print_performance_metrics(
            results,
            cumulative_pn_mppi,
            cumulative_pn_ref,
            cumulative_power_mppi,
            cumulative_power_ref,
        )

    def print_performance_metrics(
        self, results, cum_pn_mppi, cum_pn_ref, cum_power_mppi, cum_power_ref
    ):
        """打印详细的性能指标，比较MPPI与参考，包括累积值"""
        avg_photosynthesis = np.mean(results["photosynthesis"])
        avg_photo_ref = np.mean(results["photo_ref"])
        max_photosynthesis = np.max(results["photosynthesis"])
        total_photosynthesis = cum_pn_mppi[-1]
        total_photo_ref = cum_pn_ref[-1]
        total_power_mppi = cum_power_mppi[-1]
        total_power_ref = cum_power_ref[-1]
        avg_power_mppi = np.mean(results["power"])
        avg_power_ref = np.mean(results["power_ref"])

        # 相对于参考的性能改进
        photo_improvement = ((avg_photosynthesis - avg_photo_ref) / avg_photo_ref) * 100
        total_improvement = (
            (total_photosynthesis - total_photo_ref) / total_photo_ref
        ) * 100
        power_difference = (
            (total_power_mppi - total_power_ref) / total_power_ref
        ) * 100

        # 能效指标
        efficiency_mppi = total_photosynthesis / total_power_mppi
        efficiency_ref = total_photo_ref / total_power_ref
        efficiency_improvement = (
            (efficiency_mppi - efficiency_ref) / efficiency_ref
        ) * 100

        # 用于比较的RMSE
        ppfd_rmse = np.sqrt(np.mean((results["ppfd"] - results["ppfd_ref"]) ** 2))
        temp_rmse = np.sqrt(np.mean((results["temp"] - results["temp_ref"]) ** 2))
        photo_rmse = np.sqrt(
            np.mean((results["photosynthesis"] - results["photo_ref"]) ** 2)
        )
        power_rmse = np.sqrt(np.mean((results["power"] - results["power_ref"]) ** 2))

        temp_violations = np.sum(
            (results["temp"] < self.controller.temp_min)
            | (results["temp"] > self.controller.temp_max)
        )
        temp_satisfaction = 100 * (1 - temp_violations / len(results["temp"]))

        pwm_violations = np.sum(
            (results["pwm"] < self.controller.pwm_min)
            | (results["pwm"] > self.controller.pwm_max)
        )

        print(f"\n" + "=" * 80)
        print(f"MPPI光合作用最大化与参考对比")
        print(f"=" * 80)

        print(f"\n📈 光合作用性能:")
        print(f"  MPPI平均光合作用: {avg_photosynthesis:.2f} μmol/m²/s")
        print(f"  参考平均光合作用: {avg_photo_ref:.2f} μmol/m²/s")
        print(f"  改进: {photo_improvement:+.1f}% 相对于参考")
        print(f"  MPPI最大光合作用: {max_photosynthesis:.2f} μmol/m²/s")

        print(f"\n🔋 累积指标:")
        print(f"  MPPI总光合作用: {total_photosynthesis:.1f} μmol/m²")
        print(f"  参考总光合作用: {total_photo_ref:.1f} μmol/m²")
        print(f"  总Pn改进: {total_improvement:+.1f}% 相对于参考")
        print(f"  MPPI总功耗: {total_power_mppi:.1f} W·s")
        print(f"  参考总功耗: {total_power_ref:.1f} W·s")
        print(f"  功耗差异: {power_difference:+.1f}% 相对于参考")

        print(f"\n⚡ 能效:")
        print(f"  MPPI能效: {efficiency_mppi:.4f} (μmol/m²)/(W·s)")
        print(f"  参考能效: {efficiency_ref:.4f} (μmol/m²)/(W·s)")
        print(
            f"  效率改进: {efficiency_improvement:+.1f}% 相对于参考"
        )
        print(f"  MPPI平均功率: {avg_power_mppi:.1f} W")
        print(f"  参考平均功率: {avg_power_ref:.1f} W")

        print(f"\n📊 比较指标 (RMSE):")
        print(f"  PPFD与参考的偏差: {ppfd_rmse:.1f} μmol/m²/s")
        print(f"  温度与参考的偏差: {temp_rmse:.2f} °C")
        print(f"  光合作用与参考的偏差: {photo_rmse:.2f} μmol/m²/s")
        print(f"  功率与参考的偏差: {power_rmse:.2f} W")

        print(f"\n🎯 约束满足:")
        print(
            f"  温度违规: {temp_violations}/{len(results['temp'])} 步"
        )
        print(f"  温度满足度: {temp_satisfaction:.1f}%")
        print(f"  PWM违规: {pwm_violations} 步")
        print(
            f"  温度范围: {np.min(results['temp']):.1f} 至 {np.max(results['temp']):.1f}°C"
        )

        print(f"\n💡 总结:")
        if total_improvement > 0:
            print(
                f"  ✅ MPPI实现了{total_improvement:.1f}%更高的总光合作用"
            )
        else:
            print(
                f"  ❌ MPPI实现了{total_improvement:.1f}%更低的总光合作用"
            )

        if power_difference < 0:
            print(f"  ✅ MPPI使用了{abs(power_difference):.1f}%更少的功率")
        else:
            print(f"  ⚠️  MPPI使用了{power_difference:.1f}%更多的功率")

        if efficiency_improvement > 0:
            print(f"  ✅ MPPI能效提高了{efficiency_improvement:.1f}%")
        else:
            print(
                f"  ❌ MPPI能效降低了{abs(efficiency_improvement):.1f}%"
            )

        print(f"  最终代价: {results['cost'][-1]:.2e}")
        print(f"  平均PWM: {np.mean(results['pwm']):.1f}%")


# 示例用法
if __name__ == "__main__":
    # 创建植物模型
    plant = LEDPlant(
        base_ambient_temp=22.0,
        max_ppfd=700.0,
        max_power=100.0,
        thermal_resistance=1.2,
        thermal_mass=8.0,
    )

    # 创建用于光合作用最大化的MPPI控制器
    controller = LEDMPPIController(
        plant=plant, horizon=10, num_samples=1000, dt=1.0, temperature=0.5
    )

    # 配置MPPI权重以通过参考跟踪最大化光合作用
    controller.set_weights(
        Q_photo=5.0,  # 光合作用最大化的高权重
        Q_ref=25.0,  # 参考轨迹跟踪的中等权重
        R_pwm=0.001,  # 低控制惩罚
        R_dpwm=0.05,  # 平滑控制
        R_power=0.08,  # 效率的适度功耗惩罚
    )

    # 设置约束
    controller.set_constraints(pwm_min=0.0, pwm_max=100.0, temp_min=20.0, temp_max=29.0)

    # 设置MPPI参数
    controller.set_mppi_params(num_samples=1000, temperature=0.5, pwm_std=10.0)

    # 创建仿真
    simulation = LEDMPPISimulation(plant, controller)

    # 运行光合作用最大化仿真
    print("开始基于MPPI的LED控制以实现光合作用最大化...")
    results = simulation.run_simulation(duration=120, dt=1.0)

    # 绘制结果
    simulation.plot_results()
