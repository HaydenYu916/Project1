import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

# 导入 LED 仿真函数
from led import led_step, led_steady_state

# 导入光合作用预测器
try:
    from pn_prediction.predict import PhotosynthesisPredictor

    PHOTOSYNTHESIS_AVAILABLE = True
except ImportError:
    print("Warning: PhotosynthesisPredictor not available. Using simple model.")
    PHOTOSYNTHESIS_AVAILABLE = False


class LEDPlant:
    """用于 MPPI 的 LED 植物系统模型（基于外部 LED 仿真函数）"""

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

        # 若可用则初始化光合作用预测器
        if PHOTOSYNTHESIS_AVAILABLE:
            try:
                self.photo_predictor = PhotosynthesisPredictor()
                self.use_photo_model = self.photo_predictor.is_loaded
            except:
                self.use_photo_model = False
        else:
            self.use_photo_model = False

    def step(self, pwm_percent, dt=0.1):
        """使用导入的 led_step 函数对 LED 植物模型进行单步更新"""
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

    def get_photosynthesis_rate(self, ppfd, temperature, co2=400, rb_ratio=0.83):
        """优先使用光合作用预测模型；不可用时回退到简化模型"""
        if self.use_photo_model:
            try:
                return self.photo_predictor.predict(ppfd, co2, temperature, rb_ratio)
            except Exception as e:
                print(f"Warning: Photosynthesis prediction failed: {e}")
                return self.simple_photosynthesis_model(ppfd, temperature)
        else:
            print(
                "Warning: Using simple photosynthesis model - prediction model not available"
            )
            return self.simple_photosynthesis_model(ppfd, temperature)

    def simple_photosynthesis_model(self, ppfd, temperature):
        """简化光合作用模型（预测失败时的回退方案）"""
        ppfd_max = 1000  # μmol/m²/s
        pn_max = 25  # μmol/m²/s
        km = 300  # μmol/m²/s

        # 温度效应（最佳约 25°C）
        temp_factor = np.exp(-0.01 * (temperature - 25) ** 2)

        # 光照响应
        pn = (pn_max * ppfd / (km + ppfd)) * temp_factor

        return max(0, pn)

    def predict(self, pwm_sequence, initial_temp, dt=0.1):
        """给定 PWM 序列，预测未来状态轨迹（PPFD/温度/功率/光合速率）"""
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
    """LED 系统的 MPPI 控制器——以最大化光合速率为目标"""

    def __init__(self, plant, horizon=10, num_samples=1000, dt=0.1, temperature=1.0):
        self.plant = plant
        self.horizon = horizon
        self.num_samples = num_samples
        self.dt = dt
        self.temperature = temperature  # MPPI 温度参数

        # 代价函数权重——聚焦于光合最大化
        self.Q_photo = 10.0  # 对光合最大化赋予较高权重
        self.R_pwm = 0.001  # 较低的控制能量惩罚
        self.R_dpwm = 0.05  # 控制变化平滑性惩罚
        self.R_power = 0.01  # 功耗惩罚

        # 约束
        self.pwm_min = 0.0
        self.pwm_max = 80.0
        self.temp_min = 20.0
        self.temp_max = 29.0

        # 控制参数
        self.pwm_std = 15.0  # PWM 采样的标准差
        self.pwm_prev = 0.0

        # 约束违背惩罚
        self.temp_penalty = 100000.0  # 温度越界的高惩罚
        self.pwm_penalty = 1000.0  # PWM 约束违背惩罚

    def set_weights(self, Q_photo=10.0, R_pwm=0.001, R_dpwm=0.05, R_power=0.01):
        """设置 MPPI 代价函数权重（以最大化光合作用为主目标）"""
        self.Q_photo = Q_photo
        self.R_pwm = R_pwm
        self.R_dpwm = R_dpwm
        self.R_power = R_power

    def set_constraints(self, pwm_min=0.0, pwm_max=100.0, temp_min=18.0, temp_max=30.0):
        """设置 MPPI 约束（PWM 与温度上下限）"""
        self.pwm_min = pwm_min
        self.pwm_max = pwm_max
        self.temp_min = temp_min
        self.temp_max = temp_max

    def set_mppi_params(self, num_samples=1000, temperature=1.0, pwm_std=15.0):
        """设置 MPPI 算法参数（采样数量、温度系数、PWM 噪声标准差）"""
        self.num_samples = num_samples
        self.temperature = temperature
        self.pwm_std = pwm_std

    def sample_control_sequences(self, mean_sequence):
        """围绕均值序列进行控制序列采样并裁剪到约束范围"""
        # 为采样创建噪声
        noise = np.random.normal(0, self.pwm_std, (self.num_samples, self.horizon))

        # 在均值序列上叠加噪声
        samples = mean_sequence[np.newaxis, :] + noise

        # 通过裁剪施加约束
        samples = np.clip(samples, self.pwm_min, self.pwm_max)

        return samples

    def compute_cost(self, pwm_sequence, current_temp):
        """计算单条 PWM 序列的代价——以最大化光合作用为目标"""
        try:
            # 预测未来状态
            ppfd_pred, temp_pred, power_pred, photo_pred = self.plant.predict(
                pwm_sequence, current_temp, self.dt
            )

            cost = 0.0

            # 主要目标：在预测范围内最大化光合作用
            for k in range(self.horizon):
                # 以负的光合值计入以实现最大化（最小化其相反数）
                cost -= self.Q_photo * photo_pred[k]

                # 温度硬约束惩罚
                if temp_pred[k] > self.temp_max:
                    violation = temp_pred[k] - self.temp_max
                    cost += self.temp_penalty * violation**2
                if temp_pred[k] < self.temp_min:
                    violation = self.temp_min - temp_pred[k]
                    cost += self.temp_penalty * violation**2

            # 控制能量代价
            for k in range(self.horizon):
                cost += self.R_pwm * pwm_sequence[k] ** 2
                cost += self.R_power * power_pred[k] ** 2

            # 控制平滑性
            prev_pwm = self.pwm_prev
            for k in range(self.horizon):
                dpwm = pwm_sequence[k] - prev_pwm
                cost += self.R_dpwm * dpwm**2
                prev_pwm = pwm_sequence[k]

            # PWM 约束惩罚
            for k in range(self.horizon):
                if pwm_sequence[k] > self.pwm_max:
                    violation = pwm_sequence[k] - self.pwm_max
                    cost += self.pwm_penalty * violation**2
                if pwm_sequence[k] < self.pwm_min:
                    violation = self.pwm_min - pwm_sequence[k]
                    cost += self.pwm_penalty * violation**2

            return cost

        except Exception as e:
            # 对无效序列返回极大代价
            return 1e10

    def solve(self, current_temp, mean_sequence=None):
        """求解 MPPI 优化问题（最大化光合速率，返回最优首个 PWM）"""

        # 若未提供则初始化均值序列
        if mean_sequence is None:
            # 以适中的 PWM 值起始
            mean_sequence = np.ones(self.horizon) * min(40.0, self.pwm_max * 0.5)

        # 采样控制序列
        control_samples = self.sample_control_sequences(mean_sequence)

        # 计算所有样本的代价
        costs = np.zeros(self.num_samples)

        for i in range(self.num_samples):
            costs[i] = self.compute_cost(control_samples[i], current_temp)

        # 处理无穷或 NaN 的代价
        costs = np.nan_to_num(costs, nan=1e10, posinf=1e10)

        # 用 softmax 计算权重
        min_cost = np.min(costs)
        exp_costs = np.exp(-(costs - min_cost) / self.temperature)
        weights = exp_costs / np.sum(exp_costs)

        # 计算控制序列的加权平均
        optimal_sequence = np.sum(weights[:, np.newaxis] * control_samples, axis=0)

        # 应用最终约束
        optimal_sequence = np.clip(optimal_sequence, self.pwm_min, self.pwm_max)

        # 取第一步控制量
        optimal_pwm = optimal_sequence[0]

        # 温度安全检查
        _, temp_check, _, _ = self.plant.predict([optimal_pwm], current_temp, self.dt)
        if temp_check[0] > self.temp_max:
            # 紧急降幅
            optimal_pwm = max(self.pwm_min, optimal_pwm * 0.7)
            print(
                f"MPPI: Emergency PWM reduction to {optimal_pwm:.1f}% due to temperature risk"
            )

        self.pwm_prev = optimal_pwm

        # 返回附加信息
        success = True
        best_cost = np.min(costs)

        return optimal_pwm, optimal_sequence, success, best_cost, weights


class LEDMPPISimulation:
    """用于光合最大化的 MPPI 仿真环境，并与参考轨迹进行对比"""

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
        self.ppfd_ref_data = []  # 用于对比
        self.temp_ref_data = []  # 用于对比
        self.photo_ref_data = []  # 用于对比
        self.power_ref_data = []  # 参考功耗
        self.cost_data = []

        # 初始化光合预测器用于评估
        if PHOTOSYNTHESIS_AVAILABLE:
            try:
                self.pn = PhotosynthesisPredictor()
            except:
                self.pn = None
        else:
            self.pn = None

    def create_reference_trajectory(self, duration, dt):
        """生成用于对比的参考轨迹（PPFD/温度/光合/功率）"""
        time_points = np.arange(0, duration, dt)

        ppfd_ref = []
        temp_ref = []
        photo_ref = []
        power_ref = []

        current_temp = self.plant.base_ambient_temp
        for t in time_points:
            # PPFD 参考：类日照周期
            ppfd_target = 300

            # 温度参考：使用 LED 步进并随环境温度演化
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

            # 光合作用参考（仅用于对比）
            if self.pn is not None:
                photo_target = self.pn.predict(ppfd_target, 400, temp_target, 0.83)
            else:
                # 参考使用简化模型
                ppfd_max = 1000
                pn_max = 25
                km = 300
                temp_factor = np.exp(-0.01 * (temp_target - 25) ** 2)
                photo_target = max(
                    0, (pn_max * ppfd_target / (km + ppfd_target)) * temp_factor
                )

            # 计算实现参考 PPFD 所需的 PWM
            # 简化估算——实际会更复杂
            # pwm_target = min(80.0, (ppfd_target / self.plant.max_ppfd) * 100)
            # pwm_target = ppfd_target / self.plant.max_ppfd * 100

            # 计算参考功耗
            # power_target = (pwm_target / 100) * self.plant.max_power
            # PPFD 输出
            pwm_fraction = ppfd_target / self.plant.max_ppfd

            # LED 效率
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
        """运行 MPPI 仿真（目标为光合最大化，同时记录参考对比）"""

        print("Starting LED MPPI Simulation - Photosynthesis Maximization")
        print("=" * 60)
        print("Note: References are used for comparison only, not for tracking")
        print(
            f"MPPI Parameters: {self.controller.num_samples} samples, temp={self.controller.temperature}"
        )
        print(
            f"Temperature constraints: ({self.controller.temp_min}, {self.controller.temp_max})°C"
        )
        print(
            f"PWM constraints: ({self.controller.pwm_min}, {self.controller.pwm_max})%"
        )

        if self.plant.use_photo_model:
            print("Using trained photosynthesis model")
        else:
            print("Using simple photosynthesis model")

        # 生成用于对比的参考轨迹
        ppfd_ref_full, temp_ref_full, photo_ref_full, power_ref_full = (
            self.create_reference_trajectory(duration, dt)
        )

        # 重置植物模型
        self.plant.ambient_temp = self.plant.base_ambient_temp
        self.plant.time = 0.0

        # 重置控制器
        self.controller.pwm_prev = 0.0

        # 清空数据
        self.clear_data()

        # 初始化 MPPI 的均值序列
        mean_sequence = np.ones(self.controller.horizon) * 30.0
        log_file = "mppi_log.txt"
        if os.path.exists(log_file):
            os.remove(log_file)
        # 仿真循环
        steps = int(duration / dt)
        for k in range(steps):
            current_time = k * dt

            # 求解 MPPI 以最大化光合（不跟踪参考）
            pwm_optimal, optimal_sequence, success, cost, weights = (
                self.controller.solve(self.plant.ambient_temp, mean_sequence)
            )

            # 更新下一次迭代的均值序列（滚动时域）
            if len(optimal_sequence) > 1:
                mean_sequence = np.concatenate(
                    [optimal_sequence[1:], [optimal_sequence[-1]]]
                )
            else:
                mean_sequence = optimal_sequence

            # 将控制作用到植物模型
            ppfd, temp, power, photo_rate = self.plant.step(pwm_optimal, dt)

            # 存储数据（包括用于对比的参考）
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
                    f"PPFD={ppfd:3.0f}, Temp={temp:4.1f}°C {temp_status}, "
                    f"Photo={photo_rate:4.1f}, Cost={cost:.1e}"
                )
                with open("mppi_log.txt", "a+") as log_file:
                    log_file.write(
                        f"{current_time:.1f},{pwm_optimal:.1f},{ppfd:.1f},{temp:.1f},{photo_rate:.1f},{cost:.1e}\n"
                    )

        print("\nSimulation completed!")

        # 分析约束满足度
        temp_violations = np.sum(
            (np.array(self.temp_data) < self.controller.temp_min)
            | (np.array(self.temp_data) > self.controller.temp_max)
        )
        temp_satisfaction = 100 * (1 - temp_violations / len(self.temp_data))

        print(f"Temperature constraint satisfaction: {temp_satisfaction:.1f}%")
        print(
            f"Temperature range achieved: {np.min(self.temp_data):.1f} to {np.max(self.temp_data):.1f}°C"
        )
        print(f"Average photosynthesis rate: {np.mean(self.photo_data):.2f} μmol/m²/s")
        print(f"Total photosynthesis: {np.sum(self.photo_data):.1f} μmol/m²/s·s")

        return self.get_results()

    def clear_data(self):
        """清空内部数据缓存"""
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
        """返回仿真结果字典"""
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
        """绘制仿真结果：实际 vs 参考，并展示累计指标"""
        results = self.get_results()

        # 计算累计量
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
            "LED MPPI Control - Photosynthesis Maximization vs Reference (with Accumulated Metrics)",
            fontsize=16,
        )

        # PPFD 对比
        axes[0, 0].plot(
            results["time"], results["ppfd"], "g-", linewidth=2, label="MPPI (Actual)"
        )
        axes[0, 0].plot(
            results["time"],
            results["ppfd_ref"],
            "g--",
            linewidth=2,
            alpha=0.7,
            label="Reference",
        )
        axes[0, 0].set_ylabel("PPFD (μmol/m²/s)")
        axes[0, 0].set_title("PPFD: MPPI vs Reference")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 温度对比（含约束）
        axes[0, 1].plot(
            results["time"], results["temp"], "r-", linewidth=2, label="MPPI (Actual)"
        )
        axes[0, 1].plot(
            results["time"],
            results["temp_ref"],
            "r--",
            linewidth=2,
            alpha=0.7,
            label="Reference",
        )
        axes[0, 1].axhline(
            y=self.controller.temp_min,
            color="k",
            linestyle=":",
            alpha=0.7,
            label=f"Min ({self.controller.temp_min}°C)",
        )
        axes[0, 1].axhline(
            y=self.controller.temp_max,
            color="k",
            linestyle=":",
            alpha=0.7,
            label=f"Max ({self.controller.temp_max}°C)",
        )
        axes[0, 1].set_ylabel("Temperature (°C)")
        axes[0, 1].set_title("Temperature: MPPI vs Reference")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 光合对比（关键性能指标）
        axes[0, 2].plot(
            results["time"],
            results["photosynthesis"],
            "orange",
            linewidth=3,
            label="MPPI (Maximized)",
        )
        axes[0, 2].plot(
            results["time"],
            results["photo_ref"],
            "orange",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Reference",
        )
        axes[0, 2].set_ylabel("Photosynthesis (μmol/m²/s)")
        axes[0, 2].set_title("Photosynthesis: MPPI vs Reference")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # PWM 控制
        axes[1, 0].plot(results["time"], results["pwm"], "b-", linewidth=2)
        axes[1, 0].axhline(
            y=self.controller.pwm_max,
            color="k",
            linestyle=":",
            alpha=0.7,
            label=f"Max ({self.controller.pwm_max}%)",
        )
        axes[1, 0].set_ylabel("PWM (%)")
        axes[1, 0].set_title("Control Signal")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 功耗对比
        axes[1, 1].plot(
            results["time"], results["power"], "m-", linewidth=2, label="MPPI"
        )
        axes[1, 1].plot(
            results["time"],
            results["power_ref"],
            "m--",
            linewidth=2,
            alpha=0.7,
            label="Reference",
        )
        axes[1, 1].set_ylabel("Power (W)")
        axes[1, 1].set_title("Power Consumption: MPPI vs Reference")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 代价随时间变化（因最大化光合应趋于负）
        axes[1, 2].plot(results["time"], results["cost"], "purple", linewidth=2)
        axes[1, 2].set_ylabel("Cost")
        axes[1, 2].set_title("MPPI Cost Evolution")
        axes[1, 2].grid(True, alpha=0.3)

        # 新增：累计光合对比
        axes[2, 0].plot(
            results["time"],
            cumulative_pn_mppi,
            "orange",
            linewidth=3,
            label="MPPI (Accumulated)",
        )
        axes[2, 0].plot(
            results["time"],
            cumulative_pn_ref,
            "orange",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Reference (Accumulated)",
        )
        axes[2, 0].set_ylabel("Accumulated Pn (μmol/m²)")
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].set_title("Accumulated Photosynthesis: MPPI vs Reference")
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # 新增：累计功耗对比
        axes[2, 1].plot(
            results["time"],
            cumulative_power_mppi,
            "m-",
            linewidth=2,
            label="MPPI (Accumulated)",
        )
        axes[2, 1].plot(
            results["time"],
            cumulative_power_ref,
            "m--",
            linewidth=2,
            alpha=0.7,
            label="Reference (Accumulated)",
        )
        axes[2, 1].set_ylabel("Accumulated Power (W·s)")
        axes[2, 1].set_xlabel("Time (s)")
        axes[2, 1].set_title("Accumulated Power Usage: MPPI vs Reference")
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        # 新增：效率对比（随时间的 Pn/功率 比）
        efficiency_mppi = results["photosynthesis"] / np.maximum(
            results["power"], 0.1
        )  # 避免除以零
        efficiency_ref = results["photo_ref"] / np.maximum(results["power_ref"], 0.1)

        axes[2, 2].plot(
            results["time"], efficiency_mppi, "c-", linewidth=2, label="MPPI Efficiency"
        )
        axes[2, 2].plot(
            results["time"],
            efficiency_ref,
            "c--",
            linewidth=2,
            alpha=0.7,
            label="Reference Efficiency",
        )
        axes[2, 2].set_ylabel("Efficiency (Pn/Power)")
        axes[2, 2].set_xlabel("Time (s)")
        axes[2, 2].set_title("Energy Efficiency: MPPI vs Reference")
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
        """打印 MPPI 与参考的性能对比指标（包含累计值与效率）"""
        avg_photosynthesis = np.mean(results["photosynthesis"])
        avg_photo_ref = np.mean(results["photo_ref"])
        max_photosynthesis = np.max(results["photosynthesis"])
        total_photosynthesis = cum_pn_mppi[-1]
        total_photo_ref = cum_pn_ref[-1]
        total_power_mppi = cum_power_mppi[-1]
        total_power_ref = cum_power_ref[-1]
        avg_power_mppi = np.mean(results["power"])
        avg_power_ref = np.mean(results["power_ref"])

        # 相对参考的性能提升
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

        # 用于对比的 RMSE
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
        print(f"MPPI PHOTOSYNTHESIS MAXIMIZATION vs REFERENCE COMPARISON")
        print(f"=" * 80)

        print(f"\n📈 PHOTOSYNTHESIS PERFORMANCE:")
        print(f"  MPPI Average Photosynthesis: {avg_photosynthesis:.2f} μmol/m²/s")
        print(f"  Reference Average Photosynthesis: {avg_photo_ref:.2f} μmol/m²/s")
        print(f"  Improvement: {photo_improvement:+.1f}% over reference")
        print(f"  MPPI Maximum Photosynthesis: {max_photosynthesis:.2f} μmol/m²/s")

        print(f"\n🔋 ACCUMULATED METRICS:")
        print(f"  MPPI Total Photosynthesis: {total_photosynthesis:.1f} μmol/m²")
        print(f"  Reference Total Photosynthesis: {total_photo_ref:.1f} μmol/m²")
        print(f"  Total Pn Improvement: {total_improvement:+.1f}% over reference")
        print(f"  MPPI Total Power Consumption: {total_power_mppi:.1f} W·s")
        print(f"  Reference Total Power Consumption: {total_power_ref:.1f} W·s")
        print(f"  Power Usage Difference: {power_difference:+.1f}% vs reference")

        print(f"\n⚡ ENERGY EFFICIENCY:")
        print(f"  MPPI Energy Efficiency: {efficiency_mppi:.4f} (μmol/m²)/(W·s)")
        print(f"  Reference Energy Efficiency: {efficiency_ref:.4f} (μmol/m²)/(W·s)")
        print(
            f"  Efficiency Improvement: {efficiency_improvement:+.1f}% over reference"
        )
        print(f"  MPPI Average Power: {avg_power_mppi:.1f} W")
        print(f"  Reference Average Power: {avg_power_ref:.1f} W")

        print(f"\n📊 COMPARISON METRICS (RMSE):")
        print(f"  PPFD deviation from reference: {ppfd_rmse:.1f} μmol/m²/s")
        print(f"  Temperature deviation from reference: {temp_rmse:.2f} °C")
        print(f"  Photosynthesis deviation from reference: {photo_rmse:.2f} μmol/m²/s")
        print(f"  Power deviation from reference: {power_rmse:.2f} W")

        print(f"\n🎯 CONSTRAINT SATISFACTION:")
        print(
            f"  Temperature violations: {temp_violations}/{len(results['temp'])} steps"
        )
        print(f"  Temperature satisfaction: {temp_satisfaction:.1f}%")
        print(f"  PWM violations: {pwm_violations} steps")
        print(
            f"  Temperature range: {np.min(results['temp']):.1f} to {np.max(results['temp']):.1f}°C"
        )

        print(f"\n💡 SUMMARY:")
        if total_improvement > 0:
            print(
                f"  ✅ MPPI achieved {total_improvement:.1f}% higher total photosynthesis"
            )
        else:
            print(
                f"  ❌ MPPI achieved {total_improvement:.1f}% lower total photosynthesis"
            )

        if power_difference < 0:
            print(f"  ✅ MPPI used {abs(power_difference):.1f}% less power")
        else:
            print(f"  ⚠️  MPPI used {power_difference:.1f}% more power")

        if efficiency_improvement > 0:
            print(f"  ✅ MPPI was {efficiency_improvement:.1f}% more energy efficient")
        else:
            print(
                f"  ❌ MPPI was {abs(efficiency_improvement):.1f}% less energy efficient"
            )

        print(f"  Final Cost: {results['cost'][-1]:.2e}")
        print(f"  Average PWM: {np.mean(results['pwm']):.1f}%")


# 使用示例
if __name__ == "__main__":
    # 创建植物模型
    plant = LEDPlant(
        base_ambient_temp=23.0,
        max_ppfd=600.0,
        max_power=86.4,
        thermal_resistance=0.05,
        thermal_mass=150.0,
    )

    # 创建用于光合最大化的 MPPI 控制器
    controller = LEDMPPIController(
        plant=plant, horizon=10, num_samples=1000, dt=1.0, temperature=0.5
    )

    # 配置用于光合最大化的 MPPI 权重
    controller.set_weights(
        Q_photo=10.0,  # 对光合最大化的高权重
        R_pwm=0.001,  # 较低的控制惩罚
        R_dpwm=0.05,  # 平滑控制
        R_power=0.01,  # 功耗惩罚
    )

    # 设置约束
    controller.set_constraints(pwm_min=0.0, pwm_max=70.0, temp_min=20.0, temp_max=29.0)

    # 设置 MPPI 参数
    controller.set_mppi_params(num_samples=1000, temperature=0.5, pwm_std=10.0)

    # 创建仿真环境
    simulation = LEDMPPISimulation(plant, controller)

    # 运行仿真以最大化光合
    print("Starting MPPI-based LED control for photosynthesis maximization...")
    results = simulation.run_simulation(duration=120, dt=1.0)

    # 绘制结果
    simulation.plot_results()
