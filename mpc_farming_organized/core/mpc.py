import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
import os

warnings.filterwarnings("ignore")

# 从更新的代码导入LED仿真函数
from led import led_step, led_steady_state

# 导入光合作用预测器
try:
    from pn_prediction.predict import PhotosynthesisPredictor

    PHOTOSYNTHESIS_AVAILABLE = True
except ImportError:
    print("警告：PhotosynthesisPredictor不可用。使用简单模型。")
    PHOTOSYNTHESIS_AVAILABLE = False


class LEDPlant:
    """用于MPC的LED植物模型，使用导入的LED函数"""

    def __init__(
        self,
        base_ambient_temp=25.0,
        max_ppfd=500.0,
        max_power=100.0,
        thermal_resistance=2.5,
        thermal_mass=0.5,
    ):
        self.base_ambient_temp = base_ambient_temp  # 环境基准温度
        self.max_ppfd = max_ppfd                    # 最大PPFD
        self.max_power = max_power                  # 最大功率
        self.thermal_resistance = thermal_resistance # 热阻
        self.thermal_mass = thermal_mass            # 热容

        # 当前状态
        self.ambient_temp = base_ambient_temp       # 环境温度
        self.time = 0.0                            # 时间

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
        """使用导入的led_step函数进行LED植物的单步仿真"""
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

        # 如果模型可用，计算光合作用速率
        photosynthesis_rate = self.get_photosynthesis_rate(ppfd, new_ambient_temp)

        return ppfd, new_ambient_temp, power, photosynthesis_rate

    def get_photosynthesis_rate(self, ppfd, temperature, co2=400, rb_ratio=0.83):
        """使用预测器或简单模型获取光合作用速率"""
        if self.use_photo_model:
            try:
                return self.photo_predictor.predict(ppfd, co2, temperature, rb_ratio)
            except Exception as e:
                print(f"警告: 光合作用预测失败: {e}")
                # Fallback to simple model
                return self.simple_photosynthesis_model(ppfd, temperature)
        else:
            return self.simple_photosynthesis_model(ppfd, temperature)

    def simple_photosynthesis_model(self, ppfd, temperature):
        """作为备用的简单光合作用模型"""
        # Simplified Michaelis-Menten model with temperature effect
        ppfd_max = 1000  # μmol/m²/s
        pn_max = 25  # μmol/m²/s
        km = 300  # μmol/m²/s

        # Temperature effect (optimal around 25°C)
        temp_factor = np.exp(-0.01 * (temperature - 25) ** 2)

        # Light response
        pn = (pn_max * ppfd / (km + ppfd)) * temp_factor

        return max(0, pn)

    def predict(self, pwm_sequence, initial_temp, dt=0.1):
        """Predict future states given PWM sequence"""
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


class LEDMPCController:
    """Model Predictive Controller for LED system with photosynthesis optimization"""

    def __init__(self, plant, prediction_horizon=10, control_horizon=5, dt=0.1):
        self.plant = plant
        self.N_pred = prediction_horizon
        self.N_ctrl = control_horizon
        self.dt = dt

        # Weights for cost function - Prioritize photosynthesis maximization
        self.Q_ppfd = 0.5  # PPFD tracking weight (reduced)
        self.Q_temp = 0.5  # Temperature tracking weight (reduced)
        self.Q_photo = 5.0  # Photosynthesis optimization weight (increased)
        self.R_pwm = 0.01  # PWM penalty weight
        self.R_dpwm = 0.1  # PWM change penalty weight
        self.R_power = 0.001  # Power penalty weight

        # 约束条件 - Updated ambient temperature range to (20,30)°C
        self.pwm_min = 0.0
        self.pwm_max = 100.0
        self.dpwm_max = 20.0
        self.temp_min = 20.0  # Min ambient temperature (updated)
        self.temp_max = 30.0  # Max ambient temperature (updated)

        # Previous control
        self.pwm_prev = 0.0

    def set_weights(
        self, Q_ppfd=0.5, Q_temp=0.5, Q_photo=5.0, R_pwm=0.01, R_dpwm=0.1, R_power=0.001
    ):
        """Set MPC weights"""
        self.Q_ppfd = Q_ppfd
        self.Q_temp = Q_temp
        self.Q_photo = Q_photo
        self.R_pwm = R_pwm
        self.R_dpwm = R_dpwm
        self.R_power = R_power

    def set_constraints(
        self, pwm_min=0.0, pwm_max=100.0, dpwm_max=20.0, temp_min=20.0, temp_max=30.0
    ):
        """Set MPC constraints - Default ambient temp range (20,30)°C"""
        self.pwm_min = pwm_min
        self.pwm_max = pwm_max
        self.dpwm_max = dpwm_max
        self.temp_min = temp_min
        self.temp_max = temp_max

    def cost_function(self, pwm_sequence, current_temp, ppfd_ref, temp_ref, photo_ref):
        """MPC cost function with strong temperature penalty to enforce constraints"""
        # Predict future states
        ppfd_pred, temp_pred, power_pred, photo_pred = self.plant.predict(
            pwm_sequence, current_temp, self.dt
        )

        cost = 0.0

        # Objective: Maximize photosynthesis only
        for k in range(self.N_pred):
            # Reduced temperature constraint penalties (soft constraints)
            temp_violation_penalty = 100.0  # Reduced penalty for constraint violations
            if temp_pred[k] > self.temp_max:
                violation = temp_pred[k] - self.temp_max
                cost += temp_violation_penalty * violation**2
            if temp_pred[k] < self.temp_min:
                violation = self.temp_min - temp_pred[k]
                cost += temp_violation_penalty * violation**2

            # Maximize photosynthesis (primary and only tracking objective)
            cost -= self.Q_photo * photo_pred[k]

        # Control effort penalties
        for k in range(self.N_ctrl):
            cost += self.R_pwm * pwm_sequence[k] ** 2
            cost += self.R_power * power_pred[k] ** 2

        # Control change penalty
        prev_pwm = self.pwm_prev
        for k in range(self.N_ctrl):
            dpwm = pwm_sequence[k] - prev_pwm
            cost += self.R_dpwm * dpwm**2
            prev_pwm = pwm_sequence[k]

        return cost

    def constraints(self, pwm_sequence, current_temp):
        """MPC constraints - Ambient temperature constrained to (20,30)°C"""
        constraints = []

        # PWM bounds
        for k in range(self.N_ctrl):
            constraints.append(pwm_sequence[k] - self.pwm_min)
            constraints.append(self.pwm_max - pwm_sequence[k])

        # PWM change rate constraints
        prev_pwm = self.pwm_prev
        for k in range(self.N_ctrl):
            dpwm = pwm_sequence[k] - prev_pwm
            constraints.append(self.dpwm_max - abs(dpwm))
            prev_pwm = pwm_sequence[k]

        # Ambient temperature constraints (20-30°C) with safety margins
        _, temp_pred, _, _ = self.plant.predict(pwm_sequence, current_temp, self.dt)

        # Add temperature constraints with penalties for violations
        for k in range(self.N_pred):
            # Soft constraints with larger margins for feasibility
            temp_margin = 1.0  # Allow 1°C margin for solver feasibility
            constraints.append(
                temp_pred[k] - (self.temp_min - temp_margin)
            )  # temp >= 19°C
            constraints.append(
                (self.temp_max + temp_margin) - temp_pred[k]
            )  # temp <= 31°C

        return np.array(constraints)

    def solve(
        self,
        current_temp,
        ppfd_ref=None,
        temp_ref=None,
        photo_ref=None,
        initial_guess=None,
    ):
        """Solve MPC optimization problem"""

        # Extend references to prediction horizon if needed
        def extend_ref(ref):
            if ref is None:
                return None
            if len(ref) < self.N_pred:
                return np.concatenate([ref, [ref[-1]] * (self.N_pred - len(ref))])
            return ref[: self.N_pred]

        ppfd_ref = extend_ref(ppfd_ref)
        temp_ref = extend_ref(temp_ref)
        photo_ref = extend_ref(photo_ref)

        # Initial guess
        if initial_guess is None:
            initial_guess = np.ones(self.N_ctrl) * 50.0

        # Define optimization problem
        def objective(u):
            u_extended = np.concatenate([u, [u[-1]] * (self.N_pred - self.N_ctrl)])
            return self.cost_function(
                u_extended, current_temp, ppfd_ref, temp_ref, photo_ref
            )

        def constraint_func(u):
            u_extended = np.concatenate([u, [u[-1]] * (self.N_pred - self.N_ctrl)])
            return self.constraints(u_extended, current_temp)

        # Bounds
        bounds = [(self.pwm_min, self.pwm_max) for _ in range(self.N_ctrl)]

        # Constraint dictionary
        constraint_dict = {"type": "ineq", "fun": constraint_func}

        # Solve optimization with improved robustness
        try:
            # Try multiple initial guesses if optimization fails
            initial_guesses = [
                initial_guess,
                np.ones(self.N_ctrl) * 30.0,  # Conservative guess
                np.ones(self.N_ctrl) * 10.0,  # Very low guess
                np.zeros(self.N_ctrl),  # Zero guess
            ]

            best_result = None
            best_cost = float("inf")

            for guess in initial_guesses:
                try:
                    result = minimize(
                        objective,
                        guess,
                        method="SLSQP",
                        bounds=bounds,
                        constraints=constraint_dict,
                        options={"maxiter": 200, "ftol": 1e-6},
                    )

                    if result.success and result.fun < best_cost:
                        best_result = result
                        best_cost = result.fun

                except:
                    continue

            if best_result is not None and best_result.success:
                optimal_pwm = best_result.x[0]

                # Safety check: if temperature will exceed limits, reduce PWM
                _, temp_check, _, _ = self.plant.predict(
                    [optimal_pwm], current_temp, self.dt
                )
                if temp_check[0] > self.temp_max:
                    # Emergency reduction
                    optimal_pwm = max(0, optimal_pwm * 0.5)
                    print(
                        f"Emergency PWM reduction to {optimal_pwm:.1f}% due to temperature risk"
                    )

                self.pwm_prev = optimal_pwm
                return optimal_pwm, best_result.x, True
            else:
                # Fallback: reduce PWM from previous value
                fallback_pwm = max(0, self.pwm_prev * 0.8)
                print(
                    f"MPC optimization failed, using fallback PWM: {fallback_pwm:.1f}%"
                )
                self.pwm_prev = fallback_pwm
                return fallback_pwm, initial_guess, False

        except Exception as e:
            print(f"MPC optimization error: {e}")
            # Emergency fallback: significantly reduce PWM
            emergency_pwm = max(0, self.pwm_prev * 0.5)
            print(f"Emergency PWM reduction to {emergency_pwm:.1f}%")
            self.pwm_prev = emergency_pwm
            return emergency_pwm, initial_guess, False


class LEDMPCSimulation:
    """Complete MPC simulation environment with photosynthesis optimization"""

    def __init__(self, plant, controller):
        self.plant = plant
        self.controller = controller

        # Data storage
        self.time_data = []
        self.ppfd_data = []
        self.temp_data = []
        self.power_data = []
        self.pwm_data = []
        self.photo_data = []
        self.ppfd_ref_data = []
        self.temp_ref_data = []
        self.photo_ref_data = []

        # Initialize photosynthesis predictor for evaluation
        if PHOTOSYNTHESIS_AVAILABLE:
            try:
                self.pn = PhotosynthesisPredictor()
            except:
                self.pn = None
        else:
            self.pn = None

    def create_reference_trajectory(self, duration, dt):
        """Create reference trajectories with ambient temp in (20,30)°C range"""
        time_points = np.arange(0, duration, dt)

        ppfd_ref = []
        temp_ref = []
        photo_ref = []

        for t in time_points:
            # PPFD reference: daily light cycle
            ppfd_target = 300

            # Temperature reference: keep within safe range
            temp_target = (
                22.0 + (t / duration) * 6.0
            )  # Linear increase from 22°C to 28°C

            # Photosynthesis reference (for comparison only)
            if self.pn is not None:
                photo_target = self.pn.predict(ppfd_target, 400, temp_target, 0.83)
            else:
                # Use simple model for reference
                pn_max = 25
                km = 300
                temp_factor = np.exp(-0.01 * (temp_target - 25) ** 2)
                photo_target = max(
                    0, (pn_max * ppfd_target / (km + ppfd_target)) * temp_factor
                )

            ppfd_ref.append(ppfd_target)
            temp_ref.append(temp_target)
            photo_ref.append(photo_target)

        return np.array(ppfd_ref), np.array(temp_ref), np.array(photo_ref)

    def run_simulation(self, duration=120, dt=1.0):
        """Run complete MPC simulation with CSV logging"""

        print("Starting LED MPC Simulation with Photosynthesis Optimization")
        print("=" * 60)
        print("Ambient temperature constrained to (20,30)°C")

        if self.plant.use_photo_model:
            print("Using trained photosynthesis model")
        else:
            print("Using simple photosynthesis model")

        # Initialize CSV log file
        log_file = "mpc_log.txt"
        if os.path.exists(log_file):
            os.remove(log_file)

        print(f"Logging data to: {log_file}")

        # Create reference trajectories
        ppfd_ref_full, temp_ref_full, photo_ref_full = self.create_reference_trajectory(
            duration, dt
        )

        # Reset plant
        self.plant.ambient_temp = self.plant.base_ambient_temp
        self.plant.time = 0.0

        # Reset controller
        self.controller.pwm_prev = 0.0

        # Clear data
        self.clear_data()

        # Simulation loop
        steps = int(duration / dt)
        for k in range(steps):
            current_time = k * dt

            # Get current references
            k_end = min(k + self.controller.N_pred, len(ppfd_ref_full))
            ppfd_ref = ppfd_ref_full[k:k_end]
            temp_ref = temp_ref_full[k:k_end]
            photo_ref = photo_ref_full[k:k_end]

            # Solve MPC
            pwm_optimal, pwm_sequence, success = self.controller.solve(
                self.plant.ambient_temp, ppfd_ref, temp_ref, photo_ref
            )

            # Apply control to plant
            ppfd, temp, power, photo_rate = self.plant.step(pwm_optimal, dt)

            # Store data
            self.time_data.append(current_time)
            self.ppfd_data.append(ppfd)
            self.temp_data.append(temp)
            self.power_data.append(power)
            self.pwm_data.append(pwm_optimal)
            self.photo_data.append(photo_rate)
            self.ppfd_ref_data.append(ppfd_ref_full[k])
            self.temp_ref_data.append(temp_ref_full[k])
            self.photo_ref_data.append(photo_ref_full[k])

            # CSV logging every 10 steps (or every step for detailed analysis)
            if k % 10 == 0 or k == steps - 1:  # Log every 10 steps and final step
                with open(log_file, "a+") as csv_file:
                    csv_file.write(
                        f"{current_time:.1f},{pwm_optimal:.1f},{ppfd:.1f},{temp:.1f},{photo_rate:.1f},{power:.1f}\n"
                    )

            # Print progress
            if k % 10 == 0:
                print(
                    f"t={current_time:3.0f}s: PWM={pwm_optimal:5.1f}%, "
                    f"PPFD={ppfd:3.0f}, Temp={temp:4.1f}°C, "
                    f"Photo={photo_rate:4.1f}, Power={power:4.1f}W"
                )

        print("\nSimulation completed!")
        print(f"Data logged to: {log_file}")

        # Check temperature constraint violations
        temp_violations = np.sum(
            (np.array(self.temp_data) < 20.0) | (np.array(self.temp_data) > 30.0)
        )
        print(
            f"Temperature constraint violations: {temp_violations}/{len(self.temp_data)} steps"
        )

        return self.get_results()

    def clear_data(self):
        """Clear all data arrays"""
        self.time_data = []
        self.ppfd_data = []
        self.temp_data = []
        self.power_data = []
        self.pwm_data = []
        self.photo_data = []
        self.ppfd_ref_data = []
        self.temp_ref_data = []
        self.photo_ref_data = []

    def get_results(self):
        """Get simulation results"""
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
        }

    def plot_results(self):
        """Plot simulation results with photosynthesis"""
        results = self.get_results()

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(
            "LED MPC Control with Photosynthesis Optimization (Temp: 20-30°C)",
            fontsize=16,
        )

        # PPFD tracking
        axes[0, 0].plot(
            results["time"], results["ppfd"], "g-", linewidth=2, label="Actual"
        )
        axes[0, 0].plot(
            results["time"], results["ppfd_ref"], "g--", linewidth=2, label="Reference"
        )
        axes[0, 0].set_ylabel("PPFD (μmol/m²/s)")
        axes[0, 0].set_title("PPFD Tracking")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Temperature tracking with constraint bounds
        axes[0, 1].plot(
            results["time"], results["temp"], "r-", linewidth=2, label="Actual"
        )
        axes[0, 1].plot(
            results["time"], results["temp_ref"], "r--", linewidth=2, label="Reference"
        )
        axes[0, 1].axhline(
            y=20.0, color="k", linestyle=":", alpha=0.7, label="Min (20°C)"
        )
        axes[0, 1].axhline(
            y=30.0, color="k", linestyle=":", alpha=0.7, label="Max (30°C)"
        )
        axes[0, 1].set_ylabel("Temperature (°C)")
        axes[0, 1].set_title("Temperature Tracking (20-30°C)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Photosynthesis tracking
        axes[0, 2].plot(
            results["time"],
            results["photosynthesis"],
            "orange",
            linewidth=2,
            label="Actual",
        )
        axes[0, 2].plot(
            results["time"],
            results["photo_ref"],
            "orange",
            linestyle="--",
            linewidth=2,
            label="Reference",
        )
        axes[0, 2].set_ylabel("Photosynthesis (μmol/m²/s)")
        axes[0, 2].set_title("Photosynthesis Rate")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # PWM control signal
        axes[1, 0].plot(results["time"], results["pwm"], "b-", linewidth=2)
        axes[1, 0].set_ylabel("PWM (%)")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_title("Control Signal (PWM)")
        axes[1, 0].grid(True, alpha=0.3)

        # Power consumption
        axes[1, 1].plot(results["time"], results["power"], "m-", linewidth=2)
        axes[1, 1].set_ylabel("Power (W)")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_title("Power Consumption")
        axes[1, 1].grid(True, alpha=0.3)

        # Efficiency plot (photosynthesis per power)
        efficiency = results["photosynthesis"] / (
            results["power"] + 0.1
        )  # Avoid division by zero
        axes[1, 2].plot(results["time"], efficiency, "purple", linewidth=2)
        axes[1, 2].set_ylabel("Photo/Power Efficiency")
        axes[1, 2].set_xlabel("Time (s)")
        axes[1, 2].set_title("Photosynthesis Efficiency")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Calculate performance metrics
        ppfd_rmse = np.sqrt(np.mean((results["ppfd"] - results["ppfd_ref"]) ** 2))
        temp_rmse = np.sqrt(np.mean((results["temp"] - results["temp_ref"]) ** 2))
        photo_rmse = np.sqrt(
            np.mean((results["photosynthesis"] - results["photo_ref"]) ** 2)
        )
        avg_power = np.mean(results["power"])
        total_photosynthesis = np.sum(results["photosynthesis"])

        # Temperature constraint analysis
        temp_min = np.min(results["temp"])
        temp_max = np.max(results["temp"])
        temp_violations = np.sum((results["temp"] < 20.0) | (results["temp"] > 30.0))

        print(f"\nPerformance Metrics:")
        print(f"PPFD RMSE: {ppfd_rmse:.1f} μmol/m²/s")
        print(f"Temperature RMSE: {temp_rmse:.2f} °C")
        print(f"Photosynthesis RMSE: {photo_rmse:.2f} μmol/m²/s")
        print(f"Average Power: {avg_power:.1f} W")
        print(f"Total Photosynthesis: {total_photosynthesis:.1f} μmol/m²/s·s")
        print(f"Energy Efficiency: {total_photosynthesis/avg_power:.2f} (photo/power)")
        print(f"\nTemperature Constraint Analysis:")
        print(f"Temperature range: {temp_min:.1f}°C to {temp_max:.1f}°C")
        print(f"Constraint violations: {temp_violations}/{len(results['temp'])} steps")
        print(
            f"Constraint satisfaction: {100*(1-temp_violations/len(results['temp'])):.1f}%"
        )


# Example usage
if __name__ == "__main__":
    # Create plant model
    plant = LEDPlant(
        base_ambient_temp=20.0,
        max_ppfd=1000.0,
        max_power=100.0,
        thermal_resistance=1.2,
        thermal_mass=5,
    )

    # Create MPC controller
    controller = LEDMPCController(
        plant, prediction_horizon=10, control_horizon=5, dt=1.0
    )

    # Tune MPC weights for maximum photosynthesis optimization
    controller.set_weights(
        Q_ppfd=0.3, Q_temp=0.3, Q_photo=20.0, R_pwm=0.01, R_dpwm=0.1, R_power=0.001
    )

    # Set constraints with more conservative ambient temperature range
    controller.set_constraints(
        pwm_min=0.0,
        pwm_max=80.0,
        dpwm_max=10.0,
        temp_min=20.0,
        temp_max=30.0,  # Reduced max PWM and temp
    )

    # Create simulation environment
    simulation = LEDMPCSimulation(plant, controller)

    # Run simulation
    results = simulation.run_simulation(duration=120, dt=1.0)

    # Plot results
    simulation.plot_results()
