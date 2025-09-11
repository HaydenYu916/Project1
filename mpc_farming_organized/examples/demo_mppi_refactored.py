"""
演示重构后的 MPPI 控制器

此脚本展示了如何实例化和使用 `mppi.py` 中的核心类:
1. `LEDPlant`: 系统模型
2. `LEDMPPIController`: 控制器
3. `LEDMPPISimulation`: 仿真环境
"""
import sys
import os
import numpy as np

# 将核心模块添加到Python路径
# 获取'examples'目录的父目录，即项目根目录(mpc_farming_organized)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'core'))

from mppi import LEDPlant, LEDMPPIController, led_step, PHOTOSYNTHESIS_AVAILABLE, PhotosynthesisPredictor
import matplotlib.pyplot as plt

class LEDMPPISimulation:
    """MPPI simulation environment for photosynthesis maximization with
reference comparison"""

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
        self.power_ref_data = []
        self.cost_data = []

        # Initialize photosynthesis predictor for evaluation
        if PHOTOSYNTHESIS_AVAILABLE:
            try:
                self.pn = PhotosynthesisPredictor()
            except:
                self.pn = None
        else:
            self.pn = None

    def create_reference_trajectory(self, duration, dt):
        """Create reference trajectories for comparison purposes"""
        time_points = np.arange(0, duration, dt)

        ppfd_ref = []
        temp_ref = []
        photo_ref = []
        power_ref = []

        current_temp = self.plant.base_ambient_temp
        for t in time_points:
            # PPFD reference: daily light cycle
            ppfd_target = 300

            # Temperature reference: use LED step with evolving ambient temp
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

            # Photosynthesis reference (for comparison only)
            if self.pn is not None:
                photo_target = self.pn.predict(ppfd_target, 400,
temp_target, 0.83)
            else:
                # Use simple model for reference
                ppfd_max = 1000
                pn_max = 25
                km = 300
                temp_factor = np.exp(-0.01 * (temp_target - 25) ** 2)
                photo_target = max(
                    0, (pn_max * ppfd_target / (km + ppfd_target)) *
temp_factor
                )

            # Calculate required PWM for reference PPFD
            # Simplified calculation - in reality this would be more complex
            # pwm_target = min(80.0, (ppfd_target / self.plant.max_ppfd) * 100)
            # pwm_target = ppfd_target / self.plant.max_ppfd * 100

            # Calculate reference power consumption
            # power_target = (pwm_target / 100) * self.plant.max_power
            # PPFD output
            pwm_fraction = ppfd_target / self.plant.max_ppfd

            # LED efficiency
            efficiency = 0.8 + 0.2 * np.exp(-pwm_fraction * 2.0)

            # Power consumption
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
        """Run MPPI simulation for photosynthesis maximization with
reference comparison"""

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

        # Create reference trajectories for comparison
        ppfd_ref_full, temp_ref_full, photo_ref_full, power_ref_full = (
            self.create_reference_trajectory(duration, dt)
        )

        # Reset plant
        self.plant.ambient_temp = self.plant.base_ambient_temp
        self.plant.time = 0.0

        # Reset controller
        self.controller.pwm_prev = 0.0

        # Clear data
        self.clear_data()

        # Initialize mean sequence for MPPI
        mean_sequence = np.ones(self.controller.horizon) * 30.0
        log_file = "mppi_log.txt"
        if os.path.exists(log_file):
            os.remove(log_file)
        # Simulation loop
        steps = int(duration / dt)
        for k in range(steps):
            current_time = k * dt

            # Solve MPPI for photosynthesis maximization (no reference tracking)
            pwm_optimal, optimal_sequence, success, cost, weights = self.controller.solve(self.plant.ambient_temp, mean_sequence)

            # Update mean sequence for next iteration (receding horizon)
            if len(optimal_sequence) > 1:
                mean_sequence = np.concatenate(
                    [optimal_sequence[1:], [optimal_sequence[-1]]]
                )
            else:
                mean_sequence = optimal_sequence

            # Apply control to plant
            ppfd, temp, power, photo_rate = self.plant.step(pwm_optimal,
dt)

            # Store data (including references for comparison)
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

            # Print progress
            if k % 10 == 0:
                temp_status = (
                    "✓"
                    if self.controller.temp_min <= temp <=
self.controller.temp_max
                    else "✗"
                )
                print(
                    f"t={current_time:3.0f}s: PWM={pwm_optimal:5.1f}%,
"
                    f"PPFD={ppfd:3.0f}, Temp={temp:4.1f}°C {temp_status},
"
                    f"Photo={photo_rate:4.1f}, Cost={cost:.1e}"
                )
                with open("mppi_log.txt", "a+") as log_file:
                    log_file.write(
                        f"{current_time:.1f},{pwm_optimal:.1f},{ppfd:.1f},{temp:.1f},{photo_rate:.1f},{cost:.1e}\n"
                    )

        print("\nSimulation completed!")

        # Analyze constraint satisfaction
        temp_violations = np.sum(
            (np.array(self.temp_data) < self.controller.temp_min)
            | (np.array(self.temp_data) > self.controller.temp_max)
        )
        temp_satisfaction = 100 * (1 - temp_violations /
len(self.temp_data))

        print(f"Temperature constraint satisfaction:
{temp_satisfaction:.1f}%")
        print(
            f"Temperature range achieved: {np.min(self.temp_data):.1f}
to {np.max(self.temp_data):.1f}°C"
        )
        print(f"Average photosynthesis rate:
{np.mean(self.photo_data):.2f} μmol/m²/s")
        print(f"Total photosynthesis: {np.sum(self.photo_data):.1f}
μmol/m²/s·s")

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
        self.power_ref_data = []
        self.cost_data = []

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
            "power_ref": np.array(self.power_ref_data),
            "cost": np.array(self.cost_data),
        }

    def plot_results(self):
        """Plot MPPI simulation results comparing actual vs reference with
accumulated metrics"""
        results = self.get_results()

        # Calculate accumulated values
        dt = (
            results["time"][1] - results["time"][0] if
len(results["time"]) > 1 else 1.0
        )

        # Cumulative sums
        cumulative_pn_mppi = np.cumsum(results["photosynthesis"]) * dt
        cumulative_pn_ref = np.cumsum(results["photo_ref"]) * dt
        cumulative_power_mppi = np.cumsum(results["power"]) * dt
        cumulative_power_ref = np.cumsum(results["power_ref"]) * dt

        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(
            "LED MPPI Control - Photosynthesis Maximization vs Reference
(with Accumulated Metrics)",
            fontsize=16,
        )

        # PPFD comparison
        axes[0, 0].plot(
            results["time"], results["ppfd"], "g-", linewidth=2,
label="MPPI (Actual)"
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

        # Temperature comparison with constraints
        axes[0, 1].plot(
            results["time"], results["temp"], "r-", linewidth=2,
label="MPPI (Actual)"
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

        # Photosynthesis comparison (key performance metric)
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

        # PWM control
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

        # Power consumption comparison
        axes[1, 1].plot(
            results["time"], results["power"], "m-", linewidth=2,
label="MPPI"
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

        # Cost evolution (should be negative due to photosynthesis
maximization)
        axes[1, 2].plot(results["time"], results["cost"], "purple",
linewidth=2)
        axes[1, 2].set_ylabel("Cost")
        axes[1, 2].set_title("MPPI Cost Evolution")
        axes[1, 2].grid(True, alpha=0.3)

        # NEW: Accumulated Photosynthesis Comparison
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
        axes[2, 0].set_title("Accumulated Photosynthesis: MPPI vs
Reference")
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # NEW: Accumulated Power Consumption Comparison
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

        # NEW: Efficiency Comparison (Pn/Power ratio over time)
        efficiency_mppi = results["photosynthesis"] / np.maximum(
            results["power"], 0.1
        )  # Avoid division by zero
        efficiency_ref = results["photo_ref"] /
np.maximum(results["power_ref"], 0.1)

        axes[2, 2].plot(
            results["time"], efficiency_mppi, "c-", linewidth=2,
label="MPPI Efficiency"
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

        # Performance metrics
        self.print_performance_metrics(
            results,
            cumulative_pn_mppi,
            cumulative_pn_ref,
            cumulative_power_mppi,
            cumulative_power_ref,
        )

    def print_performance_metrics(
        self, results, cum_pn_mppi, cum_pn_ref, cum_power_mppi,
cum_power_ref
    ):
        """Print detailed performance metrics comparing MPPI vs reference
including accumulated values"""
        avg_photosynthesis = np.mean(results["photosynthesis"])
        avg_photo_ref = np.mean(results["photo_ref"])
        max_photosynthesis = np.max(results["photosynthesis"])
        total_photosynthesis = cum_pn_mppi[-1]
        total_photo_ref = cum_pn_ref[-1]
        total_power_mppi = cum_power_mppi[-1]
        total_power_ref = cum_power_ref[-1]
        avg_power_mppi = np.mean(results["power"])
        avg_power_ref = np.mean(results["power_ref"])

        # Performance improvement over reference
        photo_improvement = ((avg_photosynthesis - avg_photo_ref) /
avg_photo_ref) * 100
        total_improvement = (
            (total_photosynthesis - total_photo_ref) / total_photo_ref
        ) * 100
        power_difference = (
            (total_power_mppi - total_power_ref) / total_power_ref
        ) * 100

        # Energy efficiency metrics
        efficiency_mppi = total_photosynthesis / total_power_mppi
        efficiency_ref = total_photo_ref / total_power_ref
        efficiency_improvement = (
            (efficiency_mppi - efficiency_ref) / efficiency_ref
        ) * 100

        # RMSE for comparison
        ppfd_rmse = np.sqrt(np.mean((results["ppfd"] -
results["ppfd_ref"]) ** 2))
        temp_rmse = np.sqrt(np.mean((results["temp"] -
results["temp_ref"]) ** 2))
        photo_rmse = np.sqrt(
            np.mean((results["photosynthesis"] - results["photo_ref"])
** 2)
        )
        power_rmse = np.sqrt(np.mean((results["power"] -
results["power_ref"]) ** 2))

        temp_violations = np.sum(
            (results["temp"] < self.controller.temp_min)
            | (results["temp"] > self.controller.temp_max)
        )
        temp_satisfaction = 100 * (1 - temp_violations /
len(results["temp"]))

        pwm_violations = np.sum(
            (results["pwm"] < self.controller.pwm_min)
            | (results["pwm"] > self.controller.pwm_max)
        )

        print(f"\n" + "=" * 80)
        print(f"MPPI PHOTOSYNTHESIS MAXIMIZATION vs REFERENCE COMPARISON")
        print(f"=" * 80)

        print(f"\n📈 PHOTOSYNTHESIS PERFORMANCE:")
        print(f"  MPPI Average Photosynthesis: {avg_photosynthesis:.2f}
μmol/m²/s")
        print(f"  Reference Average Photosynthesis:
{avg_photo_ref:.2f} μmol/m²/s")
        print(f"  Improvement: {photo_improvement:+.1f}% over reference")
        print(f"  MPPI Maximum Photosynthesis: {max_photosynthesis:.2f}
μmol/m²/s")

        print(f"\n🔋 ACCUMULATED METRICS:")
        print(f"  MPPI Total Photosynthesis: {total_photosynthesis:.1f}
μmol/m²")
        print(f"  Reference Total Photosynthesis:
{total_photo_ref:.1f} μmol/m²")
        print(f"  Total Pn Improvement: {total_improvement:+.1f}% over
reference")
        print(f"  MPPI Total Power Consumption: {total_power_mppi:.1f} W·s")
        print(f"  Reference Total Power Consumption:
{total_power_ref:.1f} W·s")
        print(f"  Power Usage Difference: {power_difference:+.1f}% vs
reference")

        print(f"\n⚡ ENERGY EFFICIENCY:")
        print(f"  MPPI Energy Efficiency: {efficiency_mppi:.4f}
(μmol/m²)/(W·s)")
        print(f"  Reference Energy Efficiency: {efficiency_ref:.4f}
(μmol/m²)/(W·s)")
        print(
            f"  Efficiency Improvement: {efficiency_improvement:+.1f}%
over reference"
        )
        print(f"  MPPI Average Power: {avg_power_mppi:.1f} W")
        print(f"  Reference Average Power: {avg_power_ref:.1f} W")

        print(f"\n📊 COMPARISON METRICS (RMSE):")
        print(f"  PPFD deviation from reference: {ppfd_rmse:.1f}
μmol/m²/s")
        print(f"  Temperature deviation from reference: {temp_rmse:.2f}
°C")
        print(f"  Photosynthesis deviation from reference:
{photo_rmse:.2f} μmol/m²/s")
        print(f"  Power deviation from reference: {power_rmse:.2f} W")

        print(f"\n🎯 CONSTRAINT SATISFACTION:")
        print(
            f"  Temperature violations:
{temp_violations}/{len(results['temp'])} steps"
        )
        print(f"  Temperature satisfaction: {temp_satisfaction:.1f}%")
        print(f"  PWM violations: {pwm_violations} steps")
        print(
            f"  Temperature range: {np.min(results['temp']):.1f} to
{np.max(results['temp']):.1f}°C"
        )

        print(f"\n💡 SUMMARY:")
        if total_improvement > 0:
            print(
                f"  ✅ MPPI achieved {total_improvement:.1f}% higher total
photosynthesis"
            )
        else:
            print(
                f"  ❌ MPPI achieved {total_improvement:.1f}% lower total
photosynthesis"
            )

        if power_difference < 0:
            print(f"  ✅ MPPI used {abs(power_difference):.1f}% less power")
        else:
            print(f"  ⚠️  MPPI used {power_difference:.1f}% more power")

        if efficiency_improvement > 0:
            print(f"  ✅ MPPI was {efficiency_improvement:.1f}% more
energy efficient")
        else:
            print(
                f"  ❌ MPPI was {abs(efficiency_improvement):.1f}% less
energy efficient"
            )

        print(f"  Final Cost: {results['cost'][-1]:.2e}")
        print(f"  Average PWM: {np.mean(results['pwm']):.1f}%")

if __name__ == "__main__":
    # 此部分代码现在是独立的示例，演示如何使用重构后的 MPPI 模块
    
    # 1. 创建被控对象 (LED植物模型)
    plant = LEDPlant(
        base_ambient_temp=22.0,
        max_ppfd=500.0,
        max_power=100.0,
        thermal_resistance=1.2,
        thermal_mass=5.0,
    )

    # 2. 创建 MPPI 控制器
    controller = LEDMPPIController(
        plant=plant, 
        horizon=10, 
        num_samples=1000, 
        dt=1.0, 
        temperature=0.5
    )

    # 3. 配置控制器参数 (可选, 可使用默认值)
    controller.set_weights(
        Q_photo=10.0,
        R_pwm=0.001,
        R_dpwm=0.05,
        R_power=0.01,
    )
    controller.set_constraints(
        pwm_min=0.0, 
        pwm_max=70.0, 
        temp_min=20.0, 
        temp_max=29.0
    )
    controller.set_mppi_params(
        num_samples=1000, 
        temperature=0.5, 
        pwm_std=10.0
    )

    # 4. 创建并运行仿真
    # 注意: LEDMPPISimulation 类现在也位于 mppi.py 中
    # 在未来的重构中，它可能会被移动到专门的仿真工具文件中
    print("🚀 开始运行 MPPI LED 控制仿真 (光合作用最大化)...")
    simulation = LEDMPPISimulation(plant, controller)
    results = simulation.run_simulation(duration=120, dt=1.0)

    # 5. 可视化结果
    simulation.plot_results()
