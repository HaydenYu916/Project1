import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from datetime import datetime
import os
import json
import importlib.util

warnings.filterwarnings("ignore")

# Import baseline algorithm
from baseline_pn_optimizer import (
    BaselinePnOptimizer,
    BaselinePnController,
    BaselinePnSimulation,
)

# Import MPPI from mppi-power-parameterized.py
spec_mppi = importlib.util.spec_from_file_location(
    "mppi_power_param", "mppi-power-parameterized.py"
)
mppi_module = importlib.util.module_from_spec(spec_mppi)
spec_mppi.loader.exec_module(mppi_module)

LEDPlant = mppi_module.LEDPlant
LEDMPPIController = mppi_module.LEDMPPIController
LEDMPPISimulation = mppi_module.LEDMPPISimulation

# Import photosynthesis predictor
try:
    from pn_prediction.predict import PhotosynthesisPredictor

    PHOTOSYNTHESIS_AVAILABLE = True
except ImportError:
    PHOTOSYNTHESIS_AVAILABLE = False

from led import led_step


class ThreeMethodPPFDComparison:
    """
    Comprehensive comparison between three methods across different reference PPFD values:
    1. Baseline Pn Optimization
    2. MPPI Controller
    3. Reference (constant PPFD)

    Uses the same metrics as baseline_ppfd_power_analysis.py
    """

    def __init__(
        self,
        base_ambient_temp=22.0,
        max_ppfd=1000.0,
        max_power=100.0,
        thermal_resistance=1.2,
        thermal_mass=8.0,
        pn_tolerance=50.0,
    ):
        self.base_ambient_temp = base_ambient_temp
        self.max_ppfd = max_ppfd
        self.max_power = max_power
        self.thermal_resistance = thermal_resistance
        self.thermal_mass = thermal_mass
        self.pn_tolerance = pn_tolerance

        # Initialize photosynthesis predictor
        if PHOTOSYNTHESIS_AVAILABLE:
            try:
                self.photo_predictor = PhotosynthesisPredictor()
                self.use_photo_model = self.photo_predictor.is_loaded
            except:
                self.use_photo_model = False
        else:
            self.use_photo_model = False

    def get_photosynthesis_rate(self, ppfd, temperature):
        """Get photosynthesis rate using prediction model"""
        if self.use_photo_model:
            try:
                return self.photo_predictor.predict(ppfd, temperature)
            except Exception as e:
                pass

        # Simple model fallback
        ppfd_max = 1000
        pn_max = 25
        km = 300
        temp_factor = np.exp(-0.01 * (temperature - 25) ** 2)
        return max(0, (pn_max * ppfd / (km + ppfd)) * temp_factor)

    def run_baseline_method(self, reference_ppfd, duration, dt):
        """Run baseline Pn optimization method"""

        optimizer = BaselinePnOptimizer(
            base_ambient_temp=self.base_ambient_temp,
            max_ppfd=self.max_ppfd,
            max_power=self.max_power,
            thermal_resistance=self.thermal_resistance,
            thermal_mass=self.thermal_mass,
            pn_tolerance=self.pn_tolerance,
            min_ppfd_step=1.0,
        )

        controller = BaselinePnController(optimizer, fallback_pwm=30.0)
        simulation = BaselinePnSimulation(
            optimizer, controller, reference_ppfd=reference_ppfd
        )

        results = simulation.run_simulation(duration=duration, dt=dt)
        return results

    def run_mppi_method(self, reference_ppfd, duration, dt):
        """Run MPPI controller method with parameterized reference PPFD"""

        plant = LEDPlant(
            base_ambient_temp=self.base_ambient_temp,
            max_ppfd=self.max_ppfd,
            max_power=self.max_power,
            thermal_resistance=self.thermal_resistance,
            thermal_mass=self.thermal_mass,
        )

        controller = LEDMPPIController(
            plant=plant, horizon=10, num_samples=1000, dt=dt, temperature=0.5
        )

        # Configure MPPI weights for photosynthesis maximization
        controller.set_weights(
            Q_photo=5.0,
            Q_ref=25.0,
            R_pwm=0.001,
            R_dpwm=0.05,
            R_power=0.08,
        )

        # Set constraints
        controller.set_constraints(
            pwm_min=0.0, pwm_max=100.0, temp_min=20.0, temp_max=29.0
        )
        controller.set_mppi_params(num_samples=1000, temperature=0.5, pwm_std=10.0)

        simulation = LEDMPPISimulation(plant, controller, reference_ppfd=reference_ppfd)
        results = simulation.run_simulation(duration=duration, dt=dt)

        return results

    def run_reference_method(self, reference_ppfd, duration, dt):
        """Run reference method (constant PPFD)"""

        # Calculate PWM for reference PPFD
        reference_pwm = min(100.0, (reference_ppfd / self.max_ppfd) * 100)

        # Initialize state
        current_temp = self.base_ambient_temp

        # Data storage
        time_data = []
        ppfd_data = []
        temp_data = []
        power_data = []
        pwm_data = []
        photosynthesis_data = []

        # Simulation loop
        steps = int(duration / dt)
        for k in range(steps):
            current_time = k * dt

            # Apply constant reference PWM
            ppfd, new_temp, power, efficiency = led_step(
                pwm_percent=reference_pwm,
                ambient_temp=current_temp,
                base_ambient_temp=self.base_ambient_temp,
                dt=dt,
                max_ppfd=self.max_ppfd,
                max_power=self.max_power,
                thermal_resistance=self.thermal_resistance,
                thermal_mass=self.thermal_mass,
            )

            # Calculate photosynthesis
            photosynthesis_rate = self.get_photosynthesis_rate(ppfd, new_temp)

            # Update state
            current_temp = new_temp

            # Store data
            time_data.append(current_time)
            ppfd_data.append(ppfd)
            temp_data.append(new_temp)
            power_data.append(power)
            pwm_data.append(reference_pwm)
            photosynthesis_data.append(photosynthesis_rate)

        return {
            "time": np.array(time_data),
            "ppfd": np.array(ppfd_data),
            "temp": np.array(temp_data),
            "power": np.array(power_data),
            "pwm": np.array(pwm_data),
            "photosynthesis": np.array(photosynthesis_data),
        }

    def calculate_performance_metrics(self, results):
        """Calculate performance metrics matching ppfd_power_analysis.py"""

        # Basic metrics
        avg_photosynthesis = np.mean(results["photosynthesis"])
        total_photosynthesis = np.sum(results["photosynthesis"])
        avg_power = np.mean(results["power"])
        total_power = np.sum(results["power"])
        avg_pwm = np.mean(results["pwm"])
        avg_ppfd = np.mean(results["ppfd"])
        avg_temp = np.mean(results["temp"])
        max_temp = np.max(results["temp"])
        min_temp = np.min(results["temp"])

        # Energy efficiency
        energy_efficiency = total_photosynthesis / total_power if total_power > 0 else 0

        # Control variability
        pwm_std = np.std(results["pwm"])
        ppfd_std = np.std(results["ppfd"])

        # Temperature satisfaction (assuming 20-29°C constraints)
        temp_violations = np.sum((results["temp"] < 20.0) | (results["temp"] > 29.0))
        temp_satisfaction = (1 - temp_violations / len(results["temp"])) * 100

        # Control smoothness
        pwm_changes = np.diff(results["pwm"])
        control_smoothness = np.mean(np.abs(pwm_changes))

        return {
            "avg_photosynthesis": avg_photosynthesis,
            "total_photosynthesis": total_photosynthesis,
            "avg_power": avg_power,
            "total_power": total_power,
            "energy_efficiency": energy_efficiency,
            "avg_pwm": avg_pwm,
            "avg_ppfd": avg_ppfd,
            "avg_temp": avg_temp,
            "max_temp": max_temp,
            "min_temp": min_temp,
            "temp_range": max_temp - min_temp,
            "temp_satisfaction": temp_satisfaction,
            "pwm_std": pwm_std,
            "ppfd_std": ppfd_std,
            "control_smoothness": control_smoothness,
        }

    def run_ppfd_analysis(
        self,
        ppfd_range=None,
        duration=120,
        dt=1.0,
        output_dir="three_method_ppfd_comparison_results",
    ):
        """
        Run three-method comparison across different PPFD values
        with parameterized reference PPFD
        """

        if ppfd_range is None:
            ppfd_range = [200, 300, 400, 500, 600, 700]

        print("Three-Method PPFD Comparison: Baseline vs MPPI vs Reference")
        print("=" * 70)
        print(f"Testing reference PPFD values: {ppfd_range}")
        print(f"Simulation duration: {duration}s, dt: {dt}s")

        if self.use_photo_model:
            print("Using trained photosynthesis model")
        else:
            print("Using simple photosynthesis model")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize results storage
        results_summary = []

        # Run simulations for each PPFD value
        for i, ppfd_value in enumerate(ppfd_range):
            print(
                f"\n[{i+1}/{len(ppfd_range)}] Testing Reference PPFD = {ppfd_value} μmol/m²/s"
            )

            try:
                # 1. Run Baseline Method
                print("  Running Baseline Pn Optimization...")
                baseline_results = self.run_baseline_method(ppfd_value, duration, dt)
                baseline_metrics = self.calculate_performance_metrics(baseline_results)

                # 2. Run MPPI Method
                print("  Running MPPI Controller...")
                mppi_results = self.run_mppi_method(ppfd_value, duration, dt)
                mppi_metrics = self.calculate_performance_metrics(mppi_results)

                # 3. Run Reference Method
                print("  Running Reference (Constant PPFD)...")
                reference_results = self.run_reference_method(ppfd_value, duration, dt)
                reference_metrics = self.calculate_performance_metrics(
                    reference_results
                )

                # Calculate comparative metrics
                # Baseline vs Reference
                baseline_power_saved = (
                    reference_metrics["avg_power"] - baseline_metrics["avg_power"]
                )
                baseline_power_diff_percent = (
                    (baseline_metrics["avg_power"] - reference_metrics["avg_power"])
                    / reference_metrics["avg_power"]
                ) * 100
                baseline_efficiency_improvement = (
                    (
                        baseline_metrics["energy_efficiency"]
                        - reference_metrics["energy_efficiency"]
                    )
                    / reference_metrics["energy_efficiency"]
                ) * 100

                # MPPI vs Reference
                mppi_power_saved = (
                    reference_metrics["avg_power"] - mppi_metrics["avg_power"]
                )
                mppi_power_diff_percent = (
                    (mppi_metrics["avg_power"] - reference_metrics["avg_power"])
                    / reference_metrics["avg_power"]
                ) * 100
                mppi_efficiency_improvement = (
                    (
                        mppi_metrics["energy_efficiency"]
                        - reference_metrics["energy_efficiency"]
                    )
                    / reference_metrics["energy_efficiency"]
                ) * 100

                # MPPI vs Baseline
                mppi_vs_baseline_power_diff = (
                    (mppi_metrics["avg_power"] - baseline_metrics["avg_power"])
                    / baseline_metrics["avg_power"]
                ) * 100
                mppi_vs_baseline_efficiency_diff = (
                    (
                        mppi_metrics["energy_efficiency"]
                        - baseline_metrics["energy_efficiency"]
                    )
                    / baseline_metrics["energy_efficiency"]
                ) * 100
                mppi_vs_baseline_pn_diff = (
                    (
                        mppi_metrics["avg_photosynthesis"]
                        - baseline_metrics["avg_photosynthesis"]
                    )
                    / baseline_metrics["avg_photosynthesis"]
                ) * 100

                # Store comprehensive results
                result_entry = {
                    "reference_ppfd": ppfd_value,
                    # Baseline metrics
                    "baseline_avg_photosynthesis": baseline_metrics[
                        "avg_photosynthesis"
                    ],
                    "baseline_total_photosynthesis": baseline_metrics[
                        "total_photosynthesis"
                    ],
                    "baseline_avg_power": baseline_metrics["avg_power"],
                    "baseline_total_power": baseline_metrics["total_power"],
                    "baseline_energy_efficiency": baseline_metrics["energy_efficiency"],
                    "baseline_avg_pwm": baseline_metrics["avg_pwm"],
                    "baseline_avg_ppfd": baseline_metrics["avg_ppfd"],
                    "baseline_avg_temp": baseline_metrics["avg_temp"],
                    "baseline_temp_range": baseline_metrics["temp_range"],
                    "baseline_temp_satisfaction": baseline_metrics["temp_satisfaction"],
                    "baseline_control_smoothness": baseline_metrics[
                        "control_smoothness"
                    ],
                    # MPPI metrics
                    "mppi_avg_photosynthesis": mppi_metrics["avg_photosynthesis"],
                    "mppi_total_photosynthesis": mppi_metrics["total_photosynthesis"],
                    "mppi_avg_power": mppi_metrics["avg_power"],
                    "mppi_total_power": mppi_metrics["total_power"],
                    "mppi_energy_efficiency": mppi_metrics["energy_efficiency"],
                    "mppi_avg_pwm": mppi_metrics["avg_pwm"],
                    "mppi_avg_ppfd": mppi_metrics["avg_ppfd"],
                    "mppi_avg_temp": mppi_metrics["avg_temp"],
                    "mppi_temp_range": mppi_metrics["temp_range"],
                    "mppi_temp_satisfaction": mppi_metrics["temp_satisfaction"],
                    "mppi_control_smoothness": mppi_metrics["control_smoothness"],
                    # Reference metrics
                    "reference_avg_photosynthesis": reference_metrics[
                        "avg_photosynthesis"
                    ],
                    "reference_total_photosynthesis": reference_metrics[
                        "total_photosynthesis"
                    ],
                    "reference_avg_power": reference_metrics["avg_power"],
                    "reference_total_power": reference_metrics["total_power"],
                    "reference_energy_efficiency": reference_metrics[
                        "energy_efficiency"
                    ],
                    "reference_avg_pwm": reference_metrics["avg_pwm"],
                    "reference_avg_ppfd": reference_metrics["avg_ppfd"],
                    "reference_avg_temp": reference_metrics["avg_temp"],
                    "reference_temp_range": reference_metrics["temp_range"],
                    "reference_temp_satisfaction": reference_metrics[
                        "temp_satisfaction"
                    ],
                    "reference_control_smoothness": reference_metrics[
                        "control_smoothness"
                    ],
                    # Comparative metrics (vs Reference)
                    "baseline_power_saved_watts": baseline_power_saved,
                    "baseline_power_difference_percent": baseline_power_diff_percent,
                    "baseline_efficiency_improvement_percent": baseline_efficiency_improvement,
                    "mppi_power_saved_watts": mppi_power_saved,
                    "mppi_power_difference_percent": mppi_power_diff_percent,
                    "mppi_efficiency_improvement_percent": mppi_efficiency_improvement,
                    # Comparative metrics (MPPI vs Baseline)
                    "mppi_vs_baseline_power_difference_percent": mppi_vs_baseline_power_diff,
                    "mppi_vs_baseline_efficiency_difference_percent": mppi_vs_baseline_efficiency_diff,
                    "mppi_vs_baseline_photosynthesis_difference_percent": mppi_vs_baseline_pn_diff,
                }

                results_summary.append(result_entry)

                # Print brief summary
                print(
                    f"    Baseline: Power {baseline_power_diff_percent:+.1f}%, Efficiency {baseline_efficiency_improvement:+.1f}%"
                )
                print(
                    f"    MPPI:     Power {mppi_power_diff_percent:+.1f}%, Efficiency {mppi_efficiency_improvement:+.1f}%"
                )
                print(
                    f"    MPPI vs Baseline: Power {mppi_vs_baseline_power_diff:+.1f}%, Pn {mppi_vs_baseline_pn_diff:+.1f}%"
                )

            except Exception as e:
                print(f"  ERROR: Failed to run simulation for PPFD {ppfd_value}: {e}")
                continue

        # Convert to DataFrame
        df_results = pd.DataFrame(results_summary)

        if len(df_results) == 0:
            print("ERROR: No successful simulations completed!")
            return pd.DataFrame(), output_dir

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(
            output_dir, f"three_method_ppfd_comparison_{timestamp}.csv"
        )
        df_results.to_csv(csv_filename, index=False)

        print(f"\nResults saved to: {csv_filename}")

        return df_results, output_dir

    def create_comparison_plots(self, df_results, output_dir):
        """Create comprehensive comparison plots matching ppfd_power_analysis.py format"""

        if len(df_results) == 0:
            print("No results to plot!")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create figure with 6 subplots (2x3) matching ppfd_power_analysis.py format
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Three-Method PPFD Comparison: Baseline vs MPPI vs Reference", fontsize=16
        )

        ppfd_values = df_results["reference_ppfd"]

        # 1. Power Consumption vs PPFD
        axes[0, 0].plot(
            ppfd_values,
            df_results["baseline_avg_power"],
            "g-o",
            linewidth=2,
            label="Baseline",
        )
        axes[0, 0].plot(
            ppfd_values, df_results["mppi_avg_power"], "b-s", linewidth=2, label="MPPI"
        )
        axes[0, 0].plot(
            ppfd_values,
            df_results["reference_avg_power"],
            "r--^",
            linewidth=2,
            label="Reference",
        )
        axes[0, 0].set_xlabel("Reference PPFD (μmol/m²/s)")
        axes[0, 0].set_ylabel("Average Power (W)")
        axes[0, 0].set_title("Power Consumption vs PPFD")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Absolute Power Savings vs Reference
        width = (
            (ppfd_values.iloc[1] - ppfd_values.iloc[0]) * 0.35
            if len(ppfd_values) > 1
            else 30
        )
        x_positions = np.array(ppfd_values)
        axes[0, 1].bar(
            x_positions - width / 2,
            df_results["baseline_power_saved_watts"],
            color="green",
            alpha=0.7,
            width=width,
            label="Baseline Savings",
        )
        axes[0, 1].bar(
            x_positions + width / 2,
            df_results["mppi_power_saved_watts"],
            color="blue",
            alpha=0.7,
            width=width,
            label="MPPI Savings",
        )
        axes[0, 1].set_xlabel("Reference PPFD (μmol/m²/s)")
        axes[0, 1].set_ylabel("Power Saved (W)")
        axes[0, 1].set_title("Absolute Power Savings vs Reference")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color="black", linestyle="-", alpha=0.5)

        # 3. Relative Power Savings (%)
        axes[0, 2].bar(
            x_positions - width / 2,
            df_results["baseline_power_difference_percent"],
            color="green",
            alpha=0.7,
            width=width,
            label="Baseline %",
        )
        axes[0, 2].bar(
            x_positions + width / 2,
            df_results["mppi_power_difference_percent"],
            color="blue",
            alpha=0.7,
            width=width,
            label="MPPI %",
        )
        axes[0, 2].set_xlabel("Reference PPFD (μmol/m²/s)")
        axes[0, 2].set_ylabel("Power Difference (%)")
        axes[0, 2].set_title("Relative Power Savings (%)")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axhline(y=0, color="black", linestyle="-", alpha=0.5)

        # 4. Energy Efficiency vs PPFD
        axes[1, 0].plot(
            ppfd_values,
            df_results["baseline_energy_efficiency"],
            "g-o",
            linewidth=2,
            label="Baseline",
        )
        axes[1, 0].plot(
            ppfd_values,
            df_results["mppi_energy_efficiency"],
            "b-s",
            linewidth=2,
            label="MPPI",
        )
        axes[1, 0].plot(
            ppfd_values,
            df_results["reference_energy_efficiency"],
            "r--^",
            linewidth=2,
            label="Reference",
        )
        axes[1, 0].set_xlabel("Reference PPFD (μmol/m²/s)")
        axes[1, 0].set_ylabel("Energy Efficiency (μmol/m²)/(W·s)")
        axes[1, 0].set_title("Energy Efficiency vs PPFD")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Energy Efficiency Improvement vs Reference
        axes[1, 1].bar(
            x_positions - width / 2,
            df_results["baseline_efficiency_improvement_percent"],
            color="green",
            alpha=0.7,
            width=width,
            label="Baseline Improvement",
        )
        axes[1, 1].bar(
            x_positions + width / 2,
            df_results["mppi_efficiency_improvement_percent"],
            color="blue",
            alpha=0.7,
            width=width,
            label="MPPI Improvement",
        )
        axes[1, 1].set_xlabel("Reference PPFD (μmol/m²/s)")
        axes[1, 1].set_ylabel("Efficiency Improvement (%)")
        axes[1, 1].set_title("Energy Efficiency Improvement")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color="black", linestyle="-", alpha=0.5)

        # 6. Photosynthesis vs PPFD
        axes[1, 2].plot(
            ppfd_values,
            df_results["baseline_avg_photosynthesis"],
            "g-o",
            linewidth=2,
            label="Baseline",
        )
        axes[1, 2].plot(
            ppfd_values,
            df_results["mppi_avg_photosynthesis"],
            "b-s",
            linewidth=2,
            label="MPPI",
        )
        axes[1, 2].plot(
            ppfd_values,
            df_results["reference_avg_photosynthesis"],
            "r--^",
            linewidth=2,
            label="Reference",
        )
        axes[1, 2].set_xlabel("Reference PPFD (μmol/m²/s)")
        axes[1, 2].set_ylabel("Avg Photosynthesis (μmol/m²/s)")
        axes[1, 2].set_title("Photosynthesis vs PPFD")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_filename = os.path.join(
            output_dir, f"three_method_comparison_plots_{timestamp}.png"
        )
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"Plots saved to: {plot_filename}")

        plt.show()

        return plot_filename

    def print_analysis_summary(self, df_results):
        """Print comprehensive analysis summary matching ppfd_power_analysis.py format"""

        print("\n" + "=" * 90)
        print("THREE-METHOD PPFD COMPARISON ANALYSIS SUMMARY")
        print("=" * 90)

        # Overall statistics
        total_tests = len(df_results)

        # Count improvements vs reference
        baseline_power_improvements = len(
            df_results[df_results["baseline_power_difference_percent"] < 0]
        )
        mppi_power_improvements = len(
            df_results[df_results["mppi_power_difference_percent"] < 0]
        )
        baseline_efficiency_improvements = len(
            df_results[df_results["baseline_efficiency_improvement_percent"] > 0]
        )
        mppi_efficiency_improvements = len(
            df_results[df_results["mppi_efficiency_improvement_percent"] > 0]
        )

        # Count MPPI improvements vs baseline
        mppi_better_power = len(
            df_results[df_results["mppi_vs_baseline_power_difference_percent"] < 0]
        )
        mppi_better_photosynthesis = len(
            df_results[
                df_results["mppi_vs_baseline_photosynthesis_difference_percent"] > 0
            ]
        )
        mppi_better_efficiency = len(
            df_results[df_results["mppi_vs_baseline_efficiency_difference_percent"] > 0]
        )

        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total reference PPFD values tested: {total_tests}")
        print(
            f"  Baseline power improvements vs reference: {baseline_power_improvements}/{total_tests} ({100*baseline_power_improvements/total_tests:.1f}%)"
        )
        print(
            f"  MPPI power improvements vs reference: {mppi_power_improvements}/{total_tests} ({100*mppi_power_improvements/total_tests:.1f}%)"
        )
        print(
            f"  Baseline efficiency improvements vs reference: {baseline_efficiency_improvements}/{total_tests} ({100*baseline_efficiency_improvements/total_tests:.1f}%)"
        )
        print(
            f"  MPPI efficiency improvements vs reference: {mppi_efficiency_improvements}/{total_tests} ({100*mppi_efficiency_improvements/total_tests:.1f}%)"
        )
        print(
            f"  MPPI better than baseline (power): {mppi_better_power}/{total_tests} ({100*mppi_better_power/total_tests:.1f}%)"
        )
        print(
            f"  MPPI better than baseline (photosynthesis): {mppi_better_photosynthesis}/{total_tests} ({100*mppi_better_photosynthesis/total_tests:.1f}%)"
        )
        print(
            f"  MPPI better than baseline (efficiency): {mppi_better_efficiency}/{total_tests} ({100*mppi_better_efficiency/total_tests:.1f}%)"
        )

        # Best performance analysis
        best_baseline_power_idx = df_results[
            "baseline_power_difference_percent"
        ].idxmin()
        best_mppi_power_idx = df_results["mppi_power_difference_percent"].idxmin()
        best_baseline_efficiency_idx = df_results[
            "baseline_efficiency_improvement_percent"
        ].idxmax()
        best_mppi_efficiency_idx = df_results[
            "mppi_efficiency_improvement_percent"
        ].idxmax()

        print(f"\n🏆 BEST PERFORMANCE:")
        print(f"  Best BASELINE power savings:")
        print(
            f"    Reference PPFD: {df_results.loc[best_baseline_power_idx, 'reference_ppfd']:.0f} μmol/m²/s"
        )
        print(
            f"    Power saved: {df_results.loc[best_baseline_power_idx, 'baseline_power_saved_watts']:.1f}W ({df_results.loc[best_baseline_power_idx, 'baseline_power_difference_percent']:.1f}%)"
        )

        print(f"  Best MPPI power savings:")
        print(
            f"    Reference PPFD: {df_results.loc[best_mppi_power_idx, 'reference_ppfd']:.0f} μmol/m²/s"
        )
        print(
            f"    Power saved: {df_results.loc[best_mppi_power_idx, 'mppi_power_saved_watts']:.1f}W ({df_results.loc[best_mppi_power_idx, 'mppi_power_difference_percent']:.1f}%)"
        )

        print(f"  Best BASELINE efficiency improvement:")
        print(
            f"    Reference PPFD: {df_results.loc[best_baseline_efficiency_idx, 'reference_ppfd']:.0f} μmol/m²/s"
        )
        print(
            f"    Efficiency improvement: {df_results.loc[best_baseline_efficiency_idx, 'baseline_efficiency_improvement_percent']:.1f}%"
        )

        print(f"  Best MPPI efficiency improvement:")
        print(
            f"    Reference PPFD: {df_results.loc[best_mppi_efficiency_idx, 'reference_ppfd']:.0f} μmol/m²/s"
        )
        print(
            f"    Efficiency improvement: {df_results.loc[best_mppi_efficiency_idx, 'mppi_efficiency_improvement_percent']:.1f}%"
        )

        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        print(
            f"  Baseline average power savings: {df_results['baseline_power_difference_percent'].mean():.1f}% ± {df_results['baseline_power_difference_percent'].std():.1f}%"
        )
        print(
            f"  MPPI average power savings: {df_results['mppi_power_difference_percent'].mean():.1f}% ± {df_results['mppi_power_difference_percent'].std():.1f}%"
        )
        print(
            f"  Baseline average efficiency improvement: {df_results['baseline_efficiency_improvement_percent'].mean():.1f}% ± {df_results['baseline_efficiency_improvement_percent'].std():.1f}%"
        )
        print(
            f"  MPPI average efficiency improvement: {df_results['mppi_efficiency_improvement_percent'].mean():.1f}% ± {df_results['mppi_efficiency_improvement_percent'].std():.1f}%"
        )

        print(f"\n⚖️  THREE-WAY PERFORMANCE COMPARISON:")
        print(f"  📊 POWER USAGE COMPARISON (Average across all PPFD values):")
        avg_baseline_power = df_results["baseline_avg_power"].mean()
        avg_mppi_power = df_results["mppi_avg_power"].mean()
        avg_reference_power = df_results["reference_avg_power"].mean()

        print(f"    Reference Power:  {avg_reference_power:.1f}W (baseline)")
        print(
            f"    Baseline Power:   {avg_baseline_power:.1f}W ({((avg_baseline_power - avg_reference_power) / avg_reference_power * 100):+.1f}% vs Reference)"
        )
        print(
            f"    MPPI Power:       {avg_mppi_power:.1f}W ({((avg_mppi_power - avg_reference_power) / avg_reference_power * 100):+.1f}% vs Reference)"
        )
        print(
            f"    MPPI vs Baseline: {((avg_mppi_power - avg_baseline_power) / avg_baseline_power * 100):+.1f}%"
        )

        print(f"\n  🌱 PHOTOSYNTHESIS COMPARISON (Average across all PPFD values):")
        avg_baseline_pn = df_results["baseline_avg_photosynthesis"].mean()
        avg_mppi_pn = df_results["mppi_avg_photosynthesis"].mean()
        avg_reference_pn = df_results["reference_avg_photosynthesis"].mean()

        print(f"    Reference Pn:     {avg_reference_pn:.2f} μmol/m²/s (baseline)")
        print(
            f"    Baseline Pn:      {avg_baseline_pn:.2f} μmol/m²/s ({((avg_baseline_pn - avg_reference_pn) / avg_reference_pn * 100):+.1f}% vs Reference)"
        )
        print(
            f"    MPPI Pn:          {avg_mppi_pn:.2f} μmol/m²/s ({((avg_mppi_pn - avg_reference_pn) / avg_reference_pn * 100):+.1f}% vs Reference)"
        )
        print(
            f"    MPPI vs Baseline: {((avg_mppi_pn - avg_baseline_pn) / avg_baseline_pn * 100):+.1f}%"
        )

        print(f"\n  ⚡ ENERGY EFFICIENCY COMPARISON (Average across all PPFD values):")
        avg_baseline_eff = df_results["baseline_energy_efficiency"].mean()
        avg_mppi_eff = df_results["mppi_energy_efficiency"].mean()
        avg_reference_eff = df_results["reference_energy_efficiency"].mean()

        print(
            f"    Reference Efficiency: {avg_reference_eff:.4f} (μmol/m²)/(W·s) (baseline)"
        )
        print(
            f"    Baseline Efficiency:  {avg_baseline_eff:.4f} (μmol/m²)/(W·s) ({((avg_baseline_eff - avg_reference_eff) / avg_reference_eff * 100):+.1f}% vs Reference)"
        )
        print(
            f"    MPPI Efficiency:      {avg_mppi_eff:.4f} (μmol/m²)/(W·s) ({((avg_mppi_eff - avg_reference_eff) / avg_reference_eff * 100):+.1f}% vs Reference)"
        )
        print(
            f"    MPPI vs Baseline:     {((avg_mppi_eff - avg_baseline_eff) / avg_baseline_eff * 100):+.1f}%"
        )

        print(f"\n🔄 MPPI vs BASELINE DETAILED COMPARISON:")
        print(
            f"  MPPI vs Baseline power difference: {df_results['mppi_vs_baseline_power_difference_percent'].mean():.1f}% ± {df_results['mppi_vs_baseline_power_difference_percent'].std():.1f}%"
        )
        print(
            f"  MPPI vs Baseline photosynthesis difference: {df_results['mppi_vs_baseline_photosynthesis_difference_percent'].mean():.1f}% ± {df_results['mppi_vs_baseline_photosynthesis_difference_percent'].std():.1f}%"
        )
        print(
            f"  MPPI vs Baseline efficiency difference: {df_results['mppi_vs_baseline_efficiency_difference_percent'].mean():.1f}% ± {df_results['mppi_vs_baseline_efficiency_difference_percent'].std():.1f}%"
        )

        # Detailed results table
        print("\n📋 DETAILED RESULTS BY PPFD VALUE:")
        print("\n  🔋 POWER CONSUMPTION (W):")
        print(
            "PPFD  | Reference | Baseline | MPPI    | Base vs Ref | MPPI vs Ref | MPPI vs Base"
        )
        print(
            "(μmol)|    (W)    |   (W)    |   (W)   |     (%)     |     (%)     |     (%)"
        )
        print("-" * 90)

        for _, row in df_results.iterrows():
            print(
                f"{row['reference_ppfd']:5.0f} | {row['reference_avg_power']:9.1f} | {row['baseline_avg_power']:8.1f} | {row['mppi_avg_power']:7.1f} | {row['baseline_power_difference_percent']:11.1f} | {row['mppi_power_difference_percent']:11.1f} | {row['mppi_vs_baseline_power_difference_percent']:11.1f}"
            )

        print("\n  🌱 PHOTOSYNTHESIS PRODUCTION (μmol/m²/s):")
        print(
            "PPFD  | Reference |Baseline | MPPI    | Base vs Ref | MPPI vs Ref | MPPI vs Base"
        )
        print(
            "(μmol)|  (μmol/s) | (μmol/s)| (μmol/s)|     (%)     |     (%)     |     (%)"
        )
        print("-" * 90)

        for _, row in df_results.iterrows():
            baseline_vs_ref_pn = (
                (
                    row["baseline_avg_photosynthesis"]
                    - row["reference_avg_photosynthesis"]
                )
                / row["reference_avg_photosynthesis"]
            ) * 100
            mppi_vs_ref_pn = (
                (row["mppi_avg_photosynthesis"] - row["reference_avg_photosynthesis"])
                / row["reference_avg_photosynthesis"]
            ) * 100
            print(
                f"{row['reference_ppfd']:5.0f} | {row['reference_avg_photosynthesis']:9.2f} | {row['baseline_avg_photosynthesis']:7.2f} | {row['mppi_avg_photosynthesis']:7.2f} | {baseline_vs_ref_pn:11.1f} | {mppi_vs_ref_pn:11.1f} | {row['mppi_vs_baseline_photosynthesis_difference_percent']:11.1f}"
            )

        print("\n  ⚡ ENERGY EFFICIENCY ((μmol/m²)/(W·s)):")
        print(
            "PPFD  | Reference |Baseline | MPPI    | Base vs Ref | MPPI vs Ref | MPPI vs Base"
        )
        print(
            "(μmol)|    Eff    |   Eff   |   Eff   |     (%)     |     (%)     |     (%)"
        )
        print("-" * 90)

        for _, row in df_results.iterrows():
            print(
                f"{row['reference_ppfd']:5.0f} | {row['reference_energy_efficiency']:9.4f} | {row['baseline_energy_efficiency']:7.4f} | {row['mppi_energy_efficiency']:7.4f} | {row['baseline_efficiency_improvement_percent']:11.1f} | {row['mppi_efficiency_improvement_percent']:11.1f} | {row['mppi_vs_baseline_efficiency_difference_percent']:11.1f}"
            )

        # Algorithm ranking
        baseline_wins_power = sum(
            1
            for _, row in df_results.iterrows()
            if row["baseline_power_difference_percent"]
            < row["mppi_power_difference_percent"]
        )
        mppi_wins_power = total_tests - baseline_wins_power

        baseline_wins_efficiency = sum(
            1
            for _, row in df_results.iterrows()
            if row["baseline_efficiency_improvement_percent"]
            > row["mppi_efficiency_improvement_percent"]
        )
        mppi_wins_efficiency = total_tests - baseline_wins_efficiency

        print(f"\n🏁 ALGORITHM RANKING:")
        print(
            f"  Power efficiency: Baseline wins {baseline_wins_power}/{total_tests}, MPPI wins {mppi_wins_power}/{total_tests}"
        )
        print(
            f"  Energy efficiency: Baseline wins {baseline_wins_efficiency}/{total_tests}, MPPI wins {mppi_wins_efficiency}/{total_tests}"
        )

        # Overall recommendations
        avg_baseline_power_savings = df_results[
            "baseline_power_difference_percent"
        ].mean()
        avg_mppi_power_savings = df_results["mppi_power_difference_percent"].mean()
        avg_baseline_efficiency = df_results[
            "baseline_efficiency_improvement_percent"
        ].mean()
        avg_mppi_efficiency = df_results["mppi_efficiency_improvement_percent"].mean()

        print(f"\n💡 RECOMMENDATIONS:")
        if avg_baseline_power_savings < avg_mppi_power_savings:
            print(
                f"  For power savings: Use BASELINE algorithm (avg {avg_baseline_power_savings:.1f}% savings)"
            )
        else:
            print(
                f"  For power savings: Use MPPI algorithm (avg {avg_mppi_power_savings:.1f}% savings)"
            )

        if avg_baseline_efficiency > avg_mppi_efficiency:
            print(
                f"  For energy efficiency: Use BASELINE algorithm (avg {avg_baseline_efficiency:.1f}% improvement)"
            )
        else:
            print(
                f"  For energy efficiency: Use MPPI algorithm (avg {avg_mppi_efficiency:.1f}% improvement)"
            )

        # Find optimal PPFD ranges
        best_baseline_ppfd = df_results.loc[
            df_results["baseline_power_difference_percent"].idxmin(), "reference_ppfd"
        ]
        best_mppi_ppfd = df_results.loc[
            df_results["mppi_power_difference_percent"].idxmin(), "reference_ppfd"
        ]

        print(
            f"  Optimal reference PPFD for Baseline: {best_baseline_ppfd:.0f} μmol/m²/s"
        )
        print(f"  Optimal reference PPFD for MPPI: {best_mppi_ppfd:.0f} μmol/m²/s")


def main():
    """Main function to run the complete three-method PPFD comparison"""

    print("Starting Three-Method PPFD Comparison: Baseline vs MPPI vs Reference")
    print("=" * 80)

    # Create comparison instance
    comparison = ThreeMethodPPFDComparison(
        base_ambient_temp=22.0,
        max_ppfd=1000.0,
        max_power=100.0,
        thermal_resistance=1.2,
        thermal_mass=8.0,
        pn_tolerance=90.0,
    )

    # Define PPFD range to test (parameterized reference PPFD)
    ppfd_range = range(100, 1001, 100)  # 100 to 1000 μmol/m²/s in steps of 100

    # Run the comprehensive analysis
    df_results, output_dir = comparison.run_ppfd_analysis(
        ppfd_range=ppfd_range,
        duration=120,
        dt=1.0,
    )

    # Create visualizations
    if len(df_results) > 0:
        plot_filename = comparison.create_comparison_plots(df_results, output_dir)

        # Print comprehensive summary
        comparison.print_analysis_summary(df_results)

        print(f"\n✅ Three-method comparison completed successfully!")
        print(f"Results directory: {output_dir}")
    else:
        print("❌ No successful results to analyze.")

    return df_results, output_dir


if __name__ == "__main__":
    results_df, results_dir = main()
