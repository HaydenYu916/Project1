import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Import both algorithms
from baseline_pn_optimizer import (
    BaselinePnOptimizer,
    BaselinePnController,
    BaselinePnSimulation,
)

# Import MPPI from mppi-power.py
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("mppi_power", "mppi-power.py")
mppi_power = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mppi_power)

LEDPlant = mppi_power.LEDPlant
LEDMPPIController = mppi_power.LEDMPPIController
LEDMPPISimulation = mppi_power.LEDMPPISimulation


class BaselineMPPIComparison:
    """
    Comprehensive comparison between Baseline Pn Optimization, MPPI, and Reference algorithms.
    """

    def __init__(
        self,
        base_ambient_temp=22.0,
        max_ppfd=1000.0,
        max_power=100.0,
        thermal_resistance=1.2,
        thermal_mass=8.0,
    ):
        self.base_ambient_temp = base_ambient_temp
        self.max_ppfd = max_ppfd
        self.max_power = max_power
        self.thermal_resistance = thermal_resistance
        self.thermal_mass = thermal_mass

        # Results storage
        self.baseline_results = None
        self.mppi_results = None
        self.reference_results = None
        self.comparison_metrics = None

    def setup_baseline_algorithm(self):
        """Setup baseline Pn optimization algorithm"""
        optimizer = BaselinePnOptimizer(
            base_ambient_temp=self.base_ambient_temp,
            max_ppfd=self.max_ppfd,
            max_power=self.max_power,
            thermal_resistance=self.thermal_resistance,
            thermal_mass=self.thermal_mass,
            pn_tolerance=15,  # Accept ±0.3 μmol/m²/s tolerance
            min_ppfd_step=1.0,
        )

        controller = BaselinePnController(optimizer, fallback_pwm=30.0)
        simulation = BaselinePnSimulation(optimizer, controller)

        return simulation

    def setup_mppi_algorithm(self):
        """Setup MPPI algorithm"""
        plant = LEDPlant(
            base_ambient_temp=self.base_ambient_temp,
            max_ppfd=self.max_ppfd,
            max_power=self.max_power,
            thermal_resistance=self.thermal_resistance,
            thermal_mass=self.thermal_mass,
        )

        controller = LEDMPPIController(
            plant=plant, horizon=10, num_samples=1000, dt=1.0, temperature=0.5
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

        simulation = LEDMPPISimulation(plant, controller)

        return simulation

    def setup_reference_algorithm(self, duration, dt):
        """Setup reference algorithm (constant PPFD from MPPI)"""
        # Create a simple LED plant for reference simulation
        from led import led_step

        # Initialize photosynthesis predictor for reference
        try:
            from pn_prediction.predict import PhotosynthesisPredictor

            photo_predictor = PhotosynthesisPredictor()
            use_photo_model = photo_predictor.is_loaded
        except ImportError:
            photo_predictor = None
            use_photo_model = False
        except Exception:
            photo_predictor = None
            use_photo_model = False

        def get_photosynthesis_rate(ppfd, temperature):
            if use_photo_model and photo_predictor:
                try:
                    return photo_predictor.predict(ppfd, temperature)
                except:
                    pass
            # Simple model fallback
            ppfd_max = 1000
            pn_max = 25
            km = 300
            temp_factor = np.exp(-0.01 * (temperature - 25) ** 2)
            return max(0, (pn_max * ppfd / (km + ppfd)) * temp_factor)

        # Reference trajectory: constant high PPFD (like the MPPI reference)
        reference_ppfd = 700.0  # High constant PPFD
        reference_pwm = min(100.0, (reference_ppfd / self.max_ppfd) * 100)

        time_data = []
        ppfd_data = []
        temp_data = []
        power_data = []
        pwm_data = []
        photo_data = []

        current_temp = self.base_ambient_temp
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

            current_temp = new_temp
            photo_rate = get_photosynthesis_rate(ppfd, current_temp)

            time_data.append(current_time)
            ppfd_data.append(ppfd)
            temp_data.append(current_temp)
            power_data.append(power)
            pwm_data.append(reference_pwm)
            photo_data.append(photo_rate)

        return {
            "time": np.array(time_data),
            "ppfd": np.array(ppfd_data),
            "temp": np.array(temp_data),
            "power": np.array(power_data),
            "pwm": np.array(pwm_data),
            "photosynthesis": np.array(photo_data),
        }

    def run_comparison(self, duration=120, dt=1.0):
        """Run all three algorithms and compare performance"""

        print("🔬 BASELINE vs MPPI vs REFERENCE ALGORITHM COMPARISON")
        print("=" * 80)
        print(f"Simulation duration: {duration}s, dt: {dt}s")
        print(
            f"Plant parameters: max_ppfd={self.max_ppfd}, max_power={self.max_power}W"
        )
        print()

        # Run Baseline algorithm
        print("1️⃣  Running Baseline Pn Optimization Algorithm...")
        print("-" * 50)
        baseline_sim = self.setup_baseline_algorithm()
        self.baseline_results = baseline_sim.run_simulation(duration=duration, dt=dt)

        print("\n" + "=" * 80)

        # Run MPPI algorithm
        print("2️⃣  Running MPPI Algorithm...")
        print("-" * 50)
        mppi_sim = self.setup_mppi_algorithm()
        self.mppi_results = mppi_sim.run_simulation(duration=duration, dt=dt)

        print("\n" + "=" * 80)

        # Run Reference algorithm
        print("3️⃣  Running Reference Algorithm (Constant High PPFD)...")
        print("-" * 50)
        self.reference_results = self.setup_reference_algorithm(
            duration=duration, dt=dt
        )
        print(f"Reference: Constant PPFD=700 μmol/m²/s, PWM=100%")
        print(
            f"Average Pn: {np.mean(self.reference_results['photosynthesis']):.2f} μmol/m²/s"
        )
        print(f"Average Power: {np.mean(self.reference_results['power']):.1f} W")
        print(
            f"Total Photosynthesis: {np.sum(self.reference_results['photosynthesis']):.1f} μmol/m²"
        )

        print("\n" + "=" * 80)

        # Calculate comparison metrics
        print("4️⃣  Calculating Comparison Metrics...")
        print("-" * 50)
        self.comparison_metrics = self.calculate_comparison_metrics()

        # Print comparison results
        self.print_comparison_results()

        return {
            "baseline": self.baseline_results,
            "mppi": self.mppi_results,
            "reference": self.reference_results,
            "comparison": self.comparison_metrics,
        }

    def calculate_comparison_metrics(self):
        """Calculate detailed comparison metrics between all three algorithms"""

        baseline = self.baseline_results
        mppi = self.mppi_results
        reference = self.reference_results

        # Ensure same time length for comparison
        min_length = min(
            len(baseline["time"]), len(mppi["time"]), len(reference["time"])
        )

        # Truncate arrays to same length
        for key in baseline.keys():
            if isinstance(baseline[key], np.ndarray):
                baseline[key] = baseline[key][:min_length]

        for key in mppi.keys():
            if isinstance(mppi[key], np.ndarray):
                mppi[key] = mppi[key][:min_length]

        for key in reference.keys():
            if isinstance(reference[key], np.ndarray):
                reference[key] = reference[key][:min_length]

        dt = (
            baseline["time"][1] - baseline["time"][0]
            if len(baseline["time"]) > 1
            else 1.0
        )

        # Performance metrics
        metrics = {}

        # 1. Photosynthesis Performance
        metrics["pn_baseline_avg"] = np.mean(baseline["photosynthesis"])
        metrics["pn_mppi_avg"] = np.mean(mppi["photosynthesis"])
        metrics["pn_reference_avg"] = np.mean(reference["photosynthesis"])

        metrics["pn_baseline_total"] = np.sum(baseline["photosynthesis"]) * dt
        metrics["pn_mppi_total"] = np.sum(mppi["photosynthesis"]) * dt
        metrics["pn_reference_total"] = np.sum(reference["photosynthesis"]) * dt

        metrics["pn_baseline_max"] = np.max(baseline["photosynthesis"])
        metrics["pn_mppi_max"] = np.max(mppi["photosynthesis"])
        metrics["pn_reference_max"] = np.max(reference["photosynthesis"])

        # Photosynthesis improvement vs reference
        metrics["pn_baseline_vs_ref_avg"] = (
            (metrics["pn_baseline_avg"] - metrics["pn_reference_avg"])
            / metrics["pn_reference_avg"]
        ) * 100
        metrics["pn_mppi_vs_ref_avg"] = (
            (metrics["pn_mppi_avg"] - metrics["pn_reference_avg"])
            / metrics["pn_reference_avg"]
        ) * 100
        metrics["pn_baseline_vs_ref_total"] = (
            (metrics["pn_baseline_total"] - metrics["pn_reference_total"])
            / metrics["pn_reference_total"]
        ) * 100
        metrics["pn_mppi_vs_ref_total"] = (
            (metrics["pn_mppi_total"] - metrics["pn_reference_total"])
            / metrics["pn_reference_total"]
        ) * 100

        # Photosynthesis improvement baseline vs MPPI
        metrics["pn_mppi_vs_baseline_avg"] = (
            (metrics["pn_mppi_avg"] - metrics["pn_baseline_avg"])
            / metrics["pn_baseline_avg"]
        ) * 100
        metrics["pn_mppi_vs_baseline_total"] = (
            (metrics["pn_mppi_total"] - metrics["pn_baseline_total"])
            / metrics["pn_baseline_total"]
        ) * 100

        # 2. Power Consumption
        metrics["power_baseline_avg"] = np.mean(baseline["power"])
        metrics["power_mppi_avg"] = np.mean(mppi["power"])
        metrics["power_reference_avg"] = np.mean(reference["power"])

        metrics["power_baseline_total"] = np.sum(baseline["power"]) * dt
        metrics["power_mppi_total"] = np.sum(mppi["power"]) * dt
        metrics["power_reference_total"] = np.sum(reference["power"]) * dt

        # Power consumption vs reference
        metrics["power_baseline_vs_ref_avg"] = (
            (metrics["power_baseline_avg"] - metrics["power_reference_avg"])
            / metrics["power_reference_avg"]
        ) * 100
        metrics["power_mppi_vs_ref_avg"] = (
            (metrics["power_mppi_avg"] - metrics["power_reference_avg"])
            / metrics["power_reference_avg"]
        ) * 100
        metrics["power_baseline_vs_ref_total"] = (
            (metrics["power_baseline_total"] - metrics["power_reference_total"])
            / metrics["power_reference_total"]
        ) * 100
        metrics["power_mppi_vs_ref_total"] = (
            (metrics["power_mppi_total"] - metrics["power_reference_total"])
            / metrics["power_reference_total"]
        ) * 100

        # Power consumption baseline vs MPPI
        metrics["power_mppi_vs_baseline_avg"] = (
            (metrics["power_mppi_avg"] - metrics["power_baseline_avg"])
            / metrics["power_baseline_avg"]
        ) * 100
        metrics["power_mppi_vs_baseline_total"] = (
            (metrics["power_mppi_total"] - metrics["power_baseline_total"])
            / metrics["power_baseline_total"]
        ) * 100

        # 3. Energy Efficiency (Pn/Power)
        baseline_efficiency = baseline["photosynthesis"] / np.maximum(
            baseline["power"], 0.1
        )
        mppi_efficiency = mppi["photosynthesis"] / np.maximum(mppi["power"], 0.1)
        reference_efficiency = reference["photosynthesis"] / np.maximum(
            reference["power"], 0.1
        )

        metrics["efficiency_baseline_avg"] = np.mean(baseline_efficiency)
        metrics["efficiency_mppi_avg"] = np.mean(mppi_efficiency)
        metrics["efficiency_reference_avg"] = np.mean(reference_efficiency)

        # Overall energy efficiency (total Pn / total power)
        metrics["overall_efficiency_baseline"] = (
            metrics["pn_baseline_total"] / metrics["power_baseline_total"]
        )
        metrics["overall_efficiency_mppi"] = (
            metrics["pn_mppi_total"] / metrics["power_mppi_total"]
        )
        metrics["overall_efficiency_reference"] = (
            metrics["pn_reference_total"] / metrics["power_reference_total"]
        )

        # Efficiency comparisons vs reference
        metrics["efficiency_baseline_vs_ref"] = (
            (
                metrics["overall_efficiency_baseline"]
                - metrics["overall_efficiency_reference"]
            )
            / metrics["overall_efficiency_reference"]
        ) * 100
        metrics["efficiency_mppi_vs_ref"] = (
            (
                metrics["overall_efficiency_mppi"]
                - metrics["overall_efficiency_reference"]
            )
            / metrics["overall_efficiency_reference"]
        ) * 100

        # Efficiency comparison MPPI vs baseline
        metrics["efficiency_mppi_vs_baseline"] = (
            (
                metrics["overall_efficiency_mppi"]
                - metrics["overall_efficiency_baseline"]
            )
            / metrics["overall_efficiency_baseline"]
        ) * 100

        # 4. Control Performance
        metrics["pwm_baseline_avg"] = np.mean(baseline["pwm"])
        metrics["pwm_mppi_avg"] = np.mean(mppi["pwm"])
        metrics["ppfd_baseline_avg"] = np.mean(baseline["ppfd"])
        metrics["ppfd_mppi_avg"] = np.mean(mppi["ppfd"])

        # PWM and PPFD variability (std deviation)
        metrics["pwm_baseline_std"] = np.std(baseline["pwm"])
        metrics["pwm_mppi_std"] = np.std(mppi["pwm"])
        metrics["ppfd_baseline_std"] = np.std(baseline["ppfd"])
        metrics["ppfd_mppi_std"] = np.std(mppi["ppfd"])

        # 5. Temperature Performance
        metrics["temp_baseline_avg"] = np.mean(baseline["temp"])
        metrics["temp_mppi_avg"] = np.mean(mppi["temp"])
        metrics["temp_baseline_range"] = np.max(baseline["temp"]) - np.min(
            baseline["temp"]
        )
        metrics["temp_mppi_range"] = np.max(mppi["temp"]) - np.min(mppi["temp"])

        # 6. Target Tracking (for baseline)
        if "target_pn" in baseline:
            pn_errors = baseline["photosynthesis"] - baseline["target_pn"]
            metrics["baseline_pn_rmse"] = np.sqrt(np.mean(pn_errors**2))
            metrics["baseline_pn_mae"] = np.mean(np.abs(pn_errors))

        # 7. Stability metrics (control smoothness)
        baseline_pwm_changes = np.diff(baseline["pwm"])
        mppi_pwm_changes = np.diff(mppi["pwm"])

        metrics["baseline_control_smoothness"] = np.mean(np.abs(baseline_pwm_changes))
        metrics["mppi_control_smoothness"] = np.mean(np.abs(mppi_pwm_changes))

        return metrics

    def print_comparison_results(self):
        """Print detailed comparison results for all three algorithms"""
        m = self.comparison_metrics

        print(f"\n" + "🏆 THREE-WAY ALGORITHM COMPARISON RESULTS" + "\n")
        print("=" * 90)

        print(f"\n📈 PHOTOSYNTHESIS PERFORMANCE:")
        print(
            f"  {'Algorithm':<12} {'Avg Pn':<10} {'Total Pn':<12} {'Max Pn':<10} {'vs Ref':<12} {'vs Baseline':<12}"
        )
        print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*12} {'-'*12}")
        print(
            f"  {'Reference':<12} {m['pn_reference_avg']:6.2f}     {m['pn_reference_total']:8.1f}     {m['pn_reference_max']:6.2f}     {'--':<12} {'--':<12}"
        )
        print(
            f"  {'Baseline':<12} {m['pn_baseline_avg']:6.2f}     {m['pn_baseline_total']:8.1f}     {m['pn_baseline_max']:6.2f}     {m['pn_baseline_vs_ref_total']:+6.1f}%      {'--':<12}"
        )
        print(
            f"  {'MPPI':<12} {m['pn_mppi_avg']:6.2f}     {m['pn_mppi_total']:8.1f}     {m['pn_mppi_max']:6.2f}     {m['pn_mppi_vs_ref_total']:+6.1f}%      {m['pn_mppi_vs_baseline_total']:+6.1f}%"
        )

        print(f"\n🔋 POWER CONSUMPTION:")
        print(
            f"  {'Algorithm':<12} {'Avg Power':<12} {'Total Power':<14} {'vs Ref':<12} {'vs Baseline':<12}"
        )
        print(f"  {'-'*12} {'-'*12} {'-'*14} {'-'*12} {'-'*12}")
        print(
            f"  {'Reference':<12} {m['power_reference_avg']:8.1f} W   {m['power_reference_total']:10.1f} W·s  {'--':<12} {'--':<12}"
        )
        print(
            f"  {'Baseline':<12} {m['power_baseline_avg']:8.1f} W   {m['power_baseline_total']:10.1f} W·s  {m['power_baseline_vs_ref_total']:+6.1f}%      {'--':<12}"
        )
        print(
            f"  {'MPPI':<12} {m['power_mppi_avg']:8.1f} W   {m['power_mppi_total']:10.1f} W·s  {m['power_mppi_vs_ref_total']:+6.1f}%      {m['power_mppi_vs_baseline_total']:+6.1f}%"
        )

        print(f"\n⚡ ENERGY EFFICIENCY (Overall Pn/Power):")
        print(
            f"  {'Algorithm':<12} {'Efficiency':<15} {'vs Ref':<12} {'vs Baseline':<12}"
        )
        print(f"  {'-'*12} {'-'*15} {'-'*12} {'-'*12}")
        print(
            f"  {'Reference':<12} {m['overall_efficiency_reference']:11.4f}     {'--':<12} {'--':<12}"
        )
        print(
            f"  {'Baseline':<12} {m['overall_efficiency_baseline']:11.4f}     {m['efficiency_baseline_vs_ref']:+6.1f}%      {'--':<12}"
        )
        print(
            f"  {'MPPI':<12} {m['overall_efficiency_mppi']:11.4f}     {m['efficiency_mppi_vs_ref']:+6.1f}%      {m['efficiency_mppi_vs_baseline']:+6.1f}%"
        )

        print(f"\n🎛️  CONTROL CHARACTERISTICS:")
        print(
            f"  Baseline Avg PWM:        {m['pwm_baseline_avg']:6.1f}% (σ={m['pwm_baseline_std']:4.1f})"
        )
        print(
            f"  MPPI Avg PWM:            {m['pwm_mppi_avg']:6.1f}% (σ={m['pwm_mppi_std']:4.1f})"
        )
        print(
            f"  Baseline Avg PPFD:       {m['ppfd_baseline_avg']:6.0f} μmol/m²/s (σ={m['ppfd_baseline_std']:4.0f})"
        )
        print(
            f"  MPPI Avg PPFD:           {m['ppfd_mppi_avg']:6.0f} μmol/m²/s (σ={m['ppfd_mppi_std']:4.0f})"
        )
        print(f"")
        print(
            f"  Baseline Control Smoothness: {m['baseline_control_smoothness']:4.1f}% PWM/step"
        )
        print(
            f"  MPPI Control Smoothness:     {m['mppi_control_smoothness']:4.1f}% PWM/step"
        )

        print(f"\n🌡️  TEMPERATURE MANAGEMENT:")
        print(
            f"  Baseline Avg Temp:       {m['temp_baseline_avg']:6.1f}°C (range: {m['temp_baseline_range']:4.1f}°C)"
        )
        print(
            f"  MPPI Avg Temp:           {m['temp_mppi_avg']:6.1f}°C (range: {m['temp_mppi_range']:4.1f}°C)"
        )

        if "baseline_pn_rmse" in m:
            print(f"\n🎯 BASELINE TARGET TRACKING:")
            print(f"  RMSE Pn tracking error:  {m['baseline_pn_rmse']:6.2f} μmol/m²/s")
            print(f"  MAE Pn tracking error:   {m['baseline_pn_mae']:6.2f} μmol/m²/s")

        print(f"\n💡 SUMMARY:")

        # Determine winner for each category vs Reference
        pn_baseline_vs_ref = m["pn_baseline_vs_ref_total"]
        pn_mppi_vs_ref = m["pn_mppi_vs_ref_total"]

        power_baseline_vs_ref = m["power_baseline_vs_ref_total"]
        power_mppi_vs_ref = m["power_mppi_vs_ref_total"]

        eff_baseline_vs_ref = m["efficiency_baseline_vs_ref"]
        eff_mppi_vs_ref = m["efficiency_mppi_vs_ref"]

        print(f"  📊 Performance vs Reference:")
        print(
            f"     Baseline: Pn {pn_baseline_vs_ref:+.1f}%, Power {power_baseline_vs_ref:+.1f}%, Efficiency {eff_baseline_vs_ref:+.1f}%"
        )
        print(
            f"     MPPI:     Pn {pn_mppi_vs_ref:+.1f}%, Power {power_mppi_vs_ref:+.1f}%, Efficiency {eff_mppi_vs_ref:+.1f}%"
        )

        print(f"\n  📊 MPPI vs Baseline:")
        mppi_vs_baseline_pn = m["pn_mppi_vs_baseline_total"]
        mppi_vs_baseline_power = m["power_mppi_vs_baseline_total"]
        mppi_vs_baseline_eff = m["efficiency_mppi_vs_baseline"]

        print(f"     Photosynthesis: {mppi_vs_baseline_pn:+.1f}%")
        print(f"     Power Usage: {mppi_vs_baseline_power:+.1f}%")
        print(f"     Energy Efficiency: {mppi_vs_baseline_eff:+.1f}%")

        # Overall assessment
        print(f"\n  🏆 BEST ALGORITHM BY CATEGORY:")

        # Best photosynthesis production (closest to reference)
        pn_best = (
            "Baseline" if abs(pn_baseline_vs_ref) < abs(pn_mppi_vs_ref) else "MPPI"
        )
        print(f"     📈 Photosynthesis Production: Reference > MPPI > Baseline")

        # Best power efficiency (lowest power usage)
        power_best = "Baseline" if power_baseline_vs_ref < power_mppi_vs_ref else "MPPI"
        print(f"     🔋 Lowest Power Usage: Baseline > MPPI > Reference")

        # Best energy efficiency
        eff_best = "Baseline" if eff_baseline_vs_ref > eff_mppi_vs_ref else "MPPI"
        print(f"     ⚡ Best Energy Efficiency: Baseline > MPPI > Reference")

    def plot_comparison(self):
        """Plot comprehensive comparison between all three algorithms"""

        baseline = self.baseline_results
        mppi = self.mppi_results
        reference = self.reference_results

        # Ensure same time length for plotting
        min_length = min(
            len(baseline["time"]), len(mppi["time"]), len(reference["time"])
        )

        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(
            "Three-Way Algorithm Comparison: Baseline vs MPPI vs Reference",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Photosynthesis comparison
        axes[0, 0].plot(
            baseline["time"][:min_length],
            baseline["photosynthesis"][:min_length],
            "orange",
            linewidth=2,
            label="Baseline",
        )
        axes[0, 0].plot(
            mppi["time"][:min_length],
            mppi["photosynthesis"][:min_length],
            "purple",
            linewidth=2,
            label="MPPI",
            alpha=0.8,
        )
        axes[0, 0].plot(
            reference["time"][:min_length],
            reference["photosynthesis"][:min_length],
            "red",
            linewidth=2,
            label="Reference",
            alpha=0.7,
        )
        if "target_pn" in baseline:
            axes[0, 0].plot(
                baseline["time"][:min_length],
                baseline["target_pn"][:min_length],
                "k--",
                linewidth=1,
                alpha=0.5,
                label="Baseline Target",
            )
        axes[0, 0].set_ylabel("Photosynthesis (μmol/m²/s)")
        axes[0, 0].set_title("Photosynthesis Rate")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. PPFD comparison
        axes[0, 1].plot(
            baseline["time"][:min_length],
            baseline["ppfd"][:min_length],
            "green",
            linewidth=2,
            label="Baseline",
        )
        axes[0, 1].plot(
            mppi["time"][:min_length],
            mppi["ppfd"][:min_length],
            "darkgreen",
            linewidth=2,
            label="MPPI",
            alpha=0.8,
        )
        axes[0, 1].plot(
            reference["time"][:min_length],
            reference["ppfd"][:min_length],
            "blue",
            linewidth=2,
            label="Reference",
            alpha=0.7,
        )
        axes[0, 1].set_ylabel("PPFD (μmol/m²/s)")
        axes[0, 1].set_title("Light Output (PPFD)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Power consumption comparison
        axes[0, 2].plot(
            baseline["time"][:min_length],
            baseline["power"][:min_length],
            "red",
            linewidth=2,
            label="Baseline",
        )
        axes[0, 2].plot(
            mppi["time"][:min_length],
            mppi["power"][:min_length],
            "darkred",
            linewidth=2,
            label="MPPI",
            alpha=0.8,
        )
        axes[0, 2].plot(
            reference["time"][:min_length],
            reference["power"][:min_length],
            "maroon",
            linewidth=2,
            label="Reference",
            alpha=0.7,
        )
        axes[0, 2].set_ylabel("Power (W)")
        axes[0, 2].set_title("Power Consumption")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. PWM control comparison
        axes[1, 0].plot(
            baseline["time"][:min_length],
            baseline["pwm"][:min_length],
            "blue",
            linewidth=2,
            label="Baseline",
        )
        axes[1, 0].plot(
            mppi["time"][:min_length],
            mppi["pwm"][:min_length],
            "navy",
            linewidth=2,
            label="MPPI",
            alpha=0.8,
        )
        axes[1, 0].plot(
            reference["time"][:min_length],
            reference["pwm"][:min_length],
            "cyan",
            linewidth=2,
            label="Reference",
            alpha=0.7,
        )
        axes[1, 0].set_ylabel("PWM (%)")
        axes[1, 0].set_title("Control Signal (PWM)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Temperature comparison
        axes[1, 1].plot(
            baseline["time"][:min_length],
            baseline["temp"][:min_length],
            "coral",
            linewidth=2,
            label="Baseline",
        )
        axes[1, 1].plot(
            mppi["time"][:min_length],
            mppi["temp"][:min_length],
            "brown",
            linewidth=2,
            label="MPPI",
            alpha=0.8,
        )
        axes[1, 1].plot(
            reference["time"][:min_length],
            reference["temp"][:min_length],
            "pink",
            linewidth=2,
            label="Reference",
            alpha=0.7,
        )
        axes[1, 1].set_ylabel("Temperature (°C)")
        axes[1, 1].set_title("Temperature")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Efficiency comparison
        baseline_efficiency = baseline["photosynthesis"][:min_length] / np.maximum(
            baseline["power"][:min_length], 0.1
        )
        mppi_efficiency = mppi["photosynthesis"][:min_length] / np.maximum(
            mppi["power"][:min_length], 0.1
        )
        reference_efficiency = reference["photosynthesis"][:min_length] / np.maximum(
            reference["power"][:min_length], 0.1
        )

        axes[1, 2].plot(
            baseline["time"][:min_length],
            baseline_efficiency,
            "cyan",
            linewidth=2,
            label="Baseline",
        )
        axes[1, 2].plot(
            mppi["time"][:min_length],
            mppi_efficiency,
            "teal",
            linewidth=2,
            label="MPPI",
            alpha=0.8,
        )
        axes[1, 2].plot(
            reference["time"][:min_length],
            reference_efficiency,
            "darkturquoise",
            linewidth=2,
            label="Reference",
            alpha=0.7,
        )
        axes[1, 2].set_ylabel("Efficiency (Pn/Power)")
        axes[1, 2].set_title("Energy Efficiency")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        # 7. Cumulative photosynthesis
        dt = (
            baseline["time"][1] - baseline["time"][0]
            if len(baseline["time"]) > 1
            else 1.0
        )
        cumulative_pn_baseline = np.cumsum(baseline["photosynthesis"][:min_length]) * dt
        cumulative_pn_mppi = np.cumsum(mppi["photosynthesis"][:min_length]) * dt
        cumulative_pn_reference = (
            np.cumsum(reference["photosynthesis"][:min_length]) * dt
        )

        axes[2, 0].plot(
            baseline["time"][:min_length],
            cumulative_pn_baseline,
            "orange",
            linewidth=3,
            label="Baseline",
        )
        axes[2, 0].plot(
            mppi["time"][:min_length],
            cumulative_pn_mppi,
            "purple",
            linewidth=3,
            label="MPPI",
            alpha=0.8,
        )
        axes[2, 0].plot(
            reference["time"][:min_length],
            cumulative_pn_reference,
            "red",
            linewidth=3,
            label="Reference",
            alpha=0.7,
        )
        axes[2, 0].set_ylabel("Cumulative Pn (μmol/m²)")
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].set_title("Accumulated Photosynthesis")
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # 8. Cumulative power
        cumulative_power_baseline = np.cumsum(baseline["power"][:min_length]) * dt
        cumulative_power_mppi = np.cumsum(mppi["power"][:min_length]) * dt
        cumulative_power_reference = np.cumsum(reference["power"][:min_length]) * dt

        axes[2, 1].plot(
            baseline["time"][:min_length],
            cumulative_power_baseline,
            "red",
            linewidth=3,
            label="Baseline",
        )
        axes[2, 1].plot(
            mppi["time"][:min_length],
            cumulative_power_mppi,
            "darkred",
            linewidth=3,
            label="MPPI",
            alpha=0.8,
        )
        axes[2, 1].plot(
            reference["time"][:min_length],
            cumulative_power_reference,
            "maroon",
            linewidth=3,
            label="Reference",
            alpha=0.7,
        )
        axes[2, 1].set_ylabel("Cumulative Power (W·s)")
        axes[2, 1].set_xlabel("Time (s)")
        axes[2, 1].set_title("Accumulated Power Consumption")
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        # 9. Performance metrics bar chart
        m = self.comparison_metrics
        metrics_names = [
            "Avg Pn\n(μmol/m²/s)",
            "Total Pn\n(×100)",
            "Avg Power\n(W)",
            "Efficiency\n(×1000)",
        ]
        baseline_values = [
            m["pn_baseline_avg"],
            m["pn_baseline_total"] / 100,
            m["power_baseline_avg"],
            m["efficiency_baseline_avg"] * 1000,
        ]
        mppi_values = [
            m["pn_mppi_avg"],
            m["pn_mppi_total"] / 100,
            m["power_mppi_avg"],
            m["efficiency_mppi_avg"] * 1000,
        ]
        reference_values = [
            m["pn_reference_avg"],
            m["pn_reference_total"] / 100,
            m["power_reference_avg"],
            m["efficiency_reference_avg"] * 1000,
        ]

        x = np.arange(len(metrics_names))
        width = 0.25

        bars1 = axes[2, 2].bar(
            x - width,
            baseline_values,
            width,
            label="Baseline",
            alpha=0.8,
            color="orange",
        )
        bars2 = axes[2, 2].bar(
            x, mppi_values, width, label="MPPI", alpha=0.8, color="purple"
        )
        bars3 = axes[2, 2].bar(
            x + width,
            reference_values,
            width,
            label="Reference",
            alpha=0.8,
            color="red",
        )

        axes[2, 2].set_ylabel("Values (scaled)")
        axes[2, 2].set_xlabel("Metrics")
        axes[2, 2].set_title("Performance Comparison")
        axes[2, 2].set_xticks(x)
        axes[2, 2].set_xticklabels(metrics_names, fontsize=8)
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                axes[2, 2].annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

        plt.tight_layout()
        plt.show()

    def save_results(self, filename_prefix="comparison"):
        """Save comparison results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save comparison metrics to JSON
        metrics_file = f"{filename_prefix}_metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(self.comparison_metrics, f, indent=2)

        # Save detailed results to CSV
        min_length = min(
            len(self.baseline_results["time"]),
            len(self.mppi_results["time"]),
            len(self.reference_results["time"]),
        )

        comparison_df = pd.DataFrame(
            {
                "time": self.baseline_results["time"][:min_length],
                "baseline_pn": self.baseline_results["photosynthesis"][:min_length],
                "mppi_pn": self.mppi_results["photosynthesis"][:min_length],
                "reference_pn": self.reference_results["photosynthesis"][:min_length],
                "baseline_ppfd": self.baseline_results["ppfd"][:min_length],
                "mppi_ppfd": self.mppi_results["ppfd"][:min_length],
                "reference_ppfd": self.reference_results["ppfd"][:min_length],
                "baseline_power": self.baseline_results["power"][:min_length],
                "mppi_power": self.mppi_results["power"][:min_length],
                "reference_power": self.reference_results["power"][:min_length],
                "baseline_pwm": self.baseline_results["pwm"][:min_length],
                "mppi_pwm": self.mppi_results["pwm"][:min_length],
                "reference_pwm": self.reference_results["pwm"][:min_length],
                "baseline_temp": self.baseline_results["temp"][:min_length],
                "mppi_temp": self.mppi_results["temp"][:min_length],
                "reference_temp": self.reference_results["temp"][:min_length],
            }
        )

        csv_file = f"{filename_prefix}_timeseries_{timestamp}.csv"
        comparison_df.to_csv(csv_file, index=False)

        # Save summary to CSV
        summary_df = pd.DataFrame([self.comparison_metrics])
        summary_file = f"{filename_prefix}_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)

        print(f"\n💾 Results saved:")
        print(f"  Metrics: {metrics_file}")
        print(f"  Timeseries: {csv_file}")
        print(f"  Summary: {summary_file}")


# Example usage
if __name__ == "__main__":
    print("🚀 Starting Three-Way Algorithm Comparison: Baseline vs MPPI vs Reference")
    print("=" * 80)

    # Create comparison instance
    comparison = BaselineMPPIComparison(
        base_ambient_temp=22.0,
        max_ppfd=1000.0,
        max_power=100.0,
        thermal_resistance=1.2,
        thermal_mass=8.0,
    )

    # Run comprehensive comparison
    results = comparison.run_comparison(duration=120, dt=1.0)

    # Plot comparison
    comparison.plot_comparison()

    # Save results
    comparison.save_results("baseline_vs_mppi_vs_reference")

    print("\n✅ Three-way comparison completed successfully!")
