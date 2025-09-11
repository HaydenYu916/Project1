import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from datetime import datetime
import os
import json

warnings.filterwarnings("ignore")

# Import the simulation components
import sys
import importlib.util

# Load the baseline module (same as baseline_vs_mppi_comparison.py)
from baseline_pn_optimizer import (
    BaselinePnOptimizer,
    BaselinePnController,
    BaselinePnSimulation,
)

# Load the MPPI module (same as baseline_vs_mppi_comparison.py)
spec_mppi = importlib.util.spec_from_file_location("mppi_power", "mppi-power.py")
mppi_module = importlib.util.module_from_spec(spec_mppi)
spec_mppi.loader.exec_module(mppi_module)

# Classes already imported above

# Import MPPI classes
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


def run_baseline_ppfd_power_analysis(
    ppfd_range=None,
    duration=120,
    dt=1.0,
    output_dir="baseline_ppfd_analysis_results",
):
    """
    Run three-way comparison (Baseline, MPPI, Reference) across different PPFD values
    matching the format and metrics of ppfd_power_analysis.py

    Args:
        ppfd_range: List of reference PPFD values to test
        duration: Simulation duration in seconds
        dt: Time step in seconds
        output_dir: Directory to save results
    """

    if ppfd_range is None:
        ppfd_range = [200, 300, 400, 500, 600, 700]

    print("Three-Way PPFD Power Analysis: Baseline vs MPPI vs Reference")
    print("=" * 70)
    print(f"Testing PPFD values: {ppfd_range}")
    print(f"Simulation duration: {duration}s")
    print(f"Time step: {dt}s")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize results storage
    results_summary = []

    # Plant parameters (consistent across all tests)
    plant_params = {
        "base_ambient_temp": 22.0,
        "max_ppfd": 1000.0,
        "max_power": 100.0,
        "thermal_resistance": 1.2,
        "thermal_mass": 8.0,
    }

    # Controller parameters (matching baseline_vs_mppi_comparison.py)
    controller_params = {
        "pn_tolerance": 5,  # Accept ±0.3 μmol/m²/s tolerance
        "min_ppfd_step": 1.0,
        "fallback_pwm": 30.0,
    }

    # Run simulations for each PPFD value
    for i, ppfd_value in enumerate(ppfd_range):
        print(f"\n[{i+1}/{len(ppfd_range)}] Testing PPFD = {ppfd_value} μmol/m²/s")

        try:
            # 1. Run Baseline Algorithm (matching baseline_vs_mppi_comparison.py)
            print(f"Running Baseline...")
            baseline_optimizer = BaselinePnOptimizer(
                **plant_params,
                pn_tolerance=controller_params["pn_tolerance"],
                min_ppfd_step=controller_params["min_ppfd_step"],
            )
            baseline_controller = BaselinePnController(
                baseline_optimizer, fallback_pwm=controller_params["fallback_pwm"]
            )
            baseline_simulation = BaselinePnSimulation(
                baseline_optimizer, baseline_controller, reference_ppfd=ppfd_value
            )
            baseline_results = baseline_simulation.run_simulation(
                duration=duration, dt=dt
            )

            # 2. Run MPPI Algorithm (matching baseline_vs_mppi_comparison.py)
            print("  Running MPPI...")
            mppi_plant = LEDPlant(**plant_params)
            mppi_controller = LEDMPPIController(
                plant=mppi_plant, horizon=10, num_samples=1000, dt=1.0, temperature=0.5
            )

            # Configure MPPI weights for photosynthesis maximization
            mppi_controller.set_weights(
                Q_photo=5.0,
                Q_ref=25.0,
                R_pwm=0.001,
                R_dpwm=0.05,
                R_power=0.08,
            )

            # Set constraints
            mppi_controller.set_constraints(
                pwm_min=0.0, pwm_max=100.0, temp_min=20.0, temp_max=29.0
            )
            mppi_controller.set_mppi_params(
                num_samples=1000, temperature=0.5, pwm_std=10.0
            )

            mppi_simulation = LEDMPPISimulation(mppi_plant, mppi_controller)
            mppi_results = mppi_simulation.run_simulation(duration=duration, dt=dt)

            # 3. Run Reference Algorithm (constant high PPFD)
            print("  Running Reference...")
            ref_results = run_reference_simulation(
                ppfd_value, plant_params, duration, dt
            )

            # Calculate MPPI statistics manually from results
            # Get basic metrics from mppi_results
            mppi_avg_power = mppi_results["power"].mean()
            mppi_avg_photosynthesis = mppi_results["photosynthesis"].mean()
            mppi_total_power = mppi_results["power"].sum()
            mppi_avg_pwm = mppi_results["pwm"].mean()
            mppi_energy_efficiency = (
                mppi_results["photosynthesis"].sum() / mppi_results["power"].sum()
            )

            # Calculate comparison metrics vs reference
            mppi_power_saved = ref_results["power"].mean() - mppi_avg_power
            mppi_power_diff_percent = (
                (mppi_avg_power - ref_results["power"].mean())
                / ref_results["power"].mean()
            ) * 100
            mppi_efficiency_improvement = (
                (
                    mppi_energy_efficiency
                    - (ref_results["photosynthesis"].sum() / ref_results["power"].sum())
                )
                / (ref_results["photosynthesis"].sum() / ref_results["power"].sum())
            ) * 100

            # Calculate temperature satisfaction (assuming temp constraints 20-29°C)
            temp_violations = np.sum(
                (mppi_results["temp"] < 20.0) | (mppi_results["temp"] > 29.0)
            )
            temp_satisfaction = (1 - temp_violations / len(mppi_results["temp"])) * 100

            # Calculate additional baseline vs reference metrics
            baseline_vs_ref_power_diff = (
                (baseline_results["power"].mean() - ref_results["power"].mean())
                / ref_results["power"].mean()
            ) * 100
            baseline_vs_ref_efficiency = (
                baseline_results["photosynthesis"].sum()
                / baseline_results["power"].sum()
            ) / (ref_results["photosynthesis"].sum() / ref_results["power"].sum())

            # Store results matching original format
            result_entry = {
                "ppfd_reference": ppfd_value,
                # Baseline metrics
                "baseline_avg_power": baseline_results["power"].mean(),
                "baseline_avg_photosynthesis": baseline_results[
                    "photosynthesis"
                ].mean(),
                "baseline_energy_efficiency": baseline_results["photosynthesis"].sum()
                / baseline_results["power"].sum(),
                "baseline_avg_pwm": baseline_results["pwm"].mean(),
                "baseline_total_power": baseline_results["power"].sum(),
                # MPPI metrics (calculated manually)
                "mppi_avg_power": mppi_avg_power,
                "mppi_avg_photosynthesis": mppi_avg_photosynthesis,
                "mppi_energy_efficiency": mppi_energy_efficiency,
                "mppi_avg_pwm": mppi_avg_pwm,
                "mppi_total_power": mppi_total_power,
                # Reference metrics
                "reference_avg_power": ref_results["power"].mean(),
                "reference_avg_photosynthesis": ref_results["photosynthesis"].mean(),
                "reference_energy_efficiency": ref_results["photosynthesis"].sum()
                / ref_results["power"].sum(),
                "reference_avg_pwm": ref_results["pwm"].mean(),
                "reference_total_power": ref_results["power"].sum(),
                # Comparative metrics (baseline vs reference)
                "baseline_power_saved_watts": ref_results["power"].mean()
                - baseline_results["power"].mean(),
                "baseline_power_difference_percent": baseline_vs_ref_power_diff,
                "baseline_efficiency_improvement_percent": (
                    baseline_vs_ref_efficiency - 1
                )
                * 100,
                # Comparative metrics (MPPI vs reference)
                "mppi_power_saved_watts": mppi_power_saved,
                "mppi_power_difference_percent": mppi_power_diff_percent,
                "mppi_efficiency_improvement_percent": mppi_efficiency_improvement,
                # Temperature satisfaction
                "temperature_satisfaction": temp_satisfaction,
            }

            results_summary.append(result_entry)

            # Print brief summary
            baseline_power_saved = result_entry["baseline_power_saved_watts"]
            mppi_power_saved = result_entry["mppi_power_saved_watts"]
            print(
                f"  Baseline power saved: {baseline_power_saved:.1f}W ({baseline_vs_ref_power_diff:+.1f}%)"
            )
            print(
                f"  MPPI power saved: {mppi_power_saved:.1f}W ({result_entry['mppi_power_difference_percent']:+.1f}%)"
            )

        except Exception as e:
            print(f"  ERROR: Failed to run simulation for PPFD {ppfd_value}: {e}")
            continue

    # Convert to DataFrame for easier analysis
    df_results = pd.DataFrame(results_summary)

    # Check if we have any successful results
    if len(df_results) == 0:
        print("ERROR: No successful simulations completed!")
        return pd.DataFrame(), output_dir

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(
        output_dir, f"baseline_ppfd_power_analysis_{timestamp}.csv"
    )
    df_results.to_csv(csv_filename, index=False)

    print(f"\nResults saved to: {csv_filename}")

    return df_results, output_dir


def run_reference_simulation(ref_ppfd, plant_params, duration, dt):
    """Run reference algorithm (constant PPFD) matching baseline_vs_mppi_comparison.py exactly"""

    from led import led_step

    # Initialize photosynthesis predictor for reference (matching comparison file)
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
        # Simple model fallback (matching comparison file exactly)
        ppfd_max = 1000
        pn_max = 25
        km = 300
        temp_factor = np.exp(-0.01 * (temperature - 25) ** 2)
        return max(0, (pn_max * ppfd / (km + ppfd)) * temp_factor)

    # Reference trajectory: use the configurable reference PPFD
    reference_ppfd = ref_ppfd  # Use the parameter instead of hardcoded value
    reference_pwm = min(100.0, (reference_ppfd / plant_params["max_ppfd"]) * 100)

    # Initialize state
    current_temp = plant_params["base_ambient_temp"]

    # Data storage
    time_data = []
    ppfd_data = []
    temp_data = []
    power_data = []
    pwm_data = []
    photosynthesis_data = []

    # Simulation loop (matching comparison file exactly)
    steps = int(duration / dt)
    for k in range(steps):
        current_time = k * dt

        # Apply constant reference PWM (matching comparison file)
        ppfd, new_temp, power, efficiency = led_step(
            pwm_percent=reference_pwm,
            ambient_temp=current_temp,
            base_ambient_temp=plant_params["base_ambient_temp"],
            dt=dt,
            max_ppfd=plant_params["max_ppfd"],
            max_power=plant_params["max_power"],
            thermal_resistance=plant_params["thermal_resistance"],
            thermal_mass=plant_params["thermal_mass"],
        )

        # Calculate photosynthesis
        photosynthesis_rate = get_photosynthesis_rate(ppfd, new_temp)

        # Update state
        current_temp = new_temp

        # Store data (matching comparison file)
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


def create_three_way_analysis_plots(df_results, output_dir):
    """Create comprehensive plots for the three-way PPFD analysis matching original format"""

    # Check if we have results to plot
    if len(df_results) == 0:
        print("No results to plot!")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Three-Way Power Analysis: Baseline vs MPPI vs Reference", fontsize=16)

    ppfd_values = df_results["ppfd_reference"]

    # 1. Power Consumption Comparison
    axes[0, 0].plot(
        ppfd_values,
        df_results["baseline_avg_power"],
        "g-o",
        linewidth=2,
        label="Baseline Power",
    )
    axes[0, 0].plot(
        ppfd_values,
        df_results["mppi_avg_power"],
        "b-s",
        linewidth=2,
        label="MPPI Power",
    )
    axes[0, 0].plot(
        ppfd_values,
        df_results["reference_avg_power"],
        "r--^",
        linewidth=2,
        label="Reference Power",
    )
    axes[0, 0].set_xlabel("Reference PPFD (μmol/m²/s)")
    axes[0, 0].set_ylabel("Average Power (W)")
    axes[0, 0].set_title("Power Consumption vs PPFD")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Power Savings (Baseline and MPPI vs Reference)
    width = 30
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

    # 3. Power Savings Percentage
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

    # 4. Energy Efficiency Comparison
    axes[1, 0].plot(
        ppfd_values,
        df_results["baseline_energy_efficiency"],
        "g-o",
        linewidth=2,
        label="Baseline Efficiency",
    )
    axes[1, 0].plot(
        ppfd_values,
        df_results["mppi_energy_efficiency"],
        "b-s",
        linewidth=2,
        label="MPPI Efficiency",
    )
    axes[1, 0].plot(
        ppfd_values,
        df_results["reference_energy_efficiency"],
        "r--^",
        linewidth=2,
        label="Reference Efficiency",
    )
    axes[1, 0].set_xlabel("Reference PPFD (μmol/m²/s)")
    axes[1, 0].set_ylabel("Energy Efficiency (μmol/m²)/(W·s)")
    axes[1, 0].set_title("Energy Efficiency vs PPFD")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Efficiency Improvement vs Reference
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

    # 6. Photosynthesis Performance Comparison
    axes[1, 2].plot(
        ppfd_values,
        df_results["baseline_avg_photosynthesis"],
        "g-o",
        linewidth=2,
        label="Baseline Photosynthesis",
    )
    axes[1, 2].plot(
        ppfd_values,
        df_results["mppi_avg_photosynthesis"],
        "b-s",
        linewidth=2,
        label="MPPI Photosynthesis",
    )
    axes[1, 2].plot(
        ppfd_values,
        df_results["reference_avg_photosynthesis"],
        "r--^",
        linewidth=2,
        label="Reference Photosynthesis",
    )
    axes[1, 2].set_xlabel("Reference PPFD (μmol/m²/s)")
    axes[1, 2].set_ylabel("Photosynthesis Rate (μmol/m²/s)")
    axes[1, 2].set_title("Photosynthesis Performance vs PPFD")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_filename = os.path.join(
        output_dir, f"three_way_ppfd_analysis_plots_{timestamp}.png"
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Plots saved to: {plot_filename}")

    plt.show()

    return plot_filename


def print_three_way_analysis_summary(df_results):
    """Print a comprehensive summary of the three-way analysis results"""

    print("\n" + "=" * 80)
    print("THREE-WAY PPFD POWER ANALYSIS SUMMARY")
    print("=" * 80)

    # Overall statistics
    total_tests = len(df_results)
    baseline_power_saving_tests = len(
        df_results[df_results["baseline_power_difference_percent"] < 0]
    )
    mppi_power_saving_tests = len(
        df_results[df_results["mppi_power_difference_percent"] < 0]
    )
    baseline_efficiency_improving_tests = len(
        df_results[df_results["baseline_efficiency_improvement_percent"] > 0]
    )
    mppi_efficiency_improving_tests = len(
        df_results[df_results["mppi_efficiency_improvement_percent"] > 0]
    )

    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total PPFD values tested: {total_tests}")
    print(
        f"  Baseline tests with power savings: {baseline_power_saving_tests}/{total_tests} ({100*baseline_power_saving_tests/total_tests:.1f}%)"
    )
    print(
        f"  MPPI tests with power savings: {mppi_power_saving_tests}/{total_tests} ({100*mppi_power_saving_tests/total_tests:.1f}%)"
    )
    print(
        f"  Baseline tests with efficiency improvements: {baseline_efficiency_improving_tests}/{total_tests} ({100*baseline_efficiency_improving_tests/total_tests:.1f}%)"
    )
    print(
        f"  MPPI tests with efficiency improvements: {mppi_efficiency_improving_tests}/{total_tests} ({100*mppi_efficiency_improving_tests/total_tests:.1f}%)"
    )

    # Best performing PPFD values
    best_baseline_power_idx = df_results["baseline_power_difference_percent"].idxmin()
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
        f"    PPFD: {df_results.loc[best_baseline_power_idx, 'ppfd_reference']:.0f} μmol/m²/s"
    )
    print(
        f"    Power saved: {df_results.loc[best_baseline_power_idx, 'baseline_power_saved_watts']:.1f}W ({df_results.loc[best_baseline_power_idx, 'baseline_power_difference_percent']:.1f}%)"
    )

    print(f"  Best MPPI power savings:")
    print(
        f"    PPFD: {df_results.loc[best_mppi_power_idx, 'ppfd_reference']:.0f} μmol/m²/s"
    )
    print(
        f"    Power saved: {df_results.loc[best_mppi_power_idx, 'mppi_power_saved_watts']:.1f}W ({df_results.loc[best_mppi_power_idx, 'mppi_power_difference_percent']:.1f}%)"
    )

    print(f"  Best BASELINE efficiency improvement:")
    print(
        f"    PPFD: {df_results.loc[best_baseline_efficiency_idx, 'ppfd_reference']:.0f} μmol/m²/s"
    )
    print(
        f"    Efficiency improvement: {df_results.loc[best_baseline_efficiency_idx, 'baseline_efficiency_improvement_percent']:.1f}%"
    )

    print(f"  Best MPPI efficiency improvement:")
    print(
        f"    PPFD: {df_results.loc[best_mppi_efficiency_idx, 'ppfd_reference']:.0f} μmol/m²/s"
    )
    print(
        f"    Efficiency improvement: {df_results.loc[best_mppi_efficiency_idx, 'mppi_efficiency_improvement_percent']:.1f}%"
    )

    # Detailed results table
    print(f"\n📋 DETAILED RESULTS:")
    print("PPFD  | Baseline |  MPPI   | Baseline | MPPI    | Baseline | MPPI")
    print("(μmol)| Pwr Save | Pwr Save| Eff Impr | Eff Impr| Avg PWM  | Avg PWM")
    print("      |   (W)    |   (W)   |   (%)    |   (%)   |   (%)    |   (%)")
    print("-" * 80)

    for _, row in df_results.iterrows():
        print(
            f"{row['ppfd_reference']:5.0f} | {row['baseline_power_saved_watts']:8.1f} | {row['mppi_power_saved_watts']:7.1f} | {row['baseline_efficiency_improvement_percent']:8.1f} | {row['mppi_efficiency_improvement_percent']:7.1f} | {row['baseline_avg_pwm']:8.1f} | {row['mppi_avg_pwm']:7.1f}"
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

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    # Find optimal PPFD for each algorithm
    optimal_baseline_power_ppfd = df_results.loc[
        df_results["baseline_power_difference_percent"].idxmin(), "ppfd_reference"
    ]
    optimal_mppi_power_ppfd = df_results.loc[
        df_results["mppi_power_difference_percent"].idxmin(), "ppfd_reference"
    ]

    print(
        f"  For maximum BASELINE power savings: Use PPFD = {optimal_baseline_power_ppfd:.0f} μmol/m²/s"
    )
    print(
        f"  For maximum MPPI power savings: Use PPFD = {optimal_mppi_power_ppfd:.0f} μmol/m²/s"
    )

    # Algorithm comparison
    baseline_avg_savings = df_results["baseline_power_difference_percent"].mean()
    mppi_avg_savings = df_results["mppi_power_difference_percent"].mean()

    if baseline_avg_savings < mppi_avg_savings:
        print(
            f"  BASELINE algorithm shows better average power savings ({baseline_avg_savings:.1f}% vs {mppi_avg_savings:.1f}%)"
        )
    else:
        print(
            f"  MPPI algorithm shows better average power savings ({mppi_avg_savings:.1f}% vs {baseline_avg_savings:.1f}%)"
        )


def main():
    """Main function to run the complete three-way PPFD power analysis"""

    print("Starting Three-Way PPFD Power Analysis (Baseline vs MPPI vs Reference)...")

    # Define PPFD range to test
    ppfd_range = range(100, 201, 100)  # Use default range
    # ppfd_range = [700]

    # Run the analysis
    df_results, output_dir = run_baseline_ppfd_power_analysis(
        ppfd_range=ppfd_range,
        duration=120,  # Same as original
        dt=1.0,
    )

    # Create visualizations
    plot_filename = create_three_way_analysis_plots(df_results, output_dir)

    # Print summary (only if we have results)
    if len(df_results) > 0:
        print_three_way_analysis_summary(df_results)

    print(f"\nAnalysis complete! Results saved in: {output_dir}")

    return df_results, output_dir


if __name__ == "__main__":
    # Run the complete analysis
    results_df, results_dir = main()
