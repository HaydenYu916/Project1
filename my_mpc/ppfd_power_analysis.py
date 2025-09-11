import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from datetime import datetime
import os

warnings.filterwarnings("ignore")

# Import the MPPI simulation components
import sys
import importlib.util

# Load the module with hyphens in the filename
spec = importlib.util.spec_from_file_location(
    "mppi_power_parameterized", "mppi-power-parameterized.py"
)
mppi_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mppi_module)

# Import the classes
LEDPlant = mppi_module.LEDPlant
LEDMPPIController = mppi_module.LEDMPPIController
LEDMPPISimulation = mppi_module.LEDMPPISimulation


def run_ppfd_power_analysis(
    ppfd_range=None,
    duration=120,  # Same as original
    dt=1.0,  # Same as original
    output_dir="ppfd_analysis_results",
):
    """
    Run MPPI simulations across different PPFD values and analyze power savings

    Args:
        ppfd_range: List of PPFD values to test (default: 100 to 1000 in steps of 100)
        duration: Simulation duration in seconds
        dt: Time step in seconds
        output_dir: Directory to save results
    """

    if ppfd_range is None:
        ppfd_range = [100, 200, 300]  # Just 3 values for quick testing

    print("PPFD Power Savings Analysis")
    print("=" * 50)
    print(f"Testing PPFD values: {ppfd_range}")
    print(f"Simulation duration: {duration}s")
    print(f"Time step: {dt}s")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize results storage
    results_summary = []

    # Create plant model (same for all tests)
    plant = LEDPlant(
        base_ambient_temp=22.0,
        max_ppfd=1200.0,  # Higher max to accommodate 1000 PPFD
        max_power=120.0,  # Higher max power accordingly
        thermal_resistance=1.2,
        thermal_mass=8.0,
    )

    # Create MPPI controller
    controller = LEDMPPIController(
        plant=plant,
        horizon=10,  # Same as original
        num_samples=1000,  # Same as original
        dt=dt,
        temperature=0.5,
    )

    # Configure MPPI weights
    controller.set_weights(
        Q_photo=5.0,
        Q_ref=25.0,
        R_pwm=0.001,
        R_dpwm=0.05,
        R_power=0.08,
    )

    # Set constraints
    controller.set_constraints(pwm_min=0.0, pwm_max=100.0, temp_min=20.0, temp_max=29.0)

    # Set MPPI parameters
    controller.set_mppi_params(num_samples=1000, temperature=0.5, pwm_std=10.0)

    # Run simulations for each PPFD value
    for i, ppfd_value in enumerate(ppfd_range):
        print(f"\n[{i+1}/{len(ppfd_range)}] Testing PPFD = {ppfd_value} μmol/m²/s")

        try:
            # Create simulation with current PPFD
            simulation = LEDMPPISimulation(plant, controller, reference_ppfd=ppfd_value)

            # Run simulation (suppress detailed output)
            results = simulation.run_simulation(duration=duration, dt=dt)

            # Extract key metrics
            stats = simulation.statistics

            # Store results
            result_entry = {
                "ppfd_reference": ppfd_value,
                "mppi_avg_power": stats["power_consumption"]["mppi_avg_power"],
                "reference_avg_power": stats["power_consumption"][
                    "reference_avg_power"
                ],
                "power_difference_percent": stats["power_consumption"][
                    "power_difference_percent"
                ],
                "power_saved_watts": stats["power_consumption"]["reference_avg_power"]
                - stats["power_consumption"]["mppi_avg_power"],
                "total_power_mppi": stats["power_consumption"]["total_power_mppi"],
                "total_power_ref": stats["power_consumption"]["total_power_ref"],
                "total_power_saved": stats["power_consumption"]["total_power_ref"]
                - stats["power_consumption"]["total_power_mppi"],
                "mppi_avg_photosynthesis": stats["photosynthesis_performance"][
                    "mppi_avg_photosynthesis"
                ],
                "reference_avg_photosynthesis": stats["photosynthesis_performance"][
                    "reference_avg_photosynthesis"
                ],
                "photosynthesis_improvement_percent": stats[
                    "photosynthesis_performance"
                ]["improvement_percent"],
                "energy_efficiency_mppi": stats["energy_efficiency"]["mppi_efficiency"],
                "energy_efficiency_ref": stats["energy_efficiency"][
                    "reference_efficiency"
                ],
                "efficiency_improvement_percent": stats["energy_efficiency"][
                    "efficiency_improvement_percent"
                ],
                "temperature_satisfaction": stats["constraint_satisfaction"][
                    "temperature_satisfaction_percent"
                ],
                "avg_pwm": stats["control_performance"]["avg_pwm"],
            }

            results_summary.append(result_entry)

            # Print brief summary
            power_saved = result_entry["power_saved_watts"]
            power_saved_percent = result_entry["power_difference_percent"]
            efficiency_improvement = result_entry["efficiency_improvement_percent"]

            print(f"  Power saved: {power_saved:.1f}W ({power_saved_percent:+.1f}%)")
            print(f"  Efficiency improvement: {efficiency_improvement:+.1f}%")

        except Exception as e:
            print(f"  ERROR: Failed to run simulation for PPFD {ppfd_value}: {e}")
            continue

    # Convert to DataFrame for easier analysis
    df_results = pd.DataFrame(results_summary)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dir, f"ppfd_power_analysis_{timestamp}.csv")
    df_results.to_csv(csv_filename, index=False)

    print(f"\nResults saved to: {csv_filename}")

    return df_results, output_dir


def create_power_analysis_plots(df_results, output_dir):
    """Create comprehensive plots for the PPFD power analysis"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "MPPI Power Savings Analysis Across Different PPFD Values", fontsize=16
    )

    ppfd_values = df_results["ppfd_reference"]

    # 1. Power Consumption Comparison
    axes[0, 0].plot(
        ppfd_values,
        df_results["mppi_avg_power"],
        "b-o",
        linewidth=2,
        label="MPPI Power",
    )
    axes[0, 0].plot(
        ppfd_values,
        df_results["reference_avg_power"],
        "r--s",
        linewidth=2,
        label="Reference Power",
    )
    axes[0, 0].set_xlabel("Reference PPFD (μmol/m²/s)")
    axes[0, 0].set_ylabel("Average Power (W)")
    axes[0, 0].set_title("Power Consumption vs PPFD")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Power Savings
    axes[0, 1].bar(
        ppfd_values,
        df_results["power_saved_watts"],
        color=["green" if x > 0 else "red" for x in df_results["power_saved_watts"]],
        alpha=0.7,
        width=50,
    )
    axes[0, 1].set_xlabel("Reference PPFD (μmol/m²/s)")
    axes[0, 1].set_ylabel("Power Saved (W)")
    axes[0, 1].set_title("Absolute Power Savings")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color="black", linestyle="-", alpha=0.5)

    # 3. Power Savings Percentage
    axes[0, 2].bar(
        ppfd_values,
        df_results["power_difference_percent"],
        color=[
            "green" if x < 0 else "red" for x in df_results["power_difference_percent"]
        ],
        alpha=0.7,
        width=50,
    )
    axes[0, 2].set_xlabel("Reference PPFD (μmol/m²/s)")
    axes[0, 2].set_ylabel("Power Difference (%)")
    axes[0, 2].set_title("Relative Power Savings (%)")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color="black", linestyle="-", alpha=0.5)

    # 4. Energy Efficiency
    axes[1, 0].plot(
        ppfd_values,
        df_results["energy_efficiency_mppi"],
        "g-o",
        linewidth=2,
        label="MPPI Efficiency",
    )
    axes[1, 0].plot(
        ppfd_values,
        df_results["energy_efficiency_ref"],
        "m--s",
        linewidth=2,
        label="Reference Efficiency",
    )
    axes[1, 0].set_xlabel("Reference PPFD (μmol/m²/s)")
    axes[1, 0].set_ylabel("Energy Efficiency (μmol/m²)/(W·s)")
    axes[1, 0].set_title("Energy Efficiency vs PPFD")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Efficiency Improvement
    axes[1, 1].bar(
        ppfd_values,
        df_results["efficiency_improvement_percent"],
        color=[
            "green" if x > 0 else "red"
            for x in df_results["efficiency_improvement_percent"]
        ],
        alpha=0.7,
        width=50,
    )
    axes[1, 1].set_xlabel("Reference PPFD (μmol/m²/s)")
    axes[1, 1].set_ylabel("Efficiency Improvement (%)")
    axes[1, 1].set_title("Energy Efficiency Improvement")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color="black", linestyle="-", alpha=0.5)

    # 6. Photosynthesis Performance
    axes[1, 2].plot(
        ppfd_values,
        df_results["mppi_avg_photosynthesis"],
        "orange",
        linewidth=2,
        marker="o",
        label="MPPI Photosynthesis",
    )
    axes[1, 2].plot(
        ppfd_values,
        df_results["reference_avg_photosynthesis"],
        "purple",
        linewidth=2,
        marker="s",
        linestyle="--",
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
        output_dir, f"ppfd_power_analysis_plots_{timestamp}.png"
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Plots saved to: {plot_filename}")

    plt.show()

    return plot_filename


def print_power_analysis_summary(df_results):
    """Print a comprehensive summary of the power analysis results"""

    print("\n" + "=" * 80)
    print("PPFD POWER SAVINGS ANALYSIS SUMMARY")
    print("=" * 80)

    # Overall statistics
    total_tests = len(df_results)
    power_saving_tests = len(df_results[df_results["power_difference_percent"] < 0])
    efficiency_improving_tests = len(
        df_results[df_results["efficiency_improvement_percent"] > 0]
    )

    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total PPFD values tested: {total_tests}")
    print(
        f"  Tests with power savings: {power_saving_tests}/{total_tests} ({100*power_saving_tests/total_tests:.1f}%)"
    )
    print(
        f"  Tests with efficiency improvements: {efficiency_improving_tests}/{total_tests} ({100*efficiency_improving_tests/total_tests:.1f}%)"
    )

    # Best performing PPFD values
    best_power_saving_idx = df_results["power_difference_percent"].idxmin()
    best_efficiency_idx = df_results["efficiency_improvement_percent"].idxmax()

    print(f"\n🏆 BEST PERFORMANCE:")
    print(f"  Best power savings:")
    print(
        f"    PPFD: {df_results.loc[best_power_saving_idx, 'ppfd_reference']:.0f} μmol/m²/s"
    )
    print(
        f"    Power saved: {df_results.loc[best_power_saving_idx, 'power_saved_watts']:.1f}W ({df_results.loc[best_power_saving_idx, 'power_difference_percent']:.1f}%)"
    )

    print(f"  Best efficiency improvement:")
    print(
        f"    PPFD: {df_results.loc[best_efficiency_idx, 'ppfd_reference']:.0f} μmol/m²/s"
    )
    print(
        f"    Efficiency improvement: {df_results.loc[best_efficiency_idx, 'efficiency_improvement_percent']:.1f}%"
    )

    # Power savings by PPFD range
    print(f"\n💡 POWER SAVINGS BY PPFD RANGE:")
    low_ppfd = df_results[df_results["ppfd_reference"] <= 400]
    mid_ppfd = df_results[
        (df_results["ppfd_reference"] > 400) & (df_results["ppfd_reference"] <= 700)
    ]
    high_ppfd = df_results[df_results["ppfd_reference"] > 700]

    if len(low_ppfd) > 0:
        print(
            f"  Low PPFD (≤400): Average power savings = {low_ppfd['power_difference_percent'].mean():.1f}%"
        )
    if len(mid_ppfd) > 0:
        print(
            f"  Mid PPFD (400-700): Average power savings = {mid_ppfd['power_difference_percent'].mean():.1f}%"
        )
    if len(high_ppfd) > 0:
        print(
            f"  High PPFD (>700): Average power savings = {high_ppfd['power_difference_percent'].mean():.1f}%"
        )

    # Detailed results table
    print(f"\n📋 DETAILED RESULTS:")
    print("PPFD  | Power Saved | Power Saved | Efficiency | Temp Sat | Avg PWM")
    print("(μmol)| (W)        | (%)        | Improve(%) | (%)      | (%)")
    print("-" * 70)

    for _, row in df_results.iterrows():
        print(
            f"{row['ppfd_reference']:5.0f} | {row['power_saved_watts']:10.1f} | {row['power_difference_percent']:10.1f} | {row['efficiency_improvement_percent']:10.1f} | {row['temperature_satisfaction']:8.1f} | {row['avg_pwm']:7.1f}"
        )

    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print(
        f"  Average power savings: {df_results['power_difference_percent'].mean():.1f}% ± {df_results['power_difference_percent'].std():.1f}%"
    )
    print(
        f"  Average efficiency improvement: {df_results['efficiency_improvement_percent'].mean():.1f}% ± {df_results['efficiency_improvement_percent'].std():.1f}%"
    )
    print(
        f"  Average temperature satisfaction: {df_results['temperature_satisfaction'].mean():.1f}%"
    )

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    # Find optimal PPFD for power savings
    optimal_power_ppfd = df_results.loc[
        df_results["power_difference_percent"].idxmin(), "ppfd_reference"
    ]
    optimal_efficiency_ppfd = df_results.loc[
        df_results["efficiency_improvement_percent"].idxmax(), "ppfd_reference"
    ]

    print(f"  For maximum power savings: Use PPFD = {optimal_power_ppfd:.0f} μmol/m²/s")
    print(
        f"  For maximum efficiency: Use PPFD = {optimal_efficiency_ppfd:.0f} μmol/m²/s"
    )

    # General trends
    if df_results["power_difference_percent"].corr(df_results["ppfd_reference"]) < -0.5:
        print(f"  Higher PPFD values tend to show greater power savings")
    elif (
        df_results["power_difference_percent"].corr(df_results["ppfd_reference"]) > 0.5
    ):
        print(f"  Lower PPFD values tend to show greater power savings")
    else:
        print(f"  Power savings do not show a strong correlation with PPFD level")


def main():
    """Main function to run the complete PPFD power analysis"""

    print("Starting PPFD Power Savings Analysis...")

    # Define PPFD range to test
    ppfd_range = range(100, 1001, 100)  # 100 to 1000 in steps of 100

    # Run the analysis
    df_results, output_dir = run_ppfd_power_analysis(
        ppfd_range=ppfd_range,
        duration=120,  # Same as original
        dt=1.0,
    )

    # Create visualizations
    plot_filename = create_power_analysis_plots(df_results, output_dir)

    # Print summary
    print_power_analysis_summary(df_results)

    print(f"\nAnalysis complete! Results saved in: {output_dir}")

    return df_results, output_dir


if __name__ == "__main__":
    # Run the complete analysis
    results_df, results_dir = main()
