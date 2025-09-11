import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import csv
import json
from datetime import datetime

warnings.filterwarnings("ignore")

# Import LED simulation functions
from led import led_step, led_steady_state

# Import photosynthesis predictor
try:
    from pn_prediction.predict import PhotosynthesisPredictor

    PHOTOSYNTHESIS_AVAILABLE = True
except ImportError:
    print("Warning: PhotosynthesisPredictor not available. Using simple model.")
    PHOTOSYNTHESIS_AVAILABLE = False


class BaselinePnOptimizer:
    """
    Parameterized Baseline optimization algorithm that uses photosynthesis prediction model
    to find optimal PPFD values that achieve similar Pn while minimizing power consumption.

    At each temperature, the algorithm searches for the minimum PPFD that achieves
    a target photosynthesis rate with acceptable tolerance.
    """

    def __init__(
        self,
        base_ambient_temp=25.0,
        max_ppfd=500.0,
        max_power=100.0,
        thermal_resistance=2.5,
        thermal_mass=0.5,
        pn_tolerance=0.5,  # Acceptable Pn tolerance (μmol/m²/s)
        min_ppfd_step=1.0,  # Minimum PPFD search step
        reference_ppfd=700.0,  # Parameterized reference PPFD
    ):
        self.base_ambient_temp = base_ambient_temp
        self.max_ppfd = max_ppfd
        self.max_power = max_power
        self.thermal_resistance = thermal_resistance
        self.thermal_mass = thermal_mass
        self.pn_tolerance = pn_tolerance
        self.min_ppfd_step = min_ppfd_step
        self.reference_ppfd = reference_ppfd  # NEW: Parameterized reference

        # Initialize photosynthesis predictor
        if PHOTOSYNTHESIS_AVAILABLE:
            try:
                self.photo_predictor = PhotosynthesisPredictor()
                self.use_photo_model = self.photo_predictor.is_loaded
            except Exception:
                self.use_photo_model = False
        else:
            self.use_photo_model = False

        # Current state
        self.ambient_temp = base_ambient_temp
        self.time = 0.0

    def get_photosynthesis_rate(self, ppfd, temperature):
        """Get photosynthesis rate using prediction model"""
        if self.use_photo_model:
            try:
                return self.photo_predictor.predict(ppfd, temperature)
            except Exception as e:
                print(f"Warning: Photosynthesis prediction failed: {e}")
                return self.simple_photosynthesis_model(ppfd, temperature)
        else:
            return self.simple_photosynthesis_model(ppfd, temperature)

    def simple_photosynthesis_model(self, ppfd, temperature):
        """Simple photosynthesis model as fallback"""
        ppfd_max = 1000  # μmol/m²/s
        pn_max = 25  # μmol/m²/s
        km = 300  # μmol/m²/s

        # Temperature effect (optimal around 25°C)
        temp_factor = np.exp(-0.01 * (temperature - 25) ** 2)

        # Light response
        pn = (pn_max * ppfd / (km + ppfd)) * temp_factor

        return max(0, pn)

    def ppfd_to_pwm(self, ppfd):
        """Convert PPFD to PWM percentage"""
        return min(100.0, (ppfd / self.max_ppfd) * 100.0)

    def pwm_to_ppfd(self, pwm):
        """Convert PWM percentage to PPFD"""
        return (pwm / 100.0) * self.max_ppfd

    def calculate_power_consumption(self, ppfd, temperature):
        """Calculate power consumption for given PPFD and temperature"""
        pwm_percent = self.ppfd_to_pwm(ppfd)

        # Use LED model to calculate power
        _, _, power, _ = led_step(
            pwm_percent=pwm_percent,
            ambient_temp=temperature,
            base_ambient_temp=self.base_ambient_temp,
            dt=0.1,  # Small dt for steady-state approximation
            max_ppfd=self.max_ppfd,
            max_power=self.max_power,
            thermal_resistance=self.thermal_resistance,
            thermal_mass=self.thermal_mass,
        )

        return power

    def find_optimal_ppfd_for_temperature(self, temperature, target_pn):
        """
        Find the minimum PPFD that achieves target_pn ± tolerance at given temperature.

        Args:
            temperature: Ambient temperature (°C)
            target_pn: Target photosynthesis rate (μmol/m²/s)

        Returns:
            dict: Optimization results
        """

        # Binary search for minimum PPFD that achieves target_pn
        ppfd_min = 0.0
        ppfd_max = self.max_ppfd
        best_result = None

        # First check if target is achievable at max PPFD
        max_pn = self.get_photosynthesis_rate(ppfd_max, temperature)
        if max_pn < target_pn - self.pn_tolerance:
            # Target not achievable
            return {
                "optimal_ppfd": ppfd_max,
                "optimal_pwm": self.ppfd_to_pwm(ppfd_max),
                "achieved_pn": max_pn,
                "power_consumption": self.calculate_power_consumption(
                    ppfd_max, temperature
                ),
                "efficiency": max_pn
                / self.calculate_power_consumption(ppfd_max, temperature)
                if self.calculate_power_consumption(ppfd_max, temperature) > 0
                else 0,
                "success": False,
                "target_pn": target_pn,
                "temperature": temperature,
            }

        # Binary search for optimal PPFD
        tolerance_achieved = False
        iterations = 0
        max_iterations = 50

        while ppfd_max - ppfd_min > self.min_ppfd_step and iterations < max_iterations:
            ppfd_mid = (ppfd_min + ppfd_max) / 2.0
            pn_mid = self.get_photosynthesis_rate(ppfd_mid, temperature)

            if abs(pn_mid - target_pn) <= self.pn_tolerance:
                # Found acceptable solution
                power = self.calculate_power_consumption(ppfd_mid, temperature)
                best_result = {
                    "optimal_ppfd": ppfd_mid,
                    "optimal_pwm": self.ppfd_to_pwm(ppfd_mid),
                    "achieved_pn": pn_mid,
                    "power_consumption": power,
                    "efficiency": pn_mid / power if power > 0 else 0,
                    "success": True,
                    "target_pn": target_pn,
                    "temperature": temperature,
                }
                tolerance_achieved = True
                # Continue searching for even lower PPFD
                ppfd_max = ppfd_mid
            elif pn_mid < target_pn:
                # Need higher PPFD
                ppfd_min = ppfd_mid
            else:
                # Can use lower PPFD
                ppfd_max = ppfd_mid

            iterations += 1

        # If no solution within tolerance found, use the closest we got
        if not tolerance_achieved:
            ppfd_final = (ppfd_min + ppfd_max) / 2.0
            pn_final = self.get_photosynthesis_rate(ppfd_final, temperature)
            power_final = self.calculate_power_consumption(ppfd_final, temperature)

            best_result = {
                "optimal_ppfd": ppfd_final,
                "optimal_pwm": self.ppfd_to_pwm(ppfd_final),
                "achieved_pn": pn_final,
                "power_consumption": power_final,
                "efficiency": pn_final / power_final if power_final > 0 else 0,
                "success": abs(pn_final - target_pn) <= self.pn_tolerance,
                "target_pn": target_pn,
                "temperature": temperature,
            }

        return best_result

    def step(self, pwm_percent, dt=0.1):
        """Single step of baseline optimizer (for compatibility with simulation)"""
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

        # Update state
        self.ambient_temp = new_ambient_temp
        self.time += dt

        # Calculate photosynthesis rate
        photosynthesis_rate = self.get_photosynthesis_rate(ppfd, new_ambient_temp)

        return ppfd, new_ambient_temp, power, photosynthesis_rate


class BaselinePnController:
    """
    Controller that uses baseline Pn optimization to determine PWM settings.
    """

    def __init__(self, optimizer, fallback_pwm=30.0):
        self.optimizer = optimizer
        self.fallback_pwm = fallback_pwm
        self.pwm_prev = 0.0

    def solve(self, current_temp, target_pn):
        """
        Solve for optimal PWM to achieve target Pn at current temperature.
        """

        result = self.optimizer.find_optimal_ppfd_for_temperature(
            current_temp, target_pn
        )

        if result["success"]:
            optimal_pwm = result["optimal_pwm"]
        else:
            # Fallback to a conservative PWM value
            optimal_pwm = self.fallback_pwm
            print(
                f"Baseline: Fallback to {optimal_pwm}% PWM (target Pn {target_pn:.1f} not achievable at {current_temp:.1f}°C)"
            )

        self.pwm_prev = optimal_pwm

        return optimal_pwm, result


class BaselinePnSimulation:
    """
    Simulation environment for baseline Pn optimization with parameterized reference PPFD.
    """

    def __init__(self, optimizer, controller):
        self.optimizer = optimizer
        self.controller = controller

        # Data storage
        self.time_data = []
        self.ppfd_data = []
        self.temp_data = []
        self.power_data = []
        self.pwm_data = []
        self.photo_data = []
        self.target_pn_data = []
        self.efficiency_data = []
        self.optimization_results = []

    def create_target_pn_trajectory(self, duration, dt):
        """Create target photosynthesis trajectory using parameterized reference PPFD"""
        time_points = np.arange(0, duration, dt)

        # Use parameterized reference PPFD
        target_pn = []
        current_temp = self.optimizer.base_ambient_temp

        for t in time_points:
            # Use the parameterized reference PPFD
            ppfd_target = self.optimizer.reference_ppfd

            # Calculate temperature evolution with reference PPFD
            _, temp_target, _, _ = led_step(
                pwm_percent=(ppfd_target / self.optimizer.max_ppfd) * 100,
                ambient_temp=current_temp,
                base_ambient_temp=self.optimizer.base_ambient_temp,
                dt=dt,
                max_ppfd=self.optimizer.max_ppfd,
                max_power=self.optimizer.max_power,
                thermal_resistance=self.optimizer.thermal_resistance,
                thermal_mass=self.optimizer.thermal_mass,
            )
            current_temp = temp_target

            # Calculate photosynthesis target at this temperature and PPFD
            pn_target = self.optimizer.get_photosynthesis_rate(ppfd_target, temp_target)
            target_pn.append(pn_target)

        return np.array(target_pn)

    def run_simulation(self, duration=120, dt=1.0):
        """Run baseline Pn optimization simulation"""

        print(
            f"Starting Baseline Pn Optimization Simulation (Reference PPFD: {self.optimizer.reference_ppfd})"
        )
        print("=" * 70)
        print(f"Pn tolerance: ±{self.optimizer.pn_tolerance} μmol/m²/s")
        print(f"PPFD search step: {self.optimizer.min_ppfd_step} μmol/m²/s")

        if self.optimizer.use_photo_model:
            print("Using trained photosynthesis model")
        else:
            print("Using simple photosynthesis model")

        # Create target Pn trajectory based on reference PPFD
        target_pn_trajectory = self.create_target_pn_trajectory(duration, dt)

        # Reset optimizer
        self.optimizer.ambient_temp = self.optimizer.base_ambient_temp
        self.optimizer.time = 0.0

        # Reset controller
        self.controller.pwm_prev = 0.0

        # Clear data
        self.clear_data()

        # Simulation loop
        steps = int(duration / dt)
        for k in range(steps):
            current_time = k * dt
            target_pn = target_pn_trajectory[k]

            # Solve for optimal PWM using baseline algorithm
            pwm_optimal, opt_result = self.controller.solve(
                self.optimizer.ambient_temp, target_pn
            )

            # Apply control to plant
            ppfd, temp, power, photo_rate = self.optimizer.step(pwm_optimal, dt)

            # Calculate efficiency
            efficiency = photo_rate / power if power > 0 else 0

            # Store data
            self.time_data.append(current_time)
            self.ppfd_data.append(ppfd)
            self.temp_data.append(temp)
            self.power_data.append(power)
            self.pwm_data.append(pwm_optimal)
            self.photo_data.append(photo_rate)
            self.target_pn_data.append(target_pn)
            self.efficiency_data.append(efficiency)
            self.optimization_results.append(opt_result)

            # Print progress
            if k % 20 == 0:
                success_status = "✓" if opt_result["success"] else "✗"
                print(
                    f"t={current_time:3.0f}s: PWM={pwm_optimal:5.1f}%, "
                    f"PPFD={ppfd:3.0f}, Temp={temp:4.1f}°C, "
                    f"Pn={photo_rate:4.1f} (target: {target_pn:4.1f}) {success_status}, "
                    f"Eff={efficiency:.3f}"
                )

        print("\nSimulation completed!")

        # Analyze performance
        self.analyze_performance()

        return self.get_results()

    def clear_data(self):
        """Clear all data arrays"""
        self.time_data = []
        self.ppfd_data = []
        self.temp_data = []
        self.power_data = []
        self.pwm_data = []
        self.photo_data = []
        self.target_pn_data = []
        self.efficiency_data = []
        self.optimization_results = []

    def get_results(self):
        """Get simulation results"""
        return {
            "time": np.array(self.time_data),
            "ppfd": np.array(self.ppfd_data),
            "temp": np.array(self.temp_data),
            "power": np.array(self.power_data),
            "pwm": np.array(self.pwm_data),
            "photosynthesis": np.array(self.photo_data),
            "target_pn": np.array(self.target_pn_data),
            "efficiency": np.array(self.efficiency_data),
            "optimization_results": self.optimization_results,
            "reference_ppfd": self.optimizer.reference_ppfd,  # Include reference PPFD
        }

    def analyze_performance(self):
        """Analyze baseline optimization performance"""
        results = self.get_results()

        # Calculate performance metrics
        avg_photosynthesis = np.mean(results["photosynthesis"])
        total_photosynthesis = np.sum(results["photosynthesis"])
        avg_power = np.mean(results["power"])
        total_power = np.sum(results["power"])
        avg_efficiency = np.mean(results["efficiency"])

        # Target tracking performance
        pn_errors = results["photosynthesis"] - results["target_pn"]
        rmse_pn = np.sqrt(np.mean(pn_errors**2))
        mae_pn = np.mean(np.abs(pn_errors))

        # Success rate
        successful_optimizations = sum(
            1 for r in self.optimization_results if r["success"]
        )
        success_rate = 100 * successful_optimizations / len(self.optimization_results)

        print(f"\n" + "=" * 70)
        print(
            f"BASELINE PN OPTIMIZATION PERFORMANCE (Ref PPFD: {self.optimizer.reference_ppfd})"
        )
        print(f"=" * 70)

        print(f"\n📈 PHOTOSYNTHESIS PERFORMANCE:")
        print(f"  Average Photosynthesis: {avg_photosynthesis:.2f} μmol/m²/s")
        print(f"  Total Photosynthesis: {total_photosynthesis:.1f} μmol/m²")
        print(
            f"  Maximum Photosynthesis: {np.max(results['photosynthesis']):.2f} μmol/m²/s"
        )

        print(f"\n🎯 TARGET TRACKING:")
        print(f"  RMSE Pn error: {rmse_pn:.2f} μmol/m²/s")
        print(f"  MAE Pn error: {mae_pn:.2f} μmol/m²/s")
        print(f"  Success rate: {success_rate:.1f}%")

        print(f"\n🔋 POWER CONSUMPTION:")
        print(f"  Average Power: {avg_power:.1f} W")
        print(f"  Total Power: {total_power:.1f} W·s")
        print(f"  Average Efficiency: {avg_efficiency:.4f} (μmol/m²/s)/W")

        print(f"\n💡 CONTROL SUMMARY:")
        print(f"  Average PWM: {np.mean(results['pwm']):.1f}%")
        print(f"  Average PPFD: {np.mean(results['ppfd']):.1f} μmol/m²/s")
        print(
            f"  Temperature range: {np.min(results['temp']):.1f} to {np.max(results['temp']):.1f}°C"
        )


# Example usage
if __name__ == "__main__":
    # Test with different reference PPFD values
    reference_ppfd_values = [400, 500, 600, 700, 800]

    for ref_ppfd in reference_ppfd_values:
        print(f"\n{'='*80}")
        print(f"TESTING BASELINE WITH REFERENCE PPFD = {ref_ppfd}")
        print(f"{'='*80}")

        # Create baseline optimizer with parameterized reference PPFD
        optimizer = BaselinePnOptimizer(
            base_ambient_temp=22.0,
            max_ppfd=700.0,
            max_power=100.0,
            thermal_resistance=1.2,
            thermal_mass=8.0,
            pn_tolerance=0.3,
            min_ppfd_step=1.0,
            reference_ppfd=ref_ppfd,  # Parameterized reference
        )

        # Create controller
        controller = BaselinePnController(optimizer, fallback_pwm=30.0)

        # Create simulation
        simulation = BaselinePnSimulation(optimizer, controller)

        # Run simulation
        results = simulation.run_simulation(duration=60, dt=1.0)  # Shorter for testing

        print(f"\nCompleted test for reference PPFD = {ref_ppfd}")
