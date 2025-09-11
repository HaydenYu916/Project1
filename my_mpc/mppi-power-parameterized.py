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


class LEDPlant:
    """LED plant model for MPPI using imported LED functions"""

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

        # Current state
        self.ambient_temp = base_ambient_temp
        self.time = 0.0

        # Initialize photosynthesis predictor if available
        if PHOTOSYNTHESIS_AVAILABLE:
            try:
                self.photo_predictor = PhotosynthesisPredictor()
                self.use_photo_model = self.photo_predictor.is_loaded
            except:
                self.use_photo_model = False
        else:
            self.use_photo_model = False

    def step(self, pwm_percent, dt=0.1):
        """Single step of LED plant using imported led_step function"""
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

    def get_photosynthesis_rate(self, ppfd, temperature):
        """Always use photosynthesis prediction model"""
        if self.use_photo_model:
            try:
                return self.photo_predictor.predict(ppfd, temperature)
            except Exception as e:
                print(f"Warning: Photosynthesis prediction failed: {e}")
                return self.simple_photosynthesis_model(ppfd, temperature)
        else:
            print(
                "Warning: Using simple photosynthesis model - prediction model not available"
            )
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


class LEDMPPIController:
    """Model Predictive Path Integral Controller for LED system - Photosynthesis Maximization"""

    def __init__(self, plant, horizon=10, num_samples=1000, dt=0.1, temperature=1.0):
        self.plant = plant
        self.horizon = horizon
        self.num_samples = num_samples
        self.dt = dt
        self.temperature = temperature  # MPPI temperature parameter

        # Cost function weights - focused on photosynthesis maximization
        self.Q_photo = 10.0  # High weight for photosynthesis maximization
        self.Q_ref = 5.0  # Weight for reference trajectory tracking
        self.R_pwm = 0.001  # Low control effort penalty
        self.R_dpwm = 0.05  # Smooth control changes
        self.R_power = 0.1  # Moderate power consumption penalty for efficiency

        # Constraints
        self.pwm_min = 0.0
        self.pwm_max = 80.0
        self.temp_min = 20.0
        self.temp_max = 29.0

        # Control parameters
        self.pwm_std = 15.0  # Standard deviation for PWM sampling
        self.pwm_prev = 0.0

        # Constraint penalties
        self.temp_penalty = 100000.0  # Very high penalty for temperature violations
        self.pwm_penalty = 1000.0  # Penalty for PWM constraint violations

    def set_weights(
        self, Q_photo=10.0, Q_ref=5.0, R_pwm=0.001, R_dpwm=0.05, R_power=0.1
    ):
        """Set MPPI cost weights for photosynthesis maximization"""
        self.Q_photo = Q_photo
        self.Q_ref = Q_ref
        self.R_pwm = R_pwm
        self.R_dpwm = R_dpwm
        self.R_power = R_power

    def set_constraints(self, pwm_min=0.0, pwm_max=80.0, temp_min=20.0, temp_max=29.0):
        """Set MPPI constraints"""
        self.pwm_min = pwm_min
        self.pwm_max = pwm_max
        self.temp_min = temp_min
        self.temp_max = temp_max

    def set_mppi_params(self, num_samples=1000, temperature=1.0, pwm_std=15.0):
        """Set MPPI algorithm parameters"""
        self.num_samples = num_samples
        self.temperature = temperature
        self.pwm_std = pwm_std

    def sample_control_sequences(self, mean_sequence):
        """Sample control sequences around the mean"""
        # Create noise for sampling
        noise = np.random.normal(0, self.pwm_std, (self.num_samples, self.horizon))

        # Add noise to mean sequence
        samples = mean_sequence[np.newaxis, :] + noise

        # Apply constraints by clipping
        samples = np.clip(samples, self.pwm_min, self.pwm_max)

        return samples

    def compute_cost(self, pwm_sequence, current_temp, photo_ref=None):
        """Compute cost for a single PWM sequence - maximize photosynthesis with reference tracking"""
        try:
            # Predict future states
            ppfd_pred, temp_pred, power_pred, photo_pred = self.plant.predict(
                pwm_sequence, current_temp, self.dt
            )

            cost = 0.0

            # Main objective: Maximize photosynthesis over the horizon
            for k in range(self.horizon):
                # Negative photosynthesis to maximize it (minimize negative)
                cost -= self.Q_photo * photo_pred[k]

                # Reference trajectory tracking (if provided)
                if photo_ref is not None and k < len(photo_ref):
                    ref_error = photo_pred[k] - photo_ref[k]
                    cost += self.Q_ref * ref_error**2

                # Hard constraint penalties for temperature
                if temp_pred[k] > self.temp_max:
                    violation = temp_pred[k] - self.temp_max
                    cost += self.temp_penalty * violation**2
                if temp_pred[k] < self.temp_min:
                    violation = self.temp_min - temp_pred[k]
                    cost += self.temp_penalty * violation**2

            # Control effort costs
            for k in range(self.horizon):
                cost += self.R_pwm * pwm_sequence[k] ** 2
                cost += self.R_power * power_pred[k] ** 2

            # Control smoothness
            prev_pwm = self.pwm_prev
            for k in range(self.horizon):
                dpwm = pwm_sequence[k] - prev_pwm
                cost += self.R_dpwm * dpwm**2
                prev_pwm = pwm_sequence[k]

            # PWM constraint penalties
            for k in range(self.horizon):
                if pwm_sequence[k] > self.pwm_max:
                    violation = pwm_sequence[k] - self.pwm_max
                    cost += self.pwm_penalty * violation**2
                if pwm_sequence[k] < self.pwm_min:
                    violation = self.pwm_min - pwm_sequence[k]
                    cost += self.pwm_penalty * violation**2

            return cost

        except Exception:
            # Return very high cost for invalid sequences
            return 1e10

    def solve(self, current_temp, mean_sequence=None, photo_ref=None):
        """Solve MPPI optimization for photosynthesis maximization with optional reference tracking"""

        # Initialize mean sequence if not provided
        if mean_sequence is None:
            # Start with moderate PWM values
            mean_sequence = np.ones(self.horizon) * min(40.0, self.pwm_max * 0.5)

        # Sample control sequences
        control_samples = self.sample_control_sequences(mean_sequence)

        # Compute costs for all samples
        costs = np.zeros(self.num_samples)

        for i in range(self.num_samples):
            costs[i] = self.compute_cost(control_samples[i], current_temp, photo_ref)

        # Handle infinite or NaN costs
        costs = np.nan_to_num(costs, nan=1e10, posinf=1e10)

        # Compute weights using softmax
        min_cost = np.min(costs)
        exp_costs = np.exp(-(costs - min_cost) / self.temperature)
        weights = exp_costs / np.sum(exp_costs)

        # Compute weighted average of control sequences
        optimal_sequence = np.sum(weights[:, np.newaxis] * control_samples, axis=0)

        # Apply final constraints
        optimal_sequence = np.clip(optimal_sequence, self.pwm_min, self.pwm_max)

        # Get first control action
        optimal_pwm = optimal_sequence[0]

        # Safety check for temperature
        _, temp_check, _, _ = self.plant.predict([optimal_pwm], current_temp, self.dt)
        if temp_check[0] > self.temp_max:
            # Emergency reduction
            optimal_pwm = max(self.pwm_min, optimal_pwm * 0.7)
            print(
                f"MPPI: Emergency PWM reduction to {optimal_pwm:.1f}% due to temperature risk"
            )

        self.pwm_prev = optimal_pwm

        # Return additional information
        success = True
        best_cost = np.min(costs)

        return optimal_pwm, optimal_sequence, success, best_cost, weights


class LEDMPPISimulation:
    """MPPI simulation environment for photosynthesis maximization with parameterized reference PPFD"""

    def __init__(self, plant, controller, reference_ppfd=700.0):
        self.plant = plant
        self.controller = controller
        self.reference_ppfd = reference_ppfd  # Parameterized reference PPFD

        # Data storage
        self.time_data = []
        self.ppfd_data = []
        self.temp_data = []
        self.power_data = []
        self.pwm_data = []
        self.photo_data = []
        self.ppfd_ref_data = []  # Keep for comparison
        self.temp_ref_data = []  # Keep for comparison
        self.photo_ref_data = []  # Keep for comparison
        self.power_ref_data = []  # Reference power consumption
        self.cost_data = []

        # Statistics storage
        self.statistics = {}

        # Initialize photosynthesis predictor for evaluation
        if PHOTOSYNTHESIS_AVAILABLE:
            try:
                self.pn = PhotosynthesisPredictor()
            except:
                self.pn = None
        else:
            self.pn = None

    def set_reference_ppfd(self, reference_ppfd):
        """Set the reference PPFD value"""
        self.reference_ppfd = reference_ppfd
        print(f"Reference PPFD set to: {reference_ppfd} μmol/m²/s")

    def create_reference_trajectory(self, duration, dt):
        """Create reference trajectories for comparison purposes with parameterized PPFD"""
        time_points = np.arange(0, duration, dt)

        ppfd_ref = []
        temp_ref = []
        photo_ref = []
        power_ref = []

        current_temp = self.plant.base_ambient_temp
        for t in time_points:
            # PPFD reference: use parameterized value
            ppfd_target = self.reference_ppfd

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
                photo_target = self.pn.predict(ppfd_target, temp_target)
            else:
                # Use simple model for reference
                ppfd_max = 1000
                pn_max = 25
                km = 300
                temp_factor = np.exp(-0.01 * (temp_target - 25) ** 2)
                photo_target = max(
                    0, (pn_max * ppfd_target / (km + ppfd_target)) * temp_factor
                )

            # Calculate reference power consumption
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
        """Run MPPI simulation for photosynthesis maximization with reference comparison"""

        print("Starting LED MPPI Simulation - Photosynthesis Maximization")
        print("=" * 60)
        print(f"Reference PPFD: {self.reference_ppfd} μmol/m²/s")
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

            # Extract reference trajectory over horizon for MPPI
            end_idx = min(k + self.controller.horizon, len(photo_ref_full))
            photo_ref_horizon = photo_ref_full[k:end_idx]

            # Solve MPPI for photosynthesis maximization with reference tracking
            pwm_optimal, optimal_sequence, success, cost, weights = (
                self.controller.solve(
                    self.plant.ambient_temp, mean_sequence, photo_ref_horizon
                )
            )

            # Update mean sequence for next iteration (receding horizon)
            if len(optimal_sequence) > 1:
                mean_sequence = np.concatenate(
                    [optimal_sequence[1:], [optimal_sequence[-1]]]
                )
            else:
                mean_sequence = optimal_sequence

            # Apply control to plant
            ppfd, temp, power, photo_rate = self.plant.step(pwm_optimal, dt)

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

        # Calculate and store statistics
        self.calculate_statistics()

        # Analyze constraint satisfaction
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

    def calculate_statistics(self):
        """Calculate comprehensive statistics for the simulation"""
        dt = self.time_data[1] - self.time_data[0] if len(self.time_data) > 1 else 1.0

        # Convert to numpy arrays
        ppfd_data = np.array(self.ppfd_data)
        temp_data = np.array(self.temp_data)
        power_data = np.array(self.power_data)
        pwm_data = np.array(self.pwm_data)
        photo_data = np.array(self.photo_data)
        ppfd_ref_data = np.array(self.ppfd_ref_data)
        temp_ref_data = np.array(self.temp_ref_data)
        photo_ref_data = np.array(self.photo_ref_data)
        power_ref_data = np.array(self.power_ref_data)
        cost_data = np.array(self.cost_data)

        # Cumulative values
        cumulative_pn_mppi = np.sum(photo_data) * dt
        cumulative_pn_ref = np.sum(photo_ref_data) * dt
        cumulative_power_mppi = np.sum(power_data) * dt
        cumulative_power_ref = np.sum(power_ref_data) * dt

        # Performance metrics
        avg_photosynthesis = np.mean(photo_data)
        avg_photo_ref = np.mean(photo_ref_data)
        max_photosynthesis = np.max(photo_data)
        avg_power_mppi = np.mean(power_data)
        avg_power_ref = np.mean(power_ref_data)

        # Performance improvements
        photo_improvement = ((avg_photosynthesis - avg_photo_ref) / avg_photo_ref) * 100
        total_improvement = (
            (cumulative_pn_mppi - cumulative_pn_ref) / cumulative_pn_ref
        ) * 100
        power_difference = (
            (cumulative_power_mppi - cumulative_power_ref) / cumulative_power_ref
        ) * 100

        # Energy efficiency
        efficiency_mppi = cumulative_pn_mppi / cumulative_power_mppi
        efficiency_ref = cumulative_pn_ref / cumulative_power_ref
        efficiency_improvement = (
            (efficiency_mppi - efficiency_ref) / efficiency_ref
        ) * 100

        # RMSE calculations
        ppfd_rmse = np.sqrt(np.mean((ppfd_data - ppfd_ref_data) ** 2))
        temp_rmse = np.sqrt(np.mean((temp_data - temp_ref_data) ** 2))
        photo_rmse = np.sqrt(np.mean((photo_data - photo_ref_data) ** 2))
        power_rmse = np.sqrt(np.mean((power_data - power_ref_data) ** 2))

        # Constraint satisfaction
        temp_violations = np.sum(
            (temp_data < self.controller.temp_min)
            | (temp_data > self.controller.temp_max)
        )
        temp_satisfaction = 100 * (1 - temp_violations / len(temp_data))

        pwm_violations = np.sum(
            (pwm_data < self.controller.pwm_min) | (pwm_data > self.controller.pwm_max)
        )

        # Store statistics
        self.statistics = {
            "simulation_info": {
                "reference_ppfd": self.reference_ppfd,
                "duration": self.time_data[-1] if self.time_data else 0,
                "dt": dt,
                "total_steps": len(self.time_data),
                "timestamp": datetime.now().isoformat(),
            },
            "photosynthesis_performance": {
                "mppi_avg_photosynthesis": avg_photosynthesis,
                "reference_avg_photosynthesis": avg_photo_ref,
                "improvement_percent": photo_improvement,
                "max_photosynthesis": max_photosynthesis,
                "total_photosynthesis_mppi": cumulative_pn_mppi,
                "total_photosynthesis_ref": cumulative_pn_ref,
                "total_improvement_percent": total_improvement,
            },
            "power_consumption": {
                "mppi_avg_power": avg_power_mppi,
                "reference_avg_power": avg_power_ref,
                "total_power_mppi": cumulative_power_mppi,
                "total_power_ref": cumulative_power_ref,
                "power_difference_percent": power_difference,
            },
            "energy_efficiency": {
                "mppi_efficiency": efficiency_mppi,
                "reference_efficiency": efficiency_ref,
                "efficiency_improvement_percent": efficiency_improvement,
            },
            "tracking_errors": {
                "ppfd_rmse": ppfd_rmse,
                "temperature_rmse": temp_rmse,
                "photosynthesis_rmse": photo_rmse,
                "power_rmse": power_rmse,
            },
            "constraint_satisfaction": {
                "temperature_violations": int(temp_violations),
                "temperature_satisfaction_percent": temp_satisfaction,
                "pwm_violations": int(pwm_violations),
                "temperature_range": {
                    "min": float(np.min(temp_data)),
                    "max": float(np.max(temp_data)),
                },
            },
            "control_performance": {
                "final_cost": cost_data[-1] if len(cost_data) > 0 else 0,
                "avg_pwm": float(np.mean(pwm_data)),
                "pwm_range": {
                    "min": float(np.min(pwm_data)),
                    "max": float(np.max(pwm_data)),
                },
            },
        }

    def export_statistics(self, filename_prefix="mppi_stats"):
        """Export statistics to JSON and CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export JSON statistics
        json_filename = f"{filename_prefix}_{timestamp}.json"
        with open(json_filename, "w") as f:
            json.dump(self.statistics, f, indent=2)

        # Export CSV summary
        csv_filename = f"{filename_prefix}_summary_{timestamp}.csv"
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value", "Unit"])

            # Write key metrics
            writer.writerow(
                [
                    "Reference PPFD",
                    self.statistics["simulation_info"]["reference_ppfd"],
                    "μmol/m²/s",
                ]
            )
            writer.writerow(
                ["Duration", self.statistics["simulation_info"]["duration"], "s"]
            )
            writer.writerow(
                [
                    "MPPI Avg Photosynthesis",
                    f"{self.statistics['photosynthesis_performance']['mppi_avg_photosynthesis']:.2f}",
                    "μmol/m²/s",
                ]
            )
            writer.writerow(
                [
                    "Reference Avg Photosynthesis",
                    f"{self.statistics['photosynthesis_performance']['reference_avg_photosynthesis']:.2f}",
                    "μmol/m²/s",
                ]
            )
            writer.writerow(
                [
                    "Photosynthesis Improvement",
                    f"{self.statistics['photosynthesis_performance']['improvement_percent']:.1f}",
                    "%",
                ]
            )
            writer.writerow(
                [
                    "Total Photosynthesis MPPI",
                    f"{self.statistics['photosynthesis_performance']['total_photosynthesis_mppi']:.1f}",
                    "μmol/m²",
                ]
            )
            writer.writerow(
                [
                    "Total Photosynthesis Ref",
                    f"{self.statistics['photosynthesis_performance']['total_photosynthesis_ref']:.1f}",
                    "μmol/m²",
                ]
            )
            writer.writerow(
                [
                    "Total Improvement",
                    f"{self.statistics['photosynthesis_performance']['total_improvement_percent']:.1f}",
                    "%",
                ]
            )
            writer.writerow(
                [
                    "MPPI Avg Power",
                    f"{self.statistics['power_consumption']['mppi_avg_power']:.1f}",
                    "W",
                ]
            )
            writer.writerow(
                [
                    "Reference Avg Power",
                    f"{self.statistics['power_consumption']['reference_avg_power']:.1f}",
                    "W",
                ]
            )
            writer.writerow(
                [
                    "Power Difference",
                    f"{self.statistics['power_consumption']['power_difference_percent']:.1f}",
                    "%",
                ]
            )
            writer.writerow(
                [
                    "MPPI Efficiency",
                    f"{self.statistics['energy_efficiency']['mppi_efficiency']:.4f}",
                    "(μmol/m²)/(W·s)",
                ]
            )
            writer.writerow(
                [
                    "Reference Efficiency",
                    f"{self.statistics['energy_efficiency']['reference_efficiency']:.4f}",
                    "(μmol/m²)/(W·s)",
                ]
            )
            writer.writerow(
                [
                    "Efficiency Improvement",
                    f"{self.statistics['energy_efficiency']['efficiency_improvement_percent']:.1f}",
                    "%",
                ]
            )
            writer.writerow(
                [
                    "Temperature Satisfaction",
                    f"{self.statistics['constraint_satisfaction']['temperature_satisfaction_percent']:.1f}",
                    "%",
                ]
            )
            writer.writerow(
                [
                    "Temperature Violations",
                    self.statistics["constraint_satisfaction"][
                        "temperature_violations"
                    ],
                    "steps",
                ]
            )
            writer.writerow(
                [
                    "Avg PWM",
                    f"{self.statistics['control_performance']['avg_pwm']:.1f}",
                    "%",
                ]
            )

        # Export detailed time series data
        data_filename = f"{filename_prefix}_timeseries_{timestamp}.csv"
        with open(data_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Time",
                    "PPFD",
                    "Temperature",
                    "Power",
                    "PWM",
                    "Photosynthesis",
                    "PPFD_Ref",
                    "Temp_Ref",
                    "Photo_Ref",
                    "Power_Ref",
                    "Cost",
                ]
            )

            for i in range(len(self.time_data)):
                writer.writerow(
                    [
                        self.time_data[i],
                        self.ppfd_data[i],
                        self.temp_data[i],
                        self.power_data[i],
                        self.pwm_data[i],
                        self.photo_data[i],
                        self.ppfd_ref_data[i],
                        self.temp_ref_data[i],
                        self.photo_ref_data[i],
                        self.power_ref_data[i],
                        self.cost_data[i],
                    ]
                )

        print(f"\nStatistics exported:")
        print(f"  JSON: {json_filename}")
        print(f"  CSV Summary: {csv_filename}")
        print(f"  CSV Time Series: {data_filename}")

        return json_filename, csv_filename, data_filename

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
            "statistics": self.statistics,
        }

    def plot_results(self):
        """Plot MPPI simulation results comparing actual vs reference with accumulated metrics"""
        results = self.get_results()

        # Calculate accumulated values
        dt = (
            results["time"][1] - results["time"][0] if len(results["time"]) > 1 else 1.0
        )

        # Cumulative sums
        cumulative_pn_mppi = np.cumsum(results["photosynthesis"]) * dt
        cumulative_pn_ref = np.cumsum(results["photo_ref"]) * dt
        cumulative_power_mppi = np.cumsum(results["power"]) * dt
        cumulative_power_ref = np.cumsum(results["power_ref"]) * dt

        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(
            f"LED MPPI Control - Photosynthesis Maximization vs Reference (PPFD={self.reference_ppfd})",
            fontsize=16,
        )

        # PPFD comparison
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

        # Temperature comparison with constraints
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

        # Cost evolution (should be negative due to photosynthesis maximization)
        axes[1, 2].plot(results["time"], results["cost"], "purple", linewidth=2)
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
        axes[2, 0].set_title("Accumulated Photosynthesis: MPPI vs Reference")
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

        # Performance metrics
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
        """Print detailed performance metrics comparing MPPI vs reference including accumulated values"""
        stats = self.statistics

        print(f"\n" + "=" * 80)
        print(f"MPPI PHOTOSYNTHESIS MAXIMIZATION vs REFERENCE COMPARISON")
        print(f"Reference PPFD: {self.reference_ppfd} μmol/m²/s")
        print(f"=" * 80)

        print(f"\n📈 PHOTOSYNTHESIS PERFORMANCE:")
        print(
            f"  MPPI Average Photosynthesis: {stats['photosynthesis_performance']['mppi_avg_photosynthesis']:.2f} μmol/m²/s"
        )
        print(
            f"  Reference Average Photosynthesis: {stats['photosynthesis_performance']['reference_avg_photosynthesis']:.2f} μmol/m²/s"
        )
        print(
            f"  Improvement: {stats['photosynthesis_performance']['improvement_percent']:+.1f}% over reference"
        )
        print(
            f"  MPPI Maximum Photosynthesis: {stats['photosynthesis_performance']['max_photosynthesis']:.2f} μmol/m²/s"
        )

        print(f"\n🔋 ACCUMULATED METRICS:")
        print(
            f"  MPPI Total Photosynthesis: {stats['photosynthesis_performance']['total_photosynthesis_mppi']:.1f} μmol/m²"
        )
        print(
            f"  Reference Total Photosynthesis: {stats['photosynthesis_performance']['total_photosynthesis_ref']:.1f} μmol/m²"
        )
        print(
            f"  Total Pn Improvement: {stats['photosynthesis_performance']['total_improvement_percent']:+.1f}% over reference"
        )
        print(
            f"  MPPI Total Power Consumption: {stats['power_consumption']['total_power_mppi']:.1f} W·s"
        )
        print(
            f"  Reference Total Power Consumption: {stats['power_consumption']['total_power_ref']:.1f} W·s"
        )
        print(
            f"  Power Usage Difference: {stats['power_consumption']['power_difference_percent']:+.1f}% vs reference"
        )

        print(f"\n⚡ ENERGY EFFICIENCY:")
        print(
            f"  MPPI Energy Efficiency: {stats['energy_efficiency']['mppi_efficiency']:.4f} (μmol/m²)/(W·s)"
        )
        print(
            f"  Reference Energy Efficiency: {stats['energy_efficiency']['reference_efficiency']:.4f} (μmol/m²)/(W·s)"
        )
        print(
            f"  Efficiency Improvement: {stats['energy_efficiency']['efficiency_improvement_percent']:+.1f}% over reference"
        )
        print(
            f"  MPPI Average Power: {stats['power_consumption']['mppi_avg_power']:.1f} W"
        )
        print(
            f"  Reference Average Power: {stats['power_consumption']['reference_avg_power']:.1f} W"
        )

        print(f"\n📊 COMPARISON METRICS (RMSE):")
        print(
            f"  PPFD deviation from reference: {stats['tracking_errors']['ppfd_rmse']:.1f} μmol/m²/s"
        )
        print(
            f"  Temperature deviation from reference: {stats['tracking_errors']['temperature_rmse']:.2f} °C"
        )
        print(
            f"  Photosynthesis deviation from reference: {stats['tracking_errors']['photosynthesis_rmse']:.2f} μmol/m²/s"
        )
        print(
            f"  Power deviation from reference: {stats['tracking_errors']['power_rmse']:.2f} W"
        )

        print(f"\n🎯 CONSTRAINT SATISFACTION:")
        print(
            f"  Temperature violations: {stats['constraint_satisfaction']['temperature_violations']}/{stats['simulation_info']['total_steps']} steps"
        )
        print(
            f"  Temperature satisfaction: {stats['constraint_satisfaction']['temperature_satisfaction_percent']:.1f}%"
        )
        print(
            f"  PWM violations: {stats['constraint_satisfaction']['pwm_violations']} steps"
        )
        print(
            f"  Temperature range: {stats['constraint_satisfaction']['temperature_range']['min']:.1f} to {stats['constraint_satisfaction']['temperature_range']['max']:.1f}°C"
        )

        print(f"\n💡 SUMMARY:")
        if stats["photosynthesis_performance"]["total_improvement_percent"] > 0:
            print(
                f"  ✅ MPPI achieved {stats['photosynthesis_performance']['total_improvement_percent']:.1f}% higher total photosynthesis"
            )
        else:
            print(
                f"  ❌ MPPI achieved {stats['photosynthesis_performance']['total_improvement_percent']:.1f}% lower total photosynthesis"
            )

        if stats["power_consumption"]["power_difference_percent"] < 0:
            print(
                f"  ✅ MPPI used {abs(stats['power_consumption']['power_difference_percent']):.1f}% less power"
            )
        else:
            print(
                f"  ⚠️  MPPI used {stats['power_consumption']['power_difference_percent']:.1f}% more power"
            )

        if stats["energy_efficiency"]["efficiency_improvement_percent"] > 0:
            print(
                f"  ✅ MPPI was {stats['energy_efficiency']['efficiency_improvement_percent']:.1f}% more energy efficient"
            )
        else:
            print(
                f"  ❌ MPPI was {abs(stats['energy_efficiency']['efficiency_improvement_percent']):.1f}% less energy efficient"
            )

        print(f"  Final Cost: {stats['control_performance']['final_cost']:.2e}")
        print(f"  Average PWM: {stats['control_performance']['avg_pwm']:.1f}%")


# Example usage
if __name__ == "__main__":
    # Create plant model
    plant = LEDPlant(
        base_ambient_temp=22.0,
        max_ppfd=700.0,
        max_power=100.0,
        thermal_resistance=1.2,
        thermal_mass=8.0,
    )

    # Create MPPI controller for photosynthesis maximization
    controller = LEDMPPIController(
        plant=plant, horizon=10, num_samples=1000, dt=1.0, temperature=0.5
    )

    # Configure MPPI weights for photosynthesis maximization with reference tracking
    controller.set_weights(
        Q_photo=5.0,  # High weight for photosynthesis maximization
        Q_ref=25.0,  # Moderate weight for reference trajectory tracking
        R_pwm=0.001,  # Low control penalty
        R_dpwm=0.05,  # Smooth control
        R_power=0.08,  # Moderate power consumption penalty for efficiency
    )

    # Set constraints
    controller.set_constraints(pwm_min=0.0, pwm_max=100.0, temp_min=20.0, temp_max=29.0)

    # Set MPPI parameters
    controller.set_mppi_params(num_samples=1000, temperature=0.5, pwm_std=10.0)

    # Create simulation with parameterized reference PPFD
    reference_ppfd = 700.0  # This can now be easily changed
    simulation = LEDMPPISimulation(plant, controller, reference_ppfd=reference_ppfd)

    # Run simulation for photosynthesis maximization
    print(
        f"Starting MPPI-based LED control for photosynthesis maximization with reference PPFD={reference_ppfd}..."
    )
    results = simulation.run_simulation(duration=120, dt=1.0)

    # Export statistics
    json_file, csv_summary, csv_timeseries = simulation.export_statistics()

    # Plot results
    simulation.plot_results()
