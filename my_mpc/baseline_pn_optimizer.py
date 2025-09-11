import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from scipy.optimize import minimize_scalar
import json
import pandas as pd

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
    Baseline optimization algorithm that uses photosynthesis prediction model
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
        pn_tolerance=2.0,  # Acceptable Pn tolerance (%)
        min_ppfd_step=1.0,  # Minimum PPFD search step
    ):
        self.base_ambient_temp = base_ambient_temp
        self.max_ppfd = max_ppfd
        self.max_power = max_power
        self.thermal_resistance = thermal_resistance
        self.thermal_mass = thermal_mass
        self.pn_tolerance = pn_tolerance
        self.min_ppfd_step = min_ppfd_step
        
        # Initialize photosynthesis predictor
        if PHOTOSYNTHESIS_AVAILABLE:
            try:
                self.photo_predictor = PhotosynthesisPredictor()
                self.use_photo_model = self.photo_predictor.is_loaded
            except:
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
            dict: {
                'optimal_ppfd': float,
                'optimal_pwm': float,
                'achieved_pn': float,
                'power_consumption': float,
                'efficiency': float,
                'success': bool
            }
        """
        
        # Binary search for minimum PPFD that achieves target_pn
        ppfd_min = 0.0
        ppfd_max = self.max_ppfd
        best_ppfd = None
        best_result = None
        
        # First check if target is achievable at max PPFD
        max_pn = self.get_photosynthesis_rate(ppfd_max, temperature)
        tolerance_abs = target_pn * (self.pn_tolerance / 100.0)
        if max_pn < target_pn - tolerance_abs:
            # Target not achievable
            return {
                'optimal_ppfd': ppfd_max,
                'optimal_pwm': self.ppfd_to_pwm(ppfd_max),
                'achieved_pn': max_pn,
                'power_consumption': self.calculate_power_consumption(ppfd_max, temperature),
                'efficiency': max_pn / self.calculate_power_consumption(ppfd_max, temperature) if self.calculate_power_consumption(ppfd_max, temperature) > 0 else 0,
                'success': False,
                'target_pn': target_pn,
                'temperature': temperature
            }
        
        # Binary search for optimal PPFD
        tolerance_achieved = False
        iterations = 0
        max_iterations = 50
        
        while ppfd_max - ppfd_min > self.min_ppfd_step and iterations < max_iterations:
            ppfd_mid = (ppfd_min + ppfd_max) / 2.0
            pn_mid = self.get_photosynthesis_rate(ppfd_mid, temperature)
            
            if abs(pn_mid - target_pn) <= tolerance_abs:
                # Found acceptable solution
                best_ppfd = ppfd_mid
                power = self.calculate_power_consumption(ppfd_mid, temperature)
                best_result = {
                    'optimal_ppfd': ppfd_mid,
                    'optimal_pwm': self.ppfd_to_pwm(ppfd_mid),
                    'achieved_pn': pn_mid,
                    'power_consumption': power,
                    'efficiency': pn_mid / power if power > 0 else 0,
                    'success': True,
                    'target_pn': target_pn,
                    'temperature': temperature
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
            # Use the midpoint as best estimate
            ppfd_final = (ppfd_min + ppfd_max) / 2.0
            pn_final = self.get_photosynthesis_rate(ppfd_final, temperature)
            power_final = self.calculate_power_consumption(ppfd_final, temperature)
            
            best_result = {
                'optimal_ppfd': ppfd_final,
                'optimal_pwm': self.ppfd_to_pwm(ppfd_final),
                'achieved_pn': pn_final,
                'power_consumption': power_final,
                'efficiency': pn_final / power_final if power_final > 0 else 0,
                'success': abs(pn_final - target_pn) <= tolerance_abs,
                'target_pn': target_pn,
                'temperature': temperature
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
    
    For each control step, it calculates the target Pn based on reference trajectory
    and finds the minimum PPFD needed to achieve that Pn at current temperature.
    """
    
    def __init__(self, optimizer, fallback_pwm=30.0):
        self.optimizer = optimizer
        self.fallback_pwm = fallback_pwm
        self.pwm_prev = 0.0
        
    def solve(self, current_temp, target_pn):
        """
        Solve for optimal PWM to achieve target Pn at current temperature.
        
        Args:
            current_temp: Current ambient temperature (°C)
            target_pn: Target photosynthesis rate (μmol/m²/s)
            
        Returns:
            optimal_pwm: Optimal PWM percentage
            result_dict: Detailed optimization results
        """
        
        result = self.optimizer.find_optimal_ppfd_for_temperature(current_temp, target_pn)
        
        # Always use the optimal result, even if not within tolerance
        optimal_pwm = result['optimal_pwm']
        if not result['success']:
            print(f"Baseline: Using closest solution PWM={optimal_pwm:.1f}% (target Pn {target_pn:.1f} vs achieved {result['achieved_pn']:.1f} at {current_temp:.1f}°C)")
        
        self.pwm_prev = optimal_pwm
        
        return optimal_pwm, result


class BaselinePnSimulation:
    """
    Simulation environment for baseline Pn optimization with performance comparison to MPPI.
    """
    
    def __init__(self, optimizer, controller, reference_ppfd=700.0):
        self.optimizer = optimizer
        self.controller = controller
        self.reference_ppfd = reference_ppfd
        
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
        """Create target photosynthesis trajectory using configurable reference PPFD"""
        time_points = np.arange(0, duration, dt)
        
        # Use configurable reference PPFD for target trajectory
        target_pn = []
        current_temp = self.optimizer.base_ambient_temp
        
        for t in time_points:
            # Use configurable PPFD reference
            ppfd_target = self.reference_ppfd  # Configurable PPFD reference
            
            # Calculate temperature evolution with constant high PPFD
            from led import led_step
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
        
        print("Starting Baseline Pn Optimization Simulation")
        print("=" * 60)
        print(f"Pn tolerance: ±{self.optimizer.pn_tolerance}%")
        print(f"PPFD search step: {self.optimizer.min_ppfd_step} μmol/m²/s")
        
        if self.optimizer.use_photo_model:
            print("Using trained photosynthesis model")
        else:
            print("Using simple photosynthesis model")
        
        # Create target Pn trajectory
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
            if k % 10 == 0:
                success_status = "✓" if opt_result['success'] else "✗"
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
        successful_optimizations = sum(1 for r in self.optimization_results if r['success'])
        success_rate = 100 * successful_optimizations / len(self.optimization_results)
        
        print(f"\n" + "=" * 60)
        print(f"BASELINE PN OPTIMIZATION PERFORMANCE")
        print(f"=" * 60)
        
        print(f"\n📈 PHOTOSYNTHESIS PERFORMANCE:")
        print(f"  Average Photosynthesis: {avg_photosynthesis:.2f} μmol/m²/s")
        print(f"  Total Photosynthesis: {total_photosynthesis:.1f} μmol/m²")
        print(f"  Maximum Photosynthesis: {np.max(results['photosynthesis']):.2f} μmol/m²/s")
        
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
        print(f"  Temperature range: {np.min(results['temp']):.1f} to {np.max(results['temp']):.1f}°C")
    
    def plot_results(self):
        """Plot baseline optimization results"""
        results = self.get_results()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Baseline Pn Optimization Results", fontsize=16)
        
        # PPFD vs time
        axes[0, 0].plot(results["time"], results["ppfd"], "g-", linewidth=2, label="PPFD")
        axes[0, 0].set_ylabel("PPFD (μmol/m²/s)")
        axes[0, 0].set_title("PPFD Output")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Photosynthesis tracking
        axes[0, 1].plot(results["time"], results["photosynthesis"], "orange", linewidth=2, label="Actual Pn")
        axes[0, 1].plot(results["time"], results["target_pn"], "orange", linestyle="--", linewidth=2, alpha=0.7, label="Target Pn")
        axes[0, 1].set_ylabel("Photosynthesis (μmol/m²/s)")
        axes[0, 1].set_title("Photosynthesis Tracking")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Temperature
        axes[0, 2].plot(results["time"], results["temp"], "r-", linewidth=2)
        axes[0, 2].set_ylabel("Temperature (°C)")
        axes[0, 2].set_title("Temperature")
        axes[0, 2].grid(True, alpha=0.3)
        
        # PWM control
        axes[1, 0].plot(results["time"], results["pwm"], "b-", linewidth=2)
        axes[1, 0].set_ylabel("PWM (%)")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_title("Control Signal")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Power consumption
        axes[1, 1].plot(results["time"], results["power"], "m-", linewidth=2)
        axes[1, 1].set_ylabel("Power (W)")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_title("Power Consumption")
        axes[1, 1].grid(True, alpha=0.3)
        
        # Efficiency
        axes[1, 2].plot(results["time"], results["efficiency"], "c-", linewidth=2)
        axes[1, 2].set_ylabel("Efficiency (Pn/Power)")
        axes[1, 2].set_xlabel("Time (s)")
        axes[1, 2].set_title("Energy Efficiency")
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create baseline optimizer
    optimizer = BaselinePnOptimizer(
        base_ambient_temp=22.0,
        max_ppfd=700.0,
        max_power=100.0,
        thermal_resistance=1.2,
        thermal_mass=8.0,
        pn_tolerance=1.5,  # Accept ±1.5% tolerance
        min_ppfd_step=1.0,
    )
    
    # Create controller
    controller = BaselinePnController(optimizer, fallback_pwm=30.0)
    
    # Create simulation
    simulation = BaselinePnSimulation(optimizer, controller)
    
    # Run simulation
    print("Starting Baseline Pn Optimization simulation...")
    results = simulation.run_simulation(duration=120, dt=1.0)
    
    # Plot results
    simulation.plot_results()