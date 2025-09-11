import numpy as np
import warnings

# External dependencies from this repo
from led import led_step

try:
    from pn_prediction.predict import PhotosynthesisPredictor
    _PHOTO_OK = True
except Exception:
    PhotosynthesisPredictor = None  # type: ignore
    _PHOTO_OK = False


def _simple_photosynthesis(ppfd: float, temperature: float) -> float:
    """Fallback photosynthesis model (light-response with temperature factor)."""
    ppfd_max = 1000.0
    pn_max = 25.0
    km = 300.0
    temp_factor = np.exp(-0.01 * (temperature - 25.0) ** 2)
    pn = (pn_max * ppfd / (km + ppfd)) * temp_factor
    return float(max(0.0, pn))


def mppi_next_ppfd(current_ppfd: float, temperature: float, co2: float, humidity: float) -> float:
    """Compute the next PPFD setpoint via an MPPI-style step.

    Inputs are the current measurements. The function returns the predicted PPFD
    output after applying the first control in the optimized sequence.

    Parameters
    - current_ppfd: current measured PPFD (µmol/m²/s)
    - temperature: current ambient/canopy temperature (°C)
    - co2: current CO₂ (ppm)
    - humidity: current relative humidity (%). Currently unused but accepted
      for API stability.

    Returns
    - Predicted PPFD (µmol/m²/s) after applying the first optimized control.
    """

    # --- LED/plant parameters (align with mppi-power defaults) ---
    base_ambient_temp = 22.0
    max_ppfd = 700.0
    max_power = 100.0
    thermal_resistance = 1.2
    thermal_mass = 8.0

    # --- MPPI hyperparameters ---
    horizon = 10
    num_samples = 800  # slightly reduced for speed in API usage
    dt = 1.0
    lam = 0.5  # temperature parameter for softmax weighting

    # Cost weights
    Q_photo = 5.0
    R_pwm = 1e-3
    R_dpwm = 0.05
    R_power = 0.08

    # Constraints
    pwm_min = 0.0
    pwm_max = 100.0
    temp_min = 20.0
    temp_max = 29.0

    # Penalties
    temp_penalty = 100000.0
    pwm_penalty = 1000.0

    # Sampling std for PWM sequences
    pwm_std = 10.0

    # Estimate previous PWM from current PPFD for smoothness
    pwm_prev_est = float(np.clip((current_ppfd / max_ppfd) * 100.0, pwm_min, pwm_max))
    mean_sequence = np.ones(horizon, dtype=float) * pwm_prev_est

    # Photosynthesis predictor (optional)
    predictor = None
    if _PHOTO_OK:
        try:
            predictor = PhotosynthesisPredictor()
            if not getattr(predictor, "is_loaded", True):
                predictor = None
        except Exception:
            predictor = None
            warnings.warn("PhotosynthesisPredictor unavailable. Using simple model.")

    # Sample control sequences around the mean
    noise = np.random.normal(0.0, pwm_std, size=(num_samples, horizon))
    samples = np.clip(mean_sequence[None, :] + noise, pwm_min, pwm_max)

    def rollout_and_cost(pwm_seq: np.ndarray, temp0: float) -> float:
        cost = 0.0
        prev_pwm = pwm_prev_est
        temp = float(temp0)
        for k in range(horizon):
            pwm = float(pwm_seq[k])
            # Plant step
            ppfd, new_temp, power, _ = led_step(
                pwm_percent=pwm,
                ambient_temp=temp,
                base_ambient_temp=base_ambient_temp,
                dt=dt,
                max_ppfd=max_ppfd,
                max_power=max_power,
                thermal_resistance=thermal_resistance,
                thermal_mass=thermal_mass,
            )

            # Photosynthesis
            if predictor is not None:
                try:
                    pn = float(predictor.predict(ppfd, co2, new_temp, 0.83))
                except Exception:
                    pn = _simple_photosynthesis(ppfd, new_temp)
            else:
                pn = _simple_photosynthesis(ppfd, new_temp)

            # Objective: maximize photosynthesis (minimize -Q*pn)
            cost += -Q_photo * pn

            # Control effort and smoothness
            cost += R_pwm * (pwm ** 2)
            cost += R_power * (power ** 2)
            dpwm = pwm - prev_pwm
            cost += R_dpwm * (dpwm ** 2)
            prev_pwm = pwm

            # Temperature hard penalties
            if new_temp > temp_max:
                v = new_temp - temp_max
                cost += temp_penalty * (v ** 2)
            if new_temp < temp_min:
                v = temp_min - new_temp
                cost += temp_penalty * (v ** 2)

            # PWM hard penalties (should already be clipped)
            if pwm > pwm_max:
                v = pwm - pwm_max
                cost += pwm_penalty * (v ** 2)
            if pwm < pwm_min:
                v = pwm_min - pwm
                cost += pwm_penalty * (v ** 2)

            temp = float(new_temp)

        return float(cost)

    # Compute costs for all samples
    costs = np.array([rollout_and_cost(samples[i], temperature) for i in range(num_samples)])
    costs = np.nan_to_num(costs, nan=1e10, posinf=1e10)

    # MPPI weighting (softmax on negative costs)
    cmin = float(np.min(costs))
    weights = np.exp(-(costs - cmin) / max(lam, 1e-6))
    denom = float(np.sum(weights))
    if denom <= 0 or not np.isfinite(denom):
        # Degenerate case: pick argmin
        optimal_seq = samples[int(np.argmin(costs))]
    else:
        weights /= denom
        optimal_seq = np.sum(weights[:, None] * samples, axis=0)

    optimal_seq = np.clip(optimal_seq, pwm_min, pwm_max)
    optimal_pwm = float(optimal_seq[0])

    # Safety check: single-step temperature guard
    _, temp_check, _, _ = led_step(
        pwm_percent=optimal_pwm,
        ambient_temp=temperature,
        base_ambient_temp=base_ambient_temp,
        dt=dt,
        max_ppfd=max_ppfd,
        max_power=max_power,
        thermal_resistance=thermal_resistance,
        thermal_mass=thermal_mass,
    )
    if temp_check > temp_max:
        optimal_pwm = max(pwm_min, optimal_pwm * 0.7)

    # Produce predicted PPFD after applying the chosen control
    ppfd_out, _, _, _ = led_step(
        pwm_percent=optimal_pwm,
        ambient_temp=temperature,
        base_ambient_temp=base_ambient_temp,
        dt=dt,
        max_ppfd=max_ppfd,
        max_power=max_power,
        thermal_resistance=thermal_resistance,
        thermal_mass=thermal_mass,
    )

    # Clip to feasible PPFD range
    return float(np.clip(ppfd_out, 0.0, max_ppfd))

