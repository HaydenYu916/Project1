import numpy as np

from mppi_api import mppi_next_ppfd
from led import led_step


def demo(steps: int = 30, dt: float = 1.0, co2: float = 400.0, humidity: float = 60.0):
    """Run a simple closed-loop demo using mppi_next_ppfd.

    We estimate PWM from the commanded PPFD and use the LED plant step to
    evolve temperature and the measured PPFD for the next iteration.
    """

    # Keep plant parameters consistent with mppi_api.mppi_next_ppfd
    base_ambient_temp = 22.0
    max_ppfd = 700.0
    max_power = 100.0
    thermal_resistance = 1.2
    thermal_mass = 8.0

    # Initial conditions
    temp = base_ambient_temp
    current_ppfd = 0.0

    print("Demo: MPPI PPFD controller API")
    print("step  temp(°C)  meas_ppfd  cmd_ppfd  est_pwm(%)  new_temp(°C)  new_ppfd")
    print("-" * 78)

    for k in range(steps):
        # Call the API to get the next PPFD setpoint
        cmd_ppfd = mppi_next_ppfd(current_ppfd, temp, co2, humidity)

        # Estimate PWM from commanded PPFD to evolve the plant
        est_pwm = float(np.clip((cmd_ppfd / max_ppfd) * 100.0, 0.0, 100.0))

        # Plant evolution using the estimated PWM
        new_ppfd, new_temp, power, _ = led_step(
            pwm_percent=est_pwm,
            ambient_temp=temp,
            base_ambient_temp=base_ambient_temp,
            dt=dt,
            max_ppfd=max_ppfd,
            max_power=max_power,
            thermal_resistance=thermal_resistance,
            thermal_mass=thermal_mass,
        )

        print(
            f"{k:3d}  {temp:8.2f}  {current_ppfd:9.1f}  {cmd_ppfd:8.1f}  "
            f"{est_pwm:9.1f}  {new_temp:12.2f}  {new_ppfd:8.1f}"
        )

        # Update measured state for next iteration
        temp = float(new_temp)
        current_ppfd = float(new_ppfd)


if __name__ == "__main__":
    demo()

