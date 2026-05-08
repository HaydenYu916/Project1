#!/usr/bin/env python3
"""End-to-end smoke demo for the Govee MPPI stack.

What it exercises (no live hardware required):

  1. Loads the Govee xyz-aware PWM→PPFD GPR model directly.
  2. Boots an LEDPlant with --use_govee_pwm_model so PPFD→PWM inversion
     uses the GPR (with sweep + linear interpolation) instead of the
     Shelly-style calib_data.csv linear fits.
  3. Sweeps a few target PPFDs and prints the predicted (r_pwm, b_pwm).
  4. Switches to a different (x, y, z) board position and shows that the
     same PPFD target now needs different PWM commands.

To run it on the Pi:

    cd /home/pi/Desktop/Project1/MPPI-Govee/examples
    python3 govee_mppi_demo.py

It does NOT touch BLE or the spectrometer; all numbers come from the
trained joblib models.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from govee_pwmtoppfd_model import GoveePWMtoPPFDModel, GoveePosition  # noqa: E402
from mppi_v2 import LEDPlant  # noqa: E402


def demo_pwm_model_direct() -> None:
    print("=" * 60)
    print("1) Direct GoveePWMtoPPFDModel — PPFD predictions")
    print("=" * 60)
    m = GoveePWMtoPPFDModel().load()
    print(f"feature_columns: {m._meta and m._meta.get('feature_columns')}")
    for pos_name, pos in [
        ("C  z=10", GoveePosition(0, 0, 10)),
        ("C  z=6",  GoveePosition(0, 0,  6)),
        ("C  z=0",  GoveePosition(0, 0,  0)),
        ("TR z=10", GoveePosition(9.5, 7, 10)),
        ("L  z=10", GoveePosition(-9.5, 0, 10)),
    ]:
        m.set_position(pos.x_cm, pos.y_cm, pos.z_cm)
        ppfds = [round(m.predict(r_pwm=r, b_pwm=b), 2) for (r, b) in [(0, 0), (50, 0), (0, 50), (50, 50), (90, 90)]]
        print(f"  {pos_name:<10} PWM (0,0)/(50,0)/(0,50)/(50,50)/(90,90) -> PPFD {ppfds}")


def demo_inverse_via_ledplant() -> None:
    print()
    print("=" * 60)
    print("2) LEDPlant._ppfd_to_pwm — inverse via Govee GPR (real targets)")
    print("=" * 60)
    thermal_dir = os.path.normpath(os.path.join(HERE, "..", "Thermal", "exported_models"))
    plant = LEDPlant(
        target_ppfd=20.0,
        r_b_ratio=0.83,
        use_pn_model=False,
        use_sp_to_ppfd=False,
        model_dir=thermal_dir,
        x_cm=0.0, y_cm=0.0, z_cm=10.0,
    )
    for t in [5, 10, 15, 20, 25, 30]:
        r, b = plant._ppfd_to_pwm(float(t))
        print(f"  target={t:>5.1f} µmol/m²/s -> pwm_r={r:>6.2f} pwm_b={b:>6.2f}")


def demo_gain_and_auto_target() -> None:
    print()
    print("=" * 60)
    print("3) Demo gain + auto-target — works around Govee's low max PPFD")
    print("=" * 60)
    thermal_dir = os.path.normpath(os.path.join(HERE, "..", "Thermal", "exported_models"))

    print("\n[3a] demo_gain=10×, target=200 (looks like a real plant light target)")
    plant = LEDPlant(
        target_ppfd=200.0,
        r_b_ratio=0.83,
        use_pn_model=False, use_sp_to_ppfd=False, model_dir=thermal_dir,
        x_cm=0.0, y_cm=0.0, z_cm=10.0,
        demo_gain=10.0,
    )
    for t in [50, 100, 150, 200, 250]:
        r, b = plant._ppfd_to_pwm(float(t))
        eff = t / plant.demo_gain
        print(f"  scaled target={t:>5.1f} (real={eff:>5.2f}) -> pwm_r={r:>6.2f} pwm_b={b:>6.2f}")

    print("\n[3b] auto_target_ppfd=True — pick target=70% of physical max automatically")
    plant2 = LEDPlant(
        target_ppfd=999.0,  # ignored
        r_b_ratio=0.83,
        use_pn_model=False, use_sp_to_ppfd=False, model_dir=thermal_dir,
        x_cm=0.0, y_cm=0.0, z_cm=10.0,
        demo_gain=1.0,
        auto_target_ppfd=True,
        auto_target_fraction=0.7,
    )
    print(f"  resolved target_ppfd = {plant2.target_ppfd:.2f}")
    for t in [plant2.target_ppfd * f for f in (0.25, 0.5, 0.75, 1.0)]:
        r, b = plant2._ppfd_to_pwm(float(t))
        print(f"  target={t:>5.2f} -> pwm_r={r:>6.2f} pwm_b={b:>6.2f}")


def main() -> int:
    demo_pwm_model_direct()
    demo_inverse_via_ledplant()
    demo_gain_and_auto_target()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
