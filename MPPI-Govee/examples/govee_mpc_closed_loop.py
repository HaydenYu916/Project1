#!/usr/bin/env python3
"""End-to-end Govee MPPI closed-loop verification.

This is the first demo that actually exercises LEDMPPIController.solve()
on real hardware:

  while step < N:
      u = mppi.solve(current_temp)               # PPFD target
      r, b = plant._ppfd_to_pwm(u)               # via Govee xyz GPR
      govee_controller.set_led_mix(r, b)         # BLE dispatch
      sleep dt
      ppfd_meas = spec_trigger()                 # closed-loop observation
      log(target, u, r, b, predicted, measured)

The cost is configured for **pure tracking** (Q_ref dominant, Pn off,
temperature constraint loose) so the demo's behaviour reflects MPPI's
ability to chase a setpoint at this Govee position, not the full plant
biology objective. That keeps the verification narrow: did MPPI close
the loop, did its sampled-and-weighted control track the target, did
BLE/spec stay healthy.

Usage:
  cd /home/pi/Desktop/Project1/MPPI-Govee/examples
  python3 govee_mpc_closed_loop.py            # default 8 steps @ z=6
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
LED_GOVEE_SRC = os.path.join(PROJECT_ROOT, "Tool", "LED_Govee", "src")
SPEC_DIR = os.path.join(PROJECT_ROOT, "Tool", "Spectrometer")
for p in (SRC, LED_GOVEE_SRC, SPEC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import serial  # noqa: E402

from mppi_v2 import LEDPlant, LEDMPPIController  # noqa: E402
import govee_controller  # type: ignore  # noqa: E402
from lib import (  # type: ignore  # noqa: E402
    complete_spectrum_measurement,
    initialize_spectrometer_session,
)

PPFD_KEYS = (
    "PPFD(umol/㎡/s)",
    "PPFD(umol/m2/s)",
    "PPFD(umol/m²/s)",
)


@dataclass
class StepRecord:
    t_iso: str
    step: int
    target_ppfd: float
    mppi_u: float
    pwm_r: float
    pwm_b: float
    pred_ppfd: float
    measured_ppfd: Optional[float]
    err_abs: Optional[float]
    err_pct: Optional[float]
    current_temp: float
    spec_file: Optional[str]


def extract_ppfd(parsed) -> Optional[float]:
    if isinstance(parsed, dict):
        for k in PPFD_KEYS:
            if k in parsed:
                try:
                    return float(parsed[k])
                except (TypeError, ValueError):
                    pass
    return None


def extract_ppfd_from_csv(path: Optional[str]) -> Optional[float]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("PPFD("):
                    parts = line.rstrip("\n").split(",")
                    if len(parts) >= 2:
                        try:
                            return float(parts[1])
                        except ValueError:
                            return None
    except OSError:
        return None
    return None


def trigger_spec(ser):
    try:
        result = complete_spectrum_measurement(ser)
    except Exception as exc:
        print(f"  ⚠️ spec trigger failed: {exc}")
        return None, None
    if not isinstance(result, tuple) or len(result) < 1:
        return None, None
    spectrum_result = result[0]
    if not isinstance(spectrum_result, tuple) or len(spectrum_result) < 3:
        return None, None
    spec_file = spectrum_result[1] if isinstance(spectrum_result[1], str) else None
    parsed = spectrum_result[2]
    ppfd = extract_ppfd(parsed)
    if ppfd is None:
        ppfd = extract_ppfd_from_csv(spec_file)
    return ppfd, spec_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Govee MPPI closed-loop demo")
    p.add_argument("--target-ppfd", type=float, default=15.0,
                   help="Tracking target (µmol/m²/s, real units; pick within "
                        "Govee's max ~25 at z=6)")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--dt", type=float, default=12.0,
                   help="Seconds between MPPI updates (control step)")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=120)
    p.add_argument("--rb", type=float, default=0.8,
                   help="Red fraction (must be in training set; 0.8=4:1)")
    p.add_argument("--x-cm", type=float, default=0.0)
    p.add_argument("--y-cm", type=float, default=0.0)
    p.add_argument("--z-cm", type=float, default=6.0)
    p.add_argument("--spec-port", default="/dev/ttyACM1")
    p.add_argument("--demo-gain", type=float, default=1.0)
    p.add_argument("--transport", choices=["ble", "ha"], default="ha",
                   help="LED dispatch path. HA is more reliable across "
                        "consecutive writes; BLE re-discovers services every "
                        "few minutes and tends to drop mid-demo.")
    p.add_argument("--out-csv", default="/tmp/govee_mpc_closed_loop.csv")
    p.add_argument("--ambient-temp", type=float, default=25.0,
                   help="Initial ambient temp (°C). MPPI uses this in cost; "
                        "for demo without temp control, doesn't matter much.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"🎬 Govee MPC closed-loop — target={args.target_ppfd}, steps={args.steps}, "
          f"dt={args.dt}s, horizon={args.horizon}, samples={args.num_samples}")
    print(f"   pos=({args.x_cm},{args.y_cm},{args.z_cm}) cm, R:B={args.rb}")

    thermal_dir = os.path.normpath(os.path.join(HERE, "..", "Thermal", "exported_models"))
    plant = LEDPlant(
        target_ppfd=float(args.target_ppfd),
        r_b_ratio=float(args.rb),
        use_pn_model=True,            # required: predict() always evaluates Pn
        use_sp_to_ppfd=False,         # spectrum_to_ppfd not used here
        model_dir=thermal_dir,
        x_cm=args.x_cm, y_cm=args.y_cm, z_cm=args.z_cm,
        demo_gain=float(args.demo_gain),
    )
    plant.ambient_temp = float(args.ambient_temp)

    mppi = LEDMPPIController(
        plant,
        horizon=int(args.horizon),
        num_samples=int(args.num_samples),
        dt=float(args.dt),
        temperature=0.5,
    )
    # Cost: pure target-tracking. Pn off, power/du gentle, no oob penalty.
    mppi.set_weights(Q_photo=0.0, R_du=0.05, R_power=0.0, Q_ref=1.0)
    mppi.set_constraints(u_min=0.0, u_max=max(args.target_ppfd * 1.5, 30.0),
                         temp_min=0.0, temp_max=80.0)
    mppi.set_ppfd_train_range(ppfd_min=0.0, ppfd_max=200.0, R_oob=0.0)
    mppi.set_target_ppfd(float(args.target_ppfd))

    if args.transport == "ble":
        print(f"🔌 Govee BLE connect ...")
        govee_controller.connect()
    else:
        print(f"☁️  Govee transport=HA (set_led_mix_ha)")
    print(f"🔬 spectrometer @ {args.spec_port}")
    ser = serial.Serial(args.spec_port, baudrate=115200, timeout=1)
    initialize_spectrometer_session(ser)

    records: List[StepRecord] = []
    current_temp = float(args.ambient_temp)
    try:
        for step in range(1, args.steps + 1):
            u, _seq, ok, min_cost, _w = mppi.solve(current_temp)
            r_pwm, b_pwm = plant._ppfd_to_pwm(u)
            pred = float(plant.govee_pwm_model.predict(r_pwm=r_pwm, b_pwm=b_pwm)
                         if plant.govee_pwm_model else 0.0) * plant.demo_gain
            print(f"\n[step {step}/{args.steps}] mppi_u={u:.2f} -> "
                  f"pwm=({r_pwm:.2f},{b_pwm:.2f}), pred={pred:.2f}, min_cost={min_cost:.2f}")
            try:
                if args.transport == "ble":
                    govee_controller.set_led_mix(int(round(r_pwm)), int(round(b_pwm)))
                else:
                    govee_controller.set_led_mix_ha(int(round(r_pwm)), int(round(b_pwm)),
                                                    timeout_sec=10.0)
            except Exception as exc:
                print(f"  ❌ set_led_mix failed: {exc}")
                continue
            time.sleep(max(0.0, float(args.dt)))
            measured, spec_file = trigger_spec(ser)
            err_abs = err_pct = None
            if measured is not None:
                err_abs = measured - pred
                err_pct = 100.0 * err_abs / max(pred, 1e-6)
                print(f"  📊 measured={measured:.2f}, err={err_abs:+.2f} ({err_pct:+.1f}%)")
            else:
                print(f"  ⚠️ no PPFD parsed")

            records.append(StepRecord(
                t_iso=datetime.now().isoformat(timespec="seconds"),
                step=step, target_ppfd=float(args.target_ppfd),
                mppi_u=float(u), pwm_r=float(r_pwm), pwm_b=float(b_pwm),
                pred_ppfd=pred, measured_ppfd=measured,
                err_abs=err_abs, err_pct=err_pct,
                current_temp=current_temp, spec_file=spec_file,
            ))
    finally:
        try:
            if args.transport == "ble":
                govee_controller.set_led_full_white()
            else:
                govee_controller.set_led_full_white_ha(timeout_sec=10.0)
            print("\n💡 lights → full white for charging")
        except Exception as exc:
            print(f"\n⚠️ couldn't set full white: {exc}")
        if args.transport == "ble":
            try:
                govee_controller.disconnect()
            except Exception:
                pass
        try:
            ser.close()
        except Exception:
            pass

    print("\n" + "=" * 75)
    print(f"{'step':>4} {'tgt':>6} {'mppi_u':>7} {'r_pwm':>7} {'b_pwm':>7} {'pred':>6} {'meas':>7} {'err%':>7}")
    print("=" * 75)
    for r in records:
        m = f"{r.measured_ppfd:.2f}" if r.measured_ppfd is not None else "—"
        e = f"{r.err_pct:+.1f}" if r.err_pct is not None else "—"
        print(f"{r.step:>4} {r.target_ppfd:>6.2f} {r.mppi_u:>7.2f} "
              f"{r.pwm_r:>7.2f} {r.pwm_b:>7.2f} {r.pred_ppfd:>6.2f} {m:>7} {e:>7}")

    if args.out_csv and records:
        new = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
            if new:
                w.writeheader()
            for r in records:
                w.writerow(asdict(r))
        print(f"\n📝 CSV: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
