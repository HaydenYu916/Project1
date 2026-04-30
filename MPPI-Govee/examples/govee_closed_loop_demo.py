#!/usr/bin/env python3
"""Govee MPPI closed-loop demo (real hardware).

Flow per step:
  1. Pick a target PPFD from the demo schedule.
  2. Use GoveePWMtoPPFDModel inverse to compute (pwm_r, pwm_b) at the
     current (x, y, z).
  3. Dispatch via Govee BLE (govee_controller.set_led_mix) — both bars
     get the same RGB(R, 0, B) mix per the project rule.
  4. Wait settle_sec for LED to stabilize.
  5. Trigger the spectrometer once and read PPFD_spec_mean.
  6. Log: target, predicted, measured, error.

This is the simplest faithful demo: open-loop control + closed-loop
observation. Useful to (a) sanity-check the trained Govee PWM→PPFD
model in situ and (b) show MPPI's predict→actuate→observe cycle on real
hardware without dragging in the full MPPI optimization.

Defaults:
  - position: board center, z=10cm
  - R:B ratio: 0.83 (5:1 red-heavy, matches existing calib data)
  - demo_gain: 1.0 (real numbers; honest mode)
  - targets: [5, 10, 15, 20, 25] µmol/m²/s — within Govee max ~35
  - settle: 8s; spec timeout: 60s

Pre-flight:
  - Riotee collector running (per memory rule, system_manager.py)
  - Spectrometer USB at /dev/ttyACM1
  - Govee H6056 BLE in range; HA disabled or not contending for BLE

Usage:
  cd /home/pi/Desktop/Project1/MPPI-Govee/examples
  python3 govee_closed_loop_demo.py
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

import serial  # noqa: E402

from govee_pwmtoppfd_model import GoveePWMtoPPFDModel, GoveePosition  # noqa: E402
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
    target_ppfd: float
    pred_ppfd: float
    pwm_r: float
    pwm_b: float
    measured_ppfd: Optional[float]
    error_abs: Optional[float]
    error_pct: Optional[float]
    spec_file: Optional[str]


def extract_ppfd(parsed) -> Optional[float]:
    if not isinstance(parsed, dict):
        return None
    for k in PPFD_KEYS:
        if k in parsed:
            try:
                return float(parsed[k])
            except (TypeError, ValueError):
                pass
    return None


def extract_ppfd_from_csv(path: Optional[str]) -> Optional[float]:
    """Fallback: read the spectrometer's standard_csv. The PPFD line is
    `PPFD(umol/㎡/s),<value>` (the ㎡ uses a CJK square meter glyph)."""
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
                    return None
    except OSError:
        return None
    return None


def trigger_one_spec(ser) -> tuple[Optional[float], Optional[str]]:
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
    p = argparse.ArgumentParser(description="Govee closed-loop PWM→PPFD demo")
    p.add_argument("--targets", type=str, default="5,10,15,20,25",
                   help="Comma-separated PPFD targets (µmol/m²/s)")
    p.add_argument("--rb", type=float, default=0.83,
                   help="Red fraction r/(r+b), default 0.83 (≈5:1)")
    p.add_argument("--x-cm", type=float, default=0.0)
    p.add_argument("--y-cm", type=float, default=0.0)
    p.add_argument("--z-cm", type=float, default=10.0)
    p.add_argument("--settle-sec", type=float, default=8.0,
                   help="Wait after PWM dispatch before measuring")
    p.add_argument("--spec-port", default="/dev/ttyACM1")
    p.add_argument("--demo-gain", type=float, default=1.0)
    p.add_argument("--out-csv", type=str, default=None,
                   help="Append per-step records to this CSV")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    targets: List[float] = [float(x) for x in args.targets.split(",") if x.strip()]
    if not targets:
        print("no targets", file=sys.stderr)
        return 2

    print(f"🎬 Govee closed-loop demo — {len(targets)} targets at "
          f"pos=({args.x_cm},{args.y_cm},{args.z_cm}) cm, R:B={args.rb}, "
          f"demo_gain={args.demo_gain}")

    # 1) Load model + check physical max so we don't ask for the impossible
    model = GoveePWMtoPPFDModel(
        position=GoveePosition(args.x_cm, args.y_cm, args.z_cm)
    ).load()
    phys_max = max(0.0, float(
        model.predict(r_pwm=100.0 * args.rb, b_pwm=100.0 * (1 - args.rb))
    ))
    print(f"   physical max PPFD @ pos = {phys_max:.2f} µmol/m²/s")
    over = [t / args.demo_gain for t in targets if t / args.demo_gain > phys_max]
    if over:
        print(f"   ⚠️ {len(over)} target(s) exceed physical max — those will saturate")

    # 2) Connect Govee BLE
    print("🔌 connecting Govee BLE...")
    govee_controller.connect()
    print("   connected")

    # 3) Open spectrometer + init session
    print(f"🔬 opening spectrometer at {args.spec_port}")
    ser = serial.Serial(args.spec_port, baudrate=115200, timeout=1)
    initialize_spectrometer_session(ser)
    print("   session initialized")

    records: List[StepRecord] = []
    try:
        for i, target in enumerate(targets, 1):
            target_real = target / args.demo_gain
            r_arr, b_arr = _inverse_pwm(model, target_real, args.rb)
            pred = float(model.predict(r_pwm=r_arr, b_pwm=b_arr))
            print(f"\n[{i}/{len(targets)}] target={target} (real={target_real:.2f}) "
                  f"-> pwm_r={r_arr:.2f} pwm_b={b_arr:.2f}  pred_ppfd={pred:.2f}")

            # Dispatch
            try:
                govee_controller.set_led_mix(int(round(r_arr)), int(round(b_arr)))
                print(f"   ✅ set_led_mix dispatched")
            except Exception as exc:
                print(f"   ❌ set_led_mix failed: {exc}")
                continue

            # Settle
            time.sleep(max(0.0, float(args.settle_sec)))

            # Measure
            measured, spec_file = trigger_one_spec(ser)
            if measured is None:
                print(f"   ⚠️ no PPFD parsed")
                err_abs = err_pct = None
            else:
                err_abs = measured - pred
                err_pct = 100.0 * err_abs / max(pred, 1e-6)
                print(f"   📊 measured={measured:.2f}  err={err_abs:+.2f} ({err_pct:+.1f}% vs pred)")

            records.append(StepRecord(
                t_iso=datetime.now().isoformat(timespec="seconds"),
                target_ppfd=float(target),
                pred_ppfd=pred,
                pwm_r=float(r_arr), pwm_b=float(b_arr),
                measured_ppfd=measured,
                error_abs=err_abs, error_pct=err_pct,
                spec_file=spec_file,
            ))
    finally:
        # Per project rule: turn on full white for charging after a run
        try:
            govee_controller.set_led_full_white()
            print("\n💡 lights → RGB(255,255,255) full white for charging")
        except Exception as exc:
            print(f"\n⚠️ couldn't set full white: {exc}")
        try:
            govee_controller.disconnect()
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass

    # Summary
    print("\n" + "=" * 60)
    print(f"{'target':>8} {'pred':>8} {'measured':>10} {'|Δ|':>7} {'pwm_r':>7} {'pwm_b':>7}")
    print("=" * 60)
    for r in records:
        m_str = f"{r.measured_ppfd:.2f}" if r.measured_ppfd is not None else "—"
        e_str = f"{r.error_abs:+.2f}" if r.error_abs is not None else "—"
        print(f"{r.target_ppfd:>8.1f} {r.pred_ppfd:>8.2f} {m_str:>10} "
              f"{e_str:>7} {r.pwm_r:>7.2f} {r.pwm_b:>7.2f}")

    if args.out_csv:
        new = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
            if new:
                w.writeheader()
            for r in records:
                w.writerow(asdict(r))
        print(f"\n📝 CSV: {args.out_csv}")

    return 0


def _inverse_pwm(model: GoveePWMtoPPFDModel, target_real: float,
                 rb_fraction: float) -> tuple[float, float]:
    """Same inversion as LEDPlant._govee_ppfd_to_pwm but standalone."""
    import numpy as np
    rb = max(0.0, min(1.0, float(rb_fraction)))
    totals = np.linspace(0.0, 100.0, 51)
    r = totals * rb
    b = totals * (1.0 - rb)
    ppfds = np.maximum(model.predict_batch(r, b), 0.0)
    target = max(0.0, float(target_real))
    if target <= ppfds[0]:
        return float(r[0]), float(b[0])
    if target >= ppfds[-1]:
        return float(r[-1]), float(b[-1])
    order = np.argsort(ppfds)
    total_hat = float(np.interp(target, ppfds[order], totals[order]))
    return total_hat * rb, total_hat * (1.0 - rb)


if __name__ == "__main__":
    raise SystemExit(main())
