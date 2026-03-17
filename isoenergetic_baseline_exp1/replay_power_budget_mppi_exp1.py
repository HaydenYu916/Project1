#!/usr/bin/env python3
"""Replay exp1 environment with the power-budgeted MPPI objective."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "LED_MPPI_Controller" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from led import PWMtoPowerModel  # noqa: E402
from mppi_v2 import LEDMPPIController, LEDPlant  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-log",
        type=Path,
        default=ROOT / "isoenergetic_baseline_exp1" / "mppi_v2_control_log_exp1.csv",
    )
    parser.add_argument(
        "--output-log",
        type=Path,
        default=ROOT / "isoenergetic_baseline_exp1" / "mppi_v2_control_log_exp1_power_budget_replay.csv",
    )
    parser.add_argument("--target-solar-vol", type=float, default=1.6)
    parser.add_argument("--power-budget-weight", type=float, default=25.0)
    parser.add_argument("--reference-weight", type=float, default=0.0)
    parser.add_argument("--dt-s", type=float, default=900.0)
    parser.add_argument("--gap-threshold-s", type=float, default=3600.0)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--num-samples", type=int, default=700)
    parser.add_argument("--mppi-temperature", type=float, default=1.0)
    parser.add_argument("--u-std", type=float, default=0.25)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def load_env_rows(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            row["timestamp_dt"] = datetime.strptime(str(row["timestamp"]), "%Y-%m-%d %H:%M:%S")
            row["input_temp"] = float(row["input_temp"])
            row["co2_ppm"] = float(row["co2_ppm"])
            rows.append(row)
    rows.sort(key=lambda r: r["timestamp_dt"])
    return rows


def assign_segments(rows: List[Dict[str, object]], gap_threshold_s: float) -> None:
    if not rows:
        return
    seg_id = 0
    rows[0]["replay_segment_id"] = seg_id
    prev_ts = rows[0]["timestamp_dt"]
    for row in rows[1:]:
        ts = row["timestamp_dt"]
        if (ts - prev_ts).total_seconds() > gap_threshold_s:
            seg_id += 1
        row["replay_segment_id"] = seg_id
        prev_ts = ts


def build_controller(
    target_solar_vol: float,
    power_budget_weight: float,
    reference_weight: float,
    dt_s: float,
    horizon: int,
    num_samples: int,
    mppi_temperature: float,
    u_std: float,
) -> tuple[LEDPlant, LEDMPPIController, float]:
    power_model = PWMtoPowerModel(include_intercept=True).fit(
        str(ROOT / "LED_MPPI_Controller" / "data" / "calib_data.csv")
    )
    plant = LEDPlant(
        base_ambient_temp=25.0,
        max_solar_vol=2.0,
        thermal_model_type="thermal",
        model_dir=str(ROOT / "LED_MPPI_Controller" / "Thermal" / "exported_models"),
        power_model=power_model,
        r_b_ratio=0.83,
        use_solar_vol_model=True,
    )
    controller = LEDMPPIController(
        plant=plant,
        horizon=horizon,
        num_samples=num_samples,
        dt=dt_s,
        temperature=mppi_temperature,
    )
    controller.set_constraints(u_min=0.05, u_max=float(plant.max_solar_vol), temp_min=20.0, temp_max=35.0)
    controller.set_weights(Q_photo=25.0, R_du=0.01, R_power=0.0, Q_ref=reference_weight)
    r_pwm, b_pwm = plant._solar_vol_to_pwm(float(target_solar_vol))  # noqa: SLF001
    power_key = plant._get_power_model_key(plant.r_b_ratio)  # noqa: SLF001
    target_mean_power = float(plant.power_model.predict(total_pwm=float(r_pwm + b_pwm), key=power_key))
    controller.set_power_budget(target_mean_power=target_mean_power, power_budget_weight=power_budget_weight)
    controller.set_mppi_params(u_std=u_std, dt=dt_s)
    controller.set_penalties(temp_penalty=1e3)
    return plant, controller, target_mean_power


def replay(args: argparse.Namespace) -> List[Dict[str, object]]:
    rows = load_env_rows(args.env_log, max_rows=args.max_rows)
    assign_segments(rows, args.gap_threshold_s)
    plant, controller, target_mean_power = build_controller(
        target_solar_vol=args.target_solar_vol,
        power_budget_weight=args.power_budget_weight,
        reference_weight=args.reference_weight,
        dt_s=args.dt_s,
        horizon=args.horizon,
        num_samples=args.num_samples,
        mppi_temperature=args.mppi_temperature,
        u_std=args.u_std,
    )
    out_rows: List[Dict[str, object]] = []
    current_segment: Optional[int] = None
    current_temp = 25.0
    mean_sequence = np.full(controller.horizon, args.target_solar_vol, dtype=float)
    solar_ref = None
    if args.reference_weight > 0:
        solar_ref = mean_sequence.copy()

    for row in rows:
        seg_id = int(row["replay_segment_id"])
        if current_segment != seg_id:
            current_segment = seg_id
            current_temp = float(row["input_temp"])
            controller.u_prev = 0.0

        current_co2 = float(row["co2_ppm"])
        plant.co2_ppm = current_co2
        optimal_sv, optimal_seq, success, cost, _weights = controller.solve(
            current_temp=current_temp,
            mean_sequence=mean_sequence,
            solar_vol_ref_seq=solar_ref,
        )
        (_sv, temp_pred, power_pred, pn_pred, r_series, b_series) = plant.predict(
            optimal_seq,
            current_temp,
            dt=args.dt_s,
            co2_sequence=np.full(len(optimal_seq), current_co2, dtype=float),
        )

        next_temp = float(temp_pred[0])
        next_power = float(power_pred[0])
        next_pn = float(pn_pred[0])
        r_pwm = float(r_series[0])
        b_pwm = float(b_series[0])
        out_rows.append(
            {
                "timestamp": row["timestamp"],
                "sensor_timestamp": row.get("sensor_timestamp", ""),
                "input_temp": float(current_temp),
                "co2_ppm": current_co2,
                "solar_vol_cmd": float(optimal_sv),
                "r_pwm": r_pwm,
                "b_pwm": b_pwm,
                "pred_temp": next_temp,
                "pred_power": next_power,
                "pred_pn": next_pn,
                "target_solar_vol": float(args.target_solar_vol),
                "target_mean_power": target_mean_power,
                "cost": float(cost),
                "success": bool(success),
                "note": f"power_budget_seg:{seg_id}",
            }
        )
        current_temp = next_temp
    return out_rows


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "sensor_timestamp",
        "input_temp",
        "co2_ppm",
        "solar_vol_cmd",
        "r_pwm",
        "b_pwm",
        "pred_temp",
        "pred_power",
        "pred_pn",
        "target_solar_vol",
        "target_mean_power",
        "cost",
        "success",
        "note",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = replay(args)
    write_rows(args.output_log, rows)
    print(f"wrote {len(rows)} rows to {args.output_log}")


if __name__ == "__main__":
    main()
