#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isoenergetic_baseline_analysis import (  # noqa: E402
    assign_dt,
    assign_replay_segments,
    attach_dynamic_lighting_state,
    build_time_window_mask,
    choose_constant_setting,
    compute_summary,
    init_plant,
    load_log,
    load_solar_vol_ppfd_lookup,
    recompute_exogenous_pn,
    run_segmented_thermal_replay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan oracle-tuned constant PPFD baselines for EXP4.")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=ROOT.parent / "log" / "EXP4_riotee_data_ppfd_plus50_pn.csv",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "control_log", "sensor_ppfd"),
        default="sensor_ppfd",
    )
    parser.add_argument(
        "--sensor-ppfd-column",
        type=str,
        default="ppfd_adjusted",
    )
    parser.add_argument(
        "--evaluation-window-start",
        type=str,
        default="09:00:00",
    )
    parser.add_argument(
        "--evaluation-window-end",
        type=str,
        default="01:00:00",
    )
    parser.add_argument(
        "--nominal-dt",
        type=float,
        default=900.0,
    )
    parser.add_argument(
        "--ppfd-step",
        type=float,
        default=10.0,
        help="Scan resolution in umol m^-2 s^-1.",
    )
    parser.add_argument(
        "--ppfd-min",
        type=float,
        default=None,
        help="Optional lower bound for scanned constant PPFD.",
    )
    parser.add_argument(
        "--ppfd-max",
        type=float,
        default=None,
        help="Optional upper bound for scanned constant PPFD.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "exp4_oracle_constant_scan.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "exp4_oracle_constant_scan_summary.json",
    )
    return parser.parse_args()


def prepare_dataframe(args: argparse.Namespace):
    df, input_meta = load_log(
        args.log_path,
        input_format=args.input_format,
        sensor_ppfd_column=args.sensor_ppfd_column,
    )
    evaluation_mask, evaluation_meta = build_time_window_mask(
        df,
        args.evaluation_window_start,
        args.evaluation_window_end,
    )
    df = assign_dt(df, dt_mode="nominal", nominal_dt=args.nominal_dt)
    plant = init_plant()
    solar_vals, ppfd_vals = load_solar_vol_ppfd_lookup(float(plant.r_b_ratio))
    df, ppfd_meta = attach_dynamic_lighting_state(df, plant, solar_vals, ppfd_vals)
    df = assign_replay_segments(df)
    return df, plant, solar_vals, ppfd_vals, {
        **input_meta,
        **evaluation_meta,
        **ppfd_meta,
    }, evaluation_mask


def run_single_target(
    df: pd.DataFrame,
    plant,
    solar_vals,
    ppfd_vals,
    evaluation_mask,
    nominal_dt: float,
    target_ppfd: float,
) -> dict[str, float]:
    baseline = choose_constant_setting(
        df,
        plant,
        "target_ppfd",
        solar_vals,
        ppfd_vals,
        target_ppfd=target_ppfd,
        evaluation_mask=evaluation_mask,
    )
    baseline_solar_series = np.where(evaluation_mask, float(baseline["solar_vol_const"]), 0.0).astype(float)
    run_df = recompute_exogenous_pn(df, plant, baseline_solar_series=baseline_solar_series)
    summary = compute_summary(run_df, baseline, evaluation_mask=evaluation_mask)
    _, segmented_summary = run_segmented_thermal_replay(
        run_df,
        plant,
        baseline_solar_series,
        nominal_dt,
        summary["dynamic_total_energy_wh"],
        summary["baseline_total_energy_wh"],
        summary_mask=evaluation_mask,
    )
    summary.update(segmented_summary)
    summary.update(
        {
            "cumulative_dynamic_primary_umol_m2": summary["cumulative_dynamic_segmented_umol_m2"],
            "cumulative_baseline_primary_umol_m2": summary["cumulative_baseline_segmented_umol_m2"],
            "dynamic_primary_umol_m2_per_wh": summary["dynamic_segmented_umol_m2_per_wh"],
            "baseline_primary_umol_m2_per_wh": summary["baseline_segmented_umol_m2_per_wh"],
            "dynamic_minus_baseline_primary_pct": summary["dynamic_minus_baseline_segmented_pct"],
        }
    )
    return {
        "target_ppfd": float(target_ppfd),
        "baseline_solar_vol_const": float(summary["baseline_solar_vol_const"]),
        "baseline_power_const_w": float(summary["baseline_power_const_w"]),
        "pn_int_umol_m2": float(summary["cumulative_baseline_primary_umol_m2"]),
        "pn_int_mol_m2": float(summary["cumulative_baseline_primary_umol_m2"]) / 1e6,
        "energy_wh": float(summary["baseline_total_energy_wh"]),
        "ep_umol_per_wh": float(summary["baseline_primary_umol_m2_per_wh"]),
        "ep_mmol_per_wh": float(summary["baseline_primary_umol_m2_per_wh"]) / 1000.0,
        "mean_temp_c": float(summary["mean_temp_baseline_segmented_c"]),
        "max_temp_c": float(summary["max_temp_baseline_segmented_c"]),
        "pn_pct_vs_dynamic": 100.0
        * (
            float(summary["cumulative_baseline_primary_umol_m2"])
            - float(summary["cumulative_dynamic_primary_umol_m2"])
        )
        / float(summary["cumulative_dynamic_primary_umol_m2"]),
        "energy_pct_vs_dynamic": float(summary["baseline_minus_dynamic_energy_pct"]),
        "ep_pct_vs_dynamic": 100.0
        * (float(summary["baseline_primary_umol_m2_per_wh"]) - float(summary["dynamic_primary_umol_m2_per_wh"]))
        / float(summary["dynamic_primary_umol_m2_per_wh"]),
    }


def main() -> None:
    args = parse_args()
    df, plant, solar_vals, ppfd_vals, meta, evaluation_mask = prepare_dataframe(args)

    ppfd_min_raw = float(ppfd_vals.min())
    ppfd_max_raw = float(ppfd_vals.max())
    step = max(float(args.ppfd_step), 1.0)
    ppfd_min = max(int(round(ppfd_min_raw)), int(round(step)))
    ppfd_max = int(round(ppfd_max_raw))
    if args.ppfd_min is not None:
        ppfd_min = max(ppfd_min, int(round(float(args.ppfd_min))))
    if args.ppfd_max is not None:
        ppfd_max = min(ppfd_max, int(round(float(args.ppfd_max))))
    if ppfd_min > ppfd_max:
        raise ValueError("Invalid PPFD scan range after applying ppfd-min/ppfd-max")
    targets = []
    value = float(ppfd_min)
    while value <= float(ppfd_max) + 1e-9:
        targets.append(round(value, 6))
        value += step

    rows = [
        run_single_target(
            df,
            plant,
            solar_vals,
            ppfd_vals,
            evaluation_mask,
            args.nominal_dt,
            target_ppfd,
        )
        for target_ppfd in targets
    ]
    scan_df = pd.DataFrame(rows).sort_values("target_ppfd").reset_index(drop=True)

    best_by_pn = scan_df.loc[scan_df["pn_int_umol_m2"].idxmax()].to_dict()
    best_by_ep = scan_df.loc[scan_df["ep_umol_per_wh"].idxmax()].to_dict()

    summary = {
        **meta,
        "scan_ppfd_min": ppfd_min,
        "scan_ppfd_max": ppfd_max,
        "scan_ppfd_step": step,
        "num_scan_points": int(len(scan_df)),
        "oracle_best_by_pn_int": best_by_pn,
        "oracle_best_by_ep": best_by_ep,
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    scan_df.to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")
    print("\nBest by Pn int")
    print(pd.Series(best_by_pn).to_string())
    print("\nBest by EP")
    print(pd.Series(best_by_ep).to_string())


if __name__ == "__main__":
    main()
