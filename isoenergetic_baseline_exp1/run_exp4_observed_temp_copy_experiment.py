#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ISO_DIR = ROOT / "isoenergetic_baseline_exp1"
if str(ISO_DIR) not in sys.path:
    sys.path.insert(0, str(ISO_DIR))

from isoenergetic_baseline_analysis import (  # noqa: E402
    attach_dynamic_lighting_state,
    init_plant,
    load_solar_vol_ppfd_lookup,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EXP4 observed-temperature copy experiment.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT / "log" / "EXP4_riotee_data_ppfd_plus50_pn.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "log" / "EXP4_observed_tempcopy_0914_from_prevdays.csv",
    )
    parser.add_argument(
        "--daily-summary-csv",
        type=Path,
        default=ROOT / "log" / "EXP4_pn_daily_comparison_observed_tempcopy_0914_from_prevdays.csv",
    )
    parser.add_argument(
        "--time-window-start",
        type=str,
        default="09:00:00",
    )
    parser.add_argument(
        "--time-window-end",
        type=str,
        default="01:00:00",
    )
    return parser.parse_args()


def build_eval_mask(ts: pd.Series) -> pd.Series:
    hours = ts.dt.hour
    return (hours >= 9) | (hours < 1)


def seconds_since_midnight(ts: pd.Timestamp) -> float:
    return float(ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1_000_000)


def copy_temperature_by_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date_only"] = out["timestamp"].dt.date
    out["source_date"] = ""
    out["temperature_copied"] = out["temperature"]
    out["copy_applied"] = False

    target_dates = [date(2025, 11, day) for day in range(9, 15)]
    source_dates = [date(2025, 11, day) for day in range(2, 8)]
    mapping = dict(zip(target_dates, source_dates, strict=True))

    source_frames: dict[date, pd.DataFrame] = {}
    for src_date in source_dates:
        src = out.loc[out["date_only"] == src_date, ["timestamp", "temperature"]].copy()
        src["seconds_of_day"] = src["timestamp"].apply(seconds_since_midnight)
        src = src.sort_values("seconds_of_day")
        source_frames[src_date] = src.reset_index(drop=True)

    for tgt_date, src_date in mapping.items():
        tgt_mask = out["date_only"] == tgt_date
        if not tgt_mask.any():
            continue
        src = source_frames[src_date]
        src_seconds = src["seconds_of_day"].to_numpy(dtype=float)
        src_temps = src["temperature"].to_numpy(dtype=float)

        for idx, ts in out.loc[tgt_mask, "timestamp"].items():
            sec = seconds_since_midnight(ts)
            insert_at = int(np.searchsorted(src_seconds, sec))
            if insert_at <= 0:
                nearest_idx = 0
            elif insert_at >= len(src_seconds):
                nearest_idx = len(src_seconds) - 1
            else:
                prev_idx = insert_at - 1
                next_idx = insert_at
                nearest_idx = prev_idx if abs(src_seconds[prev_idx] - sec) <= abs(src_seconds[next_idx] - sec) else next_idx

            out.at[idx, "temperature_copied"] = float(src_temps[nearest_idx])
            out.at[idx, "source_date"] = str(src_date)
            out.at[idx, "copy_applied"] = True

    return out


def recompute_pn(df: pd.DataFrame) -> pd.DataFrame:
    plant = init_plant()
    solar_vals, ppfd_vals = load_solar_vol_ppfd_lookup(float(plant.r_b_ratio))
    prepared = df.rename(columns={"temperature_copied": "input_temp"}).copy()
    prepared["co2_ppm"] = pd.to_numeric(prepared["co2"], errors="coerce")
    prepared["ppfd_dynamic"] = pd.to_numeric(prepared["ppfd_adjusted"], errors="coerce")
    prepared, _ = attach_dynamic_lighting_state(prepared, plant, solar_vals, ppfd_vals)

    rb = float(plant.r_b_ratio)
    prepared["pn_recomputed_tempcopy"] = [
        plant.get_photosynthesis_rate(float(sv), float(temp), float(co2), rb)
        for sv, temp, co2 in zip(prepared["solar_vol_cmd"], prepared["input_temp"], prepared["co2_ppm"])
    ]
    return prepared


def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["日期"] = work["timestamp"].dt.date.astype(str)
    rows = []
    for day, day_df in work.groupby("日期", sort=True):
        original = day_df["pn"].mean()
        copied = day_df["pn_recomputed_tempcopy"].mean()
        diff = copied - original
        pct = 100.0 * diff / copied if copied else float("nan")
        rows.append(
            {
                "日期": day,
                "Pn均值_原始": round(float(original), 2),
                "Pn均值_温度复制实验": round(float(copied), 2),
                "Pn差值": round(float(diff), 2),
                "Pn差值_%": round(float(pct), 2),
                "数据点数": int(len(day_df)),
                "温度是否复制": "是" if bool(day_df["copy_applied"].any()) else "否",
                "复制来源日期": ",".join(sorted({str(v) for v in day_df["source_date"].dropna().unique()})),
            }
        )

    summary = pd.DataFrame(rows)
    mean_original = float(work["pn"].mean())
    mean_copied = float(work["pn_recomputed_tempcopy"].mean())
    mean_diff = mean_copied - mean_original
    mean_pct = 100.0 * mean_diff / mean_copied if mean_copied else float("nan")
    summary.loc[len(summary)] = {
        "日期": "平均",
        "Pn均值_原始": round(mean_original, 2),
        "Pn均值_温度复制实验": round(mean_copied, 2),
        "Pn差值": round(mean_diff, 2),
        "Pn差值_%": round(mean_pct, 2),
        "数据点数": int(len(work)),
        "温度是否复制": "",
        "复制来源日期": "",
    }
    return summary


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv, comment="#")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["co2"] = pd.to_numeric(df["co2"], errors="coerce")
    df["pn"] = pd.to_numeric(df["pn"], errors="coerce")
    df["ppfd_adjusted"] = pd.to_numeric(df["ppfd_adjusted"], errors="coerce")
    df = df.dropna(subset=["timestamp", "temperature", "co2", "pn", "ppfd_adjusted"]).copy()

    mask = build_eval_mask(df["timestamp"])
    df = df.loc[mask].copy().reset_index(drop=True)

    copied = copy_temperature_by_time_of_day(df)
    recomputed = recompute_pn(copied)
    summary = build_daily_summary(recomputed)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.daily_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    recomputed.to_csv(args.output_csv, index=False)
    summary.to_csv(args.daily_summary_csv, index=False, encoding="utf-8-sig")

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.daily_summary_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
