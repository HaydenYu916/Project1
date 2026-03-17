#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot EXP4 replay temperature trajectories.")
    parser.add_argument(
        "--mean-timeseries",
        type=Path,
        default=ROOT / "exp4_eval_9am_to_1am_full_replay_dynamic_vs_mean_ppfd_timeseries.csv",
        help="Timeseries CSV containing dynamic and constant_mean_ppfd replay columns.",
    )
    parser.add_argument(
        "--ppfd450-timeseries",
        type=Path,
        default=ROOT / "exp4_eval_9am_to_1am_full_replay_dynamic_vs_ppfd450_timeseries.csv",
        help="Timeseries CSV containing constant_450_ppfd replay columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "exp4_eval_9am_to_1am_full_replay_temperature.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def build_eval_mask(ts: pd.Series) -> pd.Series:
    hours = ts.dt.hour
    return (hours >= 9) | (hours < 1)


def main() -> None:
    args = parse_args()

    mean_df = pd.read_csv(args.mean_timeseries)
    ppfd450_df = pd.read_csv(args.ppfd450_timeseries)

    for df in (mean_df, ppfd450_df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    merged = mean_df[
        [
            "timestamp",
            "input_temp",
            "temp_dynamic_segmented",
            "temp_baseline_segmented",
        ]
    ].rename(
        columns={
            "input_temp": "temp_observed",
            "temp_dynamic_segmented": "temp_dynamic",
            "temp_baseline_segmented": "temp_constant_mean_ppfd",
        }
    )
    merged["temp_constant_450_ppfd"] = ppfd450_df["temp_baseline_segmented"].to_numpy()
    merged = merged.dropna(subset=["timestamp"]).copy()
    merged["eval_window"] = build_eval_mask(merged["timestamp"])

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=False, constrained_layout=True)

    full_ax = axes[0]
    full_ax.plot(merged["timestamp"], merged["temp_observed"], color="#6c757d", linewidth=1.4, label="Observed")
    full_ax.plot(merged["timestamp"], merged["temp_dynamic"], color="#1b4965", linewidth=1.8, label="Dynamic")
    full_ax.plot(
        merged["timestamp"],
        merged["temp_constant_mean_ppfd"],
        color="#2a9d8f",
        linewidth=1.8,
        label="Constant Mean PPFD",
    )
    full_ax.plot(
        merged["timestamp"],
        merged["temp_constant_450_ppfd"],
        color="#e76f51",
        linewidth=1.8,
        label="Constant 450 PPFD",
    )
    full_ax.set_title("EXP4 Temperature Replay, Full 24h Thermal Replay")
    full_ax.set_ylabel("Temperature (C)")
    full_ax.grid(True, alpha=0.25)
    full_ax.legend(ncol=4, frameon=False, loc="upper left")

    eval_df = merged.loc[merged["eval_window"]].copy()
    eval_ax = axes[1]
    eval_ax.plot(eval_df["timestamp"], eval_df["temp_observed"], color="#6c757d", linewidth=1.4, label="Observed")
    eval_ax.plot(eval_df["timestamp"], eval_df["temp_dynamic"], color="#1b4965", linewidth=1.8, label="Dynamic")
    eval_ax.plot(
        eval_df["timestamp"],
        eval_df["temp_constant_mean_ppfd"],
        color="#2a9d8f",
        linewidth=1.8,
        label="Constant Mean PPFD",
    )
    eval_ax.plot(
        eval_df["timestamp"],
        eval_df["temp_constant_450_ppfd"],
        color="#e76f51",
        linewidth=1.8,
        label="Constant 450 PPFD",
    )
    eval_ax.set_title("Evaluation Window Only, 09:00 to Next-Day 01:00")
    eval_ax.set_ylabel("Temperature (C)")
    eval_ax.set_xlabel("Timestamp")
    eval_ax.grid(True, alpha=0.25)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", rotation=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
