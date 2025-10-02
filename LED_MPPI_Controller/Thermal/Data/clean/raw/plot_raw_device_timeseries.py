#!/usr/bin/env python3
"""Plot temperature and a1_raw vs time from raw Riotee logs.

For each device_id present in the raw CSV data, this script creates a single
figure that contains one subplot per recording date. Each subplot overlays the
temperature (left y-axis) and a1_raw (right y-axis) values against timestamp.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot raw device temperature and a1_raw timeseries")
    parser.add_argument(
        "--glob",
        default="riotee_data*.csv",
        help="Pattern to match raw CSV files (default: riotee_data*.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to store output images (default: current directory)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving",
    )
    return parser.parse_args()


def load_data(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, comment="#", parse_dates=["timestamp"])
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No CSV files matched the provided pattern")
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    return combined


def sanitize_device(device_id: str) -> str:
    return device_id.replace("=", "").replace("/", "-")


def plot_device(device_id: str, device_df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    device_df = device_df.copy()
    device_df["date"] = device_df["timestamp"].dt.date
    unique_dates = sorted(device_df["date"].unique())
    if not unique_dates:
        return

    n_rows = len(unique_dates)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]

    for ax, date in zip(axes, unique_dates):
        daily_df = device_df[device_df["date"] == date]
        if daily_df.empty:
            continue

        ax.plot(daily_df["timestamp"], daily_df["temperature"], color="tab:red", label="temperature (°C)")
        ax.set_ylabel("Temperature (°C)", color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:red")

        ax2 = ax.twinx()
        ax2.plot(daily_df["timestamp"], daily_df["a1_raw"], color="tab:blue", alpha=0.7, label="a1_raw")
        ax2.set_ylabel("a1_raw", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        ax.set_title(f"{device_id} — {date}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.grid(True, axis="x", linestyle=":", linewidth=0.5)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    outfile = output_dir / f"raw_plot_{sanitize_device(device_id)}.png"
    fig.savefig(outfile, dpi=200)
    print(f"✅ Saved {outfile}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    files = sorted(Path.cwd().glob(args.glob))
    if not files:
        raise FileNotFoundError("No CSV files matched the provided pattern")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(files)

    for device_id, group in df.groupby("device_id"):
        plot_device(str(device_id), group, output_dir, args.show)


if __name__ == "__main__":
    main()
