#!/usr/bin/env python3
"""Visualize a Govee calibration run's segment_table.csv.

Produces three figures saved next to the input CSV:
  1) ppfd_vs_total.png  — PPFD vs total_pwm, one line per ratio
  2) ppfd_heatmap.png   — PPFD as a function of (pwm_r, pwm_b)
  3) ppfd_per_trigger.png — within-segment PPFD spread (n=3 triggers)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(run_dir: Path) -> pd.DataFrame:
    csv = run_dir / "segment_table.csv"
    df = pd.read_csv(csv)
    df = df[df["segment_type"] == "normal"].copy()
    df["ratio"] = df["ratio_r"].astype(str) + ":" + df["ratio_b"].astype(str)
    df["PPFD_spec_mean"] = pd.to_numeric(df["PPFD_spec_mean"], errors="coerce")
    return df


def plot_ppfd_vs_total(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for ratio, sub in df.groupby("ratio"):
        sub = sub.sort_values("total_pwm")
        ax.plot(sub["total_pwm"], sub["PPFD_spec_mean"],
                marker="o", label=f"R:B={ratio}")
    ax.set_xlabel("total PWM (pwm_r + pwm_b)")
    ax.set_ylabel("PPFD (umol/m²/s)  spectrometer mean")
    ax.set_title("PPFD vs total PWM by red:blue ratio")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, out: Path) -> None:
    rs = sorted(df["pwm_r"].unique())
    bs = sorted(df["pwm_b"].unique())
    grid = np.full((len(bs), len(rs)), np.nan)
    for _, row in df.iterrows():
        i = bs.index(row["pwm_b"])
        j = rs.index(row["pwm_r"])
        grid[i, j] = row["PPFD_spec_mean"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                   extent=[min(rs) - 2, max(rs) + 2, min(bs) - 2, max(bs) + 2])
    ax.set_xticks(rs)
    ax.set_yticks(bs)
    ax.set_xlabel("pwm_r")
    ax.set_ylabel("pwm_b")
    ax.set_title("PPFD heatmap (umol/m²/s)")
    for _, row in df.iterrows():
        v = row["PPFD_spec_mean"]
        if pd.notna(v):
            ax.text(row["pwm_r"], row["pwm_b"], f"{v:.1f}",
                    ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label="PPFD")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_trigger_spread(df: pd.DataFrame, out: Path) -> None:
    triggers = ["ppfd_1", "ppfd_2", "ppfd_3"]
    if not all(c in df.columns for c in triggers):
        return
    df_long = df[["segment_id", "ratio", "total_pwm", *triggers]].copy()
    for c in triggers:
        df_long[c] = pd.to_numeric(df_long[c], errors="coerce")
    df_long["mean"] = df_long[triggers].mean(axis=1)
    df_long["std"] = df_long[triggers].std(axis=1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(df_long["segment_id"], df_long["mean"], yerr=df_long["std"],
                fmt="o", capsize=3)
    for _, r in df_long.iterrows():
        ax.annotate(f"{r['ratio']}\nT={int(r['total_pwm'])}",
                    (r["segment_id"], r["mean"]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=6)
    ax.set_xlabel("segment_id")
    ax.set_ylabel("PPFD (umol/m²/s)")
    ax.set_title("Per-segment PPFD: mean ± std across 3 triggers")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    args = p.parse_args()
    if not args.run_dir.is_dir():
        print(f"Not a dir: {args.run_dir}", file=sys.stderr)
        return 2
    df = load(args.run_dir)
    if df.empty:
        print("No 'normal' segments found in segment_table.csv", file=sys.stderr)
        return 1
    print(f"Loaded {len(df)} normal segments from {args.run_dir.name}")
    out_total = args.run_dir / "ppfd_vs_total.png"
    out_heat = args.run_dir / "ppfd_heatmap.png"
    out_trig = args.run_dir / "ppfd_per_trigger.png"
    plot_ppfd_vs_total(df, out_total)
    plot_heatmap(df, out_heat)
    plot_trigger_spread(df, out_trig)
    print(f"Wrote: {out_total.name}, {out_heat.name}, {out_trig.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
