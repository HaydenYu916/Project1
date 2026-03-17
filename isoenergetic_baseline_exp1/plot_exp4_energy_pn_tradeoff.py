#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot EXP4 energy-Pn trade-off.")
    parser.add_argument(
        "--combined-table",
        type=Path,
        default=ROOT / "exp4_combined_comparison_with_observed_and_oracle_baseline_constant_mean_ppfd_mol_units.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "exp4_energy_pn_tradeoff_constant_only_fit_origin.png",
    )
    return parser.parse_args()


def fit_curve_through_origin(x_wh: np.ndarray, y_mol_m2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_kwh = x_wh / 1000.0
    design = np.column_stack([x_kwh, x_kwh**2])
    coeffs, *_ = np.linalg.lstsq(design, y_mol_m2, rcond=None)

    x_fit_wh = np.linspace(0.0, x_wh.max() * 1.05, 300)
    x_fit_kwh = x_fit_wh / 1000.0
    y_fit = coeffs[0] * x_fit_kwh + coeffs[1] * (x_fit_kwh**2)
    return x_fit_wh, y_fit


def main() -> None:
    args = parse_args()

    combined = pd.read_csv(args.combined_table)
    combined = combined.loc[
        combined["scenario"].isin(
            [
                "constant_mean_ppfd",
                "constant_450_ppfd",
                "oracle_constant_max_pn",
                "oracle_constant_max_ep",
            ]
        )
    ].copy()

    fig, ax = plt.subplots(figsize=(11, 7.2))

    palette = {
        "constant_mean_ppfd": "#2a9d8f",
        "constant_450_ppfd": "#e76f51",
        "oracle_constant_max_pn": "#d62828",
        "oracle_constant_max_ep": "#f4a261",
    }

    markers = {
        "constant_mean_ppfd": "D",
        "constant_450_ppfd": "^",
        "oracle_constant_max_pn": "*",
        "oracle_constant_max_ep": "P",
    }

    label_map = {
        "constant_mean_ppfd": "Constant mean PPFD",
        "constant_450_ppfd": "Constant 450 PPFD",
        "oracle_constant_max_pn": "Oracle max Pn",
        "oracle_constant_max_ep": "Oracle max EP",
    }

    x_fit_wh, y_fit = fit_curve_through_origin(
        combined["energy_wh"].to_numpy(),
        combined["pn_int_mol_m2"].to_numpy(),
    )
    ax.plot(
        x_fit_wh,
        y_fit,
        color="#264653",
        linewidth=2.2,
        label="Origin-constrained fit",
        zorder=1,
    )

    for _, row in combined.iterrows():
        scenario = row["scenario"]
        x = row["energy_wh"]
        y = row["pn_int_mol_m2"]
        ax.scatter(
            x,
            y,
            s=130 if "oracle" in scenario else 90,
            marker=markers[scenario],
            color=palette[scenario],
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
            label=label_map[scenario],
        )
        ax.annotate(
            label_map[scenario],
            (x, y),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            color=palette[scenario],
        )

    ax.set_title("EXP4 Constant-Baseline Energy-Integrated Pn Trade-off", fontsize=15, weight="bold")
    ax.set_xlabel("Energy (Wh)")
    ax.set_ylabel("Integrated Pn (mol m$^{-2}$)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_handles.append(h)
        uniq_labels.append(l)
    ax.legend(uniq_handles, uniq_labels, loc="upper left", frameon=False, ncol=2)

    ax.text(
        0.99,
        0.02,
        "Curve: quadratic fit constrained to pass through the origin",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#6c757d",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
