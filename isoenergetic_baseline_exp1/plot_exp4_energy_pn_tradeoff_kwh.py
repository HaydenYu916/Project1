#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.use14corefonts"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans", "Helvetica"]
plt.rcParams["font.serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans", "Helvetica"]
plt.rcParams["font.size"] = 22
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.default"] = "bf"
plt.rcParams["mathtext.fontset"] = "stix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot EXP4 energy-Pn trade-off with energy in kWh.")
    parser.add_argument(
        "--combined-table",
        type=Path,
        default=ROOT / "exp4_combined_comparison_with_observed_and_oracle_baseline_constant_mean_ppfd_mol_units.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "exp4_energy_kwh_pn_int_mol_m2.png",
    )
    return parser.parse_args()


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
    combined["energy_kwh"] = combined["energy_wh"] / 1000.0

    fig, ax = plt.subplots(figsize=(10.5, 6.8))

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

    for _, row in combined.iterrows():
        scenario = row["scenario"]
        x = row["energy_kwh"]
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
        )
        ax.annotate(
            label_map[scenario],
            (x, y),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color=palette[scenario],
        )

    ax.set_title("EXP4 Energy-Pn Trade-off", fontsize=22, fontweight="bold")
    ax.set_xlabel("Energy (kWh)", fontsize=22, fontweight="bold")
    ylabel = ax.set_ylabel(r"Integrated Pn (mol m$\mathbf{^{-2}}$)", fontsize=22, fontweight="bold")
    ylabel.set_fontweight("bold")
    ylabel.set_fontsize(22)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_xlim(left=4.0)
    ax.set_ylim(bottom=6.0)
    ax.tick_params(axis="x", labelsize=20, which="major")
    ax.tick_params(axis="y", labelsize=20, which="major")
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
