"""Retrain EnvtoPN with a smooth model (MLP) so Pn(T) is continuous.

Replaces the HistGBR (piecewise constant → staircase Pn) with an MLP
that interpolates smoothly between the 6 training T levels {18,20,22,24,26,30}
and extrapolates with a hand-crafted exponential decay above 30°C
(since the experiment never went past 30°C, but the GreenLight compare
sim regularly hits 32-40°C).

Output package matches the original schema in best_model_package.joblib so
no server-side code change is needed — server._load_chamber_pn picks it up.

Run from the project root:
    /home/pi/Desktop/riotee-env/bin/python3 \\
        /home/pi/Desktop/Project1/Tool/Model/EnvtoPN/train_smooth.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
ALL_DATA_CSV = "/home/pi/Desktop/Project1/MPPI-Govee/data/ALL_Data.csv"

FEATS = ["T", "CO2", "R:B", "PPFD"]
TARGET = "Pn_avg"

# ---- High-T synthetic tail params ------------------------------------------
# Training data tops out at 30°C. Real plants drop ~50% by 40°C and bottom
# out near 45°C (see the GreenLight Farquhar curve we discussed earlier).
# We augment the training set with synthetic points so the MLP learns a
# smooth decay above 30°C instead of clamping like HistGBR.
SYNTH_T_GRID = np.arange(31.0, 46.0, 1.0)            # 31, 32, ..., 45
SYNTH_T0 = 30.0                                       # decay starts here
SYNTH_TAU = 8.0                                       # τ in exp(-(T-30)/τ)


def main():
    if not os.path.exists(ALL_DATA_CSV):
        print(f"❌ ALL_Data.csv missing: {ALL_DATA_CSV}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(ALL_DATA_CSV)
    df = df[FEATS + [TARGET]].dropna().copy()
    print(f"loaded {len(df)} rows from {ALL_DATA_CSV}")
    for c in FEATS + [TARGET]:
        print(f"  {c:>6s}  min={df[c].min():>7.2f}  max={df[c].max():>7.2f}  unique={df[c].nunique()}")

    # ---- Build synthetic high-T tail ---------------------------------------
    # For each unique (CO2, R:B, PPFD) combo at T=30 (the hottest training T),
    # spawn synthetic rows at T ∈ {31,32,...,45} with Pn = Pn(30) * exp(-(T-30)/8).
    # Pn(30) is taken from the actual measurement at T=30 for that combo
    # (so the tail is anchored to real data, not a fabricated baseline).
    hot = df[df["T"] == 30.0]
    synth_rows = []
    for _, row in hot.iterrows():
        pn30 = float(row[TARGET])
        for T_hi in SYNTH_T_GRID:
            decay = float(np.exp(-(T_hi - SYNTH_T0) / SYNTH_TAU))
            synth_rows.append({
                "T": float(T_hi), "CO2": float(row["CO2"]),
                "R:B": float(row["R:B"]), "PPFD": float(row["PPFD"]),
                TARGET: pn30 * decay,
            })
    synth_df = pd.DataFrame(synth_rows)
    print(f"\nsynth tail: {len(synth_df)} rows over T ∈ [{SYNTH_T_GRID.min()}, {SYNTH_T_GRID.max()}]°C")
    print(f"  decay: Pn(T) = Pn(30) × exp(-(T-{SYNTH_T0:.0f})/{SYNTH_TAU:.0f})")
    print(f"  T=35: ×{np.exp(-(35-30)/8):.3f}   T=40: ×{np.exp(-(40-30)/8):.3f}   T=45: ×{np.exp(-(45-30)/8):.3f}")

    full = pd.concat([df, synth_df], ignore_index=True)
    print(f"\nfull training set: {len(full)} rows ({len(df)} real + {len(synth_df)} synth)")

    X = full[FEATS].values
    y = full[TARGET].values

    # ---- 5-fold CV on the REAL data only, to compare with HistGBR baseline.
    print("\n--- 5-fold CV (real data only, no synth) ---")
    real_X = df[FEATS].values
    real_y = df[TARGET].values
    fold_r2 = []
    for fold, (tr, te) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(real_X)):
        # Train on real-fold-train + ALL synth tail; eval on real-fold-test.
        # This mirrors how the final model is trained.
        X_tr = np.vstack([real_X[tr], synth_df[FEATS].values])
        y_tr = np.hstack([real_y[tr], synth_df[TARGET].values])
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(64, 64),
                activation="tanh",
                solver="adam",
                alpha=1e-3,
                max_iter=3000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=fold,
            )),
        ])
        pipe.fit(X_tr, y_tr)
        y_pr = pipe.predict(real_X[te])
        r2 = r2_score(real_y[te], y_pr)
        fold_r2.append(r2)
        print(f"  fold {fold+1}: R²={r2:.4f}")
    print(f"  mean R² = {np.mean(fold_r2):.4f}  ± {np.std(fold_r2):.4f}")

    # ---- Final fit on full set ---------------------------------------------
    print("\n--- final fit on full set (real + synth) ---")
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="tanh",
            solver="adam",
            alpha=1e-3,
            max_iter=5000,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=50,
            random_state=0,
        )),
    ])
    final_pipe.fit(X, y)

    # In-sample real-data metrics (compare with HistGBR's R²=0.9934)
    y_real_pred = final_pipe.predict(real_X)
    rmse = float(np.sqrt(mean_squared_error(real_y, y_real_pred)))
    mae  = float(mean_absolute_error(real_y, y_real_pred))
    r2   = float(r2_score(real_y, y_real_pred))
    bias = float(np.mean(y_real_pred - real_y))
    print(f"  in-sample on REAL: RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  Bias={bias:+.4f}")

    # ---- Smoothness check: T sweep at fixed (CO2=800, RB=0.83, PPFD=270) ----
    print("\n--- T sweep at PPFD=270, CO2=800, R:B=0.83 ---")
    print("    T     Pn_smooth")
    for T in [20, 22, 24, 25, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]:
        d = pd.DataFrame([{"T": T, "CO2": 800, "R:B": 0.83, "PPFD": 270}])[FEATS]
        pn = float(final_pipe.predict(d.values)[0])
        bar = "█" * int(max(0, pn))
        print(f"  {T:>4}°C  {pn:>7.3f}  {bar}")

    # ---- Save in original package schema -----------------------------------
    pkg = {
        "pipeline":        final_pipe,
        "feature_columns": FEATS,
        "model_name":      "MLP_smooth",
        "model_version":   "v2_synth_tail",
        "target_col":      TARGET,
        "input_csv":       ALL_DATA_CSV,
        "n_rows":          int(len(full)),
        "split_config":    None,
        "normalization":   "StandardScaler in pipeline before model",
        "test_metrics":    {
            "RMSE": rmse, "MAE": mae, "R2": r2, "Bias": bias,
            "CV_R2_mean": float(np.mean(fold_r2)),
            "CV_R2_std":  float(np.std(fold_r2)),
        },
    }
    out_pkg = os.path.join(MODEL_DIR, "best_model_package.joblib")
    out_meta = os.path.join(MODEL_DIR, "best_model_metadata.json")
    joblib.dump(pkg, out_pkg)
    meta = {
        "model_name":      pkg["model_name"],
        "model_version":   pkg["model_version"],
        "feature_columns": FEATS,
        "target_col":      TARGET,
        "input_csv":       ALL_DATA_CSV,
        "n_rows":          int(len(full)),
        "n_real":          int(len(df)),
        "n_synth_tail":    int(len(synth_df)),
        "synth_tail":      {
            "T_grid": SYNTH_T_GRID.tolist(),
            "T0":     SYNTH_T0,
            "tau":    SYNTH_TAU,
            "formula":"Pn(T)=Pn(30)*exp(-(T-T0)/tau)",
        },
        "normalization":   "StandardScaler in pipeline before model",
        "test_metrics":    pkg["test_metrics"],
        "trained_at":      datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n✅ saved {out_pkg}")
    print(f"✅ saved {out_meta}")


if __name__ == "__main__":
    main()
