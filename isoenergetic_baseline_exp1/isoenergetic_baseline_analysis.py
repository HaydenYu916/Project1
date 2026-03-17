#!/usr/bin/env python3
"""Archive copy of the iso-energetic baseline analysis for exp1."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd


ARCHIVE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARCHIVE_DIR.parent
SRC_DIR = PROJECT_ROOT / "LED_MPPI_Controller" / "src"
SOLAR_VOL_CSV = PROJECT_ROOT / "LED_MPPI_Controller" / "data" / "Solar_Vol_clean.csv"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from led import PWMtoPowerModel  # noqa: E402
from mppi_v2 import LEDPlant  # noqa: E402


def safe_div(numerator: float, denominator: float) -> float:
    denom = float(denominator)
    if abs(denom) <= 1e-12:
        return float("nan")
    return float(numerator) / denom


def safe_pct_change(reference: float, candidate: float) -> float:
    ref = float(reference)
    if abs(ref) <= 1e-12:
        return float("nan")
    return 100.0 * (float(candidate) - ref) / ref


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct a whole-experiment constant-light baseline from a control log."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=ARCHIVE_DIR / "mppi_v2_control_log_exp1.csv",
        help="CSV log path.",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "control_log", "sensor_ppfd"),
        default="auto",
        help="Input schema. Auto-detect by column names by default.",
    )
    parser.add_argument(
        "--sensor-ppfd-column",
        type=str,
        default=None,
        help="PPFD column for sensor_ppfd input. Auto-selects ppfd_adjusted, then ppfd.",
    )
    parser.add_argument(
        "--time-window-start",
        type=str,
        default=None,
        help="Optional daily window start in HH:MM or HH:MM:SS.",
    )
    parser.add_argument(
        "--time-window-end",
        type=str,
        default=None,
        help="Optional daily window end in HH:MM or HH:MM:SS. Supports overnight windows.",
    )
    parser.add_argument(
        "--evaluation-window-start",
        type=str,
        default=None,
        help="Optional daily window used for baseline construction and metric integration.",
    )
    parser.add_argument(
        "--evaluation-window-end",
        type=str,
        default=None,
        help="Optional daily window used for baseline construction and metric integration.",
    )
    parser.add_argument(
        "--nominal-dt",
        type=float,
        default=900.0,
        help="Nominal control interval in seconds. Default: 900.",
    )
    parser.add_argument(
        "--dt-mode",
        choices=("nominal", "timestamp"),
        default="nominal",
        help="Use a fixed interval per row, or timestamp differences.",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=("mean_ppfd", "empirical_power", "mean_solar_vol", "target_ppfd"),
        default="mean_ppfd",
        help="How to choose the constant baseline setting.",
    )
    parser.add_argument(
        "--target-ppfd",
        type=float,
        default=None,
        help="Constant PPFD target used when --baseline-mode target_ppfd.",
    )
    parser.add_argument(
        "--closed-loop",
        action="store_true",
        help="Also run a continuous carry-over thermal replay for diagnostics.",
    )
    parser.add_argument(
        "--skip-closed-loop",
        action="store_true",
        help="Deprecated. Anchored thermal replay is now the default primary result.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON summary output path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional per-step CSV output path.",
    )
    return parser.parse_args()


def infer_input_format(columns: list[str]) -> str:
    column_set = set(columns)
    if {"timestamp", "solar_vol_cmd", "r_pwm", "b_pwm", "pred_power", "input_temp", "co2_ppm"}.issubset(
        column_set
    ):
        return "control_log"
    if {"timestamp", "temperature", "co2"}.issubset(column_set) and (
        "ppfd_adjusted" in column_set or "ppfd" in column_set
    ):
        return "sensor_ppfd"
    raise ValueError(
        "Could not infer input format from columns. "
        "Expected control_log columns or sensor_ppfd columns."
    )


def load_control_log(df: pd.DataFrame, log_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    if "success" in df.columns:
        mask = df["success"].astype(str).str.lower() == "true"
        df = df.loc[mask].copy()
    else:
        df = df.copy()

    if df.empty:
        raise ValueError(f"No usable rows found in {log_path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    required = [
        "timestamp",
        "solar_vol_cmd",
        "r_pwm",
        "b_pwm",
        "pred_power",
        "input_temp",
        "co2_ppm",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = ["solar_vol_cmd", "r_pwm", "b_pwm", "pred_power", "input_temp", "co2_ppm"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["solar_vol_cmd", "pred_power", "input_temp", "co2_ppm"]).copy()
    if df.empty:
        raise ValueError("All usable rows were dropped after numeric conversion.")
    return df.reset_index(drop=True), {"input_format": "control_log"}


def resolve_sensor_ppfd_column(columns: list[str], requested: str | None) -> str:
    if requested is not None:
        if requested not in columns:
            raise ValueError(f"Requested PPFD column not found: {requested}")
        return requested
    for candidate in ("ppfd_adjusted", "ppfd"):
        if candidate in columns:
            return candidate
    raise ValueError("sensor_ppfd input requires either ppfd_adjusted or ppfd column")


def load_sensor_ppfd_log(
    df: pd.DataFrame,
    log_path: Path,
    sensor_ppfd_column: str | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    df = df.copy()
    ppfd_col = resolve_sensor_ppfd_column(df.columns.tolist(), sensor_ppfd_column)

    required = ["timestamp", "temperature", "co2", ppfd_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for sensor_ppfd input: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["input_temp"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["co2_ppm"] = pd.to_numeric(df["co2"], errors="coerce")
    df["ppfd_dynamic"] = pd.to_numeric(df[ppfd_col], errors="coerce")
    if "pn" in df.columns:
        df["pred_pn"] = pd.to_numeric(df["pn"], errors="coerce")

    df = df.dropna(subset=["timestamp", "input_temp", "co2_ppm", "ppfd_dynamic"]).copy()
    if df.empty:
        raise ValueError(f"No usable rows found in {log_path}")

    return df.reset_index(drop=True), {
        "input_format": "sensor_ppfd",
        "sensor_ppfd_column": ppfd_col,
    }


def load_log(
    log_path: Path,
    input_format: str = "auto",
    sensor_ppfd_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(log_path, comment="#")
    resolved_format = infer_input_format(df.columns.tolist()) if input_format == "auto" else input_format
    if resolved_format == "control_log":
        return load_control_log(df, log_path)
    if resolved_format == "sensor_ppfd":
        return load_sensor_ppfd_log(df, log_path, sensor_ppfd_column)
    raise ValueError(f"Unsupported input format: {resolved_format}")


def parse_time_of_day(value: str) -> time:
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid time value: {value}")
    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2]) if len(parts) == 3 else 0
    return time(hour=hour, minute=minute, second=second)


def apply_time_window(
    df: pd.DataFrame,
    start_value: str | None,
    end_value: str | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if start_value is None and end_value is None:
        return df, {
            "time_window_start": None,
            "time_window_end": None,
            "rows_before_time_window": int(len(df)),
            "rows_after_time_window": int(len(df)),
        }
    if start_value is None or end_value is None:
        raise ValueError("Both --time-window-start and --time-window-end are required together")

    start_t = parse_time_of_day(start_value)
    end_t = parse_time_of_day(end_value)
    times = df["timestamp"].dt.time
    if start_t <= end_t:
        mask = (times >= start_t) & (times < end_t)
    else:
        mask = (times >= start_t) | (times < end_t)

    filtered = df.loc[mask].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("Time window removed all rows")
    return filtered, {
        "time_window_start": start_value,
        "time_window_end": end_value,
        "rows_before_time_window": int(len(df)),
        "rows_after_time_window": int(len(filtered)),
    }


def build_time_window_mask(
    df: pd.DataFrame,
    start_value: str | None,
    end_value: str | None,
) -> tuple[np.ndarray, dict[str, object]]:
    if start_value is None and end_value is None:
        mask = np.ones(len(df), dtype=bool)
        return mask, {
            "evaluation_window_start": None,
            "evaluation_window_end": None,
            "evaluation_rows": int(mask.sum()),
        }
    if start_value is None or end_value is None:
        raise ValueError("Both --evaluation-window-start and --evaluation-window-end are required together")

    start_t = parse_time_of_day(start_value)
    end_t = parse_time_of_day(end_value)
    times = df["timestamp"].dt.time
    if start_t <= end_t:
        mask = ((times >= start_t) & (times < end_t)).to_numpy(dtype=bool)
    else:
        mask = ((times >= start_t) | (times < end_t)).to_numpy(dtype=bool)
    if not np.any(mask):
        raise ValueError("Evaluation window removed all rows")
    return mask, {
        "evaluation_window_start": start_value,
        "evaluation_window_end": end_value,
        "evaluation_rows": int(mask.sum()),
    }


def assign_dt(df: pd.DataFrame, dt_mode: str, nominal_dt: float) -> pd.DataFrame:
    out = df.copy()
    if dt_mode == "nominal":
        out["dt_s"] = float(nominal_dt)
        return out

    diffs = out["timestamp"].shift(-1) - out["timestamp"]
    dt_seconds = diffs.dt.total_seconds()
    out["dt_s"] = dt_seconds.fillna(float(nominal_dt))
    out["dt_s"] = out["dt_s"].clip(lower=0.0)
    return out


def init_plant() -> LEDPlant:
    calib_csv = PROJECT_ROOT / "LED_MPPI_Controller" / "data" / "calib_data.csv"
    model_dir = PROJECT_ROOT / "LED_MPPI_Controller" / "Thermal" / "exported_models"
    power_model = PWMtoPowerModel(include_intercept=True).fit(str(calib_csv))
    with contextlib.redirect_stdout(io.StringIO()):
        return LEDPlant(
            max_solar_vol=2.0,
            use_solar_vol_model=True,
            power_model=power_model,
            model_dir=str(model_dir),
        )


def load_solar_vol_ppfd_lookup(r_b_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    lookup = pd.read_csv(SOLAR_VOL_CSV).copy()
    for col in ("R:B", "Solar_Vol", "PPFD"):
        lookup[col] = pd.to_numeric(lookup[col], errors="coerce")
    lookup = lookup.dropna(subset=["R:B", "Solar_Vol", "PPFD"])
    lookup = lookup.loc[np.isclose(lookup["R:B"], float(r_b_ratio), atol=0.01)].copy()
    if lookup.empty:
        raise ValueError(f"No Solar_Vol/PPFD lookup rows found for R:B={r_b_ratio}")
    lookup = lookup.sort_values("Solar_Vol")
    return (
        lookup["Solar_Vol"].to_numpy(dtype=float),
        lookup["PPFD"].to_numpy(dtype=float),
    )


def solar_vol_to_ppfd(solar_vol: float, solar_vals: np.ndarray, ppfd_vals: np.ndarray) -> float:
    return float(np.interp(float(solar_vol), solar_vals, ppfd_vals))


def ppfd_to_solar_vol(ppfd: float, solar_vals: np.ndarray, ppfd_vals: np.ndarray) -> float:
    ppfd_mono = np.maximum.accumulate(ppfd_vals)
    return float(np.interp(float(ppfd), ppfd_mono, solar_vals))


def predict_power_from_solar_vol(plant: LEDPlant, solar_vol: float) -> tuple[float, float, float]:
    r_pwm, b_pwm = plant._solar_vol_to_pwm(float(solar_vol))  # noqa: SLF001
    power_key = plant._get_power_model_key(float(plant.r_b_ratio))  # noqa: SLF001
    total_pwm = float(r_pwm) + float(b_pwm)
    power = float(plant.power_model.predict(total_pwm=total_pwm, key=power_key))
    return float(r_pwm), float(b_pwm), power


def attach_dynamic_lighting_state(
    df: pd.DataFrame,
    plant: LEDPlant,
    solar_vals: np.ndarray,
    ppfd_vals: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    out = df.copy()
    if "solar_vol_cmd" in out.columns:
        out["ppfd_dynamic"] = [
            solar_vol_to_ppfd(float(sv), solar_vals, ppfd_vals) for sv in out["solar_vol_cmd"]
        ]
        return out, {
            "ppfd_lookup_min": float(np.min(ppfd_vals)),
            "ppfd_lookup_max": float(np.max(ppfd_vals)),
            "ppfd_clip_count": 0,
            "ppfd_clip_fraction_pct": 0.0,
        }

    if "ppfd_dynamic" not in out.columns:
        raise ValueError("Missing ppfd_dynamic column for sensor_ppfd input")

    ppfd_raw = pd.to_numeric(out["ppfd_dynamic"], errors="coerce").to_numpy(dtype=float)
    ppfd_min = float(np.min(ppfd_vals))
    ppfd_max = float(np.max(ppfd_vals))
    ppfd_clipped = np.clip(ppfd_raw, ppfd_min, ppfd_max)

    solar_vol_cmd = np.asarray(
        [ppfd_to_solar_vol(float(ppfd), solar_vals, ppfd_vals) for ppfd in ppfd_clipped],
        dtype=float,
    )
    rbp = [predict_power_from_solar_vol(plant, float(sv)) for sv in solar_vol_cmd]
    r_pwm = np.asarray([row[0] for row in rbp], dtype=float)
    b_pwm = np.asarray([row[1] for row in rbp], dtype=float)
    pred_power = np.asarray([row[2] for row in rbp], dtype=float)
    ppfd_model = np.asarray(
        [solar_vol_to_ppfd(float(sv), solar_vals, ppfd_vals) for sv in solar_vol_cmd],
        dtype=float,
    )

    out["ppfd_dynamic_measured"] = ppfd_raw
    out["ppfd_dynamic_clipped"] = ppfd_clipped
    out["ppfd_dynamic_model"] = ppfd_model
    out["solar_vol_cmd"] = solar_vol_cmd
    out["r_pwm"] = r_pwm
    out["b_pwm"] = b_pwm
    out["pred_power"] = pred_power

    clip_count = int(np.sum(~np.isclose(ppfd_raw, ppfd_clipped, equal_nan=True)))
    return out, {
        "ppfd_lookup_min": ppfd_min,
        "ppfd_lookup_max": ppfd_max,
        "ppfd_clip_count": clip_count,
        "ppfd_clip_fraction_pct": 100.0 * clip_count / len(out),
    }


def choose_constant_setting(
    df: pd.DataFrame,
    plant: LEDPlant,
    baseline_mode: str,
    solar_vals: np.ndarray,
    ppfd_vals: np.ndarray,
    target_ppfd: float | None = None,
    evaluation_mask: np.ndarray | None = None,
) -> dict[str, float]:
    mask = (
        np.asarray(evaluation_mask, dtype=bool)
        if evaluation_mask is not None
        else np.ones(len(df), dtype=bool)
    )
    view = df.loc[mask].copy()
    dt_weights = view["dt_s"].to_numpy(dtype=float)

    if baseline_mode == "mean_ppfd":
        target_mean_ppfd = float(np.average(view["ppfd_dynamic"], weights=dt_weights))
        sv_const = ppfd_to_solar_vol(target_mean_ppfd, solar_vals, ppfd_vals)
        r_const, b_const, power_const = predict_power_from_solar_vol(plant, sv_const)
        return {
            "solar_vol_const": sv_const,
            "r_pwm_const": r_const,
            "b_pwm_const": b_const,
            "power_const_w": power_const,
            "ppfd_const": target_mean_ppfd,
            "comparison_domain": "ppfd",
            "method": baseline_mode,
        }

    if baseline_mode == "mean_solar_vol":
        sv_const = float(np.average(view["solar_vol_cmd"], weights=dt_weights))
        r_const, b_const, power_const = predict_power_from_solar_vol(plant, sv_const)
        return {
            "solar_vol_const": sv_const,
            "r_pwm_const": r_const,
            "b_pwm_const": b_const,
            "power_const_w": power_const,
            "ppfd_const": solar_vol_to_ppfd(sv_const, solar_vals, ppfd_vals),
            "comparison_domain": "solar_vol",
            "method": baseline_mode,
        }

    if baseline_mode == "target_ppfd":
        if target_ppfd is None:
            raise ValueError("--target-ppfd is required when --baseline-mode target_ppfd")
        sv_const = ppfd_to_solar_vol(float(target_ppfd), solar_vals, ppfd_vals)
        r_const, b_const, power_const = predict_power_from_solar_vol(plant, sv_const)
        return {
            "solar_vol_const": sv_const,
            "r_pwm_const": r_const,
            "b_pwm_const": b_const,
            "power_const_w": power_const,
            "ppfd_const": float(target_ppfd),
            "comparison_domain": "ppfd",
            "method": baseline_mode,
        }

    agg = (
        view.groupby("solar_vol_cmd", as_index=False)
        .agg(
            pred_power=("pred_power", "mean"),
            r_pwm=("r_pwm", "mean"),
            b_pwm=("b_pwm", "mean"),
        )
        .sort_values("solar_vol_cmd")
    )
    sv_vals = agg["solar_vol_cmd"].to_numpy(dtype=float)
    power_vals = agg["pred_power"].to_numpy(dtype=float)
    r_vals = agg["r_pwm"].to_numpy(dtype=float)
    b_vals = agg["b_pwm"].to_numpy(dtype=float)

    power_mono = np.maximum.accumulate(power_vals)
    target_mean_power = float(view["pred_power"].mean())
    sv_const = float(np.interp(target_mean_power, power_mono, sv_vals))
    r_const = float(np.interp(sv_const, sv_vals, r_vals))
    b_const = float(np.interp(sv_const, sv_vals, b_vals))

    return {
        "solar_vol_const": sv_const,
        "r_pwm_const": r_const,
        "b_pwm_const": b_const,
        "power_const_w": target_mean_power,
        "ppfd_const": solar_vol_to_ppfd(sv_const, solar_vals, ppfd_vals),
        "comparison_domain": "power",
        "method": baseline_mode,
    }


def recompute_exogenous_pn(
    df: pd.DataFrame,
    plant: LEDPlant,
    baseline_solar_series: np.ndarray,
) -> pd.DataFrame:
    out = df.copy()
    rb = float(plant.r_b_ratio)
    out["pn_dynamic_exogenous"] = [
        plant.get_photosynthesis_rate(float(sv), float(temp), float(co2), rb)
        for sv, temp, co2 in zip(out["solar_vol_cmd"], out["input_temp"], out["co2_ppm"])
    ]
    out["pn_baseline_exogenous"] = [
        plant.get_photosynthesis_rate(float(sv), float(temp), float(co2), rb)
        for sv, temp, co2 in zip(baseline_solar_series, out["input_temp"], out["co2_ppm"])
    ]
    return out


def assign_replay_segments(df: pd.DataFrame, gap_threshold_s: float = 3600.0) -> pd.DataFrame:
    out = df.copy()
    seg_ids: list[int] = []
    current_seg = 0
    prev_ts = None
    for ts in out["timestamp"]:
        if prev_ts is not None and pd.notna(ts) and pd.notna(prev_ts):
            gap_s = float((ts - prev_ts).total_seconds())
            if gap_s > float(gap_threshold_s):
                current_seg += 1
        seg_ids.append(current_seg)
        prev_ts = ts
    out["replay_segment_id"] = seg_ids
    return out


def run_anchored_thermal_replay(
    df: pd.DataFrame,
    plant: LEDPlant,
    solar_vol_const: float,
    nominal_dt: float,
    dynamic_total_energy_wh: float,
    baseline_total_energy_wh: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    dyn_temp = []
    dyn_power = []
    dyn_pn = []
    dyn_r = []
    dyn_b = []
    bas_temp = []
    bas_power = []
    bas_pn = []
    bas_r = []
    bas_b = []

    for solar_vol, init_temp, co2 in zip(
        out["solar_vol_cmd"].to_numpy(dtype=float),
        out["input_temp"].to_numpy(dtype=float),
        out["co2_ppm"].to_numpy(dtype=float),
    ):
        (_sv, t_dyn, p_dyn, pn_dyn, r_dyn, b_dyn) = plant.predict(
            [float(solar_vol)],
            float(init_temp),
            dt=float(nominal_dt),
            co2_sequence=[float(co2)],
        )
        (_sv, t_bas, p_bas, pn_bas, r_bas, b_bas) = plant.predict(
            [float(solar_vol_const)],
            float(init_temp),
            dt=float(nominal_dt),
            co2_sequence=[float(co2)],
        )
        dyn_temp.append(float(t_dyn[0]))
        dyn_power.append(float(p_dyn[0]))
        dyn_pn.append(float(pn_dyn[0]))
        dyn_r.append(float(r_dyn[0]))
        dyn_b.append(float(b_dyn[0]))
        bas_temp.append(float(t_bas[0]))
        bas_power.append(float(p_bas[0]))
        bas_pn.append(float(pn_bas[0]))
        bas_r.append(float(r_bas[0]))
        bas_b.append(float(b_bas[0]))

    out["temp_dynamic_anchored"] = dyn_temp
    out["power_dynamic_anchored"] = dyn_power
    out["pn_dynamic_anchored"] = dyn_pn
    out["r_pwm_dynamic_anchored"] = dyn_r
    out["b_pwm_dynamic_anchored"] = dyn_b
    out["temp_baseline_anchored"] = bas_temp
    out["power_baseline_anchored"] = bas_power
    out["pn_baseline_anchored"] = bas_pn
    out["r_pwm_baseline_anchored"] = bas_r
    out["b_pwm_baseline_anchored"] = bas_b

    dt_weights = out["dt_s"].to_numpy(dtype=float)
    cum_dyn = float(np.sum(np.asarray(dyn_pn, dtype=float) * dt_weights))
    cum_bas = float(np.sum(np.asarray(bas_pn, dtype=float) * dt_weights))
    summary = {
        "cumulative_dynamic_anchored_umol_m2": cum_dyn,
        "cumulative_baseline_anchored_umol_m2": cum_bas,
        "dynamic_anchored_umol_m2_per_wh": safe_div(cum_dyn, dynamic_total_energy_wh),
        "baseline_anchored_umol_m2_per_wh": safe_div(cum_bas, baseline_total_energy_wh),
        "dynamic_minus_baseline_anchored_pct": safe_pct_change(cum_bas, cum_dyn),
        "mean_temp_dynamic_anchored_c": float(np.average(dyn_temp, weights=dt_weights)),
        "mean_temp_baseline_anchored_c": float(np.average(bas_temp, weights=dt_weights)),
        "max_temp_dynamic_anchored_c": float(np.max(dyn_temp)),
        "max_temp_baseline_anchored_c": float(np.max(bas_temp)),
    }
    return out, summary


def run_segmented_thermal_replay(
    df: pd.DataFrame,
    plant: LEDPlant,
    baseline_solar_series: np.ndarray,
    nominal_dt: float,
    dynamic_total_energy_wh: float,
    baseline_total_energy_wh: float,
    summary_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    baseline_solar_series = np.asarray(baseline_solar_series, dtype=float)
    dyn_temp = np.full(len(out), np.nan, dtype=float)
    dyn_power = np.full(len(out), np.nan, dtype=float)
    dyn_pn = np.full(len(out), np.nan, dtype=float)
    dyn_r = np.full(len(out), np.nan, dtype=float)
    dyn_b = np.full(len(out), np.nan, dtype=float)
    bas_temp = np.full(len(out), np.nan, dtype=float)
    bas_power = np.full(len(out), np.nan, dtype=float)
    bas_pn = np.full(len(out), np.nan, dtype=float)
    bas_r = np.full(len(out), np.nan, dtype=float)
    bas_b = np.full(len(out), np.nan, dtype=float)

    segment_starts: list[str] = []
    for seg_id, seg_df in out.groupby("replay_segment_id", sort=True):
        idx = seg_df.index.to_numpy(dtype=int)
        initial_temp = float(seg_df["input_temp"].iloc[0])
        co2_seq = seg_df["co2_ppm"].to_numpy(dtype=float)
        sv_dyn = seg_df["solar_vol_cmd"].to_numpy(dtype=float)
        sv_bas = baseline_solar_series[idx]
        segment_starts.append(str(seg_df["timestamp"].iloc[0]))

        (_sv_dyn, t_dyn, p_dyn, pn_dyn, r_dyn, b_dyn) = plant.predict(
            sv_dyn,
            initial_temp,
            dt=float(nominal_dt),
            co2_sequence=co2_seq,
        )
        (_sv_bas, t_bas, p_bas, pn_bas, r_bas, b_bas) = plant.predict(
            sv_bas,
            initial_temp,
            dt=float(nominal_dt),
            co2_sequence=co2_seq,
        )

        dyn_temp[idx] = np.asarray(t_dyn, dtype=float)
        dyn_power[idx] = np.asarray(p_dyn, dtype=float)
        dyn_pn[idx] = np.asarray(pn_dyn, dtype=float)
        dyn_r[idx] = np.asarray(r_dyn, dtype=float)
        dyn_b[idx] = np.asarray(b_dyn, dtype=float)
        bas_temp[idx] = np.asarray(t_bas, dtype=float)
        bas_power[idx] = np.asarray(p_bas, dtype=float)
        bas_pn[idx] = np.asarray(pn_bas, dtype=float)
        bas_r[idx] = np.asarray(r_bas, dtype=float)
        bas_b[idx] = np.asarray(b_bas, dtype=float)

    out["temp_dynamic_segmented"] = dyn_temp
    out["power_dynamic_segmented"] = dyn_power
    out["pn_dynamic_segmented"] = dyn_pn
    out["r_pwm_dynamic_segmented"] = dyn_r
    out["b_pwm_dynamic_segmented"] = dyn_b
    out["temp_baseline_segmented"] = bas_temp
    out["power_baseline_segmented"] = bas_power
    out["pn_baseline_segmented"] = bas_pn
    out["r_pwm_baseline_segmented"] = bas_r
    out["b_pwm_baseline_segmented"] = bas_b

    mask = (
        np.asarray(summary_mask, dtype=bool)
        if summary_mask is not None
        else np.ones(len(out), dtype=bool)
    )
    dt_weights = out["dt_s"].to_numpy(dtype=float)
    cum_dyn = float(np.sum(dyn_pn[mask] * dt_weights[mask]))
    cum_bas = float(np.sum(bas_pn[mask] * dt_weights[mask]))
    summary = {
        "num_replay_segments": int(out["replay_segment_id"].nunique()),
        "replay_segment_starts": segment_starts,
        "cumulative_dynamic_segmented_umol_m2": cum_dyn,
        "cumulative_baseline_segmented_umol_m2": cum_bas,
        "dynamic_segmented_umol_m2_per_wh": safe_div(cum_dyn, dynamic_total_energy_wh),
        "baseline_segmented_umol_m2_per_wh": safe_div(cum_bas, baseline_total_energy_wh),
        "dynamic_minus_baseline_segmented_pct": safe_pct_change(cum_bas, cum_dyn),
        "mean_temp_dynamic_segmented_c": float(np.average(dyn_temp[mask], weights=dt_weights[mask])),
        "mean_temp_baseline_segmented_c": float(np.average(bas_temp[mask], weights=dt_weights[mask])),
        "max_temp_dynamic_segmented_c": float(np.max(dyn_temp[mask])),
        "max_temp_baseline_segmented_c": float(np.max(bas_temp[mask])),
    }
    return out, summary


def run_closed_loop_replay(
    df: pd.DataFrame,
    plant: LEDPlant,
    solar_vol_const: float,
    nominal_dt: float,
    dynamic_total_energy_wh: float,
    baseline_total_energy_wh: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    co2_seq = df["co2_ppm"].to_numpy(dtype=float)
    sv_dyn = df["solar_vol_cmd"].to_numpy(dtype=float)
    sv_bas = np.full(len(df), float(solar_vol_const), dtype=float)
    initial_temp = float(df["input_temp"].iloc[0])

    (_sv_dyn, t_dyn, p_dyn, pn_dyn, r_dyn, b_dyn) = plant.predict(
        sv_dyn,
        initial_temp,
        dt=float(nominal_dt),
        co2_sequence=co2_seq,
    )
    (_sv_bas, t_bas, p_bas, pn_bas, r_bas, b_bas) = plant.predict(
        sv_bas,
        initial_temp,
        dt=float(nominal_dt),
        co2_sequence=co2_seq,
    )

    out = df.copy()
    out["temp_dynamic_closedloop"] = np.asarray(t_dyn, dtype=float)
    out["power_dynamic_closedloop"] = np.asarray(p_dyn, dtype=float)
    out["pn_dynamic_closedloop"] = np.asarray(pn_dyn, dtype=float)
    out["r_pwm_dynamic_closedloop"] = np.asarray(r_dyn, dtype=float)
    out["b_pwm_dynamic_closedloop"] = np.asarray(b_dyn, dtype=float)
    out["temp_baseline_closedloop"] = np.asarray(t_bas, dtype=float)
    out["power_baseline_closedloop"] = np.asarray(p_bas, dtype=float)
    out["pn_baseline_closedloop"] = np.asarray(pn_bas, dtype=float)
    out["r_pwm_baseline_closedloop"] = np.asarray(r_bas, dtype=float)
    out["b_pwm_baseline_closedloop"] = np.asarray(b_bas, dtype=float)

    cum_dyn = float(np.sum(pn_dyn) * nominal_dt)
    cum_bas = float(np.sum(pn_bas) * nominal_dt)
    summary = {
        "cumulative_dynamic_closedloop_umol_m2": cum_dyn,
        "cumulative_baseline_closedloop_umol_m2": cum_bas,
        "dynamic_closedloop_umol_m2_per_wh": safe_div(cum_dyn, dynamic_total_energy_wh),
        "baseline_closedloop_umol_m2_per_wh": safe_div(cum_bas, baseline_total_energy_wh),
        "dynamic_minus_baseline_closedloop_pct": safe_pct_change(cum_bas, cum_dyn),
        "mean_temp_dynamic_closedloop_c": float(np.average(t_dyn, weights=df["dt_s"])),
        "mean_temp_baseline_closedloop_c": float(np.average(t_bas, weights=df["dt_s"])),
    }
    return out, summary


def compute_summary(
    df: pd.DataFrame,
    baseline: dict[str, float],
    evaluation_mask: np.ndarray | None = None,
) -> dict[str, float]:
    mask = (
        np.asarray(evaluation_mask, dtype=bool)
        if evaluation_mask is not None
        else np.ones(len(df), dtype=bool)
    )
    dt = df["dt_s"].to_numpy(dtype=float)
    pred_power = df["pred_power"].to_numpy(dtype=float)
    total_duration_s = float(np.sum(dt[mask]))
    dynamic_total_energy_j = float(np.sum(pred_power[mask] * dt[mask]))
    dynamic_total_energy_wh = dynamic_total_energy_j / 3600.0
    baseline_total_energy_j = float(baseline["power_const_w"] * total_duration_s)
    baseline_total_energy_wh = baseline_total_energy_j / 3600.0

    cum_dynamic = float(np.sum(df.loc[mask, "pn_dynamic_exogenous"] * df.loc[mask, "dt_s"]))
    cum_baseline = float(np.sum(df.loc[mask, "pn_baseline_exogenous"] * df.loc[mask, "dt_s"]))
    cum_logged = (
        float(np.sum(df.loc[mask, "pred_pn"] * df.loc[mask, "dt_s"]))
        if "pred_pn" in df.columns
        else float("nan")
    )
    dynamic_mean_ppfd = float(np.average(df.loc[mask, "ppfd_dynamic"], weights=df.loc[mask, "dt_s"]))

    summary = {
        "rows": int(len(df)),
        "total_duration_s": total_duration_s,
        "total_energy_j": dynamic_total_energy_j,
        "total_energy_wh": dynamic_total_energy_wh,
        "dynamic_total_energy_j": dynamic_total_energy_j,
        "dynamic_total_energy_wh": dynamic_total_energy_wh,
        "baseline_total_energy_j": baseline_total_energy_j,
        "baseline_total_energy_wh": baseline_total_energy_wh,
        "baseline_minus_dynamic_energy_pct": safe_pct_change(dynamic_total_energy_j, baseline_total_energy_j),
        "dynamic_mean_ppfd": dynamic_mean_ppfd,
        "baseline_ppfd_const": float(baseline["ppfd_const"]),
        "baseline_solar_vol_const": float(baseline["solar_vol_const"]),
        "baseline_r_pwm_const": float(baseline["r_pwm_const"]),
        "baseline_b_pwm_const": float(baseline["b_pwm_const"]),
        "baseline_power_const_w": float(baseline["power_const_w"]),
        "baseline_method": str(baseline["method"]),
        "baseline_comparison_domain": str(baseline["comparison_domain"]),
        "cumulative_logged_pred_pn_umol_m2": cum_logged,
        "cumulative_dynamic_exogenous_umol_m2": cum_dynamic,
        "cumulative_baseline_exogenous_umol_m2": cum_baseline,
        "dynamic_exogenous_umol_m2_per_wh": safe_div(cum_dynamic, dynamic_total_energy_wh),
        "baseline_exogenous_umol_m2_per_wh": safe_div(cum_baseline, baseline_total_energy_wh),
        "dynamic_minus_baseline_exogenous_pct": safe_pct_change(cum_baseline, cum_dynamic),
    }
    summary.update(
        {
            "primary_result_mode": "segmented_thermal_replay",
            "cumulative_dynamic_primary_umol_m2": float("nan"),
            "cumulative_baseline_primary_umol_m2": float("nan"),
            "dynamic_primary_umol_m2_per_wh": float("nan"),
            "baseline_primary_umol_m2_per_wh": float("nan"),
            "dynamic_minus_baseline_primary_pct": float("nan"),
        }
    )
    return summary


def print_summary(summary: dict[str, float], dt_mode: str, baseline_mode: str) -> None:
    print("Constant-Light Baseline Summary")
    print(f"dt_mode: {dt_mode}")
    print(f"baseline_mode: {baseline_mode}")
    print(f"primary_result_mode: {summary['primary_result_mode']}")
    print(f"rows: {summary['rows']}")
    print(f"dynamic_total_energy_j: {summary['dynamic_total_energy_j']:.3f}")
    print(f"dynamic_total_energy_wh: {summary['dynamic_total_energy_wh']:.3f}")
    print(f"baseline_total_energy_j: {summary['baseline_total_energy_j']:.3f}")
    print(f"baseline_total_energy_wh: {summary['baseline_total_energy_wh']:.3f}")
    print(f"baseline_minus_dynamic_energy_pct: {summary['baseline_minus_dynamic_energy_pct']:.6f}")
    print(f"dynamic_mean_ppfd: {summary['dynamic_mean_ppfd']:.6f}")
    print(f"baseline_ppfd_const: {summary['baseline_ppfd_const']:.6f}")
    print(f"baseline_solar_vol_const: {summary['baseline_solar_vol_const']:.6f}")
    print(f"baseline_r_pwm_const: {summary['baseline_r_pwm_const']:.6f}")
    print(f"baseline_b_pwm_const: {summary['baseline_b_pwm_const']:.6f}")
    print(f"baseline_power_const_w: {summary['baseline_power_const_w']:.6f}")
    print(f"num_replay_segments: {summary['num_replay_segments']}")
    print(f"cumulative_dynamic_primary_umol_m2: {summary['cumulative_dynamic_primary_umol_m2']:.3f}")
    print(f"cumulative_baseline_primary_umol_m2: {summary['cumulative_baseline_primary_umol_m2']:.3f}")
    print(f"dynamic_primary_umol_m2_per_wh: {summary['dynamic_primary_umol_m2_per_wh']:.6f}")
    print(f"baseline_primary_umol_m2_per_wh: {summary['baseline_primary_umol_m2_per_wh']:.6f}")
    print(f"dynamic_minus_baseline_primary_pct: {summary['dynamic_minus_baseline_primary_pct']:.6f}")
    print(f"mean_temp_dynamic_segmented_c: {summary['mean_temp_dynamic_segmented_c']:.6f}")
    print(f"mean_temp_baseline_segmented_c: {summary['mean_temp_baseline_segmented_c']:.6f}")
    print(f"max_temp_dynamic_segmented_c: {summary['max_temp_dynamic_segmented_c']:.6f}")
    print(f"max_temp_baseline_segmented_c: {summary['max_temp_baseline_segmented_c']:.6f}")
    print(f"cumulative_logged_pred_pn_umol_m2: {summary['cumulative_logged_pred_pn_umol_m2']:.3f}")
    print(f"cumulative_dynamic_exogenous_umol_m2: {summary['cumulative_dynamic_exogenous_umol_m2']:.3f}")
    print(f"cumulative_baseline_exogenous_umol_m2: {summary['cumulative_baseline_exogenous_umol_m2']:.3f}")
    print(f"dynamic_exogenous_umol_m2_per_wh: {summary['dynamic_exogenous_umol_m2_per_wh']:.6f}")
    print(f"baseline_exogenous_umol_m2_per_wh: {summary['baseline_exogenous_umol_m2_per_wh']:.6f}")
    print(f"dynamic_minus_baseline_exogenous_pct: {summary['dynamic_minus_baseline_exogenous_pct']:.6f}")
    if "cumulative_dynamic_closedloop_umol_m2" in summary:
        print(f"cumulative_dynamic_closedloop_umol_m2: {summary['cumulative_dynamic_closedloop_umol_m2']:.3f}")
        print(f"cumulative_baseline_closedloop_umol_m2: {summary['cumulative_baseline_closedloop_umol_m2']:.3f}")
        print(f"dynamic_closedloop_umol_m2_per_wh: {summary['dynamic_closedloop_umol_m2_per_wh']:.6f}")
        print(f"baseline_closedloop_umol_m2_per_wh: {summary['baseline_closedloop_umol_m2_per_wh']:.6f}")
        print(f"dynamic_minus_baseline_closedloop_pct: {summary['dynamic_minus_baseline_closedloop_pct']:.6f}")


def main() -> None:
    args = parse_args()
    df, input_meta = load_log(
        args.log_path,
        input_format=args.input_format,
        sensor_ppfd_column=args.sensor_ppfd_column,
    )
    df, time_meta = apply_time_window(df, args.time_window_start, args.time_window_end)
    evaluation_mask, evaluation_meta = build_time_window_mask(
        df,
        args.evaluation_window_start,
        args.evaluation_window_end,
    )
    df = assign_dt(df, dt_mode=args.dt_mode, nominal_dt=args.nominal_dt)
    plant = init_plant()
    solar_vals, ppfd_vals = load_solar_vol_ppfd_lookup(float(plant.r_b_ratio))
    df, ppfd_meta = attach_dynamic_lighting_state(df, plant, solar_vals, ppfd_vals)
    df = assign_replay_segments(df)
    baseline = choose_constant_setting(
        df,
        plant,
        args.baseline_mode,
        solar_vals,
        ppfd_vals,
        target_ppfd=args.target_ppfd,
        evaluation_mask=evaluation_mask,
    )
    baseline_solar_series = np.where(
        evaluation_mask,
        float(baseline["solar_vol_const"]),
        0.0,
    )
    df = recompute_exogenous_pn(df, plant, baseline_solar_series=baseline_solar_series)
    df["ppfd_baseline"] = float(baseline["ppfd_const"])
    df["solar_vol_baseline_cmd"] = baseline_solar_series
    summary = compute_summary(df, baseline, evaluation_mask=evaluation_mask)
    summary.update(input_meta)
    summary.update(time_meta)
    summary.update(evaluation_meta)
    summary.update(ppfd_meta)
    df, segmented_summary = run_segmented_thermal_replay(
        df,
        plant,
        baseline_solar_series,
        args.nominal_dt,
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

    if args.closed_loop:
        df, closed_loop_summary = run_closed_loop_replay(
            df,
            plant,
            baseline["solar_vol_const"],
            args.nominal_dt,
            summary["dynamic_total_energy_wh"],
            summary["baseline_total_energy_wh"],
        )
        summary.update(closed_loop_summary)

    print_summary(summary, dt_mode=args.dt_mode, baseline_mode=args.baseline_mode)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
