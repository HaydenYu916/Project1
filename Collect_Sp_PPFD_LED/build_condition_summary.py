#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build condition_summary.csv from one run directory.

Inputs:
- sensor_timeseries.csv (raw sensor records)
- segment_table.csv (master control + spectrometer labels)

Output:
- condition_summary.csv (one row per segment)
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any


VIS_SPECTRAL_COLS = [
    "sp_415",
    "sp_445",
    "sp_480",
    "sp_515",
    "sp_555",
    "sp_590",
    "sp_630",
    "sp_680",
]

SPECTRAL_COLS = [
    *VIS_SPECTRAL_COLS,
    "sp_clear",
    "sp_nir",
]


@dataclass
class MarkerEvent:
    marker_start_idx: int
    marker_end_idx: int
    marker_end_time: datetime
    window_start_idx: int
    source: str = "marker"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build condition_summary.csv for one run")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory under outputs/, e.g. outputs/20260306_150645",
    )
    parser.add_argument("--sensor-csv", default="", help="Override sensor CSV path")
    parser.add_argument("--segment-table", default="", help="Override segment_table.csv path")
    parser.add_argument("--out", default="", help="Override output CSV path")
    parser.add_argument(
        "--marker-min-ratio",
        type=float,
        default=0.7,
        help="Minimum ratio of expected OFF/BLUE sample count to accept marker",
    )
    return parser.parse_args()


def parse_time(ts: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp: {ts}")


def parse_master_time(ts: str) -> datetime:
    s = str(ts).strip()
    if not s:
        raise ValueError("empty master timestamp")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return parse_time(s)


def safe_float(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_sensor_rows(sensor_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with sensor_csv.open("r", encoding="utf-8", newline="") as f:
        filtered = (line for line in f if line.strip() and not line.lstrip().startswith("#"))
        reader = csv.DictReader(filtered)
        for row in reader:
            try:
                row["_ts"] = parse_time(row["timestamp"])
            except Exception:
                continue
            for col in ["temperature", "spectral_gain", "sleep_time", *SPECTRAL_COLS]:
                row[col] = safe_float(row.get(col))
            row["_sum_vis"] = float(sum((row.get(col) or 0.0) for col in VIS_SPECTRAL_COLS))
            rows.append(row)
    rows.sort(key=lambda r: r["_ts"])
    return rows


def read_segment_rows(seg_csv: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with seg_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["segment_id"] = int(float(row["segment_id"]))
            for col in ["pwm_r", "pwm_b", "total_pwm", "Ts", "T_off", "T_blue", "T_settle", "T_meas"]:
                row[col] = safe_float(row.get(col))
            out.append(row)
    out.sort(key=lambda r: r["segment_id"])
    return out


def percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    x = sorted(vals)
    pos = (len(x) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(x) - 1)
    frac = pos - lo
    return x[lo] * (1 - frac) + x[hi] * frac


def detect_markers(
    sensor_rows: list[dict[str, Any]],
    ts_sec: float,
    t_off: float,
    t_blue: float,
    t_settle: float,
    t_meas: float,
    expected_count: int,
    marker_min_ratio: float,
) -> list[MarkerEvent]:
    sum_vis_vals = [r["_sum_vis"] for r in sensor_rows if isinstance(r.get("_sum_vis"), (int, float))]
    b445_vals = [r["sp_445"] for r in sensor_rows if isinstance(r.get("sp_445"), (int, float))]
    if not sum_vis_vals or not b445_vals:
        return []

    p10_sum = percentile(sum_vis_vals, 0.10)
    p90_sum = percentile(sum_vis_vals, 0.90)
    sum_span = max(1.0, p90_sum - p10_sum)
    off_sum_thresh = p10_sum + max(20.0, 0.05 * sum_span)
    p10_445 = percentile(b445_vals, 0.10)
    p90_445 = percentile(b445_vals, 0.90)
    span_445 = max(1.0, p90_445 - p10_445)
    blue_445_thresh = max(p90_445, p10_445 + max(80.0, 0.20 * span_445))

    n_off_expected = max(1, round(t_off / max(0.1, ts_sec)))
    n_blue_expected = max(1, round(t_blue / max(0.1, ts_sec)))
    n_settle_expected = max(1, round(t_settle / max(0.1, ts_sec))) if t_settle > 0 else 1
    n_off_min = max(1, round(n_off_expected * marker_min_ratio))
    n_off_max = max(n_off_expected + 3, n_off_min)
    expected_stride = max(1, round((t_off + t_blue + t_settle + t_meas) / max(0.1, ts_sec)))
    pre_steady_rows = n_off_expected + n_blue_expected + max(1, n_settle_expected - 1)

    def is_dark_run(i: int) -> bool:
        v = sensor_rows[i].get("_sum_vis")
        return isinstance(v, (int, float)) and v <= off_sum_thresh

    def has_blue_hint(i: int) -> bool:
        b = sensor_rows[i].get("sp_445")
        return (
            isinstance(b, (int, float))
            and b >= blue_445_thresh
        )

    candidates: list[MarkerEvent] = []
    i = 0
    n = len(sensor_rows)
    while i < n:
        if not is_dark_run(i):
            i += 1
            continue
        dark_start = i
        while i < n and is_dark_run(i):
            i += 1
        dark_end = i - 1
        dark_len = i - dark_start
        if dark_len < n_off_min or dark_len > n_off_max:
            continue

        blue_idx = None
        search_end = min(n - 1, dark_end + n_blue_expected + 2)
        for j in range(dark_end + 1, search_end + 1):
            if has_blue_hint(j):
                blue_idx = j
                break
        end_idx = blue_idx if blue_idx is not None else dark_end
        window_start_idx = min(n - 1, dark_start + pre_steady_rows)
        candidates.append(
            MarkerEvent(
                marker_start_idx=dark_start,
                marker_end_idx=end_idx,
                marker_end_time=sensor_rows[end_idx]["_ts"],
                window_start_idx=window_start_idx,
                source="marker" if blue_idx is not None else "dark_run",
            )
        )
        i = max(i, end_idx + 1)

    if not candidates:
        return []

    tolerance = max(2, round(expected_stride * 0.60))
    aligned: list[MarkerEvent] = [candidates[0]]
    next_candidate_idx = 1
    while len(aligned) < expected_count:
        prev = aligned[-1]
        target_start = prev.marker_start_idx + expected_stride
        best_idx = -1
        best_score = float("inf")
        j = next_candidate_idx
        while j < len(candidates):
            cand = candidates[j]
            if cand.marker_start_idx <= prev.marker_start_idx:
                j += 1
                continue
            score = abs(cand.marker_start_idx - target_start)
            if score <= tolerance and score < best_score:
                best_idx = j
                best_score = score
            if cand.marker_start_idx > target_start + tolerance:
                break
            j += 1

        if best_idx >= 0:
            chosen = candidates[best_idx]
            aligned.append(chosen)
            next_candidate_idx = best_idx + 1
            continue

        est_start = min(n - 1, target_start)
        est_end = min(n - 1, est_start + max(0, n_off_expected + n_blue_expected - 1))
        aligned.append(
            MarkerEvent(
                marker_start_idx=est_start,
                marker_end_idx=est_end,
                marker_end_time=sensor_rows[est_end]["_ts"],
                window_start_idx=min(n - 1, est_start + pre_steady_rows),
                source="marker_interpolated",
            )
        )

    return aligned[:expected_count]


def window_rows(
    sensor_rows: list[dict[str, Any]],
    start_ts: datetime,
    end_ts: datetime,
) -> list[dict[str, Any]]:
    selected = [r for r in sensor_rows if start_ts <= r["_ts"] <= end_ts]
    if selected:
        return selected
    # fallback: nearest sample to window center
    if not sensor_rows:
        return []
    center = start_ts + (end_ts - start_ts) / 2
    nearest = min(sensor_rows, key=lambda r: abs((r["_ts"] - center).total_seconds()))
    return [nearest]


def index_window_rows(
    sensor_rows: list[dict[str, Any]],
    start_idx: int,
    sample_count: int,
) -> list[dict[str, Any]]:
    if not sensor_rows:
        return []
    lo = max(0, min(len(sensor_rows) - 1, start_idx))
    hi = max(lo + 1, min(len(sensor_rows), lo + max(1, sample_count)))
    return sensor_rows[lo:hi]


def stable_window_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) < 5:
        return rows
    sums = [r.get("_sum_vis") for r in rows if isinstance(r.get("_sum_vis"), (int, float))]
    if len(sums) < 5:
        return rows
    center = median(sums)
    if center <= 0:
        return rows
    lo = center * 0.50
    hi = center * 1.50
    filtered = [r for r in rows if lo <= float(r.get("_sum_vis") or 0.0) <= hi]
    return filtered if len(filtered) >= max(3, len(rows) // 2) else rows


def mode_or_none(vals: list[float | None]) -> float | None:
    cleaned = [v for v in vals if isinstance(v, (int, float))]
    if not cleaned:
        return None
    c = Counter(cleaned).most_common(1)
    return float(c[0][0]) if c else None


def mean_or_none(vals: list[float | None]) -> float | None:
    cleaned = [v for v in vals if isinstance(v, (int, float))]
    if not cleaned:
        return None
    return float(mean(cleaned))


def ppfd_mean_from_segment_row(seg_row: dict[str, Any]) -> float | None:
    pref = safe_float(seg_row.get("PPFD_spec_mean"))
    if pref is not None:
        return pref
    vals: list[float] = []
    for k, v in seg_row.items():
        if k.startswith("ppfd_") and not k.endswith("_time"):
            idx = k.replace("ppfd_", "")
            if idx.isdigit():
                fv = safe_float(v)
                if fv is not None:
                    vals.append(fv)
    return mean_or_none(vals)


def build_summary(
    sensor_rows: list[dict[str, Any]],
    seg_rows: list[dict[str, Any]],
    markers: list[MarkerEvent],
    ts_sec: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, seg in enumerate(seg_rows):
        marker: MarkerEvent | None = markers[i] if i < len(markers) else None
        t_settle = float(seg["T_settle"] or 0.0)
        t_meas = float(seg["T_meas"] or 0.0)
        n_meas_expected = max(1, round(t_meas / max(0.1, ts_sec)))

        window_source = "marker"
        marker_end_time: datetime | None = None
        w_start: datetime
        w_end: datetime
        rows: list[dict[str, Any]]

        if marker is not None:
            marker_end_time = marker.marker_end_time
            window_source = marker.source
            rows = stable_window_rows(
                index_window_rows(sensor_rows, marker.window_start_idx, n_meas_expected)
            )
            if rows:
                w_start = rows[0]["_ts"]
                w_end = rows[-1]["_ts"]
            else:
                w_start = marker.marker_end_time + timedelta(seconds=t_settle)
                w_end = w_start + timedelta(seconds=t_meas)
        else:
            window_source = "master_time_fallback"
            start_raw = seg.get("steady_start_master_time")
            end_raw = seg.get("steady_end_master_time")
            try:
                w_start = parse_master_time(start_raw)
                w_end = parse_master_time(end_raw)
                if w_end < w_start:
                    w_end = w_start + timedelta(seconds=t_meas)
            except Exception:
                seg_start = parse_master_time(seg.get("segment_start_master_time"))
                marker_end = seg_start + timedelta(seconds=float(seg["T_off"] or 0.0) + float(seg["T_blue"] or 0.0))
                w_start = marker_end + timedelta(seconds=t_settle)
                w_end = w_start + timedelta(seconds=t_meas)
            rows = stable_window_rows(window_rows(sensor_rows, w_start, w_end))

        row = {
            "segment_id": seg["segment_id"],
            "pwm_r": int(float(seg["pwm_r"] or 0)),
            "pwm_b": int(float(seg["pwm_b"] or 0)),
            "rb_ratio_pwm": seg.get("rb_ratio_pwm", ""),
            "total_pwm": int(float(seg["total_pwm"] or 0)),
            "PPFD_spec_mean": ppfd_mean_from_segment_row(seg),
            "sensor_window_count": len(rows),
            "window_source": window_source,
            "matched_marker_index": (i + 1) if marker is not None and window_source != "master_time_fallback" else "",
            "marker_end_sensor_time": (
                marker_end_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                if marker_end_time is not None
                else ""
            ),
            "steady_start_sensor_time": w_start.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "steady_end_sensor_time": w_end.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        }
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    sensor_csv = (
        Path(args.sensor_csv).expanduser().resolve()
        if args.sensor_csv
        else (run_dir / "sensor_timeseries.csv")
    )
    if not sensor_csv.exists():
        fallback = run_dir / "logs" / "sensor_timeseries_all.csv"
        sensor_csv = fallback if fallback.exists() else sensor_csv

    segment_csv = (
        Path(args.segment_table).expanduser().resolve()
        if args.segment_table
        else (run_dir / "segment_table.csv")
    )
    out_csv = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (run_dir / "condition_summary.csv")
    )

    sensor_rows = read_sensor_rows(sensor_csv)
    seg_rows = read_segment_rows(segment_csv)
    if not sensor_rows:
        raise RuntimeError("No sensor rows loaded")
    if not seg_rows:
        raise RuntimeError("No segment rows loaded")

    ts_sec = float(seg_rows[0]["Ts"] or 5.0)
    t_off = float(seg_rows[0]["T_off"] or (2.0 * ts_sec))
    t_blue = float(seg_rows[0]["T_blue"] or (2.0 * ts_sec))
    t_settle = float(seg_rows[0]["T_settle"] or (2.0 * ts_sec))
    t_meas = float(seg_rows[0]["T_meas"] or (10.0 * ts_sec))
    markers = detect_markers(
        sensor_rows=sensor_rows,
        ts_sec=ts_sec,
        t_off=t_off,
        t_blue=t_blue,
        t_settle=t_settle,
        t_meas=t_meas,
        expected_count=len(seg_rows),
        marker_min_ratio=args.marker_min_ratio,
    )

    summary_rows = build_summary(sensor_rows, seg_rows, markers, ts_sec=ts_sec)
    write_csv(out_csv, summary_rows)

    print(f"run_dir: {run_dir}")
    print(f"sensor_rows: {len(sensor_rows)}")
    print(f"segment_rows: {len(seg_rows)}")
    print(f"detected_markers: {len(markers)}")
    print(f"paired_segments: {len(summary_rows)}")
    print(f"output: {out_csv}")

    fallback_count = sum(1 for r in summary_rows if r.get("window_source") == "master_time_fallback")
    if fallback_count > 0:
        print(f"WARNING: {fallback_count} segment(s) used master-time fallback (marker not detected).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
