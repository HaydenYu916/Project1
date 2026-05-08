#!/usr/bin/env python3
"""Build a SPtoPPFD training CSV from a Govee calibration run.

Joins:
  - Spectrometer-labeled segments from segment_table.csv (PPFD_spec_mean)
  - Riotee 8-channel sensor rows from riotee_data_all.csv (sp_415..sp_680 etc)

For each segment, takes the Riotee rows whose `timestamp` falls inside
the segment's [steady_start_master_time, steady_end_master_time] window
and averages the per-channel readings. Writes a CSV with the same
columns as Train_data-h4_pC.csv so train_h4_no_xyz.py can consume it
without changes.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from statistics import mean


SP_COLS = ["sp_415", "sp_445", "sp_480", "sp_515",
           "sp_555", "sp_590", "sp_630", "sp_680"]
EXTRA_MEAN_COLS = ["temperature", "humidity", "a1_raw", "vcap_raw",
                   "co2_ppm", "sp_clear", "sp_nir",
                   "spectral_gain", "sleep_time"]
TRAINING_FRONT_COLS = (
    ["ID"]
    + [f"{c}_mean" for c in SP_COLS]
    + ["x_mm", "y_mm", "z_mm", "PPFD_spec_mean"]
)
EXTRA_COLS = [
    "segment_id", "pwm_r", "pwm_b", "rb_ratio_pwm",
    "ppfd_red_mean", "ppfd_blue_mean",
    "row_count", "start_timestamp", "end_timestamp",
    "device_id", "update_type",
    *(f"{c}_mean" for c in EXTRA_MEAN_COLS),
    "spec_sum", "spec_per_ppfd", "clean_status", "clean_reason",
]
OUTPUT_COLS = TRAINING_FRONT_COLS + EXTRA_COLS


def parse_iso(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    # segment_table uses ISO with T; riotee CSV uses space — accept both.
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        try:
            return datetime.fromisoformat(s.replace(" ", "T"))
        except ValueError:
            return None


def to_float(s) -> float | None:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v


def fmt(v: float | None) -> str:
    if v is None:
        return ""
    return f"{v:.10f}".rstrip("0").rstrip(".") or "0"


RIOTEE_FIELDNAMES = [
    "id", "timestamp", "device_id", "update_type",
    "temperature", "humidity", "a1_raw", "vcap_raw",
    "co2_ppm", "co2_state",
    "sp_415", "sp_445", "sp_480", "sp_515",
    "sp_555", "sp_590", "sp_630", "sp_680",
    "sp_clear", "sp_nir",
    "spectral_gain", "sleep_time",
]


def load_riotee_rows(path: Path, device_filter: str | None) -> list[dict]:
    """Return rows that have non-zero spectral data, parsed timestamp, optional device filter.

    riotee_data_all.csv is appended-only with NO header line — only `# Start @...`
    and `# Stop @...` comment markers — so we supply fieldnames explicitly.
    """
    out: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(
            (ln for ln in f if ln and not ln.startswith("#")),
            fieldnames=RIOTEE_FIELDNAMES,
        )
        for r in reader:
            if device_filter and r.get("device_id") != device_filter:
                continue
            if r.get("update_type") != "COMPACT":
                continue
            ts = parse_iso(r.get("timestamp", ""))
            if ts is None:
                continue
            # Keep rows with all-zero spectral channels: under very dim
            # illumination (e.g. total_pwm=10) AS7341 legitimately reads 0
            # across all bands while the node still reports temp/vcap. These
            # are the most informative low-light training samples — dropping
            # them would bias the model away from the dim regime.
            # Skip only rows that look fully empty (no env payload either),
            # which indicates a malformed line rather than a real reading.
            if to_float(r.get("temperature")) is None \
               and to_float(r.get("vcap_raw")) is None:
                continue
            r["_ts"] = ts
            out.append(r)
    return out


def aggregate_segment(seg_id: int, seg_meta: dict, rows: list[dict]) -> dict | None:
    if not rows:
        return None
    rec: dict = {col: "" for col in OUTPUT_COLS}
    rec["ID"] = f"govee-{seg_id:02d}"
    rec["segment_id"] = str(seg_id)
    rec["pwm_r"] = seg_meta["pwm_r"]
    rec["pwm_b"] = seg_meta["pwm_b"]
    rec["rb_ratio_pwm"] = seg_meta["rb_ratio_pwm"]
    rec["PPFD_spec_mean"] = seg_meta["PPFD_spec_mean"]

    # Synthesize ppfd_red/blue from the spectrometer label split by the
    # nominal pwm ratio. The training feature_columns don't actually use
    # these, but the Train_data-h4_pC.csv schema includes them, so we keep
    # the column shape compatible.
    p = to_float(seg_meta["PPFD_spec_mean"]) or 0.0
    pr = to_float(seg_meta["pwm_r"]) or 0.0
    pb = to_float(seg_meta["pwm_b"]) or 0.0
    tot = pr + pb
    if tot > 0:
        rec["ppfd_red_mean"] = fmt(p * pr / tot)
        rec["ppfd_blue_mean"] = fmt(p * pb / tot)
    else:
        rec["ppfd_red_mean"] = "0"
        rec["ppfd_blue_mean"] = "0"

    rec["row_count"] = str(len(rows))
    rec["start_timestamp"] = rows[0]["timestamp"]
    rec["end_timestamp"] = rows[-1]["timestamp"]
    rec["device_id"] = rows[0]["device_id"]
    rec["update_type"] = rows[0]["update_type"]

    for c in SP_COLS:
        vals = [to_float(r.get(c)) for r in rows]
        rec[f"{c}_mean"] = fmt(mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None)
    for c in EXTRA_MEAN_COLS:
        vals = [to_float(r.get(c)) for r in rows]
        rec[f"{c}_mean"] = fmt(mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None)

    spec_sum = sum(to_float(rec[f"{c}_mean"]) or 0.0 for c in SP_COLS)
    rec["spec_sum"] = fmt(spec_sum)
    rec["spec_per_ppfd"] = fmt(spec_sum / p) if p > 0 else ""
    rec["clean_status"] = "keep"
    rec["clean_reason"] = ""
    spp = to_float(rec["spec_per_ppfd"])
    if spp is not None and spp < 10:
        rec["clean_status"] = "reject"
        rec["clean_reason"] = "spectral_sum_too_low_for_ppfd"
    return rec


def main() -> int:
    here = Path(__file__).resolve().parent
    base = here.parent  # Hao_SPtoPPFD_training
    project_root = base.parents[3]  # /home/pi/Desktop/Project1

    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir", type=Path,
        default=project_root / "Collect_Sp_PPFD_Govee" / "outputs"
                              / "20260426_184218_govee_calibration_full",
        help="Govee calibration run directory containing segment_table.csv",
    )
    p.add_argument(
        "--riotee-csv", type=Path,
        default=project_root / "Tool" / "Sensor_riotee_server"
                              / "logs" / "riotee_data_all.csv",
        help="Riotee data CSV (riotee_data_all.csv)",
    )
    p.add_argument("--device-id", default="T6ncwg==",
                   help="Riotee device_id to use (default: T6ncwg== — chamber sensor)")
    p.add_argument(
        "--out", type=Path,
        default=base / "data" / "Train_data-govee_calibration.csv",
    )
    p.add_argument(
        "--out-rejected", type=Path,
        default=base / "data" / "Train_data-govee_calibration_rejected.csv",
    )
    p.add_argument(
        "--out-all", type=Path,
        default=base / "data" / "Train_data-govee_calibration_all_segments.csv",
    )
    args = p.parse_args()

    seg_csv = args.run_dir / "segment_table.csv"
    if not seg_csv.is_file():
        raise SystemExit(f"missing {seg_csv}")
    if not args.riotee_csv.is_file():
        raise SystemExit(f"missing {args.riotee_csv}")

    riotee_rows = load_riotee_rows(args.riotee_csv, args.device_id)
    print(f"Loaded {len(riotee_rows)} non-zero Riotee rows for device={args.device_id!r}")

    records: list[dict] = []
    with seg_csv.open() as f:
        for r in csv.DictReader(f):
            if r.get("segment_type") != "normal":
                continue
            t0 = parse_iso(r.get("steady_start_master_time", ""))
            t1 = parse_iso(r.get("steady_end_master_time", ""))
            if t0 is None or t1 is None:
                continue
            seg_id = int(r["segment_id"])
            window_rows = [rr for rr in riotee_rows if t0 <= rr["_ts"] <= t1]
            rec = aggregate_segment(
                seg_id,
                {
                    "pwm_r": r["pwm_r"], "pwm_b": r["pwm_b"],
                    "rb_ratio_pwm": r["rb_ratio_pwm"],
                    "PPFD_spec_mean": r["PPFD_spec_mean"],
                },
                window_rows,
            )
            if rec is not None:
                records.append(rec)
            else:
                print(f"  segment {seg_id}: 0 Riotee rows in [{t0}, {t1}] — skipped")

    keep = [r for r in records if r["clean_status"] == "keep"]
    rej = [r for r in records if r["clean_status"] == "reject"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in [(args.out_all, records), (args.out, keep), (args.out_rejected, rej)]:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
            w.writeheader()
            w.writerows(rows)

    print(f"all     {len(records):3d} rows -> {args.out_all}")
    print(f"keep    {len(keep):3d} rows -> {args.out}")
    print(f"reject  {len(rej):3d} rows -> {args.out_rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
