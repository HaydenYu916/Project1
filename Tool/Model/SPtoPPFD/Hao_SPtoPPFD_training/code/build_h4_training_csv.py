#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "data" / "h4_raw.csv"
OUTPUT_CSV = BASE_DIR / "data" / "Train_data-h4_pC.csv"
ALL_SEGMENTS_OUTPUT_CSV = BASE_DIR / "data" / "Train_data-h4_pC_all_segments.csv"
REJECTED_OUTPUT_CSV = BASE_DIR / "data" / "Train_data-h4_pC_rejected.csv"

SEGMENT_RE = re.compile(
    r"segment_id=(?P<segment_id>\d+)\s+"
    r"pwm_r=(?P<pwm_r>[-+]?\d*\.?\d+)\s+"
    r"pwm_b=(?P<pwm_b>[-+]?\d*\.?\d+)\s+"
    r"rb_ratio_pwm=(?P<rb_ratio_pwm>\S+)\s+"
    r"PPFD_spec_mean=(?P<PPFD_spec_mean>[-+]?\d*\.?\d+)\s+"
    r"ppfd_red_mean=(?P<ppfd_red_mean>[-+]?\d*\.?\d+)\s+"
    r"ppfd_blue_mean=(?P<ppfd_blue_mean>[-+]?\d*\.?\d+)"
)

MEAN_SOURCE_COLS = [
    "sp_415",
    "sp_445",
    "sp_480",
    "sp_515",
    "sp_555",
    "sp_590",
    "sp_630",
    "sp_680",
    "temperature",
    "humidity",
    "a1_raw",
    "vcap_raw",
    "co2_ppm",
    "sp_clear",
    "sp_nir",
    "spectral_gain",
    "sleep_time",
]

TRAINING_FRONT_COLS = [
    "ID",
    "sp_415_mean",
    "sp_445_mean",
    "sp_480_mean",
    "sp_515_mean",
    "sp_555_mean",
    "sp_590_mean",
    "sp_630_mean",
    "sp_680_mean",
    "x_mm",
    "y_mm",
    "z_mm",
    "PPFD_spec_mean",
]


def to_float(value: str) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    return float(text)


def mean_or_blank(values: list[float | None]) -> str:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return ""
    return f"{sum(filtered) / len(filtered):.10f}".rstrip("0").rstrip(".")


def finalize_segment(segment_index: int, segment_meta: dict[str, str], rows: list[dict[str, str]]):
    if not segment_meta or not rows:
        return None

    record: dict[str, str] = {
        "ID": f"h4_pC-{segment_index}",
        "x_mm": "",
        "y_mm": "",
        "z_mm": "",
        "PPFD_spec_mean": segment_meta["PPFD_spec_mean"],
        "segment_id": segment_meta["segment_id"],
        "pwm_r": segment_meta["pwm_r"],
        "pwm_b": segment_meta["pwm_b"],
        "rb_ratio_pwm": segment_meta["rb_ratio_pwm"],
        "ppfd_red_mean": segment_meta["ppfd_red_mean"],
        "ppfd_blue_mean": segment_meta["ppfd_blue_mean"],
        "row_count": str(len(rows)),
        "start_timestamp": rows[0]["timestamp"],
        "end_timestamp": rows[-1]["timestamp"],
        "device_id": rows[0]["device_id"],
        "update_type": rows[0]["update_type"],
    }

    for col in MEAN_SOURCE_COLS:
        record[f"{col}_mean"] = mean_or_blank([to_float(row.get(col, "")) for row in rows])

    for spectral in ["sp_415", "sp_445", "sp_480", "sp_515", "sp_555", "sp_590", "sp_630", "sp_680"]:
        record[f"{spectral}_mean"] = mean_or_blank([to_float(row.get(spectral, "")) for row in rows])

    return record


def add_quality_flags(records: list[dict[str, str]]):
    spectral_mean_cols = [
        "sp_415_mean",
        "sp_445_mean",
        "sp_480_mean",
        "sp_515_mean",
        "sp_555_mean",
        "sp_590_mean",
        "sp_630_mean",
        "sp_680_mean",
    ]

    for record in records:
        spec_sum = sum(to_float(record.get(col, "")) or 0.0 for col in spectral_mean_cols)
        ppfd = to_float(record["PPFD_spec_mean"]) or 0.0
        record["spec_sum"] = f"{spec_sum:.10f}".rstrip("0").rstrip(".")
        record["spec_per_ppfd"] = (
            f"{spec_sum / ppfd:.10f}".rstrip("0").rstrip(".") if ppfd > 0 else ""
        )
        record["clean_status"] = "keep"
        record["clean_reason"] = ""

    # Heuristic for obvious label/data mismatch:
    # a few bad segments have extremely small spectral sums for very large PPFD.
    for record in records:
        spec_per_ppfd = to_float(record.get("spec_per_ppfd", ""))
        if spec_per_ppfd is not None and spec_per_ppfd < 10:
            record["clean_status"] = "reject"
            record["clean_reason"] = "spectral_sum_too_low_for_ppfd"

    return records


def main():
    lines = INPUT_CSV.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("Input CSV does not contain enough lines.")

    header = next(line for line in lines if line and not line.startswith("#"))
    fieldnames = next(csv.reader([header]))

    records = []
    pending_rows: list[dict[str, str]] = []
    segment_index = 0
    skipped_unlabeled_tail_rows = 0

    for line in lines[lines.index(header) + 1 :]:
        if not line.strip():
            continue
        if line.startswith("# SEGMENT"):
            match = SEGMENT_RE.search(line)
            if not match:
                raise ValueError(f"Could not parse segment line: {line}")

            # In this log format, each SEGMENT line labels the measurements
            # collected immediately before it, not the rows after it.
            if pending_rows:
                segment_index += 1
                record = finalize_segment(segment_index, match.groupdict(), pending_rows)
                if record is not None:
                    records.append(record)
            pending_rows = []
            continue
        if line.startswith("#"):
            continue

        row = next(csv.DictReader([header, line]))
        pending_rows.append(row)

    skipped_unlabeled_tail_rows = len(pending_rows)

    records = add_quality_flags(records)

    extra_cols = [
        "segment_id",
        "pwm_r",
        "pwm_b",
        "rb_ratio_pwm",
        "ppfd_red_mean",
        "ppfd_blue_mean",
        "row_count",
        "start_timestamp",
        "end_timestamp",
        "device_id",
        "update_type",
        "temperature_mean",
        "humidity_mean",
        "a1_raw_mean",
        "vcap_raw_mean",
        "co2_ppm_mean",
        "sp_clear_mean",
        "sp_nir_mean",
        "spectral_gain_mean",
        "sleep_time_mean",
        "spec_sum",
        "spec_per_ppfd",
        "clean_status",
        "clean_reason",
    ]

    output_cols = TRAINING_FRONT_COLS + extra_cols

    with ALL_SEGMENTS_OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_cols)
        writer.writeheader()
        writer.writerows(records)

    clean_records = [r for r in records if r["clean_status"] == "keep"]
    rejected_records = [r for r in records if r["clean_status"] == "reject"]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_cols)
        writer.writeheader()
        writer.writerows(clean_records)

    with REJECTED_OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_cols)
        writer.writeheader()
        writer.writerows(rejected_records)

    print(f"Saved {len(records)} rows to {ALL_SEGMENTS_OUTPUT_CSV}")
    print(f"Saved {len(clean_records)} cleaned rows to {OUTPUT_CSV}")
    print(f"Saved {len(rejected_records)} rejected rows to {REJECTED_OUTPUT_CSV}")
    print(f"Skipped unlabeled tail rows after last segment: {skipped_unlabeled_tail_rows}")


if __name__ == "__main__":
    main()
