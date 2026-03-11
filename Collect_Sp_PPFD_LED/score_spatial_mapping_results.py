#!/usr/bin/env python3
"""Score spatial PPFD/spectrum mapping results across heights."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    x = sorted(values)
    pos = (len(x) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return x[lo]
    frac = pos - lo
    return x[lo] * (1 - frac) + x[hi] * frac


def mean_or_none(values: list[float | None]) -> float | None:
    cleaned = [v for v in values if isinstance(v, (int, float))]
    return float(mean(cleaned)) if cleaned else None


def std_or_none(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    return math.sqrt(var)


@dataclass
class PointAggregate:
    height_mm: float
    point_id: str
    ppfd_mean: float | None
    rb_ratio_mean: float | None


def compute_rb_ratio(row: dict[str, Any]) -> float | None:
    direct = safe_float(row.get("rb_sensor"))
    if direct is not None:
        return direct

    b1 = safe_float(row.get("sp_445_mean"))
    b2 = safe_float(row.get("sp_480_mean"))
    r1 = safe_float(row.get("sp_630_mean"))
    r2 = safe_float(row.get("sp_680_mean"))
    den = (b1 or 0.0) + (b2 or 0.0)
    num = (r1 or 0.0) + (r2 or 0.0)
    if den <= 0:
        return None
    return num / den


def read_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    if not rows:
        raise RuntimeError(f"No result rows found in {path}")
    return rows


def aggregate_points(rows: list[dict[str, Any]]) -> list[PointAggregate]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        height_mm = safe_float(row.get("height_mm"))
        point_id = str(row.get("point_id") or "").strip()
        if height_mm is None or not point_id:
            continue
        grouped[(height_mm, point_id)].append(row)

    point_rows: list[PointAggregate] = []
    for (height_mm, point_id), members in sorted(grouped.items()):
        ppfd_vals = [safe_float(r.get("PPFD_spec_mean")) for r in members]
        rb_vals = [compute_rb_ratio(r) for r in members]
        point_rows.append(
            PointAggregate(
                height_mm=height_mm,
                point_id=point_id,
                ppfd_mean=mean_or_none(ppfd_vals),
                rb_ratio_mean=mean_or_none(rb_vals),
            )
        )
    return point_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score spatial mapping results")
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--score-ppfd-weight", type=float, default=0.45)
    parser.add_argument("--score-uniformity-weight", type=float, default=0.35)
    parser.add_argument("--score-spectrum-weight", type=float, default=0.20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_csv = Path(args.results_csv).expanduser().resolve()
    if not results_csv.exists():
        raise FileNotFoundError(f"results csv not found: {results_csv}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else results_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_results(results_csv)
    point_rows = aggregate_points(rows)
    if not point_rows:
        raise RuntimeError("No valid height/point rows found")

    by_height: dict[float, list[PointAggregate]] = defaultdict(list)
    for row in point_rows:
        by_height[row.height_mm].append(row)

    raw_rows: list[dict[str, Any]] = []
    for height_mm, members in sorted(by_height.items()):
        ppfd_vals = [m.ppfd_mean for m in members if m.ppfd_mean is not None]
        rb_vals = [m.rb_ratio_mean for m in members if m.rb_ratio_mean is not None]
        ppfd_mean = mean_or_none(ppfd_vals)
        ppfd_std = std_or_none(ppfd_vals)
        rb_mean = mean_or_none(rb_vals)
        rb_std = std_or_none(rb_vals)
        ppfd_min = min(ppfd_vals) if ppfd_vals else None
        ppfd_max = max(ppfd_vals) if ppfd_vals else None
        p10 = percentile(ppfd_vals, 0.10) if ppfd_vals else None
        ppfd_cv = (ppfd_std / ppfd_mean) if ppfd_std is not None and ppfd_mean not in (None, 0) else None
        rb_cv = (rb_std / rb_mean) if rb_std is not None and rb_mean not in (None, 0) else None
        raw_rows.append(
            {
                "height_mm": height_mm,
                "point_count": len(members),
                "ppfd_mean": ppfd_mean,
                "ppfd_min": ppfd_min,
                "ppfd_max": ppfd_max,
                "ppfd_std": ppfd_std,
                "ppfd_cv": ppfd_cv,
                "uniformity_min_over_mean": (ppfd_min / ppfd_mean) if ppfd_min is not None and ppfd_mean not in (None, 0) else None,
                "uniformity_p10_over_mean": (p10 / ppfd_mean) if p10 is not None and ppfd_mean not in (None, 0) else None,
                "ppfd_sum_points": sum(ppfd_vals) if ppfd_vals else None,
                "rb_ratio_mean": rb_mean,
                "rb_ratio_cv": rb_cv,
            }
        )

    ppfd_means = [r["ppfd_mean"] for r in raw_rows if isinstance(r.get("ppfd_mean"), (int, float))]
    max_ppfd_mean = max(ppfd_means) if ppfd_means else 1.0

    scored_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        ppfd_mean = safe_float(row.get("ppfd_mean")) or 0.0
        uniformity = safe_float(row.get("uniformity_min_over_mean"))
        rb_cv = safe_float(row.get("rb_ratio_cv"))
        ppfd_score = ppfd_mean / max_ppfd_mean if max_ppfd_mean > 0 else 0.0
        uniformity_score = uniformity if uniformity is not None else 0.0
        spectrum_score = 1.0 / (1.0 + max(0.0, rb_cv if rb_cv is not None else 1.0))
        overall_score = (
            (args.score_ppfd_weight * ppfd_score)
            + (args.score_uniformity_weight * uniformity_score)
            + (args.score_spectrum_weight * spectrum_score)
        )
        row = dict(row)
        row["score_ppfd"] = ppfd_score
        row["score_uniformity"] = uniformity_score
        row["score_spectrum"] = spectrum_score
        row["overall_score"] = overall_score
        scored_rows.append(row)

    scored_rows.sort(key=lambda r: (safe_float(r.get("overall_score")) or -1.0), reverse=True)
    for idx, row in enumerate(scored_rows, start=1):
        row["rank"] = idx

    write_csv(out_dir / "height_summary.csv", scored_rows)

    best = scored_rows[0]
    summary = [
        "# Spatial Mapping Recommendation",
        "",
        f"- Source: `{results_csv}`",
        f"- Best height: `{int(float(best['height_mm']))} mm`",
        f"- Overall score: `{best['overall_score']:.4f}`",
        f"- PPFD mean: `{best['ppfd_mean']:.4f}`",
        f"- Uniformity min/mean: `{best['uniformity_min_over_mean']:.4f}`" if best.get("uniformity_min_over_mean") is not None else "- Uniformity min/mean: `N/A`",
        f"- RB ratio CV: `{best['rb_ratio_cv']:.4f}`" if best.get("rb_ratio_cv") is not None else "- RB ratio CV: `N/A`",
        "",
        "## Ranking",
        "",
    ]
    for row in scored_rows:
        summary.append(
            "- {height:.0f} mm: score={score:.4f}, ppfd_mean={ppfd:.4f}, uniformity={uni}, rb_cv={rb}".format(
                height=float(row["height_mm"]),
                score=float(row["overall_score"]),
                ppfd=float(row["ppfd_mean"] or 0.0),
                uni=(
                    f"{float(row['uniformity_min_over_mean']):.4f}"
                    if row.get("uniformity_min_over_mean") is not None
                    else "N/A"
                ),
                rb=(
                    f"{float(row['rb_ratio_cv']):.4f}"
                    if row.get("rb_ratio_cv") is not None
                    else "N/A"
                ),
            )
        )

    (out_dir / "height_recommendation.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"source: {results_csv}")
    print(f"height_count: {len(scored_rows)}")
    print(f"best_height_mm: {best['height_mm']}")
    print(f"summary_csv: {out_dir / 'height_summary.csv'}")
    print(f"recommendation_md: {out_dir / 'height_recommendation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
