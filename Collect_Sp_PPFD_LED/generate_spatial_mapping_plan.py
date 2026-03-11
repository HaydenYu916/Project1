#!/usr/bin/env python3
"""Generate a spatial PPFD/spectrum mapping plan for multi-height tests."""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Point:
    point_id: str
    row_index: int
    col_index: int
    x_mm: float
    y_mm: float
    radial_mm: float
    zone: str


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("At least one integer value is required")
    return values


def safe_slug(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch.lower())
        elif ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "plan"


def classify_zone(row: int, col: int, rows: int, cols: int) -> str:
    center_row = (rows - 1) / 2.0
    center_col = (cols - 1) / 2.0
    if abs(row - center_row) < 0.5 and abs(col - center_col) < 0.5:
        return "center"

    is_top_or_bottom = row in {0, rows - 1}
    is_left_or_right = col in {0, cols - 1}
    if rows > 1 and cols > 1 and is_top_or_bottom and is_left_or_right:
        return "corner"
    return "edge"


def build_grid_points(rows: int, cols: int, spacing_x_mm: float, spacing_y_mm: float) -> list[Point]:
    x0 = -((cols - 1) * spacing_x_mm) / 2.0
    y0 = -((rows - 1) * spacing_y_mm) / 2.0
    points: list[Point] = []

    for row in range(rows):
        for col in range(cols):
            x_mm = x0 + (col * spacing_x_mm)
            y_mm = y0 + (row * spacing_y_mm)
            points.append(
                Point(
                    point_id=f"P{row + 1}{col + 1}",
                    row_index=row + 1,
                    col_index=col + 1,
                    x_mm=x_mm,
                    y_mm=y_mm,
                    radial_mm=math.hypot(x_mm, y_mm),
                    zone=classify_zone(row, col, rows, cols),
                )
            )
    return points


def priority_order(points: list[Point]) -> list[Point]:
    zone_rank = {"center": 0, "edge": 1, "corner": 2}
    return sorted(
        points,
        key=lambda p: (
            zone_rank.get(p.zone, 9),
            round(p.radial_mm, 6),
            abs(p.y_mm),
            abs(p.x_mm),
            p.point_id,
        ),
    )


def shuffled_points(
    points: list[Point],
    rng: random.Random,
    randomize_order: str,
    keep_center_first: bool,
) -> list[Point]:
    ordered = priority_order(points)
    if randomize_order == "none":
        return ordered

    if keep_center_first and ordered and ordered[0].zone == "center":
        head = [ordered[0]]
        tail = ordered[1:]
    else:
        head = []
        tail = ordered[:]

    rng.shuffle(tail)
    return head + tail


def csv_write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary_markdown(
    fixture_name: str,
    led_layout: str,
    heights_mm: list[int],
    points: list[Point],
    repeats: int,
    total_pwm: int,
    ratio_r: int,
    ratio_b: int,
    n_spec: int,
    measurement_count: int,
    randomize_order: str,
    run_name_template: str,
) -> str:
    point_ids = ", ".join(p.point_id for p in priority_order(points))
    return f"""# Spatial Mapping Plan

## Objective

Measure spatial PPFD and spectrum consistency for `{fixture_name}` with a `{led_layout}` LED layout.

The goal is to compare multiple heights above the LED plane and identify the best tradeoff between:

1. High PPFD level
2. Good spatial uniformity
3. Stable red/blue spectral ratio across points

## Recommended Design

- Heights: {", ".join(str(v) for v in heights_mm)} mm
- Grid points per height: {len(points)}
- Point order reference: {point_ids}
- Repeats per point: {repeats}
- Total measurements: {measurement_count}
- Target PWM ratio: {ratio_r}:{ratio_b}
- Total PWM: {total_pwm}
- Spectrometer triggers per run: {n_spec}
- Visit ordering: {randomize_order}

## Execution Rules

1. Fix the LED fixture pose. Do not rotate the lamp between heights.
2. Define the origin `(0, 0)` directly under the fixture center.
3. At each height, measure the full point grid at the same sensor orientation.
4. Keep the sensor plane parallel to the LED plane.
5. Use the same PWM condition for all heights when doing spatial comparison.
6. Recommended first sweep: center + edges + corners on each height.
7. If a height looks promising, add one extra repeat to verify repeatability.

## Suggested Acquisition Workflow

1. Prewarm the lamp for 3 minutes.
2. Move to the target height.
3. Visit points in the generated `measurement_plan.csv` order.
4. For each point, run one single-condition acquisition and fill the returned metrics into `measurement_results_template.csv`.

Suggested `run_name` format:

`{run_name_template}`

Suggested command template:

```bash
cd {ROOT_DIR}
python3 run_led_calibration.py \\
  --sensor-mode auto \\
  --ratios "{ratio_r}:{ratio_b}" \\
  --totals "{total_pwm}" \\
  --n-spec {n_spec} \\
  --run-name <replace_with_plan_run_name>
```

## Decision Rule

After data collection, score each height using:

1. `PPFD mean`: higher is better
2. `Uniformity = min(PPFD) / mean(PPFD)`: higher is better
3. `Spectrum stability`: lower red/blue ratio CV across points is better

Use `score_spatial_mapping_results.py` after filling the results CSV.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a spatial PPFD/spectrum mapping plan")
    parser.add_argument("--fixture-name", default="3R1B_horizontal_fixture")
    parser.add_argument("--led-layout", default="3 red + 1 blue, horizontal")
    parser.add_argument("--heights-mm", default="100,150,200,250,300")
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--grid-cols", type=int, default=3)
    parser.add_argument("--spacing-x-mm", type=float, default=50.0)
    parser.add_argument("--spacing-y-mm", type=float, default=50.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--ratio-r", type=int, default=3)
    parser.add_argument("--ratio-b", type=int, default=1)
    parser.add_argument("--total-pwm", type=int, default=100)
    parser.add_argument("--n-spec", type=int, default=3)
    parser.add_argument(
        "--randomize-order",
        choices=["none", "by_height", "global"],
        default="by_height",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-center-first", action="store_true")
    parser.add_argument("--plan-name", default="")
    parser.add_argument("--out-dir", default=str(ROOT_DIR / "plans"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.grid_rows <= 0 or args.grid_cols <= 0:
        raise ValueError("grid rows/cols must be > 0")
    if args.repeats <= 0:
        raise ValueError("repeats must be > 0")
    if args.ratio_r < 0 or args.ratio_b < 0 or (args.ratio_r == 0 and args.ratio_b == 0):
        raise ValueError("ratio must be non-negative and not both zero")
    if args.total_pwm < 0 or args.total_pwm > 100:
        raise ValueError("total-pwm must be in [0, 100]")

    heights_mm = parse_int_list(args.heights_mm)
    points = build_grid_points(args.grid_rows, args.grid_cols, args.spacing_x_mm, args.spacing_y_mm)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_name = safe_slug(args.plan_name) if args.plan_name else f"spatial_mapping_{stamp}"
    plan_dir = Path(args.out_dir).expanduser().resolve() / plan_name
    plan_dir.mkdir(parents=True, exist_ok=False)

    layout_rows = [
        {
            "point_id": p.point_id,
            "row_index": p.row_index,
            "col_index": p.col_index,
            "x_mm": round(p.x_mm, 3),
            "y_mm": round(p.y_mm, 3),
            "radial_mm": round(p.radial_mm, 3),
            "zone": p.zone,
        }
        for p in priority_order(points)
    ]

    rng = random.Random(args.seed)
    measurements: list[dict[str, object]] = []
    sequence = 0

    if args.randomize_order == "global":
        combos = [(repeat_idx, height_mm, point) for repeat_idx in range(1, args.repeats + 1) for height_mm in heights_mm for point in points]
        rng.shuffle(combos)
        ordered_combos = combos
    else:
        ordered_combos = []
        for repeat_idx in range(1, args.repeats + 1):
            for height_mm in heights_mm:
                ordered_points = shuffled_points(
                    points=points,
                    rng=rng,
                    randomize_order=args.randomize_order,
                    keep_center_first=args.keep_center_first,
                )
                ordered_combos.extend((repeat_idx, height_mm, point) for point in ordered_points)

    for repeat_idx, height_mm, point in ordered_combos:
        sequence += 1
        run_name = f"h{height_mm:03d}_{point.point_id.lower()}_r{repeat_idx}"
        measurements.append(
            {
                "sequence_id": sequence,
                "height_mm": height_mm,
                "point_id": point.point_id,
                "x_mm": round(point.x_mm, 3),
                "y_mm": round(point.y_mm, 3),
                "radial_mm": round(point.radial_mm, 3),
                "zone": point.zone,
                "repeat_index": repeat_idx,
                "ratio_r": args.ratio_r,
                "ratio_b": args.ratio_b,
                "total_pwm": args.total_pwm,
                "n_spec": args.n_spec,
                "suggested_run_name": run_name,
                "run_dir": "",
                "PPFD_spec_mean": "",
                "rb_sensor": "",
                "sp_445_mean": "",
                "sp_480_mean": "",
                "sp_630_mean": "",
                "sp_680_mean": "",
                "notes": "",
            }
        )

    summary_md = build_summary_markdown(
        fixture_name=args.fixture_name,
        led_layout=args.led_layout,
        heights_mm=heights_mm,
        points=points,
        repeats=args.repeats,
        total_pwm=args.total_pwm,
        ratio_r=args.ratio_r,
        ratio_b=args.ratio_b,
        n_spec=args.n_spec,
        measurement_count=len(measurements),
        randomize_order=args.randomize_order,
        run_name_template="h{height_mm}_p{point_id}_r{repeat_index}",
    )

    csv_write(plan_dir / "point_layout.csv", layout_rows)
    csv_write(plan_dir / "measurement_plan.csv", measurements)
    csv_write(plan_dir / "measurement_results_template.csv", measurements)
    (plan_dir / "plan_summary.md").write_text(summary_md, encoding="utf-8")

    print(f"plan_dir: {plan_dir}")
    print(f"heights: {len(heights_mm)}")
    print(f"points_per_height: {len(points)}")
    print(f"repeats: {args.repeats}")
    print(f"total_measurements: {len(measurements)}")
    print(f"measurement_plan: {plan_dir / 'measurement_plan.csv'}")
    print(f"results_template: {plan_dir / 'measurement_results_template.csv'}")
    print(f"summary: {plan_dir / 'plan_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
