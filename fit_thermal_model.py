#!/usr/bin/env python3
"""
Fit a simple discrete-time thermal model from CSV logs.

Model: T[k+1] = a*T[k] + b*u[k] + c

Inputs:
- CSV with header including: timestamp, temperature, a1_raw (default input), device_id (optional)
- Lines starting with '#' are ignored.

Outputs:
- Prints a,b,c and derived parameters: tau, K, T_env based on median dt.

Usage examples:
  python fit_thermal_model.py --csv 5-1-400_20250829_091025.csv
  python fit_thermal_model.py --csv file.csv --device L_6vSQ== --u a1_raw --normalize-u
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from math import isfinite, log
from typing import List, Optional, Tuple


@dataclass
class Record:
    t: float  # seconds from start
    temp: float
    u: float
    device_id: Optional[str]


def parse_csv(path: str, u_col: str, device_id: Optional[str]) -> Tuple[List[Record], float]:
    """Parse CSV. Returns (records, median_dt)."""
    rows: List[Record] = []

    # First pass: read rows, skip comments
    with open(path, 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        header = None
        for raw in rdr:
            if not raw:
                continue
            if raw[0].startswith('#'):
                continue
            if header is None:
                header = [h.strip() for h in raw]
                continue
            row = {header[i]: raw[i] for i in range(min(len(header), len(raw)))}

            try:
                ts = row.get('timestamp') or row.get('time') or row.get('datetime')
                if not ts:
                    continue
                # Try parse various formats
                ts_dt = None
                for fmt in (
                    '%Y-%m-%d %H:%M:%S.%f',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y/%m/%d %H:%M:%S.%f',
                    '%Y/%m/%d %H:%M:%S',
                ):
                    try:
                        ts_dt = datetime.strptime(ts.strip(), fmt)
                        break
                    except Exception:
                        pass
                if ts_dt is None:
                    continue

                temp = float(row['temperature'])
                uval = float(row[u_col])
                did = row.get('device_id') or row.get('device') or None

                rows.append((ts_dt, temp, uval, did))
            except Exception:
                continue

    if not rows:
        raise RuntimeError('No data rows parsed. Check header names and file encoding.')

    # Filter device_id if requested
    if device_id is not None:
        rows = [r for r in rows if r[3] == device_id]
        if not rows:
            raise RuntimeError(f'No rows for device_id={device_id!r}')

    # Sort by timestamp
    rows.sort(key=lambda r: r[0])

    # Build time in seconds from start
    t0 = rows[0][0]
    recs: List[Record] = []
    for ts, temp, u, did in rows:
        t = (ts - t0).total_seconds()
        recs.append(Record(t=t, temp=temp, u=u, device_id=did))

    # Compute dt list and median
    dts: List[float] = []
    for i in range(1, len(recs)):
        dts.append(max(1e-6, recs[i].t - recs[i-1].t))
    if not dts:
        raise RuntimeError('Not enough samples to compute dt')
    dts_sorted = sorted(dts)
    median_dt = dts_sorted[len(dts_sorted)//2]

    return recs, median_dt


def downsample(recs: List[Record], stride: int) -> List[Record]:
    if stride <= 1:
        return recs
    return [recs[i] for i in range(0, len(recs), stride)]


def normalize_u(recs: List[Record]) -> Tuple[List[Record], float, float]:
    u_vals = [r.u for r in recs]
    u_min, u_max = min(u_vals), max(u_vals)
    rng = max(1e-9, (u_max - u_min))
    out = [Record(t=r.t, temp=r.temp, u=(r.u - u_min)/rng, device_id=r.device_id) for r in recs]
    return out, u_min, u_max


def fit_discrete_linear(recs: List[Record]) -> Tuple[float, float, float]:
    """Least squares fit for T[k+1] = a*T[k] + b*u[k] + c"""
    import numpy as np
    T = np.array([r.temp for r in recs], dtype=float)
    U = np.array([r.u for r in recs], dtype=float)
    # Build X, y (one-step shift)
    X = np.stack([T[:-1], U[:-1], np.ones_like(U[:-1])], axis=1)
    y = T[1:]
    # Solve normal equations
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = map(float, theta)
    return a, b, c


def main():
    ap = argparse.ArgumentParser(description='Fit discrete-time thermal model from CSV logs')
    ap.add_argument('--csv', required=True, help='CSV file path')
    ap.add_argument('--device', default=None, help='device_id to filter (optional)')
    ap.add_argument('--u', default='a1_raw', help='input column name (default: a1_raw)')
    ap.add_argument('--normalize-u', action='store_true', help='normalize input to [0,1]')
    ap.add_argument('--stride', type=int, default=1, help='downsample stride (default: 1)')
    args = ap.parse_args()

    recs, median_dt = parse_csv(args.csv, args.u, args.device)
    if args.stride > 1:
        recs = downsample(recs, args.stride)
    u_min = u_max = None
    if args.normalize_u:
        recs, u_min, u_max = normalize_u(recs)

    a, b, c = fit_discrete_linear(recs)

    # Derived quantities
    tau = None
    if 0.0 < a < 1.0:
        tau = -median_dt / log(a)
    K = None
    T_env = None
    if a != 1.0:
        K = b / (1.0 - a)
        T_env = c / (1.0 - a)

    print('\n=== FIT RESULTS (Discrete Linear Model) ===')
    print(f'rows used: {len(recs)}')
    print(f'median dt: {median_dt:.6f} s')
    print(f'a: {a:.6f}, b: {b:.6f}, c: {c:.6f}')
    if tau is not None and isfinite(tau):
        print(f'tau: {tau:.3f} s  (time constant)')
    else:
        print('tau: N/A (a not in (0,1))')
    if K is not None and isfinite(K):
        print(f'K: {K:.6f}  (static gain wrt input "{args.u}")')
    else:
        print('K: N/A')
    if T_env is not None and isfinite(T_env):
        print(f'T_env: {T_env:.3f} °C  (baseline when input=0)')
    else:
        print('T_env: N/A')

    if args.normalize_u:
        print(f'input normalization: u_norm = (u - {u_min}) / ({u_max} - {u_min})')
        print('Note: K is per unit normalized input.')

    print('\nHow to use:')
    print('- Use tau as thermal time constant;')
    print('- If input is power (W), K ≈ R_th (°C/W), and C ≈ tau / R_th;')
    print('- If input is PWM/PPFD/a1_raw, K is an effective gain (°C per unit input).')


if __name__ == '__main__':
    main()

