#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用实测数据拟合 LED 热力学模型参数

支持两种数据来源：
- PPFD 模式：仅有时间、PPFD、温度 → 拟合等效 K_ppfd 与 τ
- PWM 模式：有时间、红PWM、蓝PWM、温度 → 拟合 R_th 与 τ（与 led.py 一致）

用法示例：
- 从仅有 PPFD 的数据拟合：
  python fit_thermal_model.py --input data.csv --mode ppfd --plot --out model_ppfd.json

- 从红/蓝 PWM 的数据拟合：
  python fit_thermal_model.py --input data.csv --mode pwm --base-ambient 25.0 --max-power 86.4 --plot --out model_pwm.json

CSV 列要求：
- PPFD 模式：必须包含 time_s, ppfd, temp_c
- PWM 模式：必须包含 time_s, red_pwm, blue_pwm, temp_c
"""

from __future__ import annotations

import json
import argparse
from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd

from led import (
    DEFAULT_BASE_AMBIENT_TEMP,
    DEFAULT_MAX_POWER,
)
from led_rb_control import (
    fit_rb_thermal_from_ppfd,
    fit_rb_thermal_from_pwm,
    EffectivePpfdThermalModel,
    RBLedParams,
)


def _estimate_base_from_low_light(df: pd.DataFrame, ppfd_col: str = "ppfd", temp_col: str = "temp_c") -> Optional[float]:
    """当存在低光/关灯区间时，估计环境基准温度"""
    if ppfd_col not in df.columns or temp_col not in df.columns:
        return None
    try:
        q = np.nanpercentile(df[ppfd_col].values.astype(float), 20)
        sel = df[df[ppfd_col] <= q + 1e-6][temp_col].values.astype(float)
        if sel.size >= 3:
            return float(np.nanmedian(sel))
    except Exception:
        pass
    return None


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    rmse = float(np.sqrt(np.nanmean(err ** 2)))
    mae = float(np.nanmean(np.abs(err)))
    ss_res = float(np.nansum(err ** 2))
    ss_tot = float(np.nansum((y_true - np.nanmean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def run_ppfd_mode(df: pd.DataFrame, base_ambient: float, plot: bool):
    time_s = df["time_s"].values.astype(float)
    ppfd = df["ppfd"].values.astype(float)
    temp = df["temp_c"].values.astype(float)

    model: EffectivePpfdThermalModel = fit_rb_thermal_from_ppfd(time_s, ppfd, temp, base_ambient)

    # 生成拟合下的预测温度，用于评估
    T_pred = [float(temp[0])]
    for k in range(1, len(time_s)):
        dt = float(time_s[k] - time_s[k - 1])
        T_pred.append(model.step(ppfd[k - 1], T_pred[-1], dt))
    T_pred = np.asarray(T_pred, dtype=float)
    m = _metrics(temp, T_pred)

    print("拟合结果 (PPFD 模式):")
    print(f"  base_ambient_temp = {model.base_ambient_temp:.3f} °C")
    print(f"  k_ppfd_to_temp    = {model.k_ppfd_to_temp:.6f} °C/(μmol·m⁻²·s⁻¹)")
    print(f"  time_constant_s   = {model.time_constant_s:.3f} s")
    print(f"  评估: RMSE={m['rmse']:.3f} °C, MAE={m['mae']:.3f} °C, R²={m['r2']:.4f}")

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 4))
        plt.plot(time_s, temp, label="Measured Temp")
        plt.plot(time_s, T_pred, label="Fitted Model Temp", linestyle="--")
        ax2 = plt.gca().twinx()
        ax2.plot(time_s, ppfd, color="tab:green", alpha=0.3, label="PPFD")
        plt.title("PPFD-based Thermal Fit")
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    return model, m


def run_pwm_mode(df: pd.DataFrame, base_ambient: float, max_power: float, plot: bool):
    time_s = df["time_s"].values.astype(float)
    red_pwm = df["red_pwm"].values.astype(float)
    blue_pwm = df["blue_pwm"].values.astype(float)
    temp = df["temp_c"].values.astype(float)

    params: RBLedParams = fit_rb_thermal_from_pwm(time_s, red_pwm, blue_pwm, temp, base_ambient, max_power)

    # 用拟合参数回放得到预测温度（与拟合目标一致）
    # 计算热功率 U_k
    r = np.clip(red_pwm, 0.0, 100.0)
    b = np.clip(blue_pwm, 0.0, 100.0)
    pwm_frac = np.clip((r + b) / 200.0, 0.0, 1.0)
    efficiency = 0.8 + 0.2 * np.exp(-2.0 * pwm_frac)
    efficiency = np.clip(efficiency, 1e-3, 1.0)
    power = (float(max_power) * pwm_frac) / efficiency
    U = power * (1.0 - efficiency)

    # 精确离散解模拟
    T_pred = [float(temp[0])]
    for k in range(1, len(time_s)):
        dt = float(time_s[k] - time_s[k - 1])
        a = np.exp(-dt / max(params.time_constant_s, 1e-9))
        T_next = T_pred[-1] * a + (1 - a) * (base_ambient + params.thermal_resistance * U[k - 1])
        T_pred.append(float(T_next))
    T_pred = np.asarray(T_pred, dtype=float)
    m = _metrics(temp, T_pred)

    print("拟合结果 (PWM 模式):")
    print(f"  base_ambient_temp = {params.base_ambient_temp:.3f} °C")
    print(f"  thermal_resistance= {params.thermal_resistance:.6f} K/W")
    print(f"  time_constant_s   = {params.time_constant_s:.3f} s")
    print(f"  max_power         = {params.max_power:.3f} W")
    print(f"  评估: RMSE={m['rmse']:.3f} °C, MAE={m['mae']:.3f} °C, R²={m['r2']:.4f}")

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 4))
        plt.plot(time_s, temp, label="Measured Temp")
        plt.plot(time_s, T_pred, label="Fitted Model Temp", linestyle="--")
        ax2 = plt.gca().twinx()
        ax2.plot(time_s, r + b, color="tab:orange", alpha=0.3, label="Red+Blue PWM")
        plt.title("PWM-based Thermal Fit")
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    return params, m


def main():
    ap = argparse.ArgumentParser(description="Fit LED thermal model from measured data")
    ap.add_argument("--input", required=True, help="CSV file path")
    ap.add_argument("--mode", choices=["auto", "ppfd", "pwm"], default="auto", help="data mode")
    ap.add_argument("--base-ambient", type=float, default=None, help="base ambient temperature (°C)")
    ap.add_argument("--estimate-base", action="store_true", help="estimate base ambient from low PPFD region")
    ap.add_argument("--max-power", type=float, default=DEFAULT_MAX_POWER, help="max power at 100%+100% PWM")
    ap.add_argument("--plot", action="store_true", help="plot measured vs fitted temperature")
    ap.add_argument("--out", help="output JSON file to save fitted parameters and metrics")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    cols = set(df.columns)

    # 选择模式
    mode = args.mode
    if mode == "auto":
        if {"time_s", "ppfd", "temp_c"}.issubset(cols):
            mode = "ppfd"
        elif {"time_s", "red_pwm", "blue_pwm", "temp_c"}.issubset(cols):
            mode = "pwm"
        else:
            raise SystemExit("无法自动识别模式：请提供 (time_s, ppfd, temp_c) 或 (time_s, red_pwm, blue_pwm, temp_c)")

    # 基准温度
    if args.base_ambient is not None:
        base_ambient = float(args.base_ambient)
    else:
        if args.estimate_base and "ppfd" in cols and "temp_c" in cols:
            est = _estimate_base_from_low_light(df)
            base_ambient = float(est) if est is not None else DEFAULT_BASE_AMBIENT_TEMP
        else:
            base_ambient = DEFAULT_BASE_AMBIENT_TEMP

    print(f"模式: {mode}")
    print(f"基准环境温度: {base_ambient:.2f} °C")

    result = {}
    if mode == "ppfd":
        model, metrics = run_ppfd_mode(df, base_ambient, args.plot)
        result = {"mode": "ppfd", "params": asdict(model), "metrics": metrics}
    else:
        params, metrics = run_pwm_mode(df, base_ambient, args.max_power, args.plot)
        result = {"mode": "pwm", "params": asdict(params), "metrics": metrics}

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"已保存拟合结果到: {args.out}")


if __name__ == "__main__":
    main()

