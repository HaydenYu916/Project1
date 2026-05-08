"""
Demo: compare a "default" vs "energy-saving" rule-based controller in GL-Gym,
seeded with a real chamber sensor snapshot. Produces 4-panel comparison plots
showing indoor climate, energy use, and a Pn (carbon-assimilation) proxy.

Run:
    python demo/demo_compare.py

Outputs:
    demo/output/demo_compare.png
    demo/output/demo_compare_summary.txt
"""
from __future__ import annotations

import os
import sys
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gl_gym.environments.greenlight_env import GreenLightEnv
from gl_gym.components.rule_based import RuleBasedController
from gl_gym.core.types import StepContext
from gl_gym.components.weather import WeatherRepository
from gl_gym.environments.utils import load_weather_data

import yaml as _yaml
from os.path import join as _join


def load_env_params(env_id: str, path: str) -> Dict:
    with open(_join(path, env_id + ".yml"), "r") as f:
        params = _yaml.load(f, Loader=_yaml.FullLoader)
    return params[env_id]


def load_model_hyperparams(algorithm: str, env_id: str) -> Dict:
    with open(_join("configs/agents/", algorithm + ".yml"), "r") as f:
        params = _yaml.load(f, Loader=_yaml.FullLoader)
    return params[env_id]


def build_env_kwargs(env_kwargs):
    env_kwargs = env_kwargs.copy()
    weather_repository_kwargs = env_kwargs.pop("weather_repository_kwargs")
    env_kwargs["weather_repository"] = WeatherRepository(
        weather_data_dir=weather_repository_kwargs["weather_data_dir"],
        load_weather_data_fn=eval(weather_repository_kwargs["load_weather_data_fn"]),
    )
    eval_weather_scenarios = env_kwargs.pop("eval_scenarios")
    return env_kwargs, eval_weather_scenarios

from demo.sensor_snapshot import read_sensor_snapshot
from demo.seed_env import seed_env_from_snapshot

OUT_DIR = os.path.join(REPO_ROOT, "demo", "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- demo settings ----
DEMO_DAYS = 3            # short window, fast to run, easy to read
SCENARIO = {"location": "Amsterdam", "growth_year": 2010, "start_day": 90}  # spring


def make_env(env_kwargs: Dict, season_length: int) -> GreenLightEnv:
    kw = deepcopy(env_kwargs)
    kw["season_length"] = season_length
    kw["normalize_actions"] = False  # rule-based outputs raw u
    return GreenLightEnv(**kw)


def build_step_context(env: GreenLightEnv) -> StepContext:
    return StepContext(
        t=env.timestep, dt=env.dt, Np=env.Np,
        x_prev=env.x_prev, x=env.x, u=env.u, p=env.p,
        d=env.weather_data,
        hour_of_day=env.hour_of_day, day_of_year=env.day_of_year,
    )


def run_episode(
    env_kwargs: Dict,
    rb_params: Dict,
    snapshot: Dict[str, float],
    label: str,
) -> pd.DataFrame:
    env = make_env(env_kwargs, season_length=DEMO_DAYS)
    env.reset(seed=0, options={"scenario": SCENARIO})
    seed_env_from_snapshot(env, snapshot)

    controller = RuleBasedController(**rb_params)

    # parameter indices used in rewards.py for energy terms
    p_boil_W_per_aFlr = float(env.p[108])  # boiler nominal power, W per aFlr
    p_aFlr = float(env.p[46])              # floor area scaling
    p_lamp_W_per_m2 = float(env.p[172])    # lamp electrical power, W/m²
    dt_s = float(env.dt)

    rows: List[Dict] = []
    while True:
        ctx = build_step_context(env)
        u = controller.predict(ctx)
        obs, reward, terminated, truncated, info = env.step(u.astype(np.float32))

        # raw energy in kWh/m² for this step
        lamp_kwh = u[4] * p_lamp_W_per_m2 * dt_s / 3600.0 * 1e-3
        heat_kwh = u[0] * p_boil_W_per_aFlr / p_aFlr * dt_s / 3600.0 * 1e-3

        ic = obs["IndoorClimateObservations"]
        rows.append({
            "label": label,
            "t_min": env.timestep * dt_s / 60.0,
            "co2_ppm": float(ic[0]),
            "t_air": float(ic[1]),
            "rh": float(ic[2]),
            "uBoil": float(u[0]),
            "uCO2": float(u[1]),
            "uLamp": float(u[4]),
            "uVent": float(u[3]),
            "lamp_kwh_step": float(lamp_kwh),
            "heat_kwh_step": float(heat_kwh),
            # canopy photosynthesis fluxes from CasADi aux outputs
            "Pn_gross_ch2o": float(info.get("mcAirBuf", 0.0)),     # mg{CH2O}/m²/s
            "Pn_net_co2":    float(info.get("mcAirCan", 0.0)),     # mg{CO2}/m²/s
            "resp_growth":   float(info.get("mcBufAir", 0.0)),
            "resp_maint":    float(info.get("mcOrgAir", 0.0)),
            "parCan":        float(info.get("parCan", 0.0)),
            "fruit_growth_dm": float(info.get("fruit_growth_dm", 0.0)),
            "profit": float(info.get("profit", 0.0)),
            "elec_cost": float(info.get("elec_cost", 0.0)),
            "heat_cost": float(info.get("heat_cost", 0.0)),
        })
        if terminated or truncated:
            break

    df = pd.DataFrame(rows)
    df["lamp_kwh_cum"] = df["lamp_kwh_step"].cumsum()
    df["heat_kwh_cum"] = df["heat_kwh_step"].cumsum()
    df["energy_kwh_cum"] = df["lamp_kwh_cum"] + df["heat_kwh_cum"]
    # cumulative net CO2 assimilation [mg{CO2}/m²]
    df["Pn_cum_co2_mg_m2"] = (df["Pn_net_co2"] * dt_s).cumsum()
    return df


def make_plots(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    ax = axes[0, 0]
    ax.plot(df_a["t_min"] / 60.0, df_a["t_air"], label=label_a, color="C0")
    ax.plot(df_b["t_min"] / 60.0, df_b["t_air"], label=label_b, color="C1")
    ax.set_ylabel("Indoor temperature [°C]")
    ax.set_title("Indoor air temperature")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(df_a["t_min"] / 60.0, df_a["co2_ppm"], label=label_a, color="C0")
    ax.plot(df_b["t_min"] / 60.0, df_b["co2_ppm"], label=label_b, color="C1")
    ax.set_ylabel("CO₂ [ppm]")
    ax.set_title("Indoor CO₂")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(df_a["t_min"] / 60.0, df_a["energy_kwh_cum"], label=f"{label_a} total", color="C0")
    ax.plot(df_b["t_min"] / 60.0, df_b["energy_kwh_cum"], label=f"{label_b} total", color="C1")
    ax.plot(df_a["t_min"] / 60.0, df_a["lamp_kwh_cum"], "--", color="C0", alpha=0.5, label=f"{label_a} lamp")
    ax.plot(df_b["t_min"] / 60.0, df_b["lamp_kwh_cum"], "--", color="C1", alpha=0.5, label=f"{label_b} lamp")
    ax.set_ylabel("Cumulative energy [kWh/m²]")
    ax.set_xlabel("Time [hours]")
    ax.set_title("Energy use (lower = better)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    win = max(3, len(df_a) // 100)
    ax.plot(df_a["t_min"] / 60.0, df_a["Pn_net_co2"].rolling(win, min_periods=1).mean(),
            label=label_a, color="C0")
    ax.plot(df_b["t_min"] / 60.0, df_b["Pn_net_co2"].rolling(win, min_periods=1).mean(),
            label=label_b, color="C1")
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_ylabel("Pn  [mg CO₂·m⁻²·s⁻¹]")
    ax.set_xlabel("Time [hours]")
    ax.set_title("Net canopy photosynthesis (mcAirCan, true model output)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "demo_compare.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def summarize(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str,
              snapshot: Dict[str, float]) -> str:
    e_a = df_a["energy_kwh_cum"].iloc[-1]
    e_b = df_b["energy_kwh_cum"].iloc[-1]
    pn_a = df_a["Pn_cum_co2_mg_m2"].iloc[-1]
    pn_b = df_b["Pn_cum_co2_mg_m2"].iloc[-1]
    saved_pct = (e_a - e_b) / max(e_a, 1e-9) * 100.0
    pn_pct = (pn_b - pn_a) / max(abs(pn_a), 1e-9) * 100.0

    lines = [
        "=" * 60,
        " GL-Gym demo: real-sensor-seeded comparison",
        "=" * 60,
        f"Sensor snapshot ({snapshot.get('source', '?')}):",
        f"  T_air = {snapshot['t_air']:.2f} °C",
        f"  RH    = {snapshot['rh']:.2f} %",
        f"  CO2   = {snapshot['co2_ppm']:.2f} ppm",
        f"Scenario: {SCENARIO}",
        f"Sim length: {DEMO_DAYS} days",
        "",
        f"[{label_a}]  total energy = {e_a:.4f} kWh/m²    ∫Pn = {pn_a:.2f} mg CO₂/m²",
        f"[{label_b}]  total energy = {e_b:.4f} kWh/m²    ∫Pn = {pn_b:.2f} mg CO₂/m²",
        "",
        f"Energy saved by {label_b} vs {label_a}: {saved_pct:+.1f} %",
        f"Cumulative-Pn change:                       {pn_pct:+.1f} %",
        "",
        "Pn = mcAirCan (net canopy CO₂ assimilation) read directly from the",
        "GreenLight CasADi model — the true instantaneous photosynthesis flux.",
    ]
    txt = "\n".join(lines)
    out_path = os.path.join(OUT_DIR, "demo_compare_summary.txt")
    with open(out_path, "w") as f:
        f.write(txt + "\n")
    return txt


def main() -> None:
    snapshot = read_sensor_snapshot()
    print(f"[demo] sensor snapshot: {snapshot}")

    env_kwargs = load_env_params("GreenLightEnv", "gl_gym/configs/envs/")
    env_kwargs, _scenarios = build_env_kwargs(env_kwargs)

    rb_default = load_model_hyperparams("rule_based", "GreenLightEnv")
    # energy-saving variant: lower setpoints, narrower lamp window, tolerate higher RH
    rb_eco = dict(rb_default)
    rb_eco.update({
        "temp_setpoint_day": 18.0,
        "temp_setpoint_night": 15.5,
        "lamps_on": 6,
        "lamps_off": 16,
        "rh_max": 90,
        "rhMax": 90,
    })

    print("[demo] running default rule-based ...")
    df_a = run_episode(env_kwargs, rb_default, snapshot, label="default RB")
    print("[demo] running eco rule-based ...")
    df_b = run_episode(env_kwargs, rb_eco, snapshot, label="eco RB")

    df_a.to_csv(os.path.join(OUT_DIR, "trace_default.csv"), index=False)
    df_b.to_csv(os.path.join(OUT_DIR, "trace_eco.csv"), index=False)

    png = make_plots(df_a, df_b, "default RB", "eco RB")
    print(f"[demo] wrote plot: {png}")

    summary = summarize(df_a, df_b, "default RB", "eco RB", snapshot)
    print(summary)


if __name__ == "__main__":
    main()
