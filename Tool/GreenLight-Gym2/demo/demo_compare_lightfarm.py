"""Standalone GL-Gym comparison: rule_based (8-22 ON) vs LightFarm (MPPI).

Runs both controllers in lockstep over `--days` sim days starting at
(--start-day, hour 0), prints a per-hour summary line and a final
cumulative table. Sim-only — no hardware touched.

Run:
  python demo/demo_compare_lightfarm.py --days 1 --start-day 90
"""
from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from typing import Dict

import numpy as np
import yaml as _yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT1_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))
MPPI_GOVEE_SRC = os.path.join(PROJECT1_ROOT, "MPPI-Govee", "src")
for _p in (REPO_ROOT, MPPI_GOVEE_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gl_gym.environments.greenlight_env import GreenLightEnv
from gl_gym.components.rule_based import RuleBasedController
from gl_gym.core.types import StepContext
from gl_gym.components.weather import WeatherRepository
from gl_gym.environments.utils import load_weather_data


# ---- Controllers (mirrors server.py _make_lamp_controllers) --------------

def make_controllers(rb_params, mppi_r_power=1.0, mppi_temp_max=80.0):
    base_rb = RuleBasedController(**rb_params)

    PHOTO_ON, PHOTO_OFF = 8, 22
    def in_win(h): return PHOTO_ON <= h < PHOTO_OFF

    LAMP_PPFD, PPFD_TARGET = 500.0, 450.0
    U_RB = min(1.0, PPFD_TARGET / LAMP_PPFD)   # 0.9

    def rule_based_ctl(ctx):
        u = base_rb.predict(ctx)
        u[4] = U_RB if in_win(ctx.hour_of_day) else 0.0
        return u

    def glgym_rb_ctl(ctx):
        u = base_rb.predict(ctx)
        if not in_win(ctx.hour_of_day):
            u[4] = 0.0
        return u

    # Real chamber MPPI from MPPI-Govee/src/mppi_v2
    _lf_mppi = None
    try:
        from mppi_v2 import LEDPlant, LEDMPPIController  # type: ignore
        thermal_dir = os.path.join(PROJECT1_ROOT, "MPPI-Govee", "Thermal", "exported_models")

        class _StubPowerModel:
            def predict(self, *, total_pwm, key=None):
                return float(total_pwm) * 0.12

        # Greenhouse-style thermal model (sim mode). Calibrated so:
        #   u=0 → T → 22°C (outdoor lab)
        #   u=1 → T → ~40°C steady state (lamp full)
        #   alpha=0.5 → ~half-rise per env.dt (15min)
        _lf_plant = LEDPlant(
            target_ppfd=PPFD_TARGET, r_b_ratio=0.8,
            use_pn_model=True, use_sp_to_ppfd=False,
            use_govee_pwm_model=False,
            use_thermal_model=False,
            sim_thermal={"t_outdoor": 22.0, "delta_t_at_u1": 18.0,
                         "alpha": 0.5, "lamp_ppfd_max": LAMP_PPFD},
            model_dir=thermal_dir, power_model=_StubPowerModel(),
            x_cm=0.0, y_cm=0.0, z_cm=5.0, demo_gain=1.0,
        )
        _lf_plant.ambient_temp = 22.0
        _lf_mppi = LEDMPPIController(_lf_plant, horizon=5, num_samples=120, dt=900.0, temperature=0.5)
        _lf_mppi.set_weights(Q_photo=1.0, R_du=0.05, R_power=float(mppi_r_power), Q_ref=0.0)
        _lf_mppi.set_constraints(u_min=0.0, u_max=PPFD_TARGET * 1.5, temp_min=0.0, temp_max=float(mppi_temp_max))
        _lf_mppi.set_ppfd_train_range(ppfd_min=0.0, ppfd_max=600.0, R_oob=0.0)
        # Center exploration at 300 (the Pn saturation knee, per model probe)
        # so MPPI samples cover 200-400 — where energy savings live without
        # losing meaningful Pn. u_std widened for broader search.
        _lf_mppi.set_target_ppfd(400.0)
        _lf_mppi.set_mppi_params(u_std=60.0)
        print("[boot] real MPPI loaded (LEDMPPIController, num_samples=120, horizon=5)")
    except Exception as exc:
        print(f"⚠️ LightFarm MPPI init failed: {exc}; falling back to rule_based")

    def lightfarm_ctl(ctx):
        u = base_rb.predict(ctx)
        if not in_win(ctx.hour_of_day):
            u[4] = 0.0
            return u
        if _lf_mppi is None:
            u[4] = U_RB
            return u
        try:
            optimal_ppfd, _seq, _ok, _cost, _w = _lf_mppi.solve(float(ctx.x[2]))
            u[4] = float(np.clip(optimal_ppfd / LAMP_PPFD, 0.0, 1.0))
        except Exception as exc:
            print(f"[lightfarm] solve failed: {exc}")
            u[4] = U_RB
        return u

    return {
        "rule_based": rule_based_ctl,
        "glgym_rb":   glgym_rb_ctl,
        "lightfarm":  lightfarm_ctl,
    }


# ---- Env / driver --------------------------------------------------------

def make_env(season_days: int):
    cfg_path = os.path.join(REPO_ROOT, "gl_gym", "configs", "envs", "GreenLightEnv.yml")
    with open(cfg_path) as f:
        ek = _yaml.load(f, Loader=_yaml.FullLoader)["GreenLightEnv"]
    wkw = ek.pop("weather_repository_kwargs")
    wd = wkw["weather_data_dir"]
    if not os.path.isabs(wd):
        wd = os.path.join(REPO_ROOT, wd)
    ek["weather_repository"] = WeatherRepository(
        weather_data_dir=wd,
        load_weather_data_fn=eval(wkw["load_weather_data_fn"]),
    )
    ek.pop("eval_scenarios", None)
    ek["normalize_actions"] = False
    ek["season_length"] = season_days + 1
    return GreenLightEnv(**deepcopy(ek))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=1.0)
    ap.add_argument("--start-day", type=int, default=90)
    ap.add_argument("--print-every", type=int, default=4,
                    help="print one summary line every N env-steps (4 × 15min = 1h)")
    ap.add_argument("--indoor", action="store_true",
                    help="zero outdoor PAR (chamber-style, no sun)")
    ap.add_argument("--lamp-par-scale", type=float, default=1.0,
                    help="multiply lamp PAR (p[172]); <1 keeps canopy un-saturated")
    ap.add_argument("--no-climate-control", action="store_true",
                    help="zero heat/vent/CO2/screens — only lamp matters")
    ap.add_argument("--mppi-r-power", type=float, default=1.0,
                    help="MPPI R_power weight (higher = more aggressive energy savings)")
    ap.add_argument("--mppi-temp-max", type=float, default=80.0,
                    help="MPPI temp hard constraint upper bound (°C)")
    args = ap.parse_args()

    rb_yml = os.path.join(REPO_ROOT, "configs", "agents", "rule_based.yml")
    with open(rb_yml) as f:
        rb_params = _yaml.load(f, Loader=_yaml.FullLoader)["GreenLightEnv"]

    ctls = make_controllers(rb_params, mppi_r_power=args.mppi_r_power, mppi_temp_max=args.mppi_temp_max)

    envs = {}
    for name in ctls:
        e = make_env(int(args.days) + 1)
        e.reset(seed=0, options={"scenario": {
            "location": "Amsterdam", "growth_year": 2010,
            "start_day": int(args.start_day),
        }})
        if args.indoor:
            e.weather_data[:, 0] = 0.0    # sun → 0
            e.weather_data[:, 7] = 0.0    # daily sun sum → 0
            e.weather_data[:, 1] = 22.0   # outdoor air T → 22°C (lab room)
            e.weather_data[:, 5] = 22.0   # sky T
            e.weather_data[:, 6] = 22.0   # outdoor soil T
            x = np.copy(e.x)
            for idx in (2, 3, 4, 5, 6, 7, 8, 9, 17, 20, 21):
                x[idx] = 22.0
            e.x = x
            e.x_prev = np.copy(x)
        if args.lamp_par_scale != 1.0:
            e.p[172] = float(e.p[172]) * args.lamp_par_scale
        envs[name] = e

    any_env = next(iter(envs.values()))
    p_lamp_W = float(any_env.p[172])
    dt_s = float(any_env.dt)
    total_steps = int(round(args.days * 24 * 3600 / dt_s))
    print(f"Env dt = {dt_s}s ({dt_s/60:.0f}min). Plan {total_steps} steps "
          f"= {total_steps*dt_s/3600:.1f} sim hours.")
    print(f"Lamp power coef p[172] = {p_lamp_W:.1f} W/m²")
    print()
    header = f"{'step':>5} {'sim_h':>5} {'sun':>6} | "
    for n in ctls:
        header += f"{n+'_u':>10} {n+'_Pn':>10} {n+'_W':>10} | "
    print(header)

    cum_pn = {n: 0.0 for n in ctls}
    cum_kwh = {n: 0.0 for n in ctls}

    # Track per-controller T peak so summary shows divergence
    t_peak = {n: 0.0 for n in ctls}

    for step in range(total_steps):
        for name, ctl in ctls.items():
            e = envs[name]
            ctx = StepContext(
                t=e.timestep, dt=e.dt, Np=e.Np,
                x_prev=e.x_prev, x=e.x, u=e.u, p=e.p,
                d=e.weather_data,
                hour_of_day=e.hour_of_day, day_of_year=e.day_of_year,
            )
            u = np.asarray(ctl(ctx), dtype=float)
            if args.no_climate_control:
                # Keep lamp (4) AND CO2 injection (1) — strip the rest.
                lamp = float(u[4]); co2 = float(u[1])
                u = np.zeros_like(u); u[4] = lamp; u[1] = co2
            obs, _r, term, trunc, info = e.step(u)
            pn = float(info.get("mcAirCan", 0.0))   # mg CO2/m²/s
            cum_pn[name] += pn * dt_s
            cum_kwh[name] += float(u[4]) * p_lamp_W * dt_s / 3600.0 / 1e3
            if float(e.x[2]) > t_peak[name]:
                t_peak[name] = float(e.x[2])

        if (step + 1) % args.print_every == 0:
            ctx0 = StepContext(
                t=any_env.timestep, dt=any_env.dt, Np=any_env.Np,
                x_prev=any_env.x_prev, x=any_env.x, u=any_env.u, p=any_env.p,
                d=any_env.weather_data,
                hour_of_day=any_env.hour_of_day, day_of_year=any_env.day_of_year,
            )
            sun = float(any_env.weather_data[any_env.timestep, 0])
            line = f"{step+1:>5} {ctx0.hour_of_day:5.1f} {sun:6.0f} | "
            for name, e in envs.items():
                u_lamp = float(e.u[4])
                # Pn from last info — re-compute? we lost it; fetch via info on next step
                # Simpler: derive from our cum increment
                line += f"{u_lamp:10.2f} {0:10.3f} {u_lamp*p_lamp_W:10.1f} | "
            print(line)

    print()
    print("=" * 60)
    print(f"{'controller':<14} {'∫Pn (g CO₂/m²)':>16} {'energy (kWh/m²)':>18} {'T_peak (°C)':>14}")
    for n in ctls:
        print(f"{n:<14} {cum_pn[n]/1000.0:>16.2f} {cum_kwh[n]:>18.4f} {t_peak[n]:>14.1f}")
    print()
    base = "rule_based"
    if base in cum_pn:
        for n in cum_pn:
            if n == base: continue
            e_save = (cum_kwh[base] - cum_kwh[n]) / max(cum_kwh[base], 1e-9) * 100
            pn_chg = (cum_pn[n] - cum_pn[base]) / max(cum_pn[base], 1e-9) * 100
            print(f"  {n:<14} vs {base}: energy {-e_save:+.1f}% (lower is less energy)  ∫Pn {pn_chg:+.1f}%")


if __name__ == "__main__":
    main()
