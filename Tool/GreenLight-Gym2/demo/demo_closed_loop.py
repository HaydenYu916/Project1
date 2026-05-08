"""GL-Gym → Govee closed-loop demo.

Each tick:
  1. Read live Riotee sensor (T, RH, CO2) via web API.
  2. Override the GL-Gym indoor state with the live readings.
  3. Run the RuleBasedController for one env.step() to get u[4] = lampOn ∈ [0,1].
  4. Map u[4] → target PPFD (linear: u=1 ⇒ MAX_PPFD).
  5. Invert the trained Govee PWM→PPFD model at fixed R:B=4:1 to get (R,B) PWM.
  6. POST /api/led to drive the bulbs; POST /api/measure to read actual PPFD.
  7. Append a CSV row.

Defaults: tick=15s wall, env.dt=900s sim → 60× speedup. Stops on Ctrl+C
or after --max-ticks. Talks to http://127.0.0.1:8001.
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
import urllib.request
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GLGYM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT1_SRC = os.path.join(REPO_ROOT, "MPPI-Govee", "src")
for p in (GLGYM_ROOT, PROJECT1_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

# GL-Gym
from gl_gym.environments.greenlight_env import GreenLightEnv  # noqa: E402
from gl_gym.components.rule_based import RuleBasedController  # noqa: E402
from gl_gym.core.types import StepContext  # noqa: E402
from gl_gym.components.weather import WeatherRepository  # noqa: E402
from gl_gym.environments.utils import (  # noqa: E402
    load_weather_data,
    co2ppm2dens,
    satVp,
)
import yaml as _yaml

# Govee inverse model
from govee_pwmtoppfd_model import GoveePWMtoPPFDModel, GoveePosition  # noqa: E402


# ---- defaults ------------------------------------------------------------
WEB_API = "http://127.0.0.1:8001"
SCENARIO = {"location": "Amsterdam", "growth_year": 2010, "start_day": 90}
MAX_PPFD = 60.0           # u_lamp=1.0 maps to this PPFD on canopy (matches model range)
RB_RED_FRAC = 0.8         # 4:1 → 80% red, 20% blue
SENSOR_Z_CM = 5.0         # match server's POSITION


# ---- web API helpers -----------------------------------------------------

def _api_get(path: str, timeout: float = 5.0):
    req = urllib.request.Request(f"{WEB_API}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _api_post(path: str, body: Optional[dict] = None, timeout: float = 30.0):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{WEB_API}{path}", data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def read_live_sensor():
    """Returns dict with T_air, RH, CO2_ppm or None on failure."""
    try:
        s = _api_get("/api/sensor", timeout=5.0)
        return {
            "T_air": float(s["temperature"]) if s.get("temperature") is not None else None,
            "RH":    float(s["humidity"]) if s.get("humidity") is not None else None,
            "CO2_ppm": (float(s["co2_ppm"]) if s.get("co2_ppm") not in (None,) and float(s["co2_ppm"]) > 0 else None),
        }
    except Exception as e:
        print(f"[warn] sensor read failed: {e}")
        return None


def dispatch_pwm(pwm_r: int, pwm_b: int):
    try:
        return _api_post("/api/led", {"pwm_r": pwm_r, "pwm_b": pwm_b}, timeout=15.0)
    except Exception as e:
        print(f"[warn] LED dispatch failed: {e}")
        return None


def trigger_measure(n: int = 1):
    try:
        m = _api_post("/api/measure", {"trigger_count": n}, timeout=20.0)
        return float(m["measured_ppfd"])
    except Exception as e:
        print(f"[warn] measure failed: {e}")
        return None


# ---- inverse Govee model -------------------------------------------------

class GoveeInverter:
    """Search PWM at fixed R:B ratio to hit a target PPFD.

    Cheap because GoveePWMtoPPFDModel is fast (sklearn pipeline). We do a
    small bisection over total PWM ∈ [0, 100].
    """
    def __init__(self, model: GoveePWMtoPPFDModel, red_frac: float):
        self.model = model
        self.red_frac = red_frac
        # PPFD when fully on, used to clamp targets.
        r_max, b_max = self._split(100)
        self._max_ppfd = float(model.predict(r_pwm=r_max, b_pwm=b_max))

    def _split(self, total: int):
        return (
            int(round(total * self.red_frac)),
            int(round(total * (1 - self.red_frac))),
        )

    def invert(self, target_ppfd: float) -> tuple[int, int]:
        target = max(0.0, min(target_ppfd, self._max_ppfd))
        if target <= 0.5:
            return 0, 0
        lo, hi = 0, 100
        for _ in range(8):  # ~0.4 PWM resolution after 8 bisection rounds
            mid = (lo + hi) // 2
            r, b = self._split(mid)
            p = float(self.model.predict(r_pwm=r, b_pwm=b))
            if p < target:
                lo = mid
            else:
                hi = mid
        return self._split(hi)


# ---- env helpers ---------------------------------------------------------

def load_env_kwargs(env_id: str = "GreenLightEnv") -> dict:
    cfg_path = os.path.join(GLGYM_ROOT, "gl_gym", "configs", "envs", f"{env_id}.yml")
    with open(cfg_path) as f:
        params = _yaml.load(f, Loader=_yaml.FullLoader)
    env_kwargs = params[env_id]
    weather_kwargs = env_kwargs.pop("weather_repository_kwargs")
    env_kwargs["weather_repository"] = WeatherRepository(
        weather_data_dir=weather_kwargs["weather_data_dir"],
        load_weather_data_fn=eval(weather_kwargs["load_weather_data_fn"]),
    )
    env_kwargs.pop("eval_scenarios", None)
    env_kwargs["normalize_actions"] = False
    return env_kwargs


def load_rb_params() -> dict:
    cfg_path = os.path.join(GLGYM_ROOT, "configs", "agents", "rule_based.yml")
    with open(cfg_path) as f:
        params = _yaml.load(f, Loader=_yaml.FullLoader)
    return params["GreenLightEnv"]


def make_env(season_days: int = 30) -> GreenLightEnv:
    kw = deepcopy(load_env_kwargs())
    kw["season_length"] = season_days
    return GreenLightEnv(**kw)


def build_step_context(env) -> StepContext:
    return StepContext(
        t=env.timestep, dt=env.dt, Np=env.Np,
        x_prev=env.x_prev, x=env.x, u=env.u, p=env.p,
        d=env.weather_data,
        hour_of_day=env.hour_of_day, day_of_year=env.day_of_year,
    )


def override_indoor_from_sensor(env, sensor: dict):
    """In-place: replace the indoor portion of env.x with the live sensor read.
    Mirrors `seed_env_from_snapshot` but applied every tick to keep sim
    indoor state pinned to reality."""
    t = sensor.get("T_air")
    rh = sensor.get("RH")
    co2_ppm = sensor.get("CO2_ppm")
    x = np.copy(env.x)
    if t is not None:
        x[2] = t           # tAir
        x[3] = t           # tTop
        x[4] = t + 4.0     # tCan
    if t is not None and rh is not None:
        x[15] = (rh / 100.0) * satVp(t)   # vpAir
        x[16] = x[15]
    if co2_ppm is not None and t is not None:
        co2_dens_kgm3 = co2ppm2dens(t, co2_ppm)
        x[0] = co2_dens_kgm3 * 1e6
        x[1] = x[0]
    env.x = x


# ---- main loop -----------------------------------------------------------

_stop = False
def _on_sigint(signum, frame):
    global _stop
    _stop = True
    print("\n[ctrl-c] stopping at next tick")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tick-sec", type=float, default=15.0,
                    help="wall-clock seconds between ticks (default 15)")
    ap.add_argument("--max-ticks", type=int, default=240,
                    help="stop after N ticks (default 240 = 1 sim day at default speedup)")
    ap.add_argument("--max-ppfd", type=float, default=MAX_PPFD,
                    help="PPFD value that uLamp=1.0 maps to (default 60)")
    ap.add_argument("--red-frac", type=float, default=RB_RED_FRAC,
                    help="Red/(Red+Blue) PWM fraction (default 0.8 = 4:1)")
    ap.add_argument("--measure", action="store_true",
                    help="Trigger spectrometer each tick (slower, more data)")
    ap.add_argument("--csv", type=str, default=None,
                    help="Output CSV path (default: demo/output/closed_loop_<ts>.csv)")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_sigint)

    out_dir = os.path.join(GLGYM_ROOT, "demo", "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = args.csv or os.path.join(
        out_dir, f"closed_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    print(f"[boot] loading Govee model @ z={SENSOR_Z_CM} cm")
    govee_model = GoveePWMtoPPFDModel(
        position=GoveePosition(x_cm=0.0, y_cm=0.0, z_cm=SENSOR_Z_CM)
    ).load()
    inverter = GoveeInverter(govee_model, red_frac=args.red_frac)
    print(f"        model max-PPFD @ R:B={args.red_frac}/{1-args.red_frac:.2f}: "
          f"{inverter._max_ppfd:.1f} µmol/m²/s")
    print(f"        target ceiling: {args.max_ppfd:.1f}")

    print("[boot] building GL-Gym env (Amsterdam 2010, day 90)")
    env = make_env(season_days=30)
    env.reset(seed=0, options={"scenario": SCENARIO})
    rb = RuleBasedController(**load_rb_params())

    print(f"[boot] tick={args.tick_sec}s wall, env.dt={env.dt}s sim "
          f"→ speedup ~{env.dt/args.tick_sec:.0f}×")
    print(f"[boot] max ticks: {args.max_ticks} "
          f"(~{args.max_ticks*env.dt/3600:.1f} sim hours = "
          f"{args.max_ticks*args.tick_sec/60:.1f} wall min)")
    print(f"[boot] log: {csv_path}")

    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "wall_iso", "tick", "sim_hour_of_day", "sim_day_of_year",
        "sensor_T", "sensor_RH", "sensor_CO2",
        "u_lamp", "target_ppfd", "pwm_r", "pwm_b",
        "predicted_ppfd", "measured_ppfd",
    ])
    f.flush()

    tick = 0
    try:
        while not _stop and tick < args.max_ticks:
            tick += 1
            t0 = time.monotonic()
            wall_iso = datetime.now().isoformat(timespec="seconds")

            # 1) sensor → override
            sensor = read_live_sensor() or {}
            override_indoor_from_sensor(env, sensor)

            # 2) controller
            ctx = build_step_context(env)
            u_full = np.asarray(rb.predict(ctx), dtype=float)
            u_lamp = float(u_full[4])

            # 3) target → PWM
            target = max(0.0, min(args.max_ppfd, u_lamp * args.max_ppfd))
            pwm_r, pwm_b = inverter.invert(target)
            pred = float(govee_model.predict(r_pwm=pwm_r, b_pwm=pwm_b)) if (pwm_r or pwm_b) else 0.0

            # 4) dispatch
            dispatch_pwm(pwm_r, pwm_b)

            # 5) optional measure
            measured = trigger_measure(1) if args.measure else None

            # 6) advance sim by one env.dt
            try:
                env.step(u_full)
            except Exception as e:
                print(f"[warn] env.step failed: {e}")

            row = [
                wall_iso, tick, ctx.hour_of_day, ctx.day_of_year,
                sensor.get("T_air"), sensor.get("RH"), sensor.get("CO2_ppm"),
                round(u_lamp, 4), round(target, 2), pwm_r, pwm_b,
                round(pred, 2), measured,
            ]
            w.writerow(row); f.flush()

            print(
                f"[t{tick:03d} {wall_iso}] "
                f"sim {ctx.day_of_year:.2f}d {ctx.hour_of_day:5.2f}h | "
                f"T={sensor.get('T_air','—'):>4} RH={sensor.get('RH','—'):>4} | "
                f"u={u_lamp:.2f} → tgt={target:5.1f} | "
                f"PWM=({pwm_r:3d},{pwm_b:3d}) pred={pred:5.1f}"
                + (f" meas={measured:5.1f}" if measured is not None else "")
            )

            # 7) sleep
            elapsed = time.monotonic() - t0
            if elapsed < args.tick_sec:
                time.sleep(args.tick_sec - elapsed)
    finally:
        f.close()
        # leave lights at full white (per user preference, charging mode)
        try:
            _api_post("/api/led/full_white", {}, timeout=10.0)
            print("[exit] lights → full white")
        except Exception as e:
            print(f"[exit] full-white failed: {e}")
        print(f"[done] {tick} ticks. log: {csv_path}")


if __name__ == "__main__":
    main()
