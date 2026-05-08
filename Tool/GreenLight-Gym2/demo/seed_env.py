"""
Seed a GreenLightEnv instance with real-chamber sensor snapshot.

Call AFTER env.reset() — it overwrites the indoor portion of the initial
state vector x. Crop / external-weather states are left untouched (the
chamber cannot observe them, so the model's defaults are used).

State indices (from gl_gym/environments/utils.py:init_state):
    x[0]  co2Air   [mg m^-3]    (we receive ppm; convert)
    x[1]  co2Top   = co2Air
    x[2]  tAir     [°C]
    x[3]  tTop     = tAir
    x[4]  tCan     = tAir + 4   (canopy ~ air + 4)
    x[15] vpAir    [Pa]         (we receive RH%; convert via satVp(tAir))
    x[16] vpTop    = vpAir
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from gl_gym.environments.utils import co2ppm2dens, satVp


def seed_env_from_snapshot(env, snapshot: Dict[str, float]) -> None:
    """In-place: overwrite env.x indoor entries with sensor snapshot."""
    if not hasattr(env, "x") or env.x is None:
        raise RuntimeError("env not reset; call env.reset() before seeding.")

    t_air = float(snapshot["t_air"])
    rh = float(snapshot["rh"])
    co2_ppm = float(snapshot["co2_ppm"])

    co2_dens_kgm3 = co2ppm2dens(t_air, co2_ppm)
    co2_dens_mgm3 = co2_dens_kgm3 * 1e6
    vp_air = (rh / 100.0) * satVp(t_air)

    x = np.copy(env.x)
    x[0] = co2_dens_mgm3
    x[1] = co2_dens_mgm3
    x[2] = t_air
    x[3] = t_air
    x[4] = t_air + 4.0
    x[5] = t_air      # tCovIn
    x[6] = t_air      # tCovE
    x[7] = t_air      # tThScr
    x[8] = t_air      # tFlr
    x[9] = t_air      # tPipe
    x[15] = vp_air
    x[16] = vp_air
    x[17] = t_air     # tLamp
    x[20] = t_air     # tBlScr
    x[21] = t_air + 4.0  # tCan24

    env.x = x
    env.x_prev = np.copy(x)
    env.obs = env._get_obs()
