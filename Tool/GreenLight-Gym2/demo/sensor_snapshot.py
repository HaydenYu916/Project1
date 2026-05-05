"""
Read a snapshot of the real chamber state for use as GL-Gym initial condition.

Resolution order:
  1. Live sensors via Project1.MPPI-Govee.src.sensor_reader.DemoSensorReader
     (only works when riotee_data_all.csv & co2 CSV exist on this machine).
  2. Last row of Project1/MPPI-Govee/data/ALL_Data.csv (chamber measurement
     log shipped with the repo) — used as a stand-in when hardware is not
     connected to the server.
  3. Hardcoded defaults.

Returned dict:
    { "co2_ppm": float, "t_air": float, "rh": float, "source": str }
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# After the layout flip GL-Gym lives at Project1/Tool/GreenLight-Gym2/.
# Project1 root is two levels up from REPO_ROOT.
PROJECT1_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))
PROJECT1_SRC = os.path.join(PROJECT1_ROOT, "MPPI-Govee", "src")
ALL_DATA_CSV = os.path.join(PROJECT1_ROOT, "MPPI-Govee", "data", "ALL_Data.csv")
RIOTEE_CSV = os.path.join(
    PROJECT1_ROOT, "Tool", "Sensor_riotee_server", "logs", "riotee_data_all.csv"
)

DEFAULT_SNAPSHOT = {"co2_ppm": 600.0, "t_air": 22.0, "rh": 65.0, "source": "default"}


def _try_live_sensor(co2_csv: Optional[str] = None) -> Optional[Dict[str, float]]:
    if not os.path.exists(RIOTEE_CSV):
        return None
    if PROJECT1_SRC not in sys.path:
        sys.path.insert(0, PROJECT1_SRC)
    try:
        from sensor_reader import DemoSensorReader  # type: ignore
    except Exception:
        return None
    try:
        reader = DemoSensorReader(riotee_data_path=RIOTEE_CSV, co2_data_path=co2_csv)
        co2, t_air, rh, _pipe = reader.read_indoor_climate()
        return {"co2_ppm": float(co2), "t_air": float(t_air), "rh": float(rh), "source": "live_sensor"}
    except Exception as e:
        print(f"[sensor_snapshot] live sensor read failed: {e}")
        return None


def _try_all_data_csv() -> Optional[Dict[str, float]]:
    if not os.path.exists(ALL_DATA_CSV):
        return None
    try:
        df = pd.read_csv(ALL_DATA_CSV)
        if df.empty:
            return None
        last = df.iloc[-1]
        co2 = float(last.get("CO2", DEFAULT_SNAPSHOT["co2_ppm"]))
        t = float(last.get("T", DEFAULT_SNAPSHOT["t_air"]))
        return {
            "co2_ppm": co2,
            "t_air": t,
            "rh": DEFAULT_SNAPSHOT["rh"],
            "source": f"ALL_Data.csv:row{len(df)-1}",
        }
    except Exception as e:
        print(f"[sensor_snapshot] ALL_Data.csv fallback failed: {e}")
        return None


def read_sensor_snapshot(co2_csv: Optional[str] = None) -> Dict[str, float]:
    snap = _try_live_sensor(co2_csv)
    if snap is not None:
        return snap
    snap = _try_all_data_csv()
    if snap is not None:
        return snap
    return dict(DEFAULT_SNAPSHOT)


if __name__ == "__main__":
    print(read_sensor_snapshot())
