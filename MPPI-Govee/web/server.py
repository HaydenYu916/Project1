"""Govee MPPI demo dashboard — FastAPI server.

Runs on the Pi, served over tailscale. UI lets viewers slide R/B PWM
and see live spectrometer feedback + model-predicted PPFD.

Layout (per /api/* below):
  GET  /                    → static index.html (single-page dashboard)
  GET  /api/state           → live LED + last measurement snapshot
  POST /api/led             → set (pwm_r, pwm_b); applied via Govee mix
  POST /api/measure         → trigger spectrometer once; also returns the
                              PWM-model's PPFD prediction at current state
  GET  /api/mppi            → placeholder (filled later)

Sensor placement (x_cm, y_cm, z_cm) is fixed per current memory entry
(0, 0, 6); both models were trained at the center-only z grid.

Bind to 0.0.0.0:8001 so any tailscale peer (e.g. Mac) can reach it; do
NOT use 8000 — that's already taken by the Riotee gateway.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import serial  # noqa: F401 (used indirectly by spec lib)
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SRC = HERE.parent / "src"
LED_GOVEE_SRC = PROJECT_ROOT / "Tool" / "LED_Govee" / "src"
SPEC_DIR = PROJECT_ROOT / "Tool" / "Spectrometer"
for p in (str(SRC), str(LED_GOVEE_SRC), str(SPEC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from govee_pwmtoppfd_model import GoveePWMtoPPFDModel, GoveePosition  # noqa: E402
import govee_controller  # type: ignore  # noqa: E402
from lib import (  # type: ignore  # noqa: E402
    complete_spectrum_measurement,
    initialize_spectrometer_session,
)

# ----- runtime state -------------------------------------------------------

POSITION = GoveePosition(x_cm=0.0, y_cm=0.0, z_cm=6.0)
SPEC_PORT_DEFAULT = "/dev/ttyACM1"
TRANSPORT_DEFAULT = "ha"  # "ble" or "ha" — HA is more reliable across writes


class LedSet(BaseModel):
    pwm_r: float
    pwm_b: float


class MeasureRequest(BaseModel):
    trigger_count: int = 1


_state_lock = threading.Lock()
_serial_lock = threading.Lock()
_state: dict = {
    "pwm_r": 0,
    "pwm_b": 0,
    "transport": TRANSPORT_DEFAULT,
    "position_cm": {"x": POSITION.x_cm, "y": POSITION.y_cm, "z": POSITION.z_cm},
    "predicted_ppfd": None,
    "last_measured_ppfd": None,
    "last_measured_at": None,
    "last_spec_file": None,
    "last_dispatch_at": None,
    "last_dispatch_latency_ms": None,
    "vcap_v": None,
    "vcap_at": None,
}

_pwm_model: Optional[GoveePWMtoPPFDModel] = None
_spec_ser = None


def _predict_ppfd(pwm_r: float, pwm_b: float) -> Optional[float]:
    if _pwm_model is None:
        return None
    return float(max(0.0, _pwm_model.predict(r_pwm=pwm_r, b_pwm=pwm_b)))


def _open_spec(port: str = SPEC_PORT_DEFAULT):
    global _spec_ser
    import serial as _ser
    _spec_ser = _ser.Serial(port, baudrate=115200, timeout=1)
    initialize_spectrometer_session(_spec_ser)
    return _spec_ser


def _close_spec():
    global _spec_ser
    if _spec_ser is not None:
        try:
            _spec_ser.close()
        except Exception:
            pass
        _spec_ser = None


def _vcap_from_riotee_csv() -> tuple[Optional[float], Optional[str]]:
    path = "/home/pi/Desktop/Project1/Tool/Sensor_riotee_server/logs/riotee_data_all.csv"
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            sz = f.tell()
            f.seek(max(0, sz - 4096))
            tail = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return None, None
    for line in reversed([ln for ln in tail.splitlines() if ln and not ln.startswith("#")]):
        parts = line.split(",")
        if len(parts) >= 8:
            try:
                v = float(parts[7])
                return v, parts[1]
            except ValueError:
                continue
    return None, None


def _refresh_vcap():
    v, ts = _vcap_from_riotee_csv()
    with _state_lock:
        _state["vcap_v"] = v
        _state["vcap_at"] = ts


def _dispatch_pwm(pwm_r: int, pwm_b: int, transport: str) -> dict:
    info: dict = {"transport": transport}
    if transport == "ha":
        r = govee_controller.set_led_mix_ha(pwm_r, pwm_b, timeout_sec=10.0)
        info["latency_ms"] = r.get("latency_ms")
    else:
        govee_controller.set_led_mix(pwm_r, pwm_b)
        info["latency_ms"] = None
    return info


# ----- spectrometer helpers (extract PPFD) ---------------------------------

PPFD_KEYS = (
    "PPFD(umol/㎡/s)",
    "PPFD(umol/m2/s)",
    "PPFD(umol/m²/s)",
)


def _extract_ppfd(parsed) -> Optional[float]:
    if isinstance(parsed, dict):
        for k in PPFD_KEYS:
            if k in parsed:
                try:
                    return float(parsed[k])
                except (TypeError, ValueError):
                    pass
    return None


def _extract_ppfd_from_csv(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    full = path
    if not os.path.isabs(full):
        # spec lib writes archive/.../raw_*.csv relative to cwd
        full = str(HERE / full)
    if not os.path.exists(full):
        full = path  # fall back; let it fail
    try:
        with open(full, encoding="utf-8") as f:
            for line in f:
                if line.startswith("PPFD("):
                    parts = line.rstrip("\n").split(",")
                    if len(parts) >= 2:
                        try:
                            return float(parts[1])
                        except ValueError:
                            return None
    except OSError:
        return None
    return None


def _trigger_spec_once() -> tuple[Optional[float], Optional[str]]:
    if _spec_ser is None:
        return None, None
    with _serial_lock:
        try:
            old_cwd = os.getcwd()
            os.chdir(str(HERE))
            try:
                result = complete_spectrum_measurement(_spec_ser)
            finally:
                os.chdir(old_cwd)
        except Exception as exc:
            print(f"⚠️ spec failure: {exc}")
            return None, None
    if not isinstance(result, tuple) or not result:
        return None, None
    spec_result = result[0]
    if not isinstance(spec_result, tuple) or len(spec_result) < 3:
        return None, None
    spec_file = spec_result[1] if isinstance(spec_result[1], str) else None
    parsed = spec_result[2]
    ppfd = _extract_ppfd(parsed)
    if ppfd is None:
        ppfd = _extract_ppfd_from_csv(spec_file)
    return ppfd, spec_file


# ----- FastAPI app ---------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pwm_model
    print("🌿 boot: loading PWM→PPFD model")
    _pwm_model = GoveePWMtoPPFDModel(position=POSITION).load()
    print(f"   model features={_pwm_model._meta and _pwm_model._meta.get('feature_columns')}")

    print(f"🔬 boot: opening spectrometer @ {SPEC_PORT_DEFAULT}")
    try:
        _open_spec()
    except Exception as exc:
        print(f"   ⚠️ spec open failed: {exc}; /api/measure will be unavailable until reset")

    if TRANSPORT_DEFAULT == "ble":
        print("🔌 boot: connecting Govee over BLE")
        try:
            govee_controller.connect()
        except Exception as exc:
            print(f"   ⚠️ BLE connect failed: {exc}; falling back to HA at request time")
    else:
        print("☁️ boot: Govee transport=HA (no persistent connect)")

    # Background vcap refresh
    stop_evt = threading.Event()

    def _vcap_loop():
        while not stop_evt.is_set():
            try:
                _refresh_vcap()
            except Exception:
                pass
            stop_evt.wait(5.0)

    th = threading.Thread(target=_vcap_loop, daemon=True)
    th.start()

    yield

    stop_evt.set()
    th.join(timeout=2)
    _close_spec()
    if TRANSPORT_DEFAULT == "ble":
        try:
            govee_controller.disconnect()
        except Exception:
            pass


app = FastAPI(title="Govee MPPI Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


@app.get("/")
def index():
    from fastapi.responses import FileResponse
    return FileResponse(str(HERE / "static" / "index.html"))


@app.get("/api/state")
def api_state():
    with _state_lock:
        return JSONResponse(_state.copy())


@app.post("/api/led")
def api_led(req: LedSet):
    pwm_r = int(np.clip(round(float(req.pwm_r)), 0, 100))
    pwm_b = int(np.clip(round(float(req.pwm_b)), 0, 100))
    try:
        info = _dispatch_pwm(pwm_r, pwm_b, _state["transport"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"dispatch failed: {exc}")
    pred = _predict_ppfd(pwm_r, pwm_b)
    with _state_lock:
        _state["pwm_r"] = pwm_r
        _state["pwm_b"] = pwm_b
        _state["predicted_ppfd"] = pred
        _state["last_dispatch_at"] = datetime.now().isoformat(timespec="seconds")
        _state["last_dispatch_latency_ms"] = info.get("latency_ms")
        snap = _state.copy()
    return JSONResponse(snap)


@app.post("/api/measure")
def api_measure(req: MeasureRequest):
    if _spec_ser is None:
        try:
            _open_spec()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"spectrometer not available: {exc}")
    n = max(1, min(5, int(req.trigger_count)))
    ppfds = []
    files = []
    for _ in range(n):
        p, f = _trigger_spec_once()
        if p is not None:
            ppfds.append(p)
            files.append(f)
    if not ppfds:
        raise HTTPException(status_code=500, detail="no PPFD parsed from spec")
    measured = float(np.mean(ppfds))
    with _state_lock:
        pred = _predict_ppfd(_state["pwm_r"], _state["pwm_b"])
        _state["last_measured_ppfd"] = measured
        _state["last_measured_at"] = datetime.now().isoformat(timespec="seconds")
        _state["last_spec_file"] = files[-1] if files else None
        _state["predicted_ppfd"] = pred
        snap = _state.copy()
    return JSONResponse({
        "measured_ppfd": measured,
        "samples": ppfds,
        "predicted_ppfd": pred,
        "delta_abs": (measured - pred) if pred is not None else None,
        "delta_pct": ((measured - pred) / pred * 100.0) if pred else None,
        "state": snap,
    })


@app.get("/api/mppi")
def api_mppi():
    """Placeholder for the future MPPI panel."""
    return JSONResponse({
        "status": "not_implemented",
        "note": "Connect this to LEDMPPIController.solve() in a follow-up.",
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
    )
