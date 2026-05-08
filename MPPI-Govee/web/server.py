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
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import serial  # noqa: F401 (used indirectly by spec lib)
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SRC = HERE.parent / "src"
LED_GOVEE_SRC = PROJECT_ROOT / "Tool" / "LED_Govee" / "src"
SPEC_DIR = PROJECT_ROOT / "Tool" / "Spectrometer"
GLGYM_DIR = PROJECT_ROOT / "Tool" / "GreenLight-Gym2"
for p in (str(SRC), str(LED_GOVEE_SRC), str(SPEC_DIR), str(GLGYM_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from govee_pwmtoppfd_model import GoveePWMtoPPFDModel, GoveePosition  # noqa: E402
import govee_controller  # type: ignore  # noqa: E402
from lib import (  # type: ignore  # noqa: E402
    complete_spectrum_measurement,
    initialize_spectrometer_session,
)

# ----- runtime state -------------------------------------------------------

POSITION = GoveePosition(x_cm=0.0, y_cm=0.0, z_cm=5.0)
TRANSPORT_DEFAULT = "ble"  # BLE — local, bypasses HA cloud (TLS issues on gaytime)


def _find_spec_port() -> str:
    """Locate the spectrometer's STM32 USB-CDC device, since Linux may
    re-enumerate it as ttyACM1/2/... after a replug."""
    import glob
    for path in sorted(glob.glob("/dev/ttyACM*")):
        try:
            from serial.tools import list_ports
            for info in list_ports.comports():
                if info.device == path and info.vid == 0x0483:  # STMicro
                    return path
        except Exception:
            pass
    # Fallback: assume the legacy default if probing fails.
    return "/dev/ttyACM1"


SPEC_PORT_DEFAULT = _find_spec_port()


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
    "mode": "rb_mix",  # "rb_mix" or "full_white" — UI uses this to render preview
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

# ----- MPPI runner state ---------------------------------------------------

# ----- GL-Gym closed-loop runner state -------------------------------------

_glgym_lock = threading.Lock()
_glgym_state: dict = {
    "running": False,
    "tick": 0,
    "started_at": None,
    "stopped_at": None,
    "params": None,
    "sim_hour": None,
    "sim_day": None,
    "sensor_T": None,
    "sensor_RH": None,
    "u_lamp": None,
    "target_ppfd": None,
    "pwm_r": None,
    "pwm_b": None,
    "predicted_ppfd": None,
    "measured_ppfd": None,
    "last_tick_at": None,
    "last_error": None,
}
_glgym_stop_evt = threading.Event()
_glgym_thread: Optional[threading.Thread] = None


_mppi_lock = threading.Lock()
_mppi_state: dict = {
    "running": False,
    "step": 0,
    "target_ppfd": None,
    "started_at": None,
    "stopped_at": None,
    "params": None,
    "last_u": None,
    "last_pwm_r": None,
    "last_pwm_b": None,
    "last_pred": None,
    "last_measured": None,
    "last_err_abs": None,
    "last_err_pct": None,
    "last_min_cost": None,
    "last_power_w": None,
    "last_step_at": None,
    "last_error": None,
}
_mppi_stop_evt = threading.Event()
_mppi_thread: Optional[threading.Thread] = None
_ws_clients: set = set()
_main_loop: Optional[asyncio.AbstractEventLoop] = None


# Translate common Chinese phrases that bubble up through exception
# strings from src/led.py / src/mppi_v2.py, then strip any remaining
# CJK so the dashboard stays English-only.
_CN_TO_EN = [
    ("加载热力学模型失败", "failed to load thermal model"),
    ("Govee PWMtoPPFD 加载失败", "Govee PWMtoPPFD load failed"),
    ("EnvtoPN 模型不存在", "EnvtoPN model missing"),
    ("Pn 模型不可用", "Pn model unavailable"),
    ("SPtoPPFD 加载失败", "SPtoPPFD load failed"),
    ("MPPI求解失败", "MPPI solve failed"),
    ("热力学模型参数未加载", "thermal-model params not loaded"),
    ("升温", "heating"),
    ("降温", "cooling"),
    ("功率模型未提供", "power model missing"),
    ("读取Riotee数据失败", "failed to read Riotee data"),
    ("读取CO2数据失败", "failed to read CO2 data"),
]
_CJK_RE = re.compile(r"[　-〿一-鿿＀-￯]+")


def _en(text) -> str:
    s = str(text)
    for cn, en in _CN_TO_EN:
        s = s.replace(cn, en)
    s = _CJK_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def _broadcast(event: dict) -> None:
    if _main_loop is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(_async_broadcast(event), _main_loop)
    except RuntimeError:
        pass


async def _async_broadcast(event: dict) -> None:
    dead = []
    for ws in list(_ws_clients):
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


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


RIOTEE_CSV = "/home/pi/Desktop/Project1/Tool/Sensor_riotee_server/logs/riotee_data_all.csv"
RIOTEE_DEVICE = "T6ncwg=="
# CSV column index map (mirrors Tool/Sensor_riotee_server/riotee_data_collector.py)
RIOTEE_IDX = {
    "id": 0, "timestamp": 1, "device_id": 2, "update_type": 3,
    "temperature": 4, "humidity": 5, "a1_raw": 6, "vcap_raw": 7,
    "co2_ppm": 8, "co2_state": 9,
    "sp_415": 10, "sp_445": 11, "sp_480": 12, "sp_515": 13,
    "sp_555": 14, "sp_590": 15, "sp_630": 16, "sp_680": 17,
    "sp_clear": 18, "sp_nir": 19,
    "spectral_gain": 20, "sleep_time": 21,
}
SP_CHANNELS = ["sp_415", "sp_445", "sp_480", "sp_515",
               "sp_555", "sp_590", "sp_630", "sp_680"]


def _read_riotee_tail(nbytes: int = 4096) -> list[str]:
    try:
        with open(RIOTEE_CSV, "rb") as f:
            f.seek(0, 2)
            sz = f.tell()
            f.seek(max(0, sz - nbytes))
            tail = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return []
    return [ln for ln in tail.splitlines() if ln and not ln.startswith("#")]


def _parse_riotee_row(line: str) -> Optional[dict]:
    parts = line.split(",")
    if len(parts) < 22:
        return None
    if parts[RIOTEE_IDX["device_id"]] != RIOTEE_DEVICE:
        return None
    out: dict = {"timestamp": parts[RIOTEE_IDX["timestamp"]]}
    for key in ("temperature", "humidity", "vcap_raw", "co2_ppm",
                *SP_CHANNELS, "sp_clear", "sp_nir"):
        try:
            out[key] = float(parts[RIOTEE_IDX[key]])
        except (ValueError, IndexError):
            out[key] = None
    return out


def _latest_riotee_row() -> Optional[dict]:
    for line in reversed(_read_riotee_tail(4096)):
        row = _parse_riotee_row(line)
        if row is not None:
            return row
    return None


def _riotee_history(n: int = 120) -> list[dict]:
    # ~22 cols * ~120 chars per line ≈ 2.5 KB/row; pull plenty of tail.
    lines = _read_riotee_tail(max(8192, n * 200))
    rows: list[dict] = []
    for line in reversed(lines):
        row = _parse_riotee_row(line)
        if row is not None:
            rows.append(row)
            if len(rows) >= n:
                break
    rows.reverse()
    return rows


def _vcap_from_riotee_csv() -> tuple[Optional[float], Optional[str]]:
    row = _latest_riotee_row()
    if row is None:
        return None, None
    return row.get("vcap_raw"), row.get("timestamp")


def _refresh_vcap():
    v, ts = _vcap_from_riotee_csv()
    with _state_lock:
        _state["vcap_v"] = v
        _state["vcap_at"] = ts


def _dispatch_pwm(pwm_r: int, pwm_b: int, transport: str) -> dict:
    info: dict = {"transport": transport}
    if transport == "ha":
        # 5s convergence wait — HA cloud read-back is often slow (~3-8s),
        # but POST itself lands fast; if convergence times out we still
        # treat it as a soft success because the bulb usually responds.
        # Retry once on transient SSL / network errors (HA cloud occasionally
        # rejects handshakes during load).
        try:
            r = govee_controller.set_led_mix_ha(pwm_r, pwm_b, timeout_sec=5.0)
        except TimeoutError:
            raise  # bubble; caller treats as soft warning
        except Exception as exc:
            msg = str(exc).lower()
            transient = any(s in msg for s in (
                "_ssl.c", "handshake", "ssl", "timed out", "connection reset",
                "temporarily unavailable", "bad gateway", "504", "502",
            ))
            if not transient:
                raise
            print(f"⚠️ HA transient ({exc}); retrying once after 1s")
            time.sleep(1.0)
            r = govee_controller.set_led_mix_ha(pwm_r, pwm_b, timeout_sec=5.0)
        info["latency_ms"] = r.get("latency_ms")
        return info
    # BLE: write may fail if the link dropped silently or the cached client
    # has no services. Try once, on failure rebuild the BLE client and retry.
    # Reconnect must wait — the peripheral needs ~1.5–2 s after a disconnect
    # before it accepts a new GATT connection, otherwise the retry races and
    # fails again ("failed to discover services, device disconnected").
    try:
        govee_controller.set_led_mix(pwm_r, pwm_b)
    except Exception as exc:
        msg = str(exc).lower()
        if any(s in msg for s in ("service discovery", "not connected",
                                  "ble write failed", "disconnected",
                                  "dbus", "reconnect", "no services",
                                  "discover services")):
            print(f"⚠️ BLE dispatch error ({exc}); cooling 2s + reconnecting")
            try:
                try: govee_controller.disconnect()
                except Exception: pass
                time.sleep(2.0)
                govee_controller.connect()
                govee_controller.set_led_mix(pwm_r, pwm_b)
            except Exception as exc2:
                # Don't 500 on transient BLE flakiness (e.g. user moving the
                # bar disturbs the link). Surface as a soft warning so the UI
                # banner shows once but the slider keeps responding.
                info["warn"] = f"BLE write transient: {exc2}"
                print(f"⚠️ BLE retry still failed: {exc2}")
        else:
            raise
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
    global _pwm_model, _main_loop
    _main_loop = asyncio.get_running_loop()
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
    if _mppi_state["running"]:
        raise HTTPException(status_code=409, detail="MPPI running; stop it first")
    if _glgym_state["running"]:
        raise HTTPException(status_code=409, detail="GLGym running; stop it first")
    pwm_r = int(np.clip(round(float(req.pwm_r)), 0, 100))
    pwm_b = int(np.clip(round(float(req.pwm_b)), 0, 100))
    warn: Optional[str] = None
    info: dict = {}
    try:
        info = _dispatch_pwm(pwm_r, pwm_b, _state["transport"])
        if info.get("warn"):
            warn = info["warn"]
    except TimeoutError as exc:
        # HA accepted the POST but didn't observe the bulb's state update
        # within the convergence window. Common when the Govee cloud
        # integration has lost touch with the bulb — the command was still
        # sent, so don't fail the request; surface as a soft warning.
        warn = f"HA convergence timeout (command sent): {exc}"
        print(f"⚠️ {warn}")
    except Exception as exc:
        # Treat transient network / SSL errors as soft warnings too — the
        # POST may have landed even if our read failed. UI keeps responding.
        msg = str(exc).lower()
        transient = any(s in msg for s in (
            "_ssl.c", "handshake", "ssl", "timed out", "connection reset",
            "temporarily unavailable", "bad gateway", "504", "502",
            "name or service not known", "network is unreachable",
        ))
        if transient:
            warn = f"HA network error (likely sent): {exc}"
            print(f"⚠️ {warn}")
        else:
            raise HTTPException(status_code=500, detail=f"dispatch failed: {exc}")
    pred = _predict_ppfd(pwm_r, pwm_b)
    with _state_lock:
        _state["pwm_r"] = pwm_r
        _state["pwm_b"] = pwm_b
        _state["mode"] = "rb_mix"
        _state["predicted_ppfd"] = pred
        _state["last_dispatch_at"] = datetime.now().isoformat(timespec="seconds")
        _state["last_dispatch_latency_ms"] = info.get("latency_ms")
        snap = _state.copy()
    if warn:
        snap["warn"] = warn
    return JSONResponse(snap)


@app.post("/api/led/full_white")
def api_led_full_white():
    """Set lights to full RGB(255,255,255) at max brightness (charging mode).
    Uses HA's set_led_full_white_ha or BLE's set_led_full_white depending on
    current transport. Soft-warns on HA convergence timeout."""
    if _mppi_state["running"]:
        raise HTTPException(status_code=409, detail="MPPI running; stop it first")
    if _glgym_state["running"]:
        raise HTTPException(status_code=409, detail="GLGym running; stop it first")
    transport = _state["transport"]
    warn: Optional[str] = None
    latency: Optional[float] = None
    try:
        if transport == "ha":
            r = govee_controller.set_led_full_white_ha(timeout_sec=5.0)
            latency = r.get("latency_ms")
        else:
            govee_controller.set_led_full_white()
    except TimeoutError as exc:
        warn = f"HA convergence timeout (command sent): {exc}"
        print(f"⚠️ {warn}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"full_white failed: {exc}")
    with _state_lock:
        # Mark UI state: lights at full white (charging mode). Sliders show
        # 100/100 but UI must paint preview as actual white, not R+B magenta.
        _state["pwm_r"] = 100
        _state["pwm_b"] = 100
        _state["mode"] = "full_white"
        _state["predicted_ppfd"] = _predict_ppfd(100, 100)
        _state["last_dispatch_at"] = datetime.now().isoformat(timespec="seconds")
        _state["last_dispatch_latency_ms"] = latency
        snap = _state.copy()
    if warn:
        snap["warn"] = warn
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


@app.get("/api/sensor")
def api_sensor():
    row = _latest_riotee_row()
    if row is None:
        raise HTTPException(status_code=503, detail="no recent riotee row")
    return JSONResponse(row)


@app.get("/api/sensor/history")
def api_sensor_history(n: int = 120):
    n = max(1, min(600, int(n)))
    return JSONResponse({"rows": _riotee_history(n)})


# ----- GL-Gym comparison runner -------------------------------------------
#
# Runs N envs in lockstep, each driven by a different lamp-control strategy:
#   rule_based  — GL-Gym's stock RuleBasedController (lamp depends on time +
#                 weather + indoor temp; "Dutch greenhouse" baseline)
#   schedule    — lamp on between hours_on..hours_off, ignoring weather
#   fixed_dli   — lamp on at full power until daily DLI target hit, then off
#   mppi        — supervisor: lamp on at night/dim, off when sun is bright
# All controllers share the SAME RuleBasedController for non-lamp actions
# (heating, CO2, vent, screens) so the comparison isolates lamp strategy.

_compare_lock = threading.Lock()
_compare_state: dict = {
    "running": False, "tick": 0,
    "started_at": None, "stopped_at": None,
    "params": None, "controllers": [],
    "summary": {},   # filled on stop
    "last_error": None,
}
_compare_stop_evt = threading.Event()
_compare_thread: Optional[threading.Thread] = None


def _make_lamp_controllers(rb_params: dict) -> dict:
    """Return {name: controller_callable(ctx) → u}.

    All controllers use GL-Gym's RuleBasedController for the non-lamp actions
    (heating/CO2/vent/screens) so the comparison isolates the lamp strategy.

    All controllers gated to a 14-hour photoperiod (08:00–22:00) and track
    a constant canopy PPFD target of 450 µmol/m²/s while the lamp is on.

    rule_based — open-loop: lamp at u = PPFD_TARGET / LAMP_PPFD inside the
                 window (ignores sun completely; over-shoots when sun bright).
    glgym_rb   — GL-Gym RuleBasedController (its own logic), with lamp gated
                 to the 14 h window.
    lightfarm  — MPPI: tracks total canopy PPFD = 450 by combining sun + lamp;
                 closed-form would be u = max(0,(450-sun)/LAMP_PPFD), but we
                 sample continuous u sequences and softmax-weight them.
    """
    from gl_gym.components.rule_based import RuleBasedController  # noqa: WPS433
    base_rb = RuleBasedController(**rb_params)

    PHOTOPERIOD_ON, PHOTOPERIOD_OFF = 8, 22   # 14-hour window
    LAMP_PPFD     = 500.0   # µmol/m²/s when u = 1 (max canopy lamp PPFD)
    PPFD_TARGET   = 450.0   # µmol/m²/s constant tracking target
    U_RULE_BASED  = min(1.0, PPFD_TARGET / LAMP_PPFD)   # = 0.9
    HORIZON_H     = 4       # MPPI plan horizon — 4 sim hours is plenty for tracking

    def _in_window(hour: float) -> bool:
        return PHOTOPERIOD_ON <= hour < PHOTOPERIOD_OFF

    # ---- rule_based: constant u inside window (no sun awareness) ----
    def rule_based_ctl(ctx):
        u = base_rb.predict(ctx)
        u[4] = U_RULE_BASED if _in_window(ctx.hour_of_day) else 0.0
        return u

    # ---- glgym_rb: GL-Gym RB but gated to the same 14 h window ----
    def glgym_rb_ctl(ctx):
        u = base_rb.predict(ctx)
        if not _in_window(ctx.hour_of_day):
            u[4] = 0.0
        return u

    # ---- lightfarm: REAL MPPI from MPPI-Govee/src/mppi_v2.LEDMPPIController ----
    # The user's chamber-scale MPPI: Pn-aware tracking with thermal/power
    # constraints. It outputs target PPFD (in PPFD space, not [0,1]); we map
    # back to GL-Gym's u_lamp = clip(target_ppfd / LAMP_PPFD, 0, 1).
    # Plant uses live env.x[2] (sim air temperature) as plant temp each tick.
    _lf_mppi = None
    try:
        from mppi_v2 import LEDPlant, LEDMPPIController  # noqa: WPS433
        thermal_dir = str(SRC.parent / "Thermal" / "exported_models")

        class _StubPowerModel:
            def predict(self, *, total_pwm, key=None):
                return float(total_pwm) * 0.12

        _lf_plant = LEDPlant(
            target_ppfd=PPFD_TARGET,
            r_b_ratio=0.8,
            use_pn_model=True,
            use_sp_to_ppfd=False,
            use_govee_pwm_model=False,   # sim mode — no Govee PWM inversion
            use_thermal_model=False,     # bypass chamber thermal
            sim_thermal={                # use greenhouse-style sim model
                "t_outdoor": 22.0,
                "delta_t_at_u1": 18.0,
                "alpha": 0.5,
                "lamp_ppfd_max": LAMP_PPFD,
            },
            model_dir=thermal_dir,
            power_model=_StubPowerModel(),
            x_cm=0.0, y_cm=0.0, z_cm=5.0,
            demo_gain=1.0,
        )
        _lf_plant.ambient_temp = 22.0
        _lf_mppi = LEDMPPIController(
            _lf_plant,
            horizon=5,                  # short — only first action used
            num_samples=120,            # tame from 1000 → keep solve fast
            dt=900.0,                   # match env.dt
            temperature=0.5,
        )
        # No Q_ref: don't track target, let MPPI trade Pn vs power directly.
        # R_power=5 (cranked up) → MPPI aggressively cuts u to save energy
        # whenever Pn is near saturation, so its u clearly diverges from
        # rule_based's flat 0.9 → T evolution differs visibly.
        _lf_mppi.set_weights(Q_photo=1.0, R_du=0.05, R_power=5.0, Q_ref=0.0)
        _lf_mppi.set_constraints(u_min=0.0, u_max=PPFD_TARGET * 1.5,
                                 temp_min=0.0, temp_max=80.0)
        _lf_mppi.set_ppfd_train_range(ppfd_min=0.0, ppfd_max=600.0, R_oob=0.0)
        # Center exploration in the Pn-saturation knee zone for the chamber
        # Pn model — MPPI then trades small Pn loss for energy savings.
        _lf_mppi.set_target_ppfd(400.0)
        _lf_mppi.set_mppi_params(u_std=60.0)
    except Exception as exc:
        print(f"⚠️ LightFarm MPPI init failed: {exc}; falling back to rule_based")

    def lightfarm_ctl(ctx):
        u = base_rb.predict(ctx)
        if not _in_window(ctx.hour_of_day):
            u[4] = 0.0
            return u
        if _lf_mppi is None:
            u[4] = U_RULE_BASED
            return u
        try:
            current_temp = float(ctx.x[2])
            optimal_ppfd, _seq, _ok, _cost, _w = _lf_mppi.solve(current_temp)
            u[4] = float(np.clip(optimal_ppfd / LAMP_PPFD, 0.0, 1.0))
        except Exception as exc:
            print(f"[lightfarm] solve failed: {exc}; using rule_based action")
            u[4] = U_RULE_BASED
        return u

    return {
        "rule_based": rule_based_ctl,
        "glgym_rb":   glgym_rb_ctl,
        "lightfarm":  lightfarm_ctl,
    }


class GLGymCompareStart(BaseModel):
    controllers: list[str] = ["rule_based", "glgym_rb", "lightfarm"]
    days: float = 1.0          # sim days per controller
    start_day: int = 200
    start_hour: float = 0.0    # comparisons usually run a full day from midnight
    location: str = "Amsterdam"
    growth_year: int = 2010
    tick_sec: float = 0.5      # wall-clock per tick (sim_only — fast OK)
    steps_per_tick: int = 4    # env.step calls per tick (15min × 4 = 1h sim/tick)
    indoor: bool = False       # if true: zero outdoor PAR (chamber-style, no sun)
    lamp_par_scale: float = 1.0  # multiply lamp PAR (p[172]) — <1 keeps canopy in linear region
    no_climate_control: bool = False  # if true: zero heat/vent/CO2/screen actions; only lamp matters
    # Pn evaluation source. "chamber" → user's HistGBR f(T, CO2, R:B, PPFD)
    # from Tool/Model/EnvtoPN; "greenlight" → leaf-level Farquhar P_leaf_net
    # from the GreenLight aux. Both are reported in mg{CO2}/m²/s for charting.
    pn_source: str = "chamber"
    lamp_ppfd_max: float = 500.0   # u_lamp=1.0 → this many µmol/m²/s incident PPFD
    chamber_rb: float = 0.83        # R fraction fed to chamber Pn (rule_based has no R:B)
    # Multiplier applied to chamber Pn so its magnitude is comparable to a
    # canopy-scale flux. The chamber model returns single-leaf Pn; multiplying
    # by ~effective LAI (typical greenhouse tomato 2-4) gives canopy Pn.
    chamber_lai_scale: float = 3.0


# ----- Lazy-loaded chamber Pn pipeline (Tool/Model/EnvtoPN HistGBR) -----------
_chamber_pn_pipeline = None
_chamber_pn_features = None
_chamber_pn_load_lock = threading.Lock()
_ENV_TO_PN_PACKAGE = str(PROJECT_ROOT / "Tool" / "Model" / "EnvtoPN" / "model" / "best_model_package.joblib")


def _load_chamber_pn():
    """Load and cache the user's chamber Pn pipeline (HistGBR over [T,CO2,R:B,PPFD])."""
    global _chamber_pn_pipeline, _chamber_pn_features
    if _chamber_pn_pipeline is not None:
        return _chamber_pn_pipeline, _chamber_pn_features
    with _chamber_pn_load_lock:
        if _chamber_pn_pipeline is not None:
            return _chamber_pn_pipeline, _chamber_pn_features
        if not os.path.exists(_ENV_TO_PN_PACKAGE):
            return None, None
        import joblib  # noqa: WPS433
        pkg = joblib.load(_ENV_TO_PN_PACKAGE)
        pipe = pkg["pipeline"]
        feats = list(pkg["feature_columns"])
        # sklearn 1.3 → 1.5 compat shim (mirrors mppi_v2._init_pn_model:288-307).
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: WPS433
            for _, est in pipe.steps:
                if isinstance(est, HistGradientBoostingRegressor):
                    for attr, default in (
                        ("_preprocessor", None),
                        ("_categorical_features", None),
                        ("_known_categories", None),
                        ("is_categorical_", None),
                    ):
                        if not hasattr(est, attr):
                            setattr(est, attr, default)
        except Exception:
            pass
        _chamber_pn_pipeline = pipe
        _chamber_pn_features = feats
        print(f"[compare] loaded chamber Pn model: features={feats}")
        return pipe, feats


def _compare_run(params: dict) -> None:
    import yaml as _yaml
    from copy import deepcopy
    from gl_gym.environments.greenlight_env import GreenLightEnv  # noqa: WPS433
    from gl_gym.core.types import StepContext  # noqa: WPS433
    from gl_gym.components.weather import WeatherRepository  # noqa: WPS433
    from gl_gym.environments.utils import load_weather_data  # noqa: WPS433

    glgym_root = str(GLGYM_DIR)
    ctl_names = list(params["controllers"]) or ["rule_based"]
    days = float(params["days"])
    start_day = int(params["start_day"])
    start_hour = float(params["start_hour"])
    tick_sec = float(params["tick_sec"])
    steps_per_tick = max(1, int(params["steps_per_tick"]))

    # Pn evaluation knobs (default to chamber HistGBR — matches the user's
    # f(PPFD, R:B, CO2, T) calibration; "greenlight" falls back to the env's
    # leaf-level Farquhar P_leaf_net).
    pn_source = str(params.get("pn_source", "chamber")).lower()
    lamp_ppfd_max = float(params.get("lamp_ppfd_max", 500.0))
    chamber_rb_default = float(params.get("chamber_rb", 0.83))
    chamber_lai_scale = max(0.0, float(params.get("chamber_lai_scale", 3.0)))
    chamber_pipe = chamber_feats = None
    if pn_source == "chamber":
        try:
            chamber_pipe, chamber_feats = _load_chamber_pn()
        except Exception as exc:
            print(f"[compare] chamber Pn load failed, falling back to greenlight: {exc}")
            chamber_pipe = None
        if chamber_pipe is None:
            print("[compare] chamber Pn unavailable → using greenlight P_leaf_net")
            pn_source = "greenlight"
    # Imports for chamber Pn evaluation (only used when pn_source=="chamber").
    try:
        from gl_gym.environments.utils import co2dens2ppm  # noqa: WPS433
    except Exception:
        co2dens2ppm = None  # type: ignore
    try:
        import pandas as _pd  # noqa: WPS433
    except Exception:
        _pd = None  # type: ignore

    try:
        # Load env config + RB params
        cfg = os.path.join(glgym_root, "gl_gym", "configs", "envs", "GreenLightEnv.yml")
        with open(cfg) as f:
            ek = _yaml.load(f, Loader=_yaml.FullLoader)["GreenLightEnv"]
        wkw = ek.pop("weather_repository_kwargs")
        wd = wkw["weather_data_dir"]
        if not os.path.isabs(wd):
            wd = os.path.join(glgym_root, wd)
        ek["weather_repository"] = WeatherRepository(
            weather_data_dir=wd,
            load_weather_data_fn=eval(wkw["load_weather_data_fn"]),
        )
        ek.pop("eval_scenarios", None)
        ek["normalize_actions"] = False
        ek["season_length"] = max(1, int(days) + 1)

        rb_yml = os.path.join(glgym_root, "configs", "agents", "rule_based.yml")
        with open(rb_yml) as f:
            rb_params = _yaml.load(f, Loader=_yaml.FullLoader)["GreenLightEnv"]

        all_ctls = _make_lamp_controllers(rb_params)
        ctls = {n: all_ctls[n] for n in ctl_names if n in all_ctls}
        if not ctls:
            raise ValueError(f"unknown controllers: {ctl_names}")

        indoor = bool(params.get("indoor", False))
        lamp_par_scale = float(params.get("lamp_par_scale", 1.0))
        # One env per controller (fresh state per run, same scenario).
        envs = {}
        for name in ctls:
            e = GreenLightEnv(**deepcopy(ek))
            e.reset(seed=0, options={"scenario": {
                "location": params["location"],
                "growth_year": int(params["growth_year"]),
                "start_day": start_day,
            }})
            if indoor:
                try:
                    e.weather_data[:, 0] = 0.0    # sun → 0
                    e.weather_data[:, 7] = 0.0    # daily sun sum → 0
                    e.weather_data[:, 1] = 22.0   # outdoor air T → 22°C
                    e.weather_data[:, 5] = 22.0   # sky T → 22°C (same as room)
                    e.weather_data[:, 6] = 22.0   # outdoor soil T → 22°C
                    # Force initial indoor state to 22°C lab room — overrides
                    # the cold start from weather data at midnight.
                    x = np.copy(e.x)
                    x[2] = 22.0   # tAir
                    x[3] = 22.0   # tTop
                    x[4] = 22.0   # tCan (canopy)
                    x[5] = 22.0   # tCovIn
                    x[6] = 22.0   # tCovE
                    x[7] = 22.0   # tThScr
                    x[8] = 22.0   # tFlr
                    x[9] = 22.0   # tPipe
                    x[17] = 22.0  # tLamp
                    x[20] = 22.0  # tBlScr
                    x[21] = 22.0  # tCan24
                    e.x = x
                    e.x_prev = np.copy(x)
                except Exception as exc:
                    print(f"[compare] indoor seed failed: {exc}")
            if lamp_par_scale != 1.0:
                # Scale lamp electrical power → less PAR per u_lamp → canopy
                # stays in the linear (un-saturated) region so that small u
                # differences between controllers translate to clear Pn diffs.
                try:
                    e.p[172] = float(e.p[172]) * lamp_par_scale
                except Exception as exc:
                    print(f"[compare] lamp scaling failed: {exc}")
            # Fast-forward to start_hour with zero action
            if 0 < start_hour < 24:
                ff = int(round(start_hour * 3600 / float(e.dt)))
                u_zero = np.zeros(e.u.shape, dtype=float)
                for _ in range(ff):
                    e.step(u_zero)
            envs[name] = e

        # Energy/Pn coefficients
        any_env = next(iter(envs.values()))
        p_lamp_W_per_m2 = float(any_env.p[172])
        dt_s = float(any_env.dt)
        max_ticks = max(1, int(round(days * 24 * 3600 / dt_s / steps_per_tick)))
    except Exception as exc:
        with _compare_lock:
            _compare_state["running"] = False
            _compare_state["last_error"] = _en(f"compare init failed: {exc}")
        _broadcast({"type": "glgym_compare_error", "msg": _en(f"init failed: {exc}")})
        return

    no_climate = bool(params.get("no_climate_control", False))
    # Cumulative trackers
    cum_pn = {n: 0.0 for n in ctls}
    cum_energy_kwh = {n: 0.0 for n in ctls}

    tick = 0
    try:
        while not _compare_stop_evt.is_set() and tick < max_ticks:
            tick += 1
            t0 = time.monotonic()
            per_ctl = {}
            for name, ctl in ctls.items():
                e = envs[name]
                # Build context just-in-time
                ctx = StepContext(
                    t=e.timestep, dt=e.dt, Np=e.Np,
                    x_prev=e.x_prev, x=e.x, u=e.u, p=e.p,
                    d=e.weather_data,
                    hour_of_day=e.hour_of_day, day_of_year=e.day_of_year,
                )
                u = np.asarray(ctl(ctx), dtype=float)
                if no_climate:
                    # Strip heat/vent/screens but KEEP lamp (idx 4) and CO2
                    # injection (idx 1) so the canopy doesn't suffocate from
                    # CO2 depletion under sustained lamp use. All controllers
                    # use base_rb's CO2 logic for fair comparison.
                    lamp = float(u[4])
                    co2  = float(u[1])
                    u = np.zeros_like(u)
                    u[4] = lamp
                    u[1] = co2
                u_lamp_avg = 0.0
                pn_step_sum = 0.0
                lamp_w_step_sum = 0.0
                ok = True
                for _ in range(steps_per_tick):
                    obs, _r, term, trunc, info = e.step(u)
                    u_lamp_avg += float(u[4])
                    # ---- Pn this sub-step (mg{CO2}/m²/s) -----------------
                    if pn_source == "chamber" and chamber_pipe is not None and _pd is not None:
                        # User's chamber HistGBR: Pn = f(T, CO2, R:B, PPFD).
                        # PPFD scaled from lamp action (NOT from lamp_par_scale —
                        # that knob exists only to amplify GreenLight's thermal
                        # forcing; the chamber model lives in chamber-PPFD units).
                        ppfd_inc = max(0.0, float(u[4])) * lamp_ppfd_max
                        t_air = float(e.x[2])
                        co2_ppm = (
                            float(co2dens2ppm(t_air, 1e-6 * float(e.x[0])))
                            if co2dens2ppm is not None else 800.0
                        )
                        feats_in = {"T": t_air, "CO2": co2_ppm,
                                    "R:B": chamber_rb_default, "PPFD": ppfd_inc}
                        df = _pd.DataFrame([feats_in])[chamber_feats]
                        pn_umol = max(0.0, float(chamber_pipe.predict(df)[0]))  # µmol/m²/s leaf
                        # Leaf → canopy via effective LAI scale, then µmol → mg.
                        pn_step_sum += pn_umol * chamber_lai_scale * 44e-3      # mg{CO2}/m²/s canopy
                    else:
                        # Leaf-level Farquhar from GreenLight aux (already mg/m²/s).
                        pn_step_sum += float(info.get("P_leaf_net",
                                                      info.get("mcAirCan", 0.0)))
                    lamp_w_step_sum += float(u[4]) * p_lamp_W_per_m2
                    if term or trunc:
                        ok = False
                        break
                k = max(1, steps_per_tick)
                u_lamp_avg /= k
                pn_avg = pn_step_sum / k             # avg over the tick
                lamp_w_avg = lamp_w_step_sum / k
                # cumulate
                cum_pn[name] += pn_avg * dt_s * k          # mg CO2/m²
                cum_energy_kwh[name] += lamp_w_avg * dt_s * k / 3600.0 / 1e3
                per_ctl[name] = {
                    "u_lamp": round(u_lamp_avg, 3),
                    "pn": round(pn_avg, 4),
                    "power_w": round(lamp_w_avg, 1),
                    "t_air":   round(float(e.x[2]), 2),
                    "cum_pn": round(cum_pn[name], 1),
                    "cum_energy_kwh": round(cum_energy_kwh[name], 4),
                }
                if not ok:
                    break

            ctx0 = StepContext(
                t=any_env.timestep, dt=any_env.dt, Np=any_env.Np,
                x_prev=any_env.x_prev, x=any_env.x, u=any_env.u, p=any_env.p,
                d=any_env.weather_data,
                hour_of_day=any_env.hour_of_day, day_of_year=any_env.day_of_year,
            )
            ts = datetime.now().isoformat(timespec="seconds")
            with _compare_lock:
                _compare_state["tick"] = tick
            _broadcast({
                "type": "glgym_compare_tick", "tick": tick,
                "sim_hour": float(ctx0.hour_of_day),
                "sim_day":  float(ctx0.day_of_year),
                "controllers": per_ctl, "t_iso": ts,
            })

            elapsed = time.monotonic() - t0
            if elapsed < tick_sec:
                if _compare_stop_evt.wait(tick_sec - elapsed):
                    break
    finally:
        summary = {n: {"cum_pn_mgCO2_m2": cum_pn[n],
                       "cum_energy_kwh_m2": cum_energy_kwh[n]} for n in ctls}
        with _compare_lock:
            _compare_state["running"] = False
            _compare_state["stopped_at"] = datetime.now().isoformat(timespec="seconds")
            _compare_state["summary"] = summary
        _broadcast({"type": "glgym_compare_stopped", "tick": tick,
                    "stopped_at": _compare_state["stopped_at"],
                    "summary": summary})


@app.get("/api/glgym/compare")
def api_glgym_compare():
    with _compare_lock:
        return JSONResponse(_compare_state.copy())


@app.post("/api/glgym/compare/start")
def api_glgym_compare_start(req: GLGymCompareStart):
    global _compare_thread
    with _compare_lock:
        if _compare_state["running"]:
            raise HTTPException(status_code=409, detail="compare already running")
    if _glgym_state["running"] or _mppi_state["running"]:
        raise HTTPException(status_code=409, detail="another loop is running; stop it first")
    _compare_stop_evt.clear()
    p = req.model_dump()
    with _compare_lock:
        _compare_state.update({
            "running": True, "tick": 0,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "stopped_at": None, "params": p,
            "controllers": p["controllers"], "summary": {}, "last_error": None,
        })
    _compare_thread = threading.Thread(target=_compare_run, args=(p,), daemon=True)
    _compare_thread.start()
    _broadcast({"type": "glgym_compare_started", "params": p,
                "started_at": _compare_state["started_at"]})
    return JSONResponse({"ok": True, "params": p})


@app.post("/api/glgym/compare/stop")
def api_glgym_compare_stop():
    if not _compare_state["running"]:
        return JSONResponse({"ok": True, "note": "not running"})
    _compare_stop_evt.set()
    return JSONResponse({"ok": True})


# ----- GL-Gym closed-loop runner ------------------------------------------

class GLGymStart(BaseModel):
    tick_sec: float = 15.0       # wall-clock seconds between ticks
    max_ticks: int = 240         # total ticks
    max_ppfd: float = 60.0       # u_lamp=1 ⇒ this PPFD target
    red_frac: float = 0.5        # R:B PWM split (0.5 = 1:1, 0.8 = 4:1)
    start_day: int = 90          # day-of-year sim start (90 ≈ April 1)
    location: str = "Amsterdam"
    growth_year: int = 2010
    measure: bool = True         # trigger spectrometer each tick (ignored if sim_only)
    sim_only: bool = False       # if true: skip ALL Govee dispatch + spec measure
    steps_per_tick: int = 1      # advance N env.step() per tick (>1 = faster sim time)
    start_hour: float = 8.0      # fast-forward sim past midnight to this hour-of-day


def _glgym_run(params: dict) -> None:
    """Background closed-loop: GL-Gym sim drives Govee via the same code
    paths as /api/led + /api/measure. Streams tick events on /ws/glgym."""
    import yaml as _yaml
    from copy import deepcopy
    from gl_gym.environments.greenlight_env import GreenLightEnv  # noqa: WPS433
    from gl_gym.components.rule_based import RuleBasedController  # noqa: WPS433
    from gl_gym.core.types import StepContext  # noqa: WPS433
    from gl_gym.components.weather import WeatherRepository  # noqa: WPS433
    from gl_gym.environments.utils import (  # noqa: WPS433
        load_weather_data, co2ppm2dens, satVp,
    )

    tick_sec  = float(params["tick_sec"])
    max_ticks = int(params["max_ticks"])
    max_ppfd  = float(params["max_ppfd"])
    red_frac  = float(params["red_frac"])
    start_day = int(params["start_day"])
    sim_only  = bool(params.get("sim_only", False))
    do_measure = bool(params.get("measure", True)) and not sim_only
    steps_per_tick = max(1, int(params.get("steps_per_tick", 1)))
    start_hour = float(params.get("start_hour", 8.0))

    glgym_root = str(GLGYM_DIR)

    def _split_pwm(total: int) -> tuple[int, int]:
        return (
            int(round(total * red_frac)),
            int(round(total * (1 - red_frac))),
        )

    def _invert_ppfd_to_pwm(target: float) -> tuple[int, int]:
        if target <= 0.5 or _pwm_model is None:
            return 0, 0
        lo, hi = 0, 100
        for _ in range(8):
            mid = (lo + hi) // 2
            r, b = _split_pwm(mid)
            p = float(_pwm_model.predict(r_pwm=r, b_pwm=b))
            if p < target:
                lo = mid
            else:
                hi = mid
        return _split_pwm(hi)

    try:
        # Load env config (mirrors demo/demo_compare.py path).
        cfg = os.path.join(glgym_root, "gl_gym", "configs", "envs", "GreenLightEnv.yml")
        with open(cfg) as f:
            ek = _yaml.load(f, Loader=_yaml.FullLoader)["GreenLightEnv"]
        wkw = ek.pop("weather_repository_kwargs")
        # Resolve weather_data_dir relative to glgym_root (server cwd is
        # MPPI-Govee/web/, but the config uses 'gl_gym/data/weather/').
        wd = wkw["weather_data_dir"]
        if not os.path.isabs(wd):
            wd = os.path.join(glgym_root, wd)
        ek["weather_repository"] = WeatherRepository(
            weather_data_dir=wd,
            load_weather_data_fn=eval(wkw["load_weather_data_fn"]),
        )
        ek.pop("eval_scenarios", None)
        ek["normalize_actions"] = False
        ek["season_length"] = 30
        env = GreenLightEnv(**deepcopy(ek))
        env.reset(seed=0, options={"scenario": {
            "location": params["location"],
            "growth_year": int(params["growth_year"]),
            "start_day": start_day,
        }})

        # Rule-based controller params
        rb_yml = os.path.join(glgym_root, "configs", "agents", "rule_based.yml")
        with open(rb_yml) as f:
            rb_params = _yaml.load(f, Loader=_yaml.FullLoader)["GreenLightEnv"]
        rb = RuleBasedController(**rb_params)

        # Fast-forward env to start_hour (skips midnight → daytime so the
        # rule-based controller actually turns lamps on right away).
        if 0 < start_hour < 24:
            ff_steps = int(round(start_hour * 3600 / float(env.dt)))
            try:
                u_zero = np.zeros(env.u.shape, dtype=float)
                for _ in range(ff_steps):
                    env.step(u_zero)
            except Exception as exc:
                print(f"[glgym] fast-forward failed: {exc}")
    except Exception as exc:
        with _glgym_lock:
            _glgym_state["running"] = False
            _glgym_state["last_error"] = _en(f"glgym init failed: {exc}")
        _broadcast({"type": "glgym_error", "msg": _en(f"init failed: {exc}")})
        return

    # Optional one-shot indoor seed from live sensor (hardware mode only).
    # We do NOT override per-tick: the rule-based controller has a
    # "lamp off when temp is low" check that breaks if we keep pinning
    # the sim's t_air to a 20 °C ambient — better to let GL-Gym evolve
    # its own thermal state and just record what real sensors read for
    # logging/comparison.
    if not sim_only:
        try:
            row0 = _latest_riotee_row() or {}
            t0v = row0.get("temperature")
            rh0v = row0.get("humidity")
            co2v = row0.get("co2_ppm")
            co2v = co2v if (co2v is not None and co2v > 0) else None
            x = np.copy(env.x)
            if t0v is not None:
                x[2] = t0v; x[3] = t0v; x[4] = t0v + 4.0
            if t0v is not None and rh0v is not None:
                x[15] = (rh0v / 100.0) * satVp(t0v); x[16] = x[15]
            if co2v is not None and t0v is not None:
                x[0] = co2ppm2dens(t0v, co2v) * 1e6; x[1] = x[0]
            env.x = x
            env.x_prev = np.copy(x)
            env.obs = env._get_obs()
            print(f"[glgym] seeded env from sensor: T={t0v} RH={rh0v} CO2={co2v}")
        except Exception as exc:
            print(f"[glgym] seed failed: {exc}")

    tick = 0
    try:
        while not _glgym_stop_evt.is_set() and tick < max_ticks:
            tick += 1
            t0 = time.monotonic()

            # Read live sensor for logging/UI only (no env override).
            row = _latest_riotee_row() or {}
            t_air = row.get("temperature")
            rh    = row.get("humidity")

            # 2) controller
            try:
                ctx = StepContext(
                    t=env.timestep, dt=env.dt, Np=env.Np,
                    x_prev=env.x_prev, x=env.x, u=env.u, p=env.p,
                    d=env.weather_data,
                    hour_of_day=env.hour_of_day, day_of_year=env.day_of_year,
                )
                u_full = np.asarray(rb.predict(ctx), dtype=float)
                u_lamp = float(u_full[4])
            except Exception as exc:
                _broadcast({"type": "glgym_error", "msg": _en(f"controller failed: {exc}")})
                with _glgym_lock:
                    _glgym_state["last_error"] = _en(f"controller failed: {exc}")
                break

            # 3) target PPFD → PWM
            target = max(0.0, min(max_ppfd, u_lamp * max_ppfd))
            pwm_r, pwm_b = _invert_ppfd_to_pwm(target)
            pred = float(_pwm_model.predict(r_pwm=pwm_r, b_pwm=pwm_b)) if _pwm_model else 0.0

            # 4) dispatch (skip in sim_only)
            measured: Optional[float] = None
            if not sim_only:
                try:
                    _dispatch_pwm(pwm_r, pwm_b, _state["transport"])
                    with _state_lock:
                        _state["pwm_r"] = pwm_r
                        _state["pwm_b"] = pwm_b
                        _state["mode"] = "rb_mix"
                        _state["predicted_ppfd"] = pred
                        _state["last_dispatch_at"] = datetime.now().isoformat(timespec="seconds")
                except Exception as exc:
                    _broadcast({"type": "glgym_error", "msg": _en(f"dispatch failed: {exc}")})

                # 5) measure (optional)
                if do_measure:
                    try:
                        m, _f = _trigger_spec_once()
                        measured = m
                        if m is not None:
                            with _state_lock:
                                _state["last_measured_ppfd"] = m
                                _state["last_measured_at"] = datetime.now().isoformat(timespec="seconds")
                    except Exception as exc:
                        print(f"[glgym] measure failed: {exc}")

            # 6) advance sim by steps_per_tick env.step()
            try:
                for _ in range(steps_per_tick):
                    env.step(u_full)
            except Exception as exc:
                _broadcast({"type": "glgym_error", "msg": _en(f"env.step failed: {exc}")})
                break

            ts = datetime.now().isoformat(timespec="seconds")
            with _glgym_lock:
                _glgym_state.update({
                    "tick": tick,
                    "sim_hour": float(ctx.hour_of_day),
                    "sim_day":  float(ctx.day_of_year),
                    "sensor_T": t_air, "sensor_RH": rh,
                    "u_lamp": u_lamp, "target_ppfd": target,
                    "pwm_r": pwm_r, "pwm_b": pwm_b,
                    "predicted_ppfd": pred, "measured_ppfd": measured,
                    "last_tick_at": ts,
                })
            _broadcast({
                "type": "glgym_tick", "tick": tick,
                "sim_hour": float(ctx.hour_of_day),
                "sim_day":  float(ctx.day_of_year),
                "sensor_T": t_air, "sensor_RH": rh,
                "u_lamp": u_lamp, "target_ppfd": target,
                "pwm_r": pwm_r, "pwm_b": pwm_b,
                "predicted_ppfd": pred, "measured_ppfd": measured,
                "t_iso": ts,
            })

            # 7) sleep
            elapsed = time.monotonic() - t0
            if elapsed < tick_sec:
                if _glgym_stop_evt.wait(tick_sec - elapsed):
                    break
    finally:
        try:
            govee_controller.set_led_full_white_ha(timeout_sec=5.0)
            print("💡 GLGym finally: lights → full white for charging")
        except Exception as exc:
            print(f"⚠️ GLGym finally: full_white failed: {exc}")
        with _glgym_lock:
            _glgym_state["running"] = False
            _glgym_state["stopped_at"] = datetime.now().isoformat(timespec="seconds")
        _broadcast({"type": "glgym_stopped", "tick": tick,
                    "stopped_at": _glgym_state["stopped_at"]})


@app.get("/api/glgym")
def api_glgym():
    with _glgym_lock:
        return JSONResponse(_glgym_state.copy())


@app.post("/api/glgym/start")
def api_glgym_start(req: GLGymStart):
    global _glgym_thread
    with _glgym_lock:
        if _glgym_state["running"]:
            raise HTTPException(status_code=409, detail="GLGym already running")
        if _mppi_state["running"]:
            raise HTTPException(status_code=409, detail="MPPI is running; stop it first")
        _glgym_stop_evt.clear()
        params = req.model_dump()
        _glgym_state.update({
            "running": True, "tick": 0,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "stopped_at": None, "params": params,
            "sim_hour": None, "sim_day": None,
            "sensor_T": None, "sensor_RH": None,
            "u_lamp": None, "target_ppfd": None,
            "pwm_r": None, "pwm_b": None,
            "predicted_ppfd": None, "measured_ppfd": None,
            "last_tick_at": None, "last_error": None,
        })
    _glgym_thread = threading.Thread(target=_glgym_run, args=(params,), daemon=True)
    _glgym_thread.start()
    _broadcast({"type": "glgym_started", "params": params,
                "started_at": _glgym_state["started_at"]})
    return JSONResponse({"ok": True, "params": params})


@app.post("/api/glgym/stop")
def api_glgym_stop():
    if not _glgym_state["running"]:
        return JSONResponse({"ok": True, "note": "not running"})
    _glgym_stop_evt.set()
    return JSONResponse({"ok": True})


class MppiStart(BaseModel):
    target_ppfd: float = 15.0
    horizon: int = 5
    num_samples: int = 120
    dt: float = 12.0
    rb: float = 0.8
    steps: int = 0  # 0 = run until /stop
    demo_gain: float = 1.0
    ambient_temp: float = 25.0
    measure: bool = True


def _mppi_run(params: dict) -> None:
    """Background MPPI loop. Mirrors examples/govee_mpc_closed_loop.py but
    streams events over WebSocket and dispatches via the same /api/led path
    the manual sliders use (HA transport, set_led_mix_ha)."""
    from mppi_v2 import LEDPlant, LEDMPPIController  # noqa: WPS433
    target = float(params["target_ppfd"])
    thermal_dir = str(HERE.parent / "Thermal" / "exported_models")

    class _StubPowerModel:
        """Linear PWM→W stub. Keeps LEDPlant.predict() from raising
        ('功率模型未提供') so MPPI rollouts actually populate cost.
        Govee H6056 ≈ 12 W per bar @ 100 PWM, so total_pwm 0..200 maps
        to roughly 0..24 W. Magnitude only matters if R_power>0 or a
        power_budget is set; for the tracking-only config we use here
        it's effectively a no-op."""
        def predict(self, *, total_pwm, key=None):
            return float(total_pwm) * 0.12

    try:
        plant = LEDPlant(
            target_ppfd=target,
            r_b_ratio=float(params["rb"]),
            use_pn_model=True,
            use_sp_to_ppfd=False,
            model_dir=thermal_dir,
            power_model=_StubPowerModel(),
            x_cm=POSITION.x_cm, y_cm=POSITION.y_cm, z_cm=POSITION.z_cm,
            demo_gain=float(params.get("demo_gain", 1.0)),
        )
        plant.ambient_temp = float(params.get("ambient_temp", 25.0))
        mppi = LEDMPPIController(
            plant,
            horizon=int(params["horizon"]),
            num_samples=int(params["num_samples"]),
            dt=float(params["dt"]),
            temperature=0.5,
        )
        mppi.set_weights(Q_photo=0.0, R_du=0.05, R_power=0.0, Q_ref=1.0)
        mppi.set_constraints(u_min=0.0, u_max=max(target * 1.5, 30.0),
                             temp_min=0.0, temp_max=80.0)
        mppi.set_ppfd_train_range(ppfd_min=0.0, ppfd_max=200.0, R_oob=0.0)
        mppi.set_target_ppfd(target)
    except Exception as exc:
        with _mppi_lock:
            _mppi_state["running"] = False
            _mppi_state["last_error"] = _en(f"init failed: {exc}")
        _broadcast({"type": "error", "msg": _en(f"init failed: {exc}")})
        return

    current_temp = float(params.get("ambient_temp", 25.0))
    max_steps = int(params.get("steps") or 0)
    do_measure = bool(params.get("measure", True))
    step = 0
    try:
        while not _mppi_stop_evt.is_set():
            if max_steps and step >= max_steps:
                break
            step += 1
            try:
                u, _seq, _ok, min_cost, _w = mppi.solve(current_temp)
                r_pwm, b_pwm = plant._ppfd_to_pwm(u)
                pred = float(plant.govee_pwm_model.predict(
                    r_pwm=r_pwm, b_pwm=b_pwm)) * plant.demo_gain
            except Exception as exc:
                _broadcast({"type": "error", "msg": _en(f"solve failed: {exc}")})
                with _mppi_lock:
                    _mppi_state["last_error"] = _en(f"solve failed: {exc}")
                break

            r_int = int(np.clip(round(r_pwm), 0, 100))
            b_int = int(np.clip(round(b_pwm), 0, 100))
            power_w = float((r_pwm + b_pwm) * 0.12)
            try:
                info = _dispatch_pwm(r_int, b_int, _state["transport"])
            except Exception as exc:
                _broadcast({"type": "error", "msg": _en(f"dispatch failed: {exc}")})
                continue
            with _state_lock:
                _state["pwm_r"] = r_int
                _state["pwm_b"] = b_int
                _state["predicted_ppfd"] = pred
                _state["last_dispatch_at"] = datetime.now().isoformat(timespec="seconds")
                _state["last_dispatch_latency_ms"] = info.get("latency_ms")

            if _mppi_stop_evt.wait(timeout=float(params["dt"])):
                break

            measured: Optional[float] = None
            spec_file: Optional[str] = None
            if do_measure:
                measured, spec_file = _trigger_spec_once()
                if measured is not None:
                    with _state_lock:
                        _state["last_measured_ppfd"] = measured
                        _state["last_measured_at"] = datetime.now().isoformat(timespec="seconds")
                        _state["last_spec_file"] = spec_file
            err_abs = (measured - pred) if measured is not None else None
            err_pct = (100.0 * err_abs / max(pred, 1e-6)) if err_abs is not None else None
            ts = datetime.now().isoformat(timespec="seconds")

            with _mppi_lock:
                _mppi_state.update({
                    "step": step,
                    "last_u": float(u),
                    "last_pwm_r": float(r_pwm),
                    "last_pwm_b": float(b_pwm),
                    "last_pred": pred,
                    "last_measured": measured,
                    "last_err_abs": err_abs,
                    "last_err_pct": err_pct,
                    "last_min_cost": float(min_cost),
                    "last_power_w": power_w,
                    "last_step_at": ts,
                })
            _broadcast({
                "type": "step",
                "step": step,
                "target_ppfd": target,
                "mppi_u": float(u),
                "pwm_r": float(r_pwm),
                "pwm_b": float(b_pwm),
                "pred_ppfd": pred,
                "measured_ppfd": measured,
                "err_abs": err_abs,
                "err_pct": err_pct,
                "min_cost": float(min_cost),
                "power_w": power_w,
                "t_iso": ts,
            })
    finally:
        try:
            govee_controller.set_led_full_white_ha(timeout_sec=10.0)
            print("💡 MPPI finally: lights → full white for charging")
        except Exception as exc:
            print(f"⚠️ MPPI finally: full_white failed: {exc}")
        with _mppi_lock:
            _mppi_state["running"] = False
            _mppi_state["stopped_at"] = datetime.now().isoformat(timespec="seconds")
        _broadcast({"type": "stopped", "step": step,
                    "stopped_at": _mppi_state["stopped_at"]})


@app.get("/api/mppi")
def api_mppi():
    with _mppi_lock:
        return JSONResponse(_mppi_state.copy())


@app.post("/api/mppi/start")
def api_mppi_start(req: MppiStart):
    global _mppi_thread
    with _mppi_lock:
        if _mppi_state["running"]:
            raise HTTPException(status_code=409, detail="MPPI already running")
        if _glgym_state["running"]:
            raise HTTPException(status_code=409, detail="GLGym is running; stop it first")
        _mppi_stop_evt.clear()
        params = req.model_dump()
        _mppi_state.update({
            "running": True,
            "step": 0,
            "target_ppfd": req.target_ppfd,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "stopped_at": None,
            "params": params,
            "last_u": None, "last_pwm_r": None, "last_pwm_b": None,
            "last_pred": None, "last_measured": None,
            "last_err_abs": None, "last_err_pct": None,
            "last_min_cost": None, "last_power_w": None,
            "last_step_at": None, "last_error": None,
        })
    _mppi_thread = threading.Thread(target=_mppi_run, args=(params,), daemon=True)
    _mppi_thread.start()
    _broadcast({"type": "started", "params": params,
                "started_at": _mppi_state["started_at"]})
    return JSONResponse({"ok": True, "params": params})


@app.post("/api/mppi/stop")
def api_mppi_stop():
    if not _mppi_state["running"]:
        return JSONResponse({"ok": True, "note": "not running"})
    _mppi_stop_evt.set()
    return JSONResponse({"ok": True})


@app.websocket("/ws/mppi")
async def ws_mppi(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        with _mppi_lock:
            snap = _mppi_state.copy()
        await ws.send_json({"type": "snapshot", "state": snap})
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _ws_clients.discard(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
    )
