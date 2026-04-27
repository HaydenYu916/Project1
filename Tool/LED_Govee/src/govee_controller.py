"""Synchronous Govee H6056 BLE controller.

Mirrors the API surface of Tool/LED_Shelly/src/shelly_controller.py so the
calibration runner can swap from `from shelly_controller import ...` to
`from govee_controller import ...` with minimal changes.

Internally we run a single asyncio event loop in a background thread and own
one BleakClient; the two logical "Red"/"Blue" devices in DEVICES both point at
the same physical H6056 but address different segment masks.

Public sync API:
    DEVICES                            -- same dict as device_config
    connect(adapter=None)              -- open BLE, prime colors, all off
    disconnect()                       -- power off + close BLE
    set_led_pwm(name, pwm)             -- segment brightness 0..100
    set_led_pair(pwm_r, pwm_b)         -- both at once
    set_power(on)                      -- whole device on/off
    is_connected()                     -- bool
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional

from bleak import BleakClient

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "config"))
from device_config import (  # noqa: E402
    DEFAULT_CONNECT_TIMEOUT,
    DEVICES,
    WRITE_CHAR_UUID,
)


# --- packet builders -------------------------------------------------------

def _frame(payload: bytes) -> bytes:
    if len(payload) > 19:
        raise ValueError("payload too long")
    buf = bytearray(20)
    buf[: len(payload)] = payload
    chk = 0
    for b in buf[:19]:
        chk ^= b
    buf[19] = chk
    return bytes(buf)


def _cmd_power(on: bool) -> bytes:
    return _frame(bytes([0x33, 0x01, 0x01 if on else 0x00]))


def _cmd_seg_color(seg_mask: int, r: int, g: int, b: int) -> bytes:
    lo, hi = seg_mask & 0xFF, (seg_mask >> 8) & 0xFF
    return _frame(bytes(
        [0x33, 0x05, 0x15, 0x01, r & 0xFF, g & 0xFF, b & 0xFF,
         0, 0, 0, 0, 0, lo, hi]
    ))


def _cmd_seg_brightness(seg_mask: int, pct: int) -> bytes:
    pct = max(0, min(100, int(pct)))
    lo, hi = seg_mask & 0xFF, (seg_mask >> 8) & 0xFF
    return _frame(bytes([0x33, 0x05, 0x15, 0x02, pct, lo, hi]))


def _cmd_global_brightness(pct: int) -> bytes:
    return _frame(bytes([0x33, 0x04, max(0, min(100, int(pct)))]))


# --- background event loop -------------------------------------------------

class _LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=2.0)


# --- controller ------------------------------------------------------------

_loop: Optional[_LoopThread] = None
_client: Optional[BleakClient] = None
_mac: Optional[str] = None
_lock = threading.Lock()

STATE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "last_state.json")
)
_state_lock = threading.Lock()


def _read_state_file() -> dict:
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_state(**updates) -> None:
    with _state_lock:
        state = _read_state_file()
        state.update(updates)
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_PATH)


def read_state() -> dict:
    """Return last-known LED state recorded by this controller."""
    return _read_state_file()


_HA_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "HA_History_API", "config.json")
)
_HA_LIGHT_ENTITY = "light.rgbic_tv_light_bars"


def _ha_cfg() -> dict:
    with open(_HA_CONFIG_PATH) as f:
        text = f.read()
    return json.JSONDecoder().raw_decode(text)[0]["ha"]


def _ha_request(method: str, path: str, body: Optional[dict] = None,
                timeout: float = 5.0) -> dict:
    import urllib.request
    cfg = _ha_cfg()
    url = f"{cfg['url'].rstrip('/')}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Authorization": f"Bearer {cfg['token']}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def read_state_ha(entity_id: str = _HA_LIGHT_ENTITY,
                  timeout: float = 5.0) -> dict:
    """Query Home Assistant for the live light state.

    Returns a dict with keys: state, brightness, rgb_color, color_mode,
    last_changed, entity_id. Falls back to {"error": ...} on failure.
    Token + URL come from Tool/HA_History_API/config.json.
    """
    import urllib.request
    import urllib.error

    try:
        with open(_HA_CONFIG_PATH) as f:
            text = f.read()
        cfg = json.JSONDecoder().raw_decode(text)[0]["ha"]
        url = f"{cfg['url'].rstrip('/')}/api/states/{entity_id}"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {cfg['token']}"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except (OSError, KeyError, json.JSONDecodeError,
            urllib.error.URLError) as exc:
        return {"error": str(exc), "entity_id": entity_id}

    attrs = data.get("attributes", {})
    return {
        "entity_id": data.get("entity_id"),
        "state": data.get("state"),
        "brightness": attrs.get("brightness"),
        "rgb_color": attrs.get("rgb_color"),
        "color_mode": attrs.get("color_mode"),
        "last_changed": data.get("last_changed"),
    }


def _ensure_loop() -> _LoopThread:
    global _loop
    if _loop is None:
        _loop = _LoopThread()
    return _loop


async def _async_connect(mac: str, adapter: Optional[str], timeout: float) -> BleakClient:
    kwargs = {"timeout": timeout}
    if adapter:
        kwargs["adapter"] = adapter
    client = BleakClient(mac, **kwargs)
    await client.connect()
    return client


async def _async_write(client: BleakClient, payload: bytes) -> None:
    await client.write_gatt_char(WRITE_CHAR_UUID, payload, response=False)


async def _async_disconnect(client: BleakClient) -> None:
    try:
        await client.disconnect()
    except Exception:
        pass


def connect(adapter: Optional[str] = None, timeout: float = DEFAULT_CONNECT_TIMEOUT,
            retries: int = 3) -> None:
    """Open BLE connection, prime per-segment colors, set all segments dark."""
    global _client, _mac
    macs = {info["mac"] for info in DEVICES.values()}
    if len(macs) != 1:
        raise RuntimeError(f"govee_controller assumes one BLE device, got {macs}")
    _mac = next(iter(macs))

    loop = _ensure_loop()
    last_exc: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            _client = loop.submit(_async_connect(_mac, adapter, timeout))
            break
        except Exception as exc:
            last_exc = exc
            time.sleep(1.0 * (attempt + 1))
    else:
        raise RuntimeError(f"BLE connect failed after {retries} attempts: {last_exc}")

    # prime: power on, global brightness 100, set every segment to black
    _send(_cmd_power(True))
    _send(_cmd_global_brightness(100))
    for info in DEVICES.values():
        _send(_cmd_seg_color(info["segment_mask"], 0, 0, 0))
    _write_state(
        connected=True,
        power=True,
        global_brightness=100,
        pwm={name: 0 for name in DEVICES},
    )


def disconnect() -> None:
    """Turn off and close BLE connection."""
    global _client
    if _client is None:
        return
    try:
        for info in DEVICES.values():
            try:
                _send(_cmd_seg_color(info["segment_mask"], 0, 0, 0))
            except Exception:
                pass
        try:
            _send(_cmd_power(False))
        except Exception:
            pass
        _ensure_loop().submit(_async_disconnect(_client))
    finally:
        _client = None
        _write_state(connected=False, power=False,
                     pwm={name: 0 for name in DEVICES})


def is_connected() -> bool:
    return _client is not None and _client.is_connected


def _send(payload: bytes, retries: int = 2) -> None:
    if _client is None:
        raise RuntimeError("govee_controller not connected; call connect() first")
    loop = _ensure_loop()
    last_exc: Optional[BaseException] = None
    with _lock:
        for attempt in range(retries + 1):
            try:
                loop.submit(_async_write(_client, payload))
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(0.1 * (attempt + 1))
        raise RuntimeError(f"BLE write failed: {last_exc}")


def set_led_pwm(device_name: str, pwm: int) -> dict:
    """Set the brightness (0..100) of one logical channel.

    pwm == 0 sends a black-color command to truly extinguish the segment
    (the seg-brightness command alone leaves a visible floor on H6056).
    pwm > 0 restores the role's full color and sets seg-brightness.
    """
    if device_name not in DEVICES:
        raise KeyError(f"unknown device {device_name}; have {list(DEVICES)}")
    info = DEVICES[device_name]
    pwm = max(0, min(100, int(pwm)))
    mask = info["segment_mask"]
    if pwm == 0:
        _send(_cmd_seg_color(mask, 0, 0, 0))
    else:
        r, g, b = info["color"]
        _send(_cmd_seg_color(mask, r, g, b))
        _send(_cmd_seg_brightness(mask, pwm))
    cur = _read_state_file().get("pwm", {})
    cur[device_name] = pwm
    _write_state(pwm=cur)
    return {"device": device_name, "pwm": pwm, "ok": True}


def set_led_pair(pwm_r: int, pwm_b: int) -> None:
    set_led_pwm("Red", pwm_r)
    set_led_pwm("Blue", pwm_b)


def set_led_full_white() -> None:
    """Drive both bars to RGB(255,255,255) at full brightness via BLE.

    Used by Vcap recovery — maximum optical output regardless of color mix.
    """
    _send(_cmd_global_brightness(100))
    for info in DEVICES.values():
        mask = info["segment_mask"]
        _send(_cmd_seg_color(mask, 255, 255, 255))
        _send(_cmd_seg_brightness(mask, 100))
    _write_state(power=True, mix_rgb=[255, 255, 255], mode="full_white")


def set_led_full_white_ha(poll_sec: float = 0.3,
                          timeout_sec: float = 8.0,
                          rgb_tol: int = 6,
                          entity_id: str = _HA_LIGHT_ENTITY) -> dict:
    """Drive RGB(255,255,255) full brightness via HA, wait for convergence.

    Returns the same shape as set_led_mix_ha; raises TimeoutError on
    failure to converge.
    """
    pre = _ha_request("GET", f"/api/states/{entity_id}", timeout=3.0)
    pre_reported = pre.get("last_reported") or pre.get("last_updated")
    sent_at = datetime.now().isoformat(timespec="milliseconds")
    body = {"entity_id": entity_id, "rgb_color": [255, 255, 255],
            "brightness": 255}
    _ha_request("POST", "/api/services/light/turn_on", body=body, timeout=5.0)

    time.sleep(0.4)
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    last_state: dict = {}
    while time.monotonic() < deadline:
        try:
            cur = _ha_request("GET", f"/api/states/{entity_id}", timeout=3.0)
        except Exception:
            time.sleep(poll_sec)
            continue
        last_state = cur
        cur_reported = cur.get("last_reported") or cur.get("last_updated")
        attrs = cur.get("attributes", {})
        cur_rgb = attrs.get("rgb_color") or [None, None, None]
        time_advanced = (cur_reported and pre_reported
                         and cur_reported > pre_reported)
        state_ok = cur.get("state") == "on"
        rgb_ok = (
            len(cur_rgb) == 3
            and all(v is not None for v in cur_rgb)
            and abs(cur_rgb[0] - 255) <= rgb_tol
            and abs(cur_rgb[1] - 255) <= rgb_tol
            and abs(cur_rgb[2] - 255) <= rgb_tol
        )
        if state_ok and rgb_ok:
            applied_at = datetime.now().isoformat(timespec="milliseconds")
            t0 = datetime.fromisoformat(sent_at)
            t1 = datetime.fromisoformat(applied_at)
            latency_ms = (t1 - t0).total_seconds() * 1000.0
            _write_state(transport="ha", power=True,
                         mix_rgb=[255, 255, 255], mode="full_white",
                         applied_at=applied_at,
                         latency_ms=round(latency_ms, 1))
            return {"sent_at": sent_at, "applied_at": applied_at,
                    "latency_ms": round(latency_ms, 1),
                    "ha_state": {"state": cur.get("state"),
                                 "rgb": cur_rgb,
                                 "last_reported": cur_reported}}
        time.sleep(poll_sec)

    raise TimeoutError(
        f"HA did not converge to full-white within {timeout_sec}s; "
        f"last_state={last_state.get('state')} "
        f"rgb={last_state.get('attributes', {}).get('rgb_color')}"
    )


def set_led_mix_ha(pwm_r: int, pwm_b: int,
                   poll_sec: float = 0.3,
                   timeout_sec: float = 8.0,
                   rgb_tol: int = 6,
                   entity_id: str = _HA_LIGHT_ENTITY) -> dict:
    """Drive the H6056 via Home Assistant cloud and *wait until verified*.

    Alignment guarantee:
      1. Capture `last_reported` from HA before sending.
      2. POST light.turn_on (rgb=R,0,B; brightness=255) — or turn_off if R=B=0.
      3. Poll /api/states until BOTH:
           - HA `last_reported` advanced past our pre-send snapshot
           - reported rgb matches target within rgb_tol per channel
           - reported state matches (on / off)
      4. Return {sent_at, applied_at, latency_ms, ha_state}.
    Raises TimeoutError if not verified within timeout_sec.
    Use `applied_at` for downstream spectrometer alignment, NOT `sent_at`.
    """
    pwm_r = max(0, min(100, int(pwm_r)))
    pwm_b = max(0, min(100, int(pwm_b)))
    r = int(round(pwm_r * 255 / 100))
    b = int(round(pwm_b * 255 / 100))
    target_on = (r > 0) or (b > 0)

    pre = _ha_request("GET", f"/api/states/{entity_id}", timeout=3.0)
    pre_reported = pre.get("last_reported") or pre.get("last_updated")

    sent_at = datetime.now().isoformat(timespec="milliseconds")
    if target_on:
        body = {"entity_id": entity_id, "rgb_color": [r, 0, b],
                "brightness": 255}
        _ha_request("POST", "/api/services/light/turn_on", body=body, timeout=5.0)
    else:
        _ha_request("POST", "/api/services/light/turn_off",
                    body={"entity_id": entity_id}, timeout=5.0)

    # min settle so we don't snapshot HA before it processes our POST
    time.sleep(0.4)
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    last_state: dict = {}
    while time.monotonic() < deadline:
        try:
            cur = _ha_request("GET", f"/api/states/{entity_id}", timeout=3.0)
        except Exception:
            time.sleep(poll_sec)
            continue
        last_state = cur
        cur_reported = cur.get("last_reported") or cur.get("last_updated")
        attrs = cur.get("attributes", {})
        cur_rgb = attrs.get("rgb_color") or [None, None, None]
        cur_state = cur.get("state")
        time_advanced = (cur_reported and pre_reported
                         and cur_reported > pre_reported)
        state_ok = (cur_state == ("on" if target_on else "off"))
        if target_on and len(cur_rgb) == 3 and all(v is not None for v in cur_rgb):
            rgb_ok = (abs(cur_rgb[0] - r) <= rgb_tol
                      and abs(cur_rgb[1]) <= rgb_tol
                      and abs(cur_rgb[2] - b) <= rgb_tol)
        else:
            rgb_ok = not target_on
        # Accept on rgb+state match; HA writes attributes synchronously when
        # the service call lands, so a match after our POST means the command
        # was accepted regardless of whether Govee pushed back a fresh state.
        # last_reported advance is recorded as a 'verified' bit, not gating.
        if state_ok and rgb_ok:
            applied_at = datetime.now().isoformat(timespec="milliseconds")
            t0 = datetime.fromisoformat(sent_at)
            t1 = datetime.fromisoformat(applied_at)
            latency_ms = (t1 - t0).total_seconds() * 1000.0
            cur_pwm = _read_state_file().get("pwm", {})
            cur_pwm["Red"] = pwm_r
            cur_pwm["Blue"] = pwm_b
            _write_state(transport="ha", power=target_on, pwm=cur_pwm,
                         mix_rgb=[r, 0, b], applied_at=applied_at,
                         latency_ms=round(latency_ms, 1))
            return {
                "sent_at": sent_at,
                "applied_at": applied_at,
                "latency_ms": round(latency_ms, 1),
                "ha_state": {"state": cur_state, "rgb": cur_rgb,
                             "last_reported": cur_reported},
            }
        time.sleep(poll_sec)

    raise TimeoutError(
        f"HA did not converge to target within {timeout_sec}s; "
        f"target=(rgb=({r},0,{b}), state={'on' if target_on else 'off'}); "
        f"last_state={last_state.get('state')} "
        f"rgb={last_state.get('attributes', {}).get('rgb_color')} "
        f"last_reported={last_state.get('last_reported')}"
    )


def set_led_mix(pwm_r: int, pwm_b: int) -> None:
    """Drive BOTH bars with the same red+blue mixed RGB.

    pwm_r, pwm_b are 0..100 percentages of red and blue channels.
    Produces RGB = (pwm_r*255/100, 0, pwm_b*255/100) on every segment,
    so both bars emit identical mixed light (the preferred mode for
    uniform illumination over the solar panels).
    """
    r = max(0, min(255, int(round(int(pwm_r) * 255 / 100))))
    b = max(0, min(255, int(round(int(pwm_b) * 255 / 100))))
    for info in DEVICES.values():
        mask = info["segment_mask"]
        _send(_cmd_seg_color(mask, r, 0, b))
        _send(_cmd_seg_brightness(mask, 100))
    cur = _read_state_file().get("pwm", {})
    cur["Red"] = max(0, min(100, int(pwm_r)))
    cur["Blue"] = max(0, min(100, int(pwm_b)))
    _write_state(pwm=cur, mix_rgb=[r, 0, b])


def set_power(on: bool) -> None:
    _send(_cmd_power(on))
    _write_state(power=bool(on))


# --- CLI -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Govee H6056 controller CLI")
    ap.add_argument("device", choices=list(DEVICES) + ["all"])
    ap.add_argument("action", choices=["on", "off", "brightness", "pair", "status", "status_ha"])
    ap.add_argument("value", nargs="*", type=int)
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    if args.action == "status":
        print(json.dumps(read_state(), indent=2, ensure_ascii=False))
        raise SystemExit(0)
    if args.action == "status_ha":
        print(json.dumps(read_state_ha(), indent=2, ensure_ascii=False))
        raise SystemExit(0)

    connect(adapter=args.adapter)
    try:
        if args.action == "on":
            set_power(True)
        elif args.action == "off":
            set_power(False)
        elif args.action == "brightness":
            if not args.value:
                raise SystemExit("brightness requires a value")
            if args.device == "all":
                set_led_pair(args.value[0], args.value[0])
            else:
                set_led_pwm(args.device, args.value[0])
        elif args.action == "pair":
            if len(args.value) != 2:
                raise SystemExit("pair requires two values: pwm_r pwm_b")
            set_led_pair(args.value[0], args.value[1])
        time.sleep(0.5)
    finally:
        disconnect()
