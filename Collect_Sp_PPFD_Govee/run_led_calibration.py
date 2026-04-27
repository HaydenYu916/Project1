#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LED calibration data acquisition runner (Govee H6056 BLE variant).

This script orchestrates three systems:
1) Govee H6056 LED segment PWM control over BLE (red front bar + blue back bar),
2) spectrometer PPFD sampling,
3) optional Riotee sensor collector process.

Design goal:
- Segment-level alignment with optical marker pattern (OFF -> BLUE -> TARGET).
- BLE writes are fire-and-forget plus a fixed settle delay; we do not poll
  status because H6056 does not expose reliable read-back.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
TOOL_DIR = ROOT_DIR.parent / "Tool"
LED_SRC_DIR = TOOL_DIR / "LED_Govee" / "src"
SPEC_DIR = TOOL_DIR / "Spectrometer"
RIOTEE_DIR = TOOL_DIR / "Sensor_riotee_server"

for path in (LED_SRC_DIR, SPEC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import serial
except Exception as exc:
    print(f"ERROR: pyserial import failed: {exc}")
    raise

try:
    from serial.tools import list_ports
except Exception:
    list_ports = None

try:
    import govee_controller as _gc
    from govee_controller import DEVICES
except Exception as exc:
    print(f"ERROR: LED controller import failed: {exc}")
    raise

try:
    from lib import complete_spectrum_measurement, initialize_spectrometer_session
except Exception as exc:
    print(f"ERROR: Spectrometer lib import failed: {exc}")
    raise


PPFD_CANDIDATE_KEYS = (
    "PPFD(umol/㎡/s)",
    "PPFD(umol/m2/s)",
    "PPFD(umol/m²/s)",
)

PPFD_BLUE_CANDIDATE_KEYS = (
    "PPFD-B(umol/㎡/s)",
    "PPFD-B(umol/m2/s)",
    "PPFD-B(umol/m²/s)",
)

PPFD_RED_CANDIDATE_KEYS = (
    "PPFD-R(umol/㎡/s)",
    "PPFD-R(umol/m2/s)",
    "PPFD-R(umol/m²/s)",
)

SPECTROMETER_USB_IDS = {(0x0483, 0x5741)}  # STM32
RIOTEE_USB_IDS = {(0x1209, 0xC8A2)}


def now_iso_ms() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def iso_diff_ms(start_iso: str, end_iso: str) -> float | None:
    try:
        return (datetime.fromisoformat(end_iso) - datetime.fromisoformat(start_iso)).total_seconds() * 1000.0
    except Exception:
        return None


def clamp_pwm(value: int) -> int:
    return max(0, min(100, int(value)))


def safe_float(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_ratio_list(text: str) -> list[tuple[int, int]]:
    ratios: list[tuple[int, int]] = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid ratio token: {token!r}. Expected format like 2:1")
        r = int(parts[0].strip())
        b = int(parts[1].strip())
        if r < 0 or b < 0:
            raise ValueError(f"Ratio must be non-negative: {token!r}")
        if r == 0 and b == 0:
            raise ValueError("Ratio cannot be 0:0")
        ratios.append((r, b))
    if not ratios:
        raise ValueError("At least one ratio must be provided")
    return ratios


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        val = int(token)
        if val < 0 or val > 100:
            raise ValueError(f"Total PWM must be in [0,100], got {val}")
        values.append(val)
    if not values:
        raise ValueError("At least one total PWM value must be provided")
    return values


def ratio_to_pwm(r: int, b: int, total: int) -> tuple[int, int]:
    den = r + b
    pwm_r = int(round(total * r / den))
    pwm_b = int(round(total * b / den))
    return clamp_pwm(pwm_r), clamp_pwm(pwm_b)


def rb_ratio_from_pwm(pwm_r: int, pwm_b: int) -> str:
    if pwm_b == 0:
        return "inf" if pwm_r > 0 else "0"
    return f"{pwm_r / pwm_b:.6f}"


@dataclass
class TimingConfig:
    Ts: float
    T_off: float
    T_blue: float
    T_settle: float
    T_meas: float
    T_seg: float
    N_spec: int


@dataclass
class SegmentCondition:
    segment_id: int
    segment_type: str
    ratio_r: int
    ratio_b: int
    total_pwm: int
    pwm_r: int
    pwm_b: int
    condition_index: int


def log(msg: str) -> None:
    print(f"[{now_iso_ms()}] {msg}", flush=True)


def retry_call(fn, attempts: int = 5, base_delay_sec: float = 0.4):
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: PERF203
            last_err = exc
            sleep_s = base_delay_sec * (2 ** i)
            time.sleep(sleep_s)
    if last_err is None:
        raise RuntimeError("Unknown retry failure")
    raise last_err


def set_led_pwm(device_name: str, pwm: int) -> dict[str, Any]:
    pwm_int = clamp_pwm(pwm)
    return retry_call(lambda: _gc.set_led_pwm(device_name, pwm_int))


TRANSPORT = "ble"  # set by main(); "ble" or "ha"
HA_TIMEOUT_SEC = 8.0
HA_POLL_SEC = 0.3

# Vcap guard config (set in main from argparse)
VCAP_CSV_PATH: str | None = None
VCAP_LOW_V = 3.3
VCAP_RECOVER_V = 4.5
VCAP_MAX_WAIT_SEC = 600.0
VCAP_POLL_SEC = 2.0
VCAP_FRESH_SEC = 120.0
# During Vcap recovery we drive RGB(255,255,255) full brightness — max optical
# output across all channels — instead of just red+blue mix. Any spectrometer
# data taken during this period must be discarded; recovery only happens
# BETWEEN segments, so segment_table.csv stays clean by construction.


def read_latest_vcap_from_csv(path: str,
                              fresh_sec: float = VCAP_FRESH_SEC) -> tuple[float | None, str | None]:
    """Return (vcap_volts, timestamp_str) from the riotee_data_all.csv tail.

    Returns (None, None) if the file is missing, has no rows, the last
    parseable row is older than fresh_sec, or vcap_raw is non-numeric.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(size, 8192)
            f.seek(size - chunk)
            tail = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return (None, None)

    lines = [ln for ln in tail.splitlines() if ln and not ln.startswith("#")]
    for ln in reversed(lines):
        parts = ln.split(",")
        if len(parts) < 8:
            continue
        ts_str, vcap_str = parts[1], parts[7]
        try:
            vcap = float(vcap_str)
        except ValueError:
            continue
        if vcap <= 0.0:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            return (vcap, ts_str)
        if (datetime.now() - ts).total_seconds() > fresh_sec:
            return (None, ts_str)
        return (vcap, ts_str)
    return (None, None)


def vcap_guard_recover(context: str) -> dict[str, Any] | None:
    """If Vcap is below VCAP_LOW_V, drive lights to max and wait for recovery.

    Caller MUST re-apply the target PWM after this returns (the normal
    apply_led_pair_with_policy in the next step does that).
    Returns telemetry dict on activation, or None if no action was needed
    (vcap above threshold, stale, or guard disabled).
    """
    if not VCAP_CSV_PATH:
        return None
    vcap, ts = read_latest_vcap_from_csv(VCAP_CSV_PATH, fresh_sec=VCAP_FRESH_SEC)
    if vcap is None:
        log(f"vcap-guard[{context}]: skip (no fresh reading; last_ts={ts})")
        return None
    if vcap >= VCAP_LOW_V:
        return None

    log("===== TEST INTERRUPTED: Vcap recovery =====")
    log(f"vcap-guard[{context}]: vcap={vcap:.3f}V < {VCAP_LOW_V}V; "
        f"driving lights to RGB(255,255,255) full white until vcap >= "
        f"{VCAP_RECOVER_V}V. Any spectrometer/sensor readings during this "
        f"window are NOT included in segment_table.csv (recovery only runs "
        f"between segments).")
    started = now_iso_ms()
    if TRANSPORT == "ha":
        retry_call(lambda: _gc.set_led_full_white_ha(
            poll_sec=HA_POLL_SEC, timeout_sec=HA_TIMEOUT_SEC,
        ))
    else:
        retry_call(lambda: _gc.set_led_full_white())
    deadline = time.monotonic() + max(1.0, float(VCAP_MAX_WAIT_SEC))
    last_v = vcap
    while time.monotonic() < deadline:
        time.sleep(max(0.5, float(VCAP_POLL_SEC)))
        v, _ = read_latest_vcap_from_csv(VCAP_CSV_PATH, fresh_sec=VCAP_FRESH_SEC)
        if v is None:
            continue
        last_v = v
        if v >= VCAP_RECOVER_V:
            ended = now_iso_ms()
            duration_s = None
            try:
                duration_s = (datetime.fromisoformat(ended)
                              - datetime.fromisoformat(started)).total_seconds()
            except Exception:
                pass
            log(f"vcap-guard[{context}]: recovered to {v:.3f}V; "
                f"interrupted for {duration_s}s; resuming test (data during "
                f"interruption discarded)")
            log("===== TEST RESUMED =====")
            return {
                "phase": "vcap_recover", "context": context,
                "vcap_start": vcap, "vcap_end": v,
                "started_at": started, "ended_at": ended,
                "interrupted_sec": duration_s,
                "led_mode": "full_white_rgb_255_255_255",
                "data_discarded": True,
            }
    raise RuntimeError(
        f"vcap-guard[{context}]: did not reach {VCAP_RECOVER_V}V within "
        f"{VCAP_MAX_WAIT_SEC}s (last vcap={last_v:.3f}V); aborting to avoid "
        f"recording aligned-but-undervolted data"
    )


def set_led_pair(pwm_r: int, pwm_b: int) -> dict[str, Any] | None:
    """Drive both bars with the same red+blue mixed RGB.

    BLE: fire-and-forget; alignment uses caller-supplied settle_ms.
    HA : verifies HA state changed (last_reported advanced + rgb match)
         and returns {sent_at, applied_at, latency_ms, ha_state}, so the
         caller can use applied_at for spectrometer alignment.
    """
    pr = clamp_pwm(pwm_r)
    pb = clamp_pwm(pwm_b)
    if TRANSPORT == "ha":
        return retry_call(
            lambda: _gc.set_led_mix_ha(
                pr, pb,
                poll_sec=HA_POLL_SEC,
                timeout_sec=HA_TIMEOUT_SEC,
            )
        )
    retry_call(lambda: _gc.set_led_mix(pr, pb))
    return None


def apply_led_pair_with_policy(
    pwm_r: int,
    pwm_b: int,
    failure_policy: str,
    retry_sec: float,
    context: str,
    settle_ms: int,
) -> dict[str, Any]:
    """Apply a (pwm_r, pwm_b) target and report timing for alignment.

    BLE path: send, sleep settle_ms (fixed), return.
    HA path : send + poll HA until verified; settle_ms ignored because we
              already wait for ground-truth convergence (and record real
              latency for downstream alignment).
    On send failure, 'abort' re-raises; 'wait' retries every retry_sec.
    """
    interval = max(0.5, float(retry_sec))
    settle_sec = max(0.0, float(settle_ms) / 1000.0)
    while True:
        request_master_time = now_iso_ms()
        try:
            ha_info = set_led_pair(pwm_r, pwm_b)
            if TRANSPORT == "ha" and ha_info is not None:
                applied = ha_info.get("applied_at") or now_iso_ms()
                return {
                    "request_master_time": request_master_time,
                    "applied_master_time": applied,
                    "transport": "ha",
                    "ha_latency_ms": ha_info.get("latency_ms"),
                    "ha_state": ha_info.get("ha_state"),
                    "red_status": {"brightness": clamp_pwm(pwm_r), "output": pwm_r > 0, "raw": None},
                    "blue_status": {"brightness": clamp_pwm(pwm_b), "output": pwm_b > 0, "raw": None},
                }
            time.sleep(settle_sec)
            return {
                "request_master_time": request_master_time,
                "applied_master_time": now_iso_ms(),
                "transport": "ble",
                "red_status": {"brightness": clamp_pwm(pwm_r), "output": pwm_r > 0, "raw": None},
                "blue_status": {"brightness": clamp_pwm(pwm_b), "output": pwm_b > 0, "raw": None},
            }
        except Exception as exc:
            if failure_policy == "abort":
                raise
            log(
                f"WARNING: LED set failed during {context}: {exc}. "
                f"Keep waiting {interval:.1f}s and retry..."
            )
            time.sleep(interval)


def ensure_led_off() -> None:
    try:
        set_led_pair(0, 0)
    except Exception as exc:
        log(f"WARNING: failed to force LEDs off: {exc}")


def extract_ppfd_from_parsed(parsed: dict[str, Any] | None) -> float | None:
    return extract_metric_from_parsed(parsed, PPFD_CANDIDATE_KEYS)


def extract_metric_from_parsed(
    parsed: dict[str, Any] | None,
    candidate_keys: tuple[str, ...],
) -> float | None:
    if not isinstance(parsed, dict):
        return None
    metrics = parsed.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    for key in candidate_keys:
        if key in metrics:
            try:
                return float(metrics[key])
            except Exception:
                continue
    return None


def _close_serial_quietly(ser: serial.Serial | None) -> None:
    if ser is None:
        return
    try:
        if ser.is_open:
            ser.close()
    except Exception:
        pass


@contextlib.contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _port_usb_id(port_info: Any) -> tuple[int | None, int | None]:
    return getattr(port_info, "vid", None), getattr(port_info, "pid", None)


def resolve_spectrometer_port(preferred_port: str) -> str:
    """Resolve spectrometer serial port, avoiding known Riotee gateway ports."""
    if list_ports is None:
        return preferred_port

    ports = list(list_ports.comports())
    by_dev = {p.device: p for p in ports}
    preferred = by_dev.get(preferred_port)

    if preferred is not None:
        usb_id = _port_usb_id(preferred)
        if usb_id in RIOTEE_USB_IDS:
            log(
                f"WARNING: {preferred_port} is Riotee Gateway ({preferred.description}), "
                "auto-switching to spectrometer port."
            )
        else:
            return preferred_port

    for p in ports:
        usb_id = _port_usb_id(p)
        if usb_id in SPECTROMETER_USB_IDS:
            if p.device != preferred_port:
                log(f"Auto-detected spectrometer port: {p.device} ({p.description})")
            return p.device

    return preferred_port


def open_spectrometer_port(
    port: str,
    baudrate: int = 115200,
    timeout: float = 1.0,
    retries: int = 8,
    retry_sec: float = 2.0,
    fail_ok: bool = False,
) -> serial.Serial | None:
    attempts = max(1, int(retries))
    interval = max(0.1, float(retry_sec))
    last_err: Exception | None = None
    for idx in range(1, attempts + 1):
        try:
            ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
            log(f"Spectrometer connected: {port} (attempt {idx}/{attempts})")
            return ser
        except Exception as exc:
            last_err = exc
            log(
                f"WARNING: failed to open spectrometer {port} "
                f"(attempt {idx}/{attempts}): {exc}"
            )
            if idx < attempts:
                time.sleep(interval)

    if fail_ok:
        return None
    if last_err is None:
        raise RuntimeError(f"Failed to open spectrometer: {port}")
    raise RuntimeError(f"Failed to open spectrometer: {port}: {last_err}")


def initialize_spectrometer_on_connection(
    ser: serial.Serial | None,
    serial_lock: threading.Lock | None = None,
    reason: str = "",
) -> bool:
    if ser is None or not getattr(ser, "is_open", False):
        return False
    lock_ctx = serial_lock if serial_lock is not None else contextlib.nullcontext()
    msg_reason = f" ({reason})" if reason else ""
    log(f"Initialize spectrometer now{msg_reason}: send 8C 00 connect command")
    try:
        with lock_ctx:
            initialize_spectrometer_session(ser)
        return True
    except Exception as exc:
        log(f"WARNING: spectrometer init failed{msg_reason}: {exc}")
        return False


def trigger_spectrometer_ppfd(
    ser: serial.Serial | None,
    spec_port: str,
    run_dir: Path,
    meas_retries: int = 3,
    meas_retry_sec: float = 1.0,
    open_retries: int = 8,
    open_retry_sec: float = 2.0,
    serial_lock: threading.Lock | None = None,
) -> tuple[float | None, float | None, float | None, str | None, serial.Serial | None]:
    attempts = max(1, int(meas_retries))
    retry_interval = max(0.1, float(meas_retry_sec))
    current_ser = ser
    serial_exc_type = getattr(serial, "SerialException", Exception)

    for idx in range(1, attempts + 1):
        if current_ser is None or not getattr(current_ser, "is_open", False):
            current_ser = open_spectrometer_port(
                spec_port,
                retries=open_retries,
                retry_sec=open_retry_sec,
                fail_ok=True,
            )
            if current_ser is None:
                log(
                    f"WARNING: spectrometer unavailable before trigger "
                    f"(attempt {idx}/{attempts})"
                )
                if idx < attempts:
                    time.sleep(retry_interval)
                continue
            initialize_spectrometer_on_connection(
                current_ser,
                serial_lock=serial_lock,
                reason="after reconnect",
            )

        try:
            lock_ctx = serial_lock if serial_lock is not None else contextlib.nullcontext()
            with lock_ctx:
                with _pushd(run_dir):
                    result = complete_spectrum_measurement(current_ser)
            if not isinstance(result, tuple) or len(result) < 1:
                return None, None, None, None, current_ser

            spectrum_result = result[0]
            if not isinstance(spectrum_result, tuple) or len(spectrum_result) < 3:
                return None, None, None, None, current_ser

            standard_csv_path = spectrum_result[1]
            if isinstance(standard_csv_path, str) and standard_csv_path:
                path_obj = Path(standard_csv_path)
                if not path_obj.is_absolute():
                    path_obj = (run_dir / path_obj).resolve()
                else:
                    path_obj = path_obj.resolve()

                # Force spectrometer csv under current run_dir even if library wrote elsewhere.
                try:
                    path_obj.relative_to(run_dir.resolve())
                    normalized = path_obj
                except ValueError:
                    rel = Path("archive") / datetime.now().strftime("%Y-%m-%d") / "standard_csv" / path_obj.name
                    normalized = (run_dir / rel).resolve()
                    normalized.parent.mkdir(parents=True, exist_ok=True)
                    if path_obj.exists() and path_obj != normalized:
                        shutil.copy2(path_obj, normalized)

                try:
                    standard_csv_path = str(normalized.relative_to(run_dir.resolve()))
                except ValueError:
                    standard_csv_path = str(normalized)
            parsed = spectrum_result[2]
            ppfd = extract_ppfd_from_parsed(parsed)
            ppfd_blue = extract_metric_from_parsed(parsed, PPFD_BLUE_CANDIDATE_KEYS)
            ppfd_red = extract_metric_from_parsed(parsed, PPFD_RED_CANDIDATE_KEYS)
            if ppfd is None:
                return None, ppfd_blue, ppfd_red, standard_csv_path, current_ser
            return ppfd, ppfd_blue, ppfd_red, standard_csv_path, current_ser
        except Exception as exc:
            is_serial_error = isinstance(exc, serial_exc_type) or isinstance(exc, OSError)
            msg = str(exc)
            if (
                "Input/output error" in msg
                or "device reports readiness to read but returned no data" in msg
                or "write failed" in msg
            ):
                is_serial_error = True

            log(
                f"WARNING: spectrometer trigger failed (attempt {idx}/{attempts}): {exc}"
            )
            if is_serial_error:
                _close_serial_quietly(current_ser)
                current_ser = None
                log("Spectrometer serial reset requested after I/O failure.")
            if idx < attempts:
                time.sleep(retry_interval)

    return None, None, None, None, current_ser


def create_segment_conditions(
    ratios: list[tuple[int, int]],
    totals: list[int],
    drift_every: int,
    drift_pwm_r: int,
    drift_pwm_b: int,
) -> list[SegmentCondition]:
    segment_id = 0
    normal_idx = 0
    out: list[SegmentCondition] = []

    for ratio_r, ratio_b in ratios:
        for total_pwm in totals:
            normal_idx += 1
            segment_id += 1
            pwm_r, pwm_b = ratio_to_pwm(ratio_r, ratio_b, total_pwm)
            out.append(
                SegmentCondition(
                    segment_id=segment_id,
                    segment_type="normal",
                    ratio_r=ratio_r,
                    ratio_b=ratio_b,
                    total_pwm=total_pwm,
                    pwm_r=pwm_r,
                    pwm_b=pwm_b,
                    condition_index=normal_idx,
                )
            )
            if drift_every > 0 and normal_idx % drift_every == 0:
                segment_id += 1
                out.append(
                    SegmentCondition(
                        segment_id=segment_id,
                        segment_type="drift_ref",
                        ratio_r=drift_pwm_r,
                        ratio_b=drift_pwm_b,
                        total_pwm=drift_pwm_r + drift_pwm_b,
                        pwm_r=clamp_pwm(drift_pwm_r),
                        pwm_b=clamp_pwm(drift_pwm_b),
                        condition_index=normal_idx,
                    )
                )
    return out


def start_sensor_collector(run_dir: Path, comment: str) -> tuple[subprocess.Popen[str], Path]:
    script = RIOTEE_DIR / "riotee_data_collector.py"
    if not script.exists():
        raise FileNotFoundError(f"Riotee collector not found: {script}")

    cmd = [sys.executable, str(script), "sensor_timeseries", comment]
    log(f"Starting sensor collector: {' '.join(cmd)}")
    sensor_log = run_dir / "sensor_collector.log"
    log_fp = sensor_log.open("a", encoding="utf-8")

    proc = subprocess.Popen(
        cmd,
        cwd=str(run_dir),
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    time.sleep(3.0)
    if proc.poll() is not None:
        log_fp.close()
        raise RuntimeError(f"Sensor collector exited early. Check log: {sensor_log}")
    setattr(proc, "_sensor_log_fp", log_fp)

    expected_all_csv = run_dir / "logs" / "sensor_timeseries_all.csv"
    return proc, expected_all_csv


def stop_sensor_collector(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return

    if proc.poll() is None:
        log("Stopping sensor collector...")
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    log_fp = getattr(proc, "_sensor_log_fp", None)
    if log_fp is not None:
        try:
            log_fp.close()
        except Exception:
            pass


def copy_sensor_timeseries(expected_all_csv: Path, run_dir: Path) -> Path | None:
    dst = run_dir / "sensor_timeseries.csv"
    if not expected_all_csv.exists():
        log(f"WARNING: sensor raw CSV not found: {expected_all_csv}")
        return None
    shutil.copy2(expected_all_csv, dst)
    log(f"sensor_timeseries.csv generated: {dst}")
    return dst


def _count_sensor_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    count = 0
    header_seen = False
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                first = row[0].strip() if row else ""
                if not first:
                    continue
                if first.startswith("#"):
                    continue
                if not header_seen:
                    header_seen = True
                    continue
                count += 1
    except Exception:
        return 0
    return count


def latest_sensor_sleep_time(sensor_csv_path: Path) -> float | None:
    if not sensor_csv_path.exists():
        return None
    last_sleep: float | None = None
    try:
        with sensor_csv_path.open("r", encoding="utf-8", newline="") as f:
            filtered = (line for line in f if line.strip() and not line.lstrip().startswith("#"))
            reader = csv.DictReader(filtered)
            for row in reader:
                val = safe_float(row.get("sleep_time"))
                if isinstance(val, (int, float)) and val > 0:
                    last_sleep = float(val)
    except Exception:
        return None
    return last_sleep


def wait_for_sensor_data(
    sensor_csv_path: Path,
    sensor_proc: subprocess.Popen[str] | None = None,
    poll_sec: float = 2.0,
    charge_after_sec: float = 3600.0,
) -> None:
    interval = max(0.5, float(poll_sec))
    charge_pwm_r = 50
    charge_pwm_b = 50
    start_mono = time.monotonic()
    charge_enabled = False
    next_charge_try_mono = start_mono + max(0.0, float(charge_after_sec))
    log(
        "Waiting for first sensor data row before starting experiment "
        f"(csv: {sensor_csv_path})"
    )
    last_log_mono = 0.0
    while True:
        if sensor_proc is not None and sensor_proc.poll() is not None:
            raise RuntimeError(
                "Sensor collector exited while waiting for first sensor row. "
                "Check sensor_collector.log in run directory."
            )
        row_count = _count_sensor_rows(sensor_csv_path)
        if row_count > 0:
            log(f"Sensor data ready: detected {row_count} row(s). Continue.")
            return

        now_mono = time.monotonic()
        if now_mono >= next_charge_try_mono and not charge_enabled:
            elapsed = now_mono - start_mono
            log(
                "No sensor data for {:.1f}s (likely first run / long idle). "
                "Set LEDs to charge mode pwm_r={}, pwm_b={} and keep waiting.".format(
                    elapsed, charge_pwm_r, charge_pwm_b
                )
            )
            try:
                set_led_pair(charge_pwm_r, charge_pwm_b)
                charge_enabled = True
            except Exception as exc:
                log(f"WARNING: failed to enable charge mode: {exc}")
                next_charge_try_mono = now_mono + 10.0

        if now_mono - last_log_mono >= 10.0:
            log("No sensor data yet, keep waiting...")
            last_log_mono = now_mono
        time.sleep(interval)


def try_set_sensor_sleep(
    ts_seconds: int,
    target_device: str,
    retries: int = 6,
    retry_interval_sec: float = 5.0,
) -> bool:
    script = RIOTEE_DIR / "riotee_controller.py"
    if not script.exists():
        log(f"WARNING: sensor controller script missing: {script}")
        return False
    attempts = max(1, int(retries))
    interval = max(0.0, float(retry_interval_sec))
    cmd = [sys.executable, str(script), target_device, "sleep", str(ts_seconds)]

    for idx in range(1, attempts + 1):
        log(
            f"Setting sensor sleep_time={ts_seconds}s "
            f"(attempt {idx}/{attempts}) via: {' '.join(cmd)}"
        )
        run = subprocess.run(cmd, capture_output=True, text=True)
        out = run.stdout.strip()
        err = run.stderr.strip()

        if run.returncode == 0:
            if out:
                log(out)
            return True

        log("WARNING: failed to set sensor sleep time")
        if out:
            log(f"stdout: {out}")
        if err:
            log(f"stderr: {err}")

        if idx < attempts:
            time.sleep(interval)
    return False


def compute_timing(ts: float, n_spec: int, time_scale: float = 1.0) -> TimingConfig:
    t_off = 2.0 * ts
    t_blue = 2.0 * ts
    t_settle = max(10.0, 2.0 * ts)
    t_meas = 10.0 * ts
    scale = max(0.01, float(time_scale))
    t_off *= scale
    t_blue *= scale
    t_settle *= scale
    t_meas *= scale
    t_seg = t_off + t_blue + t_settle + t_meas
    return TimingConfig(
        Ts=ts,
        T_off=t_off,
        T_blue=t_blue,
        T_settle=t_settle,
        T_meas=t_meas,
        T_seg=t_seg,
        N_spec=n_spec,
    )


def build_segment_table_header(n_spec: int) -> list[str]:
    base = [
        "segment_id",
        "segment_type",
        "condition_index",
        "ratio_r",
        "ratio_b",
        "pwm_r",
        "pwm_b",
        "total_pwm",
        "rb_ratio_pwm",
        "segment_start_master_time",
        "marker_off_request_master_time",
        "marker_off_start_master_time",
        "marker_off_delay_ms",
        "marker_blue_request_master_time",
        "marker_blue_start_master_time",
        "marker_blue_delay_ms",
        "marker_end_master_time",
        "target_pwm_request_master_time",
        "target_pwm_applied_master_time",
        "target_pwm_delay_ms",
        "steady_start_master_time",
        "steady_end_master_time",
        "segment_end_master_time",
        "Ts",
        "T_off",
        "T_blue",
        "T_settle",
        "T_meas",
        "N_spec",
    ]
    for i in range(1, n_spec + 1):
        base.extend([f"ppfd_{i}", f"ppfd_{i}_time", f"spec_file_{i}"])
    base.extend(["ppfd_valid_count", "PPFD_spec_mean"])
    return base


def run_single_segment(
    segment: SegmentCondition,
    timing: TimingConfig,
    run_dir: Path,
    spec_ser: serial.Serial | None,
    spec_port: str,
    spec_meas_retries: int,
    spec_meas_retry_sec: float,
    spec_open_retries: int,
    spec_open_retry_sec: float,
    serial_lock: threading.Lock | None,
    led_failure_policy: str,
    led_retry_sec: float,
    led_settle_ms: int,
) -> tuple[dict[str, Any], serial.Serial | None]:
    log(
        "Segment {sid} [{typ}] -> target pwm_r={r}, pwm_b={b}, ratio={rr}:{rb}, total={tot}".format(
            sid=segment.segment_id,
            typ=segment.segment_type,
            r=segment.pwm_r,
            b=segment.pwm_b,
            rr=segment.ratio_r,
            rb=segment.ratio_b,
            tot=segment.total_pwm,
        )
    )

    segment_start_time = now_iso_ms()
    write_sensor_control_state(
        run_dir,
        pwm_r=0,
        pwm_b=0,
        phase="marker_off",
        segment_id=segment.segment_id,
    )
    off_apply = apply_led_pair_with_policy(
        0,
        0,
        failure_policy=led_failure_policy,
        retry_sec=led_retry_sec,
        context=f"segment {segment.segment_id} marker OFF",
        settle_ms=led_settle_ms,
    )
    marker_off_request = off_apply["request_master_time"]
    marker_off_start = off_apply["applied_master_time"]
    time.sleep(timing.T_off)

    write_sensor_control_state(
        run_dir,
        pwm_r=0,
        pwm_b=100,
        phase="marker_blue",
        segment_id=segment.segment_id,
    )
    blue_apply = apply_led_pair_with_policy(
        0,
        100,
        failure_policy=led_failure_policy,
        retry_sec=led_retry_sec,
        context=f"segment {segment.segment_id} marker BLUE",
        settle_ms=led_settle_ms,
    )
    marker_blue_request = blue_apply["request_master_time"]
    marker_blue_start = blue_apply["applied_master_time"]
    time.sleep(timing.T_blue)
    marker_end_time = now_iso_ms()

    write_sensor_control_state(
        run_dir,
        pwm_r=segment.pwm_r,
        pwm_b=segment.pwm_b,
        phase="target",
        segment_id=segment.segment_id,
    )
    target_apply = apply_led_pair_with_policy(
        segment.pwm_r,
        segment.pwm_b,
        failure_policy=led_failure_policy,
        retry_sec=led_retry_sec,
        context=f"segment {segment.segment_id} target PWM",
        settle_ms=led_settle_ms,
    )
    target_pwm_request = target_apply["request_master_time"]
    target_pwm_applied = target_apply["applied_master_time"]
    time.sleep(timing.T_settle)

    marker_off_delay_ms = iso_diff_ms(marker_off_request, marker_off_start)
    marker_blue_delay_ms = iso_diff_ms(marker_blue_request, marker_blue_start)
    target_pwm_delay_ms = iso_diff_ms(target_pwm_request, target_pwm_applied)

    steady_start_time = now_iso_ms()
    steady_window_mono = time.monotonic()

    ppfd_values: list[float | None] = []
    ppfd_blue_values: list[float | None] = []
    ppfd_red_values: list[float | None] = []
    ppfd_times: list[str | None] = []
    spec_files: list[str | None] = []

    offsets = [timing.T_meas * (i + 1) / (timing.N_spec + 1) for i in range(timing.N_spec)]
    for idx, offset_sec in enumerate(offsets, start=1):
        target = steady_window_mono + offset_sec
        now_mono = time.monotonic()
        if target > now_mono:
            time.sleep(target - now_mono)

        trigger_time = now_iso_ms()
        log(f"  -> spectrometer trigger {idx}/{timing.N_spec}")
        ppfd_val, ppfd_blue_val, ppfd_red_val, spec_file, spec_ser = trigger_spectrometer_ppfd(
            spec_ser,
            spec_port=spec_port,
            run_dir=run_dir,
            meas_retries=spec_meas_retries,
            meas_retry_sec=spec_meas_retry_sec,
            open_retries=spec_open_retries,
            open_retry_sec=spec_open_retry_sec,
            serial_lock=serial_lock,
        )
        if ppfd_val is None:
            log(f"  -> trigger {idx} failed to parse PPFD")
        else:
            log(f"  -> trigger {idx} PPFD={ppfd_val:.6f}")
        ppfd_values.append(ppfd_val)
        ppfd_blue_values.append(ppfd_blue_val)
        ppfd_red_values.append(ppfd_red_val)
        ppfd_times.append(trigger_time)
        spec_files.append(spec_file)

    elapsed_meas = time.monotonic() - steady_window_mono
    if elapsed_meas < timing.T_meas:
        time.sleep(timing.T_meas - elapsed_meas)
    steady_end_time = now_iso_ms()
    segment_end_time = now_iso_ms()

    valid_ppfd = [v for v in ppfd_values if isinstance(v, (int, float))]
    valid_ppfd_blue = [v for v in ppfd_blue_values if isinstance(v, (int, float))]
    valid_ppfd_red = [v for v in ppfd_red_values if isinstance(v, (int, float))]
    ppfd_mean = float(mean(valid_ppfd)) if valid_ppfd else None
    ppfd_blue_mean = float(mean(valid_ppfd_blue)) if valid_ppfd_blue else None
    ppfd_red_mean = float(mean(valid_ppfd_red)) if valid_ppfd_red else None

    summary_payload = {
        "comment_id": f"{segment.segment_id}-{now_iso_ms()}",
        "segment_id": segment.segment_id,
        "pwm_r": segment.pwm_r,
        "pwm_b": segment.pwm_b,
        "rb_ratio_pwm": rb_ratio_from_pwm(segment.pwm_r, segment.pwm_b),
        "PPFD_spec_mean": ppfd_mean,
        "ppfd_red_mean": ppfd_red_mean,
        "ppfd_blue_mean": ppfd_blue_mean,
    }
    write_pending_segment_comment(run_dir, summary_payload)

    row: dict[str, Any] = {
        "segment_id": segment.segment_id,
        "segment_type": segment.segment_type,
        "condition_index": segment.condition_index,
        "ratio_r": segment.ratio_r,
        "ratio_b": segment.ratio_b,
        "pwm_r": segment.pwm_r,
        "pwm_b": segment.pwm_b,
        "total_pwm": segment.total_pwm,
        "rb_ratio_pwm": rb_ratio_from_pwm(segment.pwm_r, segment.pwm_b),
        "segment_start_master_time": segment_start_time,
        "marker_off_request_master_time": marker_off_request,
        "marker_off_start_master_time": marker_off_start,
        "marker_off_delay_ms": marker_off_delay_ms,
        "marker_blue_request_master_time": marker_blue_request,
        "marker_blue_start_master_time": marker_blue_start,
        "marker_blue_delay_ms": marker_blue_delay_ms,
        "marker_end_master_time": marker_end_time,
        "target_pwm_request_master_time": target_pwm_request,
        "target_pwm_applied_master_time": target_pwm_applied,
        "target_pwm_delay_ms": target_pwm_delay_ms,
        "steady_start_master_time": steady_start_time,
        "steady_end_master_time": steady_end_time,
        "segment_end_master_time": segment_end_time,
        "Ts": timing.Ts,
        "T_off": timing.T_off,
        "T_blue": timing.T_blue,
        "T_settle": timing.T_settle,
        "T_meas": timing.T_meas,
        "N_spec": timing.N_spec,
        "ppfd_valid_count": len(valid_ppfd),
        "PPFD_spec_mean": ppfd_mean,
    }
    for i in range(timing.N_spec):
        n = i + 1
        row[f"ppfd_{n}"] = ppfd_values[i]
        row[f"ppfd_{n}_time"] = ppfd_times[i]
        row[f"spec_file_{n}"] = spec_files[i]
    return row, spec_ser


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_sensor_control_state(
    run_dir: Path,
    *,
    pwm_r: int,
    pwm_b: int,
    phase: str,
    segment_id: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "updated_at": now_iso_ms(),
        "pwm_r": int(clamp_pwm(pwm_r)),
        "pwm_b": int(clamp_pwm(pwm_b)),
        "phase": phase,
        "segment_id": int(segment_id) if segment_id is not None else None,
    }
    if extra:
        payload.update(extra)
    save_json(run_dir / "current_led_state.json", payload)


def write_pending_segment_comment(run_dir: Path, payload: dict[str, Any]) -> None:
    save_json(run_dir / "pending_segment_comment.json", payload)


def build_condition_summary_for_run(run_dir: Path) -> bool:
    script = ROOT_DIR / "build_condition_summary.py"
    if not script.exists():
        log(f"WARNING: condition summary script not found: {script}")
        return False

    cmd = [sys.executable, str(script), "--run-dir", str(run_dir)]
    log(f"Generating condition_summary.csv: {' '.join(cmd)}")
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        log("WARNING: condition summary generation failed")
        if run.stdout.strip():
            log(f"stdout: {run.stdout.strip()}")
        if run.stderr.strip():
            log(f"stderr: {run.stderr.strip()}")
        return False

    if run.stdout.strip():
        for line in run.stdout.strip().splitlines():
            log(f"[condition_summary] {line}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LED marker-based calibration data collection")
    parser.add_argument("--ts", type=float, default=10.0, help="Sensor sleep_time in seconds (default: 10)")
    parser.add_argument(
        "--ratios",
        default="1:0,4:1,2:1,1:1,1:2,1:4,0:1",
        help="Comma-separated ratio list, e.g. 1:0,4:1,1:1,0:1",
    )
    parser.add_argument(
        "--totals",
        default="20,40,60,80",
        help="Comma-separated total PWM list in [0,100], e.g. 20,40,60,80",
    )
    parser.add_argument("--n-spec", type=int, default=3, help="Spectrometer triggers per segment (default: 3)")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Scale factor for T_off/T_blue/T_settle/T_meas (default: 1.0)",
    )
    parser.add_argument(
        "--quick-run",
        action="store_true",
        help="Quick smoke run preset: same as --time-scale 0.1 --prewarm-min 0",
    )
    parser.add_argument("--prewarm-min", type=float, default=3.0, help="Prewarm minutes at pwm 50/50")
    parser.add_argument("--prewarm-r", type=int, default=50, help="Prewarm red PWM")
    parser.add_argument("--prewarm-b", type=int, default=50, help="Prewarm blue PWM")
    parser.add_argument(
        "--led-failure-policy",
        choices=["wait", "abort"],
        default="wait",
        help="LED BLE failure policy (default: wait)",
    )
    parser.add_argument(
        "--led-retry-sec",
        type=float,
        default=5.0,
        help="Retry interval seconds when LED BLE write fails and policy=wait (default: 5.0)",
    )
    parser.add_argument(
        "--led-settle-ms",
        type=int,
        default=300,
        help="Fixed delay after each BLE write before treating it as applied (default: 300)",
    )
    parser.add_argument(
        "--ble-adapter",
        default=None,
        help="HCI adapter for BLE (e.g. hci0, hci1). Default: system default.",
    )
    parser.add_argument("--drift-every", type=int, default=0, help="Insert drift segment every N normal segments")
    parser.add_argument("--drift-pwm-r", type=int, default=60, help="Drift reference red PWM")
    parser.add_argument("--drift-pwm-b", type=int, default=60, help="Drift reference blue PWM")
    parser.add_argument(
        "--spec-port",
        default="/dev/ttySpectrometer",
        help="Spectrometer serial port (default: /dev/ttySpectrometer)",
    )
    parser.add_argument(
        "--spec-open-retries",
        type=int,
        default=8,
        help="Retries when opening spectrometer serial (default: 8)",
    )
    parser.add_argument(
        "--spec-open-retry-sec",
        type=float,
        default=2.0,
        help="Retry interval seconds when opening spectrometer serial (default: 2.0)",
    )
    parser.add_argument(
        "--spec-meas-retries",
        type=int,
        default=3,
        help="Retries per PPFD trigger when spectrometer read fails (default: 3)",
    )
    parser.add_argument(
        "--spec-meas-retry-sec",
        type=float,
        default=1.0,
        help="Retry interval seconds per PPFD trigger retry (default: 1.0)",
    )
    parser.add_argument("--sensor-mode", choices=["auto", "external", "off"], default="auto")
    parser.add_argument(
        "--sensor-charge-after-sec",
        type=float,
        default=3600.0,
        help="No-sensor-data timeout before enabling charge mode (default: 3600s)",
    )
    parser.add_argument(
        "--set-sensor-sleep",
        action="store_true",
        help="Send sensor sleep_time command before run (default: off)",
    )
    parser.add_argument(
        "--sensor-device-id",
        default="all",
        help="Riotee controller device target when --set-sensor-sleep is enabled",
    )
    parser.add_argument(
        "--sensor-sleep-retries",
        type=int,
        default=6,
        help="Retry count for setting sensor sleep_time when --set-sensor-sleep is enabled",
    )
    parser.add_argument(
        "--sensor-sleep-retry-sec",
        type=float,
        default=5.0,
        help="Retry interval seconds for setting sensor sleep_time when --set-sensor-sleep is enabled",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "outputs"),
        help="Base output directory (run folder will be created inside)",
    )
    parser.add_argument(
        "--skip-condition-summary",
        action="store_true",
        help="Skip auto-generation of condition_summary.csv at run end",
    )
    parser.add_argument("--run-name", default="", help="Optional run name suffix")
    parser.add_argument(
        "--transport", choices=["ble", "ha"], default="ble",
        help="LED control path: ble (default, local BLE) or ha (Home Assistant cloud, verifies state convergence for alignment)",
    )
    parser.add_argument(
        "--ha-timeout-sec", type=float, default=8.0,
        help="HA path: max seconds to wait for HA state convergence (default: 8.0)",
    )
    parser.add_argument(
        "--ha-poll-sec", type=float, default=0.3,
        help="HA path: poll interval while waiting for convergence (default: 0.3)",
    )
    parser.add_argument(
        "--vcap-guard", action="store_true",
        help="Before each segment, check Riotee Vcap. If < --vcap-low, force lights to max until >= --vcap-recover.",
    )
    parser.add_argument(
        "--vcap-csv",
        default=str((TOOL_DIR / "Sensor_riotee_server" / "logs" / "riotee_data_all.csv").resolve()),
        help="Path to riotee_data_all.csv for Vcap monitoring",
    )
    parser.add_argument("--vcap-low", type=float, default=3.3,
                        help="Trigger threshold (V); recover if Vcap < this")
    parser.add_argument("--vcap-recover", type=float, default=4.5,
                        help="Resume threshold (V); proceed when Vcap >= this")
    parser.add_argument("--vcap-max-wait-sec", type=float, default=600.0,
                        help="Max seconds to wait for Vcap recovery before aborting")
    parser.add_argument("--vcap-poll-sec", type=float, default=2.0,
                        help="Vcap poll interval while waiting for recovery")
    parser.add_argument("--vcap-fresh-sec", type=float, default=120.0,
                        help="A Vcap reading older than this is treated as stale and ignored")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.ts <= 0:
        raise ValueError("--ts must be > 0")
    if args.n_spec <= 0:
        raise ValueError("--n-spec must be > 0")

    effective_time_scale = args.time_scale
    if args.quick_run:
        if abs(args.time_scale - 1.0) < 1e-12:
            effective_time_scale = 0.1
        if abs(args.prewarm_min - 3.0) < 1e-12:
            args.prewarm_min = 0.0
        log(
            "Quick run enabled: time-scale={:.3f}, prewarm-min={:.3f}".format(
                effective_time_scale, args.prewarm_min
            )
        )

    ratios = parse_ratio_list(args.ratios)
    totals = parse_int_list(args.totals)
    timing = compute_timing(args.ts, args.n_spec, time_scale=effective_time_scale)
    args.spec_port = resolve_spectrometer_port(args.spec_port)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_stamp}_{args.run_name}" if args.run_name else run_stamp
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    segment_table_path = run_dir / "segment_table.csv"
    run_config_path = run_dir / "run_config.json"
    run_log_path = run_dir / "run_log.txt"

    segments = create_segment_conditions(
        ratios=ratios,
        totals=totals,
        drift_every=args.drift_every,
        drift_pwm_r=args.drift_pwm_r,
        drift_pwm_b=args.drift_pwm_b,
    )

    config_payload = {
        "created_at": now_iso_ms(),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "timing": asdict(timing),
        "ratios": [{"r": r, "b": b} for r, b in ratios],
        "totals": totals,
        "segment_count": len(segments),
        "segments_preview": [asdict(s) for s in segments[: min(10, len(segments))]],
        "args": vars(args),
        "devices": DEVICES,
    }
    save_json(run_config_path, config_payload)

    with run_log_path.open("a", encoding="utf-8") as run_log:
        run_log.write(f"[{now_iso_ms()}] run_dir={run_dir}\n")
        run_log.write(f"[{now_iso_ms()}] segment_count={len(segments)}\n")

    sensor_proc: subprocess.Popen[str] | None = None
    sensor_all_csv: Path | None = None
    spec_ser: serial.Serial | None = None
    spec_lock = threading.Lock()

    try:
        log(f"Run dir: {run_dir}")
        log(f"Segments: {len(segments)}")
        log(
            "Timing Ts={Ts}s, T_off={T_off}s, T_blue={T_blue}s, T_settle={T_settle}s, T_meas={T_meas}s".format(
                **asdict(timing)
            )
        )

        if args.sensor_mode == "auto":
            sensor_proc, sensor_all_csv = start_sensor_collector(run_dir, comment=f"run={run_name}")
        elif args.sensor_mode == "external":
            log("Sensor mode=external, expecting collector started separately.")
        else:
            log("Sensor mode=off, sensor CSV will not be generated.")

        write_sensor_control_state(run_dir, pwm_r=0, pwm_b=0, phase="idle", segment_id=None)

        if args.set_sensor_sleep:
            if float(args.ts).is_integer():
                try_set_sensor_sleep(
                    int(args.ts),
                    target_device=args.sensor_device_id,
                    retries=args.sensor_sleep_retries,
                    retry_interval_sec=args.sensor_sleep_retry_sec,
                )
            else:
                log("WARNING: --ts is not an integer; skip setting device sleep_time command.")
        else:
            log("Skip setting sensor sleep_time (assume sensor already configured).")

        if args.sensor_mode == "auto" and sensor_all_csv is not None:
            wait_for_sensor_data(
                sensor_all_csv,
                sensor_proc=sensor_proc,
                poll_sec=2.0,
                charge_after_sec=args.sensor_charge_after_sec,
            )
            sensor_ts = latest_sensor_sleep_time(sensor_all_csv)
            if isinstance(sensor_ts, (int, float)):
                if abs(sensor_ts - timing.Ts) > 0.5:
                    log(
                        "WARNING: sensor sleep_time={}s differs from run Ts={}s. "
                        "Marker/window timing may be mismatched.".format(sensor_ts, timing.Ts)
                    )
                if timing.T_off < sensor_ts or timing.T_blue < sensor_ts:
                    log(
                        "WARNING: marker duration too short for sensor interval. "
                        f"T_off={timing.T_off:.2f}s, T_blue={timing.T_blue:.2f}s, "
                        f"sensor sleep_time={sensor_ts:.2f}s"
                    )

        log(f"Opening spectrometer port: {args.spec_port}")
        spec_ser = open_spectrometer_port(
            args.spec_port,
            retries=args.spec_open_retries,
            retry_sec=args.spec_open_retry_sec,
            fail_ok=True,
        )
        if spec_ser is None:
            log("WARNING: spectrometer not available at startup; will retry during triggers.")
        else:
            initialize_spectrometer_on_connection(
                spec_ser,
                serial_lock=spec_lock,
                reason="startup",
            )

        global TRANSPORT, HA_TIMEOUT_SEC, HA_POLL_SEC
        global VCAP_CSV_PATH, VCAP_LOW_V, VCAP_RECOVER_V, VCAP_MAX_WAIT_SEC, VCAP_POLL_SEC, VCAP_FRESH_SEC
        TRANSPORT = args.transport
        HA_TIMEOUT_SEC = float(args.ha_timeout_sec)
        HA_POLL_SEC = float(args.ha_poll_sec)
        if args.vcap_guard:
            VCAP_CSV_PATH = args.vcap_csv
            VCAP_LOW_V = float(args.vcap_low)
            VCAP_RECOVER_V = float(args.vcap_recover)
            VCAP_MAX_WAIT_SEC = float(args.vcap_max_wait_sec)
            VCAP_POLL_SEC = float(args.vcap_poll_sec)
            VCAP_FRESH_SEC = float(args.vcap_fresh_sec)
            log(f"Vcap guard ON: low={VCAP_LOW_V}V recover={VCAP_RECOVER_V}V "
                f"csv={VCAP_CSV_PATH} fresh<={VCAP_FRESH_SEC}s")
        else:
            VCAP_CSV_PATH = None
            log("Vcap guard OFF (use --vcap-guard to enable)")
        if TRANSPORT == "ble":
            log(f"Connecting to Govee H6056 over BLE (adapter={args.ble_adapter or 'default'})")
            _gc.connect(adapter=args.ble_adapter)
        else:
            log(f"LED transport=ha (HA convergence wait, timeout={HA_TIMEOUT_SEC}s, poll={HA_POLL_SEC}s); skipping BLE connect")

        log("Force LED OFF before prewarm")
        write_sensor_control_state(run_dir, pwm_r=0, pwm_b=0, phase="prewarm_off", segment_id=None)
        apply_led_pair_with_policy(
            0,
            0,
            failure_policy=args.led_failure_policy,
            retry_sec=args.led_retry_sec,
            context="prewarm start force OFF",
            settle_ms=args.led_settle_ms,
        )

        prewarm_sec = max(0.0, args.prewarm_min * 60.0)
        if prewarm_sec > 0:
            log(
                f"Prewarm {args.prewarm_min:.2f} min at pwm_r={clamp_pwm(args.prewarm_r)}, "
                f"pwm_b={clamp_pwm(args.prewarm_b)}"
            )
            write_sensor_control_state(
                run_dir,
                pwm_r=clamp_pwm(args.prewarm_r),
                pwm_b=clamp_pwm(args.prewarm_b),
                phase="prewarm",
                segment_id=None,
            )
            apply_led_pair_with_policy(
                clamp_pwm(args.prewarm_r),
                clamp_pwm(args.prewarm_b),
                failure_policy=args.led_failure_policy,
                retry_sec=args.led_retry_sec,
                context="prewarm setpoint",
                settle_ms=args.led_settle_ms,
            )
            time.sleep(prewarm_sec)

        header = build_segment_table_header(args.n_spec)
        with segment_table_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            for seg in segments:
                if sensor_proc is not None and sensor_proc.poll() is not None:
                    raise RuntimeError(
                        "Sensor collector stopped unexpectedly. "
                        "Check sensor_collector.log in run directory."
                    )
                vcap_guard_recover(context=f"segment_{getattr(seg, 'segment_id', '?')}")
                row, spec_ser = run_single_segment(
                    segment=seg,
                    timing=timing,
                    run_dir=run_dir,
                    spec_ser=spec_ser,
                    spec_port=args.spec_port,
                    spec_meas_retries=args.spec_meas_retries,
                    spec_meas_retry_sec=args.spec_meas_retry_sec,
                    spec_open_retries=args.spec_open_retries,
                    spec_open_retry_sec=args.spec_open_retry_sec,
                    serial_lock=spec_lock,
                    led_failure_policy=args.led_failure_policy,
                    led_retry_sec=args.led_retry_sec,
                    led_settle_ms=args.led_settle_ms,
                )
                writer.writerow(row)
                f.flush()

        log(f"segment_table.csv written: {segment_table_path}")

    except KeyboardInterrupt:
        log("Interrupted by user.")
        return 130
    finally:
        with contextlib.suppress(Exception):
            write_sensor_control_state(run_dir, pwm_r=0, pwm_b=0, phase="finished", segment_id=None)
        ensure_led_off()
        if TRANSPORT == "ble":
            with contextlib.suppress(Exception):
                _gc.disconnect()
        _close_serial_quietly(spec_ser)
        stop_sensor_collector(sensor_proc)
        if sensor_all_csv is not None:
            copied = copy_sensor_timeseries(sensor_all_csv, run_dir)
            if copied is not None and not args.skip_condition_summary:
                build_condition_summary_for_run(run_dir)

    log("Completed.")
    log(f"Output directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
