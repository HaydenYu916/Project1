#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LED calibration data acquisition runner.

This script orchestrates three systems:
1) Shelly LED PWM control (red + blue),
2) spectrometer PPFD sampling,
3) optional Riotee sensor collector process.

Design goal:
- Segment-level alignment with optical marker pattern (OFF -> BLUE -> TARGET).
- Do not align by BLE receive time.
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
LED_SRC_DIR = TOOL_DIR / "LED_Shelly" / "src"
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
    from shelly_controller import DEVICES, rpc
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


def parse_pair_list(text: str) -> list[tuple[int, int]]:
    """Parse direct (R, B) PWM pairs, e.g. '60,80;80,60;100,100'.

    Each pair is separated by ';' (or whitespace/newline). Within a pair,
    R and B are separated by ','. Both values must be in [0, 100].
    Returns [(R, B), ...] with the order preserved.
    """
    pairs: list[tuple[int, int]] = []
    for sep in ("\n", "\t"):
        text = text.replace(sep, ";")
    for item in text.split(";"):
        token = item.strip()
        if not token:
            continue
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair token: {token!r}. Expected 'R,B'")
        r = int(parts[0].strip())
        b = int(parts[1].strip())
        if not (0 <= r <= 100) or not (0 <= b <= 100):
            raise ValueError(f"Pair PWM must be in [0,100], got ({r},{b})")
        if r == 0 and b == 0:
            raise ValueError("Pair (0,0) is invalid; use a non-zero PWM")
        pairs.append((r, b))
    if not pairs:
        raise ValueError("At least one (R,B) pair must be provided")
    return pairs


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


def set_led_pwm(device_name: str, pwm: int, transition_ms: int = 0) -> dict[str, Any]:
    pwm_int = clamp_pwm(pwm)
    ip = DEVICES[device_name]
    payload = {"id": 0, "on": pwm_int > 0, "brightness": pwm_int, "transition": int(transition_ms)}
    resp = retry_call(lambda: rpc(ip, "Light.Set", payload))
    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(f"{device_name} RPC error: {resp['error']}")
    return resp if isinstance(resp, dict) else {"raw": resp}


def set_led_pair(pwm_r: int, pwm_b: int, transition_ms: int = 0) -> None:
    set_led_pwm("Red", pwm_r, transition_ms=transition_ms)
    set_led_pwm("Blue", pwm_b, transition_ms=transition_ms)


def get_led_status(device_name: str) -> dict[str, Any]:
    ip = DEVICES[device_name]
    status = retry_call(lambda: rpc(ip, "Shelly.GetStatus"))
    if not isinstance(status, dict):
        raise RuntimeError(f"{device_name} status returned non-dict response: {status!r}")
    if status.get("error"):
        raise RuntimeError(f"{device_name} status RPC error: {status['error']}")

    light = status.get("light:0", {})
    if not isinstance(light, dict):
        raise RuntimeError(f"{device_name} status missing light:0 payload")

    brightness = safe_float(light.get("brightness"))
    output = light.get("output")
    return {
        "brightness": int(round(brightness)) if brightness is not None else None,
        "output": output,
        "raw": status,
    }


def _status_matches_target(status: dict[str, Any], target_pwm: int) -> bool:
    target = clamp_pwm(target_pwm)
    brightness = status.get("brightness")
    output = status.get("output")

    if target <= 0:
        if brightness == 0:
            return True
        return output is False

    return brightness == target and output is True


def wait_for_led_pair_target(
    pwm_r: int,
    pwm_b: int,
    timeout_sec: float,
    poll_sec: float,
    stable_reads: int,
    transition_ms: int = 0,
) -> dict[str, Any]:
    interval = max(0.05, float(poll_sec))
    required_matches = max(1, int(stable_reads))
    deadline = time.monotonic() + max(float(timeout_sec), (max(0, int(transition_ms)) / 1000.0) + 1.0)
    consecutive_matches = 0
    last_red: dict[str, Any] | None = None
    last_blue: dict[str, Any] | None = None

    while time.monotonic() <= deadline:
        last_red = get_led_status("Red")
        last_blue = get_led_status("Blue")
        red_ok = _status_matches_target(last_red, pwm_r)
        blue_ok = _status_matches_target(last_blue, pwm_b)
        if red_ok and blue_ok:
            consecutive_matches += 1
            if consecutive_matches >= required_matches:
                return {
                    "applied_master_time": now_iso_ms(),
                    "red_status": last_red,
                    "blue_status": last_blue,
                }
        else:
            consecutive_matches = 0
        time.sleep(interval)

    raise RuntimeError(
        "LED status did not reach target within timeout: "
        f"target_r={clamp_pwm(pwm_r)}, target_b={clamp_pwm(pwm_b)}, "
        f"last_red={last_red}, last_blue={last_blue}"
    )


def apply_led_pair_with_policy(
    pwm_r: int,
    pwm_b: int,
    transition_ms: int,
    failure_policy: str,
    retry_sec: float,
    context: str,
    status_timeout_sec: float,
    status_poll_sec: float,
    status_stable_reads: int,
) -> dict[str, Any]:
    interval = max(0.5, float(retry_sec))
    while True:
        request_master_time = now_iso_ms()
        try:
            set_led_pair(pwm_r, pwm_b, transition_ms=transition_ms)
            status_info = wait_for_led_pair_target(
                pwm_r=pwm_r,
                pwm_b=pwm_b,
                timeout_sec=status_timeout_sec,
                poll_sec=status_poll_sec,
                stable_reads=status_stable_reads,
                transition_ms=transition_ms,
            )
            status_info["request_master_time"] = request_master_time
            return status_info
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
        set_led_pair(0, 0, transition_ms=0)
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
    pairs: list[tuple[int, int]] | None = None,
) -> list[SegmentCondition]:
    """Build the segment list from either explicit (R,B) pairs OR a
    ratios×totals cartesian grid.

    When `pairs` is given (and non-empty) it is used verbatim and the
    ratios/totals arguments are ignored — this is the only way to sample
    the high-power region (R≥50 AND B≥50) since ratio×total is bounded
    by total ≤ 100. Each pair becomes one normal segment; ratio_r/
    ratio_b columns store the raw PWM values for downstream filtering.
    """
    segment_id = 0
    normal_idx = 0
    out: list[SegmentCondition] = []

    iterator: list[tuple[int, int, int, int, int]]
    if pairs:
        iterator = [(r, b, r + b, r, b) for (r, b) in pairs]
    else:
        iterator = []
        for ratio_r, ratio_b in ratios:
            for total_pwm in totals:
                pwm_r, pwm_b = ratio_to_pwm(ratio_r, ratio_b, total_pwm)
                iterator.append((pwm_r, pwm_b, total_pwm, ratio_r, ratio_b))

    for pwm_r, pwm_b, total_pwm, ratio_r_col, ratio_b_col in iterator:
        normal_idx += 1
        segment_id += 1
        out.append(
            SegmentCondition(
                segment_id=segment_id,
                segment_type="normal",
                ratio_r=ratio_r_col,
                ratio_b=ratio_b_col,
                total_pwm=total_pwm,
                pwm_r=clamp_pwm(pwm_r),
                pwm_b=clamp_pwm(pwm_b),
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
                set_led_pair(charge_pwm_r, charge_pwm_b, transition_ms=0)
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
    transition_ms: int,
    led_status_timeout_sec: float,
    led_status_poll_sec: float,
    led_status_stable_reads: int,
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
        transition_ms=0,
        failure_policy=led_failure_policy,
        retry_sec=led_retry_sec,
        context=f"segment {segment.segment_id} marker OFF",
        status_timeout_sec=led_status_timeout_sec,
        status_poll_sec=led_status_poll_sec,
        status_stable_reads=led_status_stable_reads,
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
        transition_ms=transition_ms,
        failure_policy=led_failure_policy,
        retry_sec=led_retry_sec,
        context=f"segment {segment.segment_id} marker BLUE",
        status_timeout_sec=led_status_timeout_sec,
        status_poll_sec=led_status_poll_sec,
        status_stable_reads=led_status_stable_reads,
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
        transition_ms=transition_ms,
        failure_policy=led_failure_policy,
        retry_sec=led_retry_sec,
        context=f"segment {segment.segment_id} target PWM",
        status_timeout_sec=led_status_timeout_sec,
        status_poll_sec=led_status_poll_sec,
        status_stable_reads=led_status_stable_reads,
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
    parser.add_argument(
        "--pairs",
        default=None,
        help=(
            "Direct (R,B) PWM pairs separated by ';'. If set, --ratios and "
            "--totals are ignored. This is the only way to sample R≥50 AND "
            "B≥50 (ratios×totals is capped at total≤100). "
            "Example: '60,60;80,80;100,100;60,80;80,60'"
        ),
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
    parser.add_argument("--transition-ms", type=int, default=0, help="Shelly transition duration in ms")
    parser.add_argument(
        "--led-failure-policy",
        choices=["wait", "abort"],
        default="wait",
        help="LED RPC failure policy (default: wait)",
    )
    parser.add_argument(
        "--led-retry-sec",
        type=float,
        default=5.0,
        help="Retry interval seconds when LED RPC fails and policy=wait (default: 5.0)",
    )
    parser.add_argument(
        "--led-status-timeout-sec",
        type=float,
        default=8.0,
        help="Timeout seconds to wait until Shelly status reaches target brightness (default: 8.0)",
    )
    parser.add_argument(
        "--led-status-poll-sec",
        type=float,
        default=0.2,
        help="Polling interval seconds for Shelly.GetStatus confirmation (default: 0.2)",
    )
    parser.add_argument(
        "--led-status-stable-reads",
        type=int,
        default=2,
        help="Consecutive matching Shelly.GetStatus reads required to confirm target PWM (default: 2)",
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
    pairs = parse_pair_list(args.pairs) if args.pairs else None
    if pairs:
        log(f"Direct-pairs mode: {len(pairs)} (R,B) segments; --ratios/--totals ignored")
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
        pairs=pairs,
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

        log("Force LED OFF before prewarm")
        write_sensor_control_state(run_dir, pwm_r=0, pwm_b=0, phase="prewarm_off", segment_id=None)
        apply_led_pair_with_policy(
            0,
            0,
            transition_ms=0,
            failure_policy=args.led_failure_policy,
            retry_sec=args.led_retry_sec,
            context="prewarm start force OFF",
            status_timeout_sec=args.led_status_timeout_sec,
            status_poll_sec=args.led_status_poll_sec,
            status_stable_reads=args.led_status_stable_reads,
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
                transition_ms=args.transition_ms,
                failure_policy=args.led_failure_policy,
                retry_sec=args.led_retry_sec,
                context="prewarm setpoint",
                status_timeout_sec=args.led_status_timeout_sec,
                status_poll_sec=args.led_status_poll_sec,
                status_stable_reads=args.led_status_stable_reads,
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
                    transition_ms=args.transition_ms,
                    led_status_timeout_sec=args.led_status_timeout_sec,
                    led_status_poll_sec=args.led_status_poll_sec,
                    led_status_stable_reads=args.led_status_stable_reads,
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
