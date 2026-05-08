import csv
import datetime
import json
import logging
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib

import paho.mqtt.client as mqtt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
MODEL_BASE_DIR = PROJECT_ROOT / "Tool" / "Model"
SHELLY_BASE_DIR = PROJECT_ROOT / "Tool" / "LED_Shelly"
RIOTEE_SERVER_DIR = PROJECT_ROOT / "Tool" / "Sensor_riotee_server"
os.environ.setdefault("LIVE_LOGS_DIR", str(RIOTEE_SERVER_DIR / "logs"))
os.environ.setdefault("LIVE_CSV_PATH", str(RIOTEE_SERVER_DIR / "logs" / "riotee_data_all.csv"))
for module_dir in (
    BASE_DIR,
    MODEL_BASE_DIR / "PWMtoPPFD",
    MODEL_BASE_DIR / "SPtoPPFD",
    MODEL_BASE_DIR / "EnvtoPN",
    SHELLY_BASE_DIR / "src",
    SHELLY_BASE_DIR / "config",
    RIOTEE_SERVER_DIR,
):
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)


import predict_pwm_from_ppfd as pwm_model
import predict_sp_to_ppfd as sp_model
import predict_env_to_pn as pn_model
from riotee_live_api import get_device_latest_data as get_local_riotee_device_latest_data


try:
    from shelly_controller import DEVICES, rpc
except ImportError:
    DEVICES = {"Red": "dev_red", "Blue": "dev_blue"}

    def rpc(dev, cmd, params):
        logger.debug("[MOCK] dev=%s cmd=%s params=%s", dev, cmd, params)


TIMEZONE = ZoneInfo("Australia/Sydney")

CONFIG_PATH = BASE_DIR / "edge_config.toml"
DEFAULT_SENSOR_FIELD_MAP = {
    "power": "Power_now_w",
}


def load_edge_config():
    with CONFIG_PATH.open("rb") as f:
        raw_config = tomllib.load(f)

    mqtt_cfg = raw_config.get("mqtt", {})
    topic_cfg = raw_config.get("topics", {})
    runtime_cfg = raw_config.get("runtime", {})
    co2_cfg = raw_config.get("co2", {})
    sensor_topics = raw_config.get("sensor_topics", {})

    missing_sensor_keys = [
        key for key in DEFAULT_SENSOR_FIELD_MAP if key not in sensor_topics
    ]
    if missing_sensor_keys:
        raise ValueError(
            f"edge_config.toml missing sensor_topics entries: {', '.join(missing_sensor_keys)}"
        )

    topic_map = {
        sensor_topics[sensor_key]: field_name
        for sensor_key, field_name in DEFAULT_SENSOR_FIELD_MAP.items()
    }

    return {
        "timezone": raw_config.get("timezone", {}).get("name", "Australia/Sydney"),
        "mqtt_broker_ip": mqtt_cfg.get("broker", "azure.nocolor.cc"),
        "mqtt_port": int(mqtt_cfg.get("port", 1883)),
        "mqtt_user": mqtt_cfg.get("username", ""),
        "mqtt_pass": mqtt_cfg.get("password", ""),
        "topic_cmd": topic_cfg.get("command", "growbox/commands/setpoints"),
        "topic_state": topic_cfg.get("state", "growbox/state/aggregated"),
        "use_fixed_co2": bool(co2_cfg.get("use_fixed", True)),
        "fixed_co2_ppm": float(co2_cfg.get("fixed_ppm", 400.0)),
        "offline_timeout_seconds": int(runtime_cfg.get("offline_timeout_seconds", 1800)),
        "publish_interval_seconds": int(runtime_cfg.get("publish_interval_seconds", 900)),
        "topic_map": topic_map,
        "power_topic": sensor_topics["power"],
        "temperature_topic": sensor_topics["temperature"],
        "sensor_topics": sensor_topics,
    }


EDGE_CONFIG = load_edge_config()
TIMEZONE = ZoneInfo(EDGE_CONFIG["timezone"])

# --- 1. DIRECTORY & LOGGING SETUP ---
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
SENSOR_SNAPSHOT_DIR = LOG_DIR / "sensor_snapshots"
SENSOR_SNAPSHOT_DIR.mkdir(exist_ok=True)
CONTROL_STATE_FILE = LOG_DIR / "edge_control_state.json"

timestamp_str = datetime.datetime.now(TIMEZONE).strftime("%Y%m%d")
log_filename = LOG_DIR / f"edge_node_{timestamp_str}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("GreenhouseEdge")

SENSOR_SNAPSHOT_HEADERS = [
    "id",
    "timestamp",
    "temperature",
    "humidity",
    "a1_raw",
    "vcap_raw",
    "sleep_time",
    "sp_415",
    "sp_445",
    "sp_480",
    "sp_515",
    "sp_555",
    "sp_590",
    "sp_630",
    "sp_680",
    "co2_ppm",
    "ppfd_pred",
    "pn_pred",
    "power_now_w",
    "target_ppfd",
    "actual_ppfd",
    "red_pwm",
    "blue_pwm",
]


def sensor_snapshot_csv_path_for_day(ts=None):
    ts = datetime.datetime.now(TIMEZONE) if ts is None else ts
    return SENSOR_SNAPSHOT_DIR / f"{ts.strftime('%Y%m%d')}.csv"


def ensure_sensor_snapshot_csv_file(ts=None):
    ts = datetime.datetime.now(TIMEZONE) if ts is None else ts
    snapshot_csv = sensor_snapshot_csv_path_for_day(ts)
    if snapshot_csv.exists():
        current_header = None
        try:
            with snapshot_csv.open(mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or row[0].startswith("#"):
                        continue
                    current_header = row
                    break
        except Exception as exc:
            logger.warning("Failed to inspect existing sensor snapshot header: %s", exc)

        if current_header == SENSOR_SNAPSHOT_HEADERS:
            return snapshot_csv

        legacy_stamp = datetime.datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")
        backup_file = SENSOR_SNAPSHOT_DIR / f"{snapshot_csv.stem}_legacy_{legacy_stamp}.csv"
        snapshot_csv.rename(backup_file)
        logger.warning(
            "Sensor snapshot header changed; moved previous file to %s",
            backup_file,
        )

    start_line = f"# Start @ {ts.strftime('%Y-%m-%d %H:%M:%S')} - edge_sensor_snapshot"
    with snapshot_csv.open(mode="w", newline="", encoding="utf-8") as f:
        f.write(start_line + "\n")
        csv.writer(f).writerow(SENSOR_SNAPSHOT_HEADERS)
    return snapshot_csv


def next_sensor_snapshot_id(snapshot_csv):
    if not snapshot_csv.exists():
        return 1

    last_id = 0
    try:
        with snapshot_csv.open(mode="r", newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("#") or row[0] == "id":
                    continue
                try:
                    last_id = int(row[0])
                except ValueError:
                    continue
    except Exception:
        logger.exception("Failed to inspect snapshot CSV for next id: %s", snapshot_csv)
    return last_id + 1


def sensor_snapshot_row_from_state(row_id, ts, snapshot_state):
    return [
        row_id,
        ts.strftime("%Y-%m-%d %H:%M:%S"),
        safe_csv_value(snapshot_state.get("Tleaf")),
        safe_csv_value(snapshot_state.get("RH")),
        safe_csv_value(snapshot_state.get("a1_raw")),
        safe_csv_value(snapshot_state.get("vcap_raw")),
        safe_csv_value(snapshot_state.get("sleep_time")),
        safe_csv_value(snapshot_state.get("sp_415")),
        safe_csv_value(snapshot_state.get("sp_445")),
        safe_csv_value(snapshot_state.get("sp_480")),
        safe_csv_value(snapshot_state.get("sp_515")),
        safe_csv_value(snapshot_state.get("sp_555")),
        safe_csv_value(snapshot_state.get("sp_590")),
        safe_csv_value(snapshot_state.get("sp_630")),
        safe_csv_value(snapshot_state.get("sp_680")),
        safe_csv_value(snapshot_state.get("Ci")),
        safe_csv_value(snapshot_state.get("PPFD_pred")),
        safe_csv_value(snapshot_state.get("Pn_pred")),
        safe_csv_value(snapshot_state.get("Power_now_w")),
        snapshot_state.get("requested_target_ppfd", 0),
        snapshot_state.get("actual_target_ppfd", 0),
        snapshot_state.get("current_red_pwm", 0),
        snapshot_state.get("current_blue_pwm", 0),
    ]


def flush_sensor_snapshot(ts, snapshot_state):
    snapshot_csv = ensure_sensor_snapshot_csv_file(ts)
    try:
        row_id = next_sensor_snapshot_id(snapshot_csv)
        with snapshot_csv.open(mode="a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(sensor_snapshot_row_from_state(row_id, ts, snapshot_state))
    except Exception:
        logger.exception("Failed to write sensor snapshot row for timestamp=%s", ts)


def current_snapshot_state():
    actual_ppfd = applied_ppfd_for_pwm(
        env_state.get("current_red_pwm", 0),
        env_state.get("current_blue_pwm", 0),
    )
    return {
        "Tleaf": env_state.get("Tleaf"),
        "RH": env_state.get("RH"),
        "a1_raw": env_state.get("a1_raw"),
        "vcap_raw": env_state.get("vcap_raw"),
        "sleep_time": env_state.get("sleep_time"),
        "sp_415": env_state.get("sp_415"),
        "sp_445": env_state.get("sp_445"),
        "sp_480": env_state.get("sp_480"),
        "sp_515": env_state.get("sp_515"),
        "sp_555": env_state.get("sp_555"),
        "sp_590": env_state.get("sp_590"),
        "sp_630": env_state.get("sp_630"),
        "sp_680": env_state.get("sp_680"),
        "Ci": env_state.get("Ci"),
        "PPFD_pred": env_state.get("PPFD_pred"),
        "Pn_pred": env_state.get("Pn_pred"),
        "Power_now_w": env_state.get("Power_now_w"),
        "requested_target_ppfd": int_target_ppfd(env_state.get("target_ppfd", 0.0)),
        "actual_target_ppfd": int_target_ppfd(actual_ppfd),
        "current_red_pwm": env_state.get("current_red_pwm", 0),
        "current_blue_pwm": env_state.get("current_blue_pwm", 0),
    }


def get_pwm_saturation(red_pwm, blue_pwm):
    red_sat = int(red_pwm >= 100)
    blue_sat = int(blue_pwm >= 100)
    if red_sat and blue_sat:
        label = "both"
    elif red_sat:
        label = "red"
    elif blue_sat:
        label = "blue"
    else:
        label = "none"
    return red_sat or blue_sat, label


def clamp_pwm_value(value):
    try:
        pwm_value = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    if pwm_value <= 0:
        return 0
    return max(4, min(100, pwm_value))


def effective_target_ppfd(requested_ppfd, red_pwm, blue_pwm):
    requested_ppfd = float(requested_ppfd or 0.0)
    red_pwm = clamp_pwm_value(red_pwm)
    blue_pwm = clamp_pwm_value(blue_pwm)

    if red_pwm == 0 and blue_pwm == 0:
        return 0.0

    model = globals().get("pwm_pkg")
    if not model:
        return requested_ppfd

    try:
        actual_ppfd = float(model["a_r"]) * float(red_pwm) + float(model["b_b"]) * float(blue_pwm)
    except Exception:
        logger.exception(
            "Failed to map applied PWM back to PPFD; using requested target_ppfd instead."
        )
        return requested_ppfd

    return actual_ppfd if abs(actual_ppfd - requested_ppfd) > 1e-6 else requested_ppfd


def applied_ppfd_for_pwm(red_pwm, blue_pwm):
    red_pwm = clamp_pwm_value(red_pwm)
    blue_pwm = clamp_pwm_value(blue_pwm)

    if red_pwm == 0 and blue_pwm == 0:
        return 0.0

    model = globals().get("pwm_pkg")
    if not model:
        return 0.0

    try:
        return float(model["a_r"]) * float(red_pwm) + float(model["b_b"]) * float(blue_pwm)
    except Exception:
        logger.exception("Failed to map applied PWM back to actual PPFD.")
        return 0.0


def int_target_ppfd(value):
    try:
        return int(round(float(value or 0.0)))
    except (TypeError, ValueError):
        return 0


pending_sensor_snapshot_second = None
pending_sensor_snapshot_state = None
LIGHT_STATUS_TIMEOUT_SECONDS = 5.0
LIGHT_STATUS_POLL_SECONDS = 0.2
LIGHT_APPLY_MAX_ATTEMPTS = 3
LIGHT_APPLY_RETRY_DELAY_SECONDS = 1.0


def queue_sensor_snapshot(now=None):
    global pending_sensor_snapshot_second, pending_sensor_snapshot_state

    now = datetime.datetime.now(TIMEZONE) if now is None else now
    current_second = now.replace(microsecond=0)

    if pending_sensor_snapshot_second is None:
        pending_sensor_snapshot_second = current_second
    elif current_second != pending_sensor_snapshot_second:
        if pending_sensor_snapshot_state is not None:
            flush_sensor_snapshot(pending_sensor_snapshot_second, pending_sensor_snapshot_state)
        pending_sensor_snapshot_second = current_second

    pending_sensor_snapshot_state = current_snapshot_state()


def flush_sensor_snapshot_if_due(now=None):
    global pending_sensor_snapshot_second, pending_sensor_snapshot_state

    if pending_sensor_snapshot_second is None or pending_sensor_snapshot_state is None:
        return

    now = datetime.datetime.now(TIMEZONE) if now is None else now
    current_second = now.replace(microsecond=0)
    if current_second <= pending_sensor_snapshot_second:
        return

    flush_sensor_snapshot(pending_sensor_snapshot_second, pending_sensor_snapshot_state)
    pending_sensor_snapshot_second = None
    pending_sensor_snapshot_state = None


def flush_pending_sensor_snapshot():
    global pending_sensor_snapshot_second, pending_sensor_snapshot_state

    if pending_sensor_snapshot_second is None or pending_sensor_snapshot_state is None:
        return

    flush_sensor_snapshot(pending_sensor_snapshot_second, pending_sensor_snapshot_state)
    pending_sensor_snapshot_second = None
    pending_sensor_snapshot_state = None


# --- 3. CONFIGURATION ---
MQTT_BROKER_IP = EDGE_CONFIG["mqtt_broker_ip"]
MQTT_PORT = EDGE_CONFIG["mqtt_port"]
MQTT_USER = EDGE_CONFIG["mqtt_user"]
MQTT_PASS = EDGE_CONFIG["mqtt_pass"]
USE_FIXED_CO2 = EDGE_CONFIG["use_fixed_co2"]
FIXED_CO2_PPM = EDGE_CONFIG["fixed_co2_ppm"]

TOPIC_CMD = EDGE_CONFIG["topic_cmd"]
TOPIC_STATE = EDGE_CONFIG["topic_state"]
TOPIC_MAP = EDGE_CONFIG["topic_map"]
POWER_TOPIC = EDGE_CONFIG["power_topic"]
LOCAL_DEVICE_TOPIC_ID = EDGE_CONFIG["temperature_topic"].split("/")[1]
LOCAL_DEVICE_ID = f"{LOCAL_DEVICE_TOPIC_ID}=="

STATE_REQUIRED_FIELDS = ["Tleaf", "PPFD_pred", "Pn_pred"]
if not USE_FIXED_CO2:
    STATE_REQUIRED_FIELDS.append("Ci")
ALL_SENSOR_FIELDS = list(TOPIC_MAP.values())
LOCAL_SENSOR_EXTRA_FIELDS = [
    "a1_raw",
    "vcap_raw",
    "sleep_time",
    "sp_415",
    "sp_445",
    "sp_480",
    "sp_515",
    "sp_555",
    "sp_590",
    "sp_630",
    "sp_680",
]

# Global State Tracker
env_state = {k: None for k in ALL_SENSOR_FIELDS + LOCAL_SENSOR_EXTRA_FIELDS}
env_state.update(
    {
        "Ci": FIXED_CO2_PPM if USE_FIXED_CO2 else None,
        "current_red_pwm": 0,
        "current_blue_pwm": 0,
        "target_ppfd": 0.0,
        "PPFD_pred": 0.0,
        "Pn_pred": 0.0,
    }
)

# Runtime status
OFFLINE_TIMEOUT_SECONDS = EDGE_CONFIG["offline_timeout_seconds"]
PUBLISH_INTERVAL = EDGE_CONFIG["publish_interval_seconds"]
last_command_time = time.time()
is_offline_mode = False
mqtt_connected = False
startup_time = time.time()
last_local_sensor_timestamp = None
STATE_HISTORY_RETENTION_SECONDS = 15 * 60
SHORT_WINDOW_SECONDS = 3 * 60
LONG_WINDOW_SECONDS = 15 * 60
state_history = deque()


def format_field_list(fields):
    return ",".join(fields) if fields else "none"


def get_missing_fields(fields):
    missing_fields = []
    for field in fields:
        value = env_state.get(field)
        if value is None:
            missing_fields.append(field)
            continue
        if isinstance(value, (int, float)) and not math.isfinite(value):
            missing_fields.append(field)
    return missing_fields


def record_state_snapshot():
    snapshot = {
        "timestamp": time.time(),
        "Tleaf": env_state.get("Tleaf"),
        "Ci": env_state.get("Ci"),
        "PPFD_pred": env_state.get("PPFD_pred"),
        "Pn_pred": env_state.get("Pn_pred"),
        "Power_now_w": env_state.get("Power_now_w"),
    }
    state_history.append(snapshot)
    cutoff = snapshot["timestamp"] - STATE_HISTORY_RETENTION_SECONDS
    while state_history and state_history[0]["timestamp"] < cutoff:
        state_history.popleft()


def _window_snapshots(window_seconds):
    if not state_history:
        return []
    cutoff = time.time() - window_seconds
    return [snapshot for snapshot in state_history if snapshot["timestamp"] >= cutoff]


def _average_from_window(field_name, window_seconds, fallback_value):
    values = []
    for snapshot in _window_snapshots(window_seconds):
        value = snapshot.get(field_name)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    if values:
        return sum(values) / len(values)
    return fallback_value


def _delta_from_window(field_name, window_seconds, fallback_value=0.0):
    current_value = env_state.get(field_name)
    if not isinstance(current_value, (int, float)) or not math.isfinite(current_value):
        return fallback_value

    earliest_value = None
    for snapshot in _window_snapshots(window_seconds):
        value = snapshot.get(field_name)
        if isinstance(value, (int, float)) and math.isfinite(value):
            earliest_value = float(value)
            break

    if earliest_value is None:
        return fallback_value
    return float(current_value) - earliest_value


def build_server_payload():
    now = datetime.datetime.now(TIMEZONE)
    pwm_saturated, pwm_saturation_label = get_pwm_saturation(
        env_state["current_red_pwm"],
        env_state["current_blue_pwm"],
    )
    last_target_ppfd = float(env_state.get("target_ppfd", 0.0) or 0.0)
    cur_r = float(env_state["current_red_pwm"] or 0)
    cur_b = float(env_state["current_blue_pwm"] or 0)
    total_pwm = cur_r + cur_b
    last_blue_share = (cur_b / total_pwm) if total_pwm > 0 else 0.0
    return {
        "local_time": now.strftime("%H:%M"),
        "timezone": EDGE_CONFIG["timezone"],
        "is_day": 1 if 7 <= now.hour < 23 else 0,
        "tleaf_now": env_state["Tleaf"],
        "co2_now": env_state["Ci"],
        "ppfd_now": env_state["PPFD_pred"],
        "pn_now": env_state["Pn_pred"],
        "power_now_w": env_state["Power_now_w"],
        "current_red_pwm": env_state["current_red_pwm"],
        "current_blue_pwm": env_state["current_blue_pwm"],
        "pwm_saturated": int(pwm_saturated),
        "pwm_saturation_label": pwm_saturation_label,
        "tleaf_avg_3min": _average_from_window("Tleaf", SHORT_WINDOW_SECONDS, env_state["Tleaf"]),
        "pn_avg_3min": _average_from_window("Pn_pred", SHORT_WINDOW_SECONDS, env_state["Pn_pred"]),
        "tleaf_delta_15min": _delta_from_window("Tleaf", LONG_WINDOW_SECONDS),
        "pn_delta_15min": _delta_from_window("Pn_pred", LONG_WINDOW_SECONDS),
        "last_target_ppfd": last_target_ppfd,
        "last_blue_share": last_blue_share,
        "sensor_data_valid": current_data_valid(),
        "missing_fields": get_missing_fields(STATE_REQUIRED_FIELDS),
    }


def current_data_valid():
    return not get_missing_fields(STATE_REQUIRED_FIELDS)


def safe_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return f"{value:.2f}" if math.isfinite(value) else ""
    return value


def persist_control_state():
    payload = {
        "target_ppfd": float(env_state.get("target_ppfd", 0.0) or 0.0),
        "current_red_pwm": clamp_pwm_value(env_state.get("current_red_pwm", 0)),
        "current_blue_pwm": clamp_pwm_value(env_state.get("current_blue_pwm", 0)),
        "saved_at": datetime.datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        CONTROL_STATE_FILE.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to persist control state to %s", CONTROL_STATE_FILE)


def read_last_control_state_from_snapshots():
    snapshot_csv = sensor_snapshot_csv_path_for_day()
    if not snapshot_csv.exists():
        return None

    try:
        with snapshot_csv.open(mode="r", newline="", encoding="utf-8") as f:
            data_rows = [
                row for row in csv.reader(f)
                if row and not row[0].startswith("#")
            ]
        if not data_rows:
            return None

        header = data_rows[0]
        rows = data_rows[1:]
        header_index = {name: idx for idx, name in enumerate(header)}
        target_idx = header_index.get("target_ppfd")
        if target_idx is None:
            target_idx = header_index.get("last_target_ppfd")
        red_idx = header_index.get("red_pwm")
        blue_idx = header_index.get("blue_pwm")

        if target_idx is None or red_idx is None or blue_idx is None:
            return None

        for row in reversed(rows):
            if len(row) <= max(target_idx, red_idx, blue_idx):
                continue
            target_ppfd = safe_float(row[target_idx])
            red_pwm = clamp_pwm_value(row[red_idx] or 0)
            blue_pwm = clamp_pwm_value(row[blue_idx] or 0)
            if target_ppfd is None:
                continue
            if target_ppfd > 0 or red_pwm > 0 or blue_pwm > 0:
                return {
                    "target_ppfd": target_ppfd,
                    "current_red_pwm": red_pwm,
                    "current_blue_pwm": blue_pwm,
                }
    except Exception:
        logger.exception("Failed to read fallback control state from %s", snapshot_csv)
    return None


def restore_control_state():
    payload = None
    if CONTROL_STATE_FILE.exists():
        try:
            payload = json.loads(CONTROL_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to restore control state from %s", CONTROL_STATE_FILE)

    if payload:
        env_state["target_ppfd"] = float(payload.get("target_ppfd", 0.0) or 0.0)
        env_state["current_red_pwm"] = clamp_pwm_value(payload.get("current_red_pwm", 0))
        env_state["current_blue_pwm"] = clamp_pwm_value(payload.get("current_blue_pwm", 0))

    if (
        env_state.get("target_ppfd", 0.0) == 0.0
        and env_state.get("current_red_pwm", 0) == 0
        and env_state.get("current_blue_pwm", 0) == 0
    ):
        fallback_state = read_last_control_state_from_snapshots()
        if fallback_state:
            env_state["target_ppfd"] = float(fallback_state["target_ppfd"])
            env_state["current_red_pwm"] = clamp_pwm_value(fallback_state["current_red_pwm"])
            env_state["current_blue_pwm"] = clamp_pwm_value(fallback_state["current_blue_pwm"])
            logger.info(
                "Restored control state from sensor snapshots: target_ppfd=%.2f red_pwm=%d blue_pwm=%d",
                env_state["target_ppfd"],
                env_state["current_red_pwm"],
                env_state["current_blue_pwm"],
            )
            return

    logger.info(
        "Restored control state from disk: target_ppfd=%.2f red_pwm=%d blue_pwm=%d",
        env_state["target_ppfd"],
        env_state["current_red_pwm"],
        env_state["current_blue_pwm"],
    )


def apply_current_light_state():
    red_pwm = clamp_pwm_value(env_state.get("current_red_pwm", 0))
    blue_pwm = clamp_pwm_value(env_state.get("current_blue_pwm", 0))
    apply_and_confirm_light_state(
        float(env_state.get("target_ppfd", 0.0) or 0.0),
        red_pwm,
        blue_pwm,
        "restored",
    )


def apply_device_command(device_key, command, params):
    if device_key not in DEVICES:
        logger.debug("Skipping %s command because the device is not configured.", device_key)
        return None

    response = rpc(DEVICES[device_key], command, params)
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(str(response["error"]))
    return response


def get_device_light_status(device_key):
    if device_key not in DEVICES:
        return None

    response = rpc(DEVICES[device_key], "Shelly.GetStatus")
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(str(response["error"]))
    if not isinstance(response, dict):
        raise RuntimeError(
            f"Unexpected status payload type for {device_key}: {type(response).__name__}"
        )

    light_status = response.get("light:0", {})
    if not isinstance(light_status, dict):
        raise RuntimeError(f"Missing light:0 status payload for {device_key}")
    return light_status


def light_status_matches_target(light_status, target_pwm):
    brightness = safe_float(light_status.get("brightness"))
    output = bool(light_status.get("output"))
    if target_pwm <= 0:
        return (brightness in (None, 0)) or (brightness == 0 and not output)
    return brightness is not None and int(round(brightness)) == int(target_pwm) and output


def apply_and_confirm_light_state(target_ppfd, red_pwm, blue_pwm, source_label):
    desired = {
        "Red": int(red_pwm),
        "Blue": int(blue_pwm),
    }
    last_status = {}
    last_send_errors = {}

    for attempt in range(1, LIGHT_APPLY_MAX_ATTEMPTS + 1):
        send_errors = {}

        for device_key, pwm_value in desired.items():
            try:
                apply_device_command(
                    device_key,
                    "Light.Set",
                    {"id": 0, "on": pwm_value > 0, "brightness": pwm_value},
                )
            except Exception as exc:
                send_errors[device_key] = str(exc)
                logger.exception(
                    "Failed to send %s light command for %s on attempt %d/%d: target_ppfd=%.2f requested_pwm=%d",
                    source_label,
                    device_key,
                    attempt,
                    LIGHT_APPLY_MAX_ATTEMPTS,
                    target_ppfd,
                    pwm_value,
                )

        deadline = time.time() + LIGHT_STATUS_TIMEOUT_SECONDS
        last_status = {}
        while time.time() < deadline:
            all_matched = True
            for device_key, pwm_value in desired.items():
                try:
                    light_status = get_device_light_status(device_key)
                except Exception:
                    logger.exception(
                        "Failed to read %s status while verifying PWM on attempt %d/%d",
                        device_key,
                        attempt,
                        LIGHT_APPLY_MAX_ATTEMPTS,
                    )
                    all_matched = False
                    continue

                last_status[device_key] = light_status
                if not light_status_matches_target(light_status, pwm_value):
                    all_matched = False

            if all_matched:
                env_state["current_red_pwm"] = desired["Red"]
                env_state["current_blue_pwm"] = desired["Blue"]
                if send_errors:
                    logger.warning(
                        "Confirmed requested %s light state on attempt %d/%d despite send errors: target_ppfd=%.2f errors=%s",
                        source_label,
                        attempt,
                        LIGHT_APPLY_MAX_ATTEMPTS,
                        target_ppfd,
                        send_errors,
                    )
                persist_control_state()
                return True

            time.sleep(LIGHT_STATUS_POLL_SECONDS)

        last_send_errors = send_errors
        if attempt < LIGHT_APPLY_MAX_ATTEMPTS:
            logger.warning(
                "Retrying %s light command after attempt %d/%d: target_ppfd=%.2f requested_red=%d requested_blue=%d last_red=%s last_blue=%s send_errors=%s",
                source_label,
                attempt,
                LIGHT_APPLY_MAX_ATTEMPTS,
                target_ppfd,
                red_pwm,
                blue_pwm,
                last_status.get("Red"),
                last_status.get("Blue"),
                send_errors,
            )
            time.sleep(LIGHT_APPLY_RETRY_DELAY_SECONDS)

    logger.warning(
        "Light status did not reach requested %s target after %d attempts: target_ppfd=%.2f requested_red=%d requested_blue=%d last_red=%s last_blue=%s send_errors=%s",
        source_label,
        LIGHT_APPLY_MAX_ATTEMPTS,
        target_ppfd,
        red_pwm,
        blue_pwm,
        last_status.get("Red"),
        last_status.get("Blue"),
        last_send_errors,
    )
    return False


def parse_nonnegative_number(raw_value, field_name, upper_bound=None):
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc

    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    if upper_bound is not None and value > upper_bound:
        raise ValueError(f"{field_name} must be <= {upper_bound}")

    return value


def safe_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_latest_local_sensor_row():
    try:
        return get_local_riotee_device_latest_data(LOCAL_DEVICE_ID, include_spectral=True, include_config=False)
    except Exception:
        logger.exception("Failed to fetch local Riotee data via riotee_live_api.")
        return None


def compute_local_ppfd_pn(row):
    spectral = row.get("spectral") or {}
    spectrum = {key: safe_float(spectral.get(key)) for key in (
        "sp_415", "sp_445", "sp_480", "sp_515", "sp_555", "sp_590", "sp_630", "sp_680"
    )}
    has_spectral_data = all(value is not None for value in spectrum.values())
    if not has_spectral_data:
        return None, None

    temp = safe_float(row.get("temperature"))
    if temp is None:
        return None, None

    co2 = FIXED_CO2_PPM if USE_FIXED_CO2 else safe_float(row.get("co2_ppm"))
    if co2 is None:
        return None, None

    try:
        sp_input = {
            "sp_415_mean": spectrum["sp_415"],
            "sp_445_mean": spectrum["sp_445"],
            "sp_480_mean": spectrum["sp_480"],
            "sp_515_mean": spectrum["sp_515"],
            "sp_555_mean": spectrum["sp_555"],
            "sp_590_mean": spectrum["sp_590"],
            "sp_630_mean": spectrum["sp_630"],
            "sp_680_mean": spectrum["sp_680"],
        }
        sp_df = pd.DataFrame([sp_input])
        X_sp = sp_model.prepare_features(sp_df, sp_pkg_local["feature_columns"])
        ppfd_pred = float(sp_pkg_local["pipeline"].predict(X_sp)[0])

        pn_input = {
            "T": temp,
            "CO2": co2,
            "R:B": 0.75,
            "PPFD": ppfd_pred,
        }
        pn_df = pd.DataFrame([pn_input])
        X_pn = pn_model.prepare_features(pn_df, pn_pkg_local["feature_columns"])
        pn_pred = float(pn_pkg_local["pipeline"].predict(X_pn)[0])
        return ppfd_pred, pn_pred
    except Exception:
        logger.exception("Failed to compute local PPFD/Pn from Riotee CSV row.")
        return None, None


def sync_local_sensor_state():
    global last_local_sensor_timestamp

    row = read_latest_local_sensor_row()
    if row is None:
        return

    row_timestamp = row.get("timestamp")
    if row_timestamp == last_local_sensor_timestamp:
        return

    spectral = row.get("spectral") or {}
    env_state["Tleaf"] = safe_float(row.get("temperature"))
    env_state["RH"] = safe_float(row.get("humidity"))
    env_state["a1_raw"] = safe_float(row.get("a1_raw"))
    env_state["vcap_raw"] = safe_float(row.get("vcap_raw"))
    env_state["sleep_time"] = safe_float(row.get("sleep_time"))
    for field_name in (
        "sp_415", "sp_445", "sp_480", "sp_515", "sp_555", "sp_590", "sp_630", "sp_680"
    ):
        env_state[field_name] = safe_float(spectral.get(field_name))
    env_state["Ci"] = FIXED_CO2_PPM if USE_FIXED_CO2 else safe_float(row.get("co2_ppm"))

    ppfd_pred, pn_pred = compute_local_ppfd_pn(row)
    if ppfd_pred is not None:
        env_state["PPFD_pred"] = ppfd_pred
    if pn_pred is not None:
        env_state["Pn_pred"] = pn_pred

    last_local_sensor_timestamp = row_timestamp
    queue_sensor_snapshot()
    record_state_snapshot()


# Load PWM model
logger.info("Loading PWM control model...")
try:
    pwm_pkg = pwm_model.load_model(None)
except Exception:
    logger.exception("Failed to load PWM model package.")
    raise

try:
    sp_pkg_local = sp_model.load_package(sp_model.DEFAULT_MODEL_PACKAGE)
    pn_pkg_local = pn_model.load_package(pn_model.DEFAULT_MODEL_PACKAGE)
except Exception:
    logger.exception("Failed to load local SP/PN model packages.")
    raise

restore_control_state()


# --- 4. CORE FUNCTIONS ---
def calculate_and_publish_state(client):
    if USE_FIXED_CO2:
        env_state["Ci"] = FIXED_CO2_PPM
        record_state_snapshot()

    missing_model_fields = get_missing_fields(STATE_REQUIRED_FIELDS)
    missing_sensor_fields = get_missing_fields(ALL_SENSOR_FIELDS)

    if missing_model_fields:
        logger.warning(
            "Skipping state publish because required sensor data is missing: %s",
            format_field_list(missing_model_fields),
        )
        return

    payload = build_server_payload()

    event = "state_computed_not_published"
    if mqtt_connected:
        try:
            publish_info = client.publish(TOPIC_STATE, json.dumps(payload))
            if publish_info.rc == mqtt.MQTT_ERR_SUCCESS:
                event = "state_published"
                logger.info(
                    "Published greenhouse state from MQTT-derived metrics: ppfd=%.2f pn=%.2f target_ppfd=%.2f offline=%s missing=%s",
                        env_state["PPFD_pred"],
                        env_state["Pn_pred"],
                        env_state["target_ppfd"],
                    int(is_offline_mode),
                    format_field_list(missing_sensor_fields),
                )
            else:
                logger.warning(
                    "Computed greenhouse state but MQTT publish returned rc=%s",
                    publish_info.rc,
                )
        except Exception:
            logger.exception("Computed greenhouse state but MQTT publish failed.")
    else:
        logger.warning("MQTT disconnected; state snapshot prepared locally without publishing.")

def run_fallback_control():
    global env_state, is_offline_mode

    now = datetime.datetime.now(TIMEZONE)
    is_day = 7 <= now.hour < 23

    safe_ppfd = 250.0 if is_day else 0.0

    logger.warning(
        "Applying fallback control: is_day=%s safe_ppfd=%.1f",
        int(is_day),
        safe_ppfd,
    )
    env_state["target_ppfd"] = float(safe_ppfd)

    if safe_ppfd > 0:
        try:
            res = pwm_model.build_result(safe_ppfd, pwm_pkg.get("default_blue_share", 0.0), pwm_pkg)
            red_pwm = clamp_pwm_value(res["red_pwm"])
            blue_pwm = clamp_pwm_value(res["blue_pwm"])
        except Exception:
            logger.exception("Fallback PPFD-to-PWM conversion failed; forcing LEDs off.")
            red_pwm, blue_pwm = 0, 0
    else:
        red_pwm, blue_pwm = 0, 0

    if not apply_and_confirm_light_state(safe_ppfd, red_pwm, blue_pwm, "fallback"):
        logger.warning(
            "Fallback light command was not confirmed by Shelly status: target_ppfd=%.2f red_pwm=%d blue_pwm=%d",
            safe_ppfd,
            red_pwm,
            blue_pwm,
        )

def on_message(client, userdata, msg):
    global last_command_time, is_offline_mode, env_state

    topic = msg.topic
    payload = msg.payload.decode(errors="replace")

    if topic == TOPIC_CMD:
        last_command_time = time.time()
        if is_offline_mode:
            logger.info("Cloud command stream resumed; leaving fallback control mode.")
            is_offline_mode = False

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Ignoring invalid JSON command on topic=%s payload=%r", topic, payload)
            return

        try:
            target_ppfd = parse_nonnegative_number(data.get("target_ppfd", 0), "target_ppfd")
        except ValueError as exc:
            logger.warning("Ignoring command with invalid values: %s payload=%s", exc, payload)
            return

        max_blue_share = float(pwm_pkg.get("max_blue_share", 0.5))
        try:
            blue_share = float(data.get("blue_share", pwm_pkg.get("default_blue_share", 0.0)))
        except (TypeError, ValueError):
            blue_share = float(pwm_pkg.get("default_blue_share", 0.0))
        blue_share = max(0.0, min(blue_share, max_blue_share))

        # On restart, keep the last non-zero light level unless the broker
        # provides a fresh non-retained command or an explicit retained zero.
        if time.time() - startup_time < 15 and target_ppfd == 0 and getattr(msg, "retain", False):
            logger.info("Ignoring retained startup zero cloud command: payload=%s", payload)
            return

        if target_ppfd > 0:
            try:
                res = pwm_model.build_result(target_ppfd, blue_share, pwm_pkg)
                red_pwm = clamp_pwm_value(res["red_pwm"])
                blue_pwm = clamp_pwm_value(res["blue_pwm"])
            except Exception:
                logger.exception(
                    "Failed to convert target_ppfd=%.2f blue_share=%.3f into LED PWM values; command ignored.",
                    target_ppfd, blue_share,
                )
                return
        else:
            red_pwm, blue_pwm = 0, 0

        env_state["target_ppfd"] = float(target_ppfd)

        pwm_saturated, pwm_saturation_label = get_pwm_saturation(red_pwm, blue_pwm)

        logger.info(
            "Applying cloud command: target_ppfd=%.2f red_pwm=%d blue_pwm=%d saturated=%s",
            target_ppfd,
            red_pwm,
            blue_pwm,
            pwm_saturation_label,
        )

        if not apply_and_confirm_light_state(target_ppfd, red_pwm, blue_pwm, "cloud"):
            logger.warning(
                "Skipped PWM state update because Shelly status did not confirm cloud target: target_ppfd=%.2f red_pwm=%d blue_pwm=%d",
                target_ppfd,
                red_pwm,
                blue_pwm,
            )
        return

    if topic != POWER_TOPIC:
        logger.debug("Ignoring unmapped MQTT topic: %s", topic)
        return

    try:
        value = float(payload)
    except ValueError:
        logger.warning("Ignoring non-numeric sensor payload on topic=%s payload=%r", topic, payload)
        return

    if not math.isfinite(value):
        logger.warning("Ignoring non-finite sensor payload on topic=%s payload=%r", topic, payload)
        return

    env_state["Power_now_w"] = value
    queue_sensor_snapshot()
    record_state_snapshot()


def on_connect(client, userdata, flags, reason_code, properties):
    global mqtt_connected

    mqtt_connected = True
    logger.info(
        "Connected to MQTT broker at %s:%s with reason_code=%s",
        MQTT_BROKER_IP,
        MQTT_PORT,
        reason_code,
    )
    apply_current_light_state()
    client.subscribe(TOPIC_CMD)
    client.subscribe(POWER_TOPIC)


def on_disconnect(client, userdata, flags, reason_code, properties):
    global mqtt_connected

    mqtt_connected = False
    if reason_code == 0:
        logger.info("Disconnected from MQTT broker cleanly.")
    else:
        logger.warning(
            "MQTT disconnected unexpectedly with reason_code=%s; automatic reconnect will continue.",
            reason_code,
        )


def next_publish_epoch(after_epoch=None):
    base_epoch = time.time() if after_epoch is None else float(after_epoch)
    return math.floor(base_epoch / PUBLISH_INTERVAL) * PUBLISH_INTERVAL + PUBLISH_INTERVAL


def format_local_timestamp(epoch_seconds):
    return datetime.datetime.fromtimestamp(epoch_seconds, TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.reconnect_delay_set(min_delay=1, max_delay=60)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    logger.info("Starting MQTT loop and connecting to %s:%s", MQTT_BROKER_IP, MQTT_PORT)
    client.connect_async(MQTT_BROKER_IP, MQTT_PORT, 60)
    client.loop_start()

    next_publish_time = next_publish_epoch()
    logger.info(
        "State uploads are aligned to %d-second wall-clock boundaries; next publish at %s",
        PUBLISH_INTERVAL,
        format_local_timestamp(next_publish_time),
    )

    try:
        while True:
            current_time = time.time()
            sync_local_sensor_state()
            flush_sensor_snapshot_if_due(datetime.datetime.now(TIMEZONE))

            if current_time - last_command_time > OFFLINE_TIMEOUT_SECONDS and not is_offline_mode:
                is_offline_mode = True
                run_fallback_control()

            if current_time >= next_publish_time:
                calculate_and_publish_state(client)
                next_publish_time = next_publish_epoch(next_publish_time)
                logger.info(
                    "Next scheduled state publish at %s",
                    format_local_timestamp(next_publish_time),
                )

            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Shutting down greenhouse edge node.")
    finally:
        persist_control_state()
        flush_pending_sensor_snapshot()
        client.loop_stop()
        client.disconnect()
