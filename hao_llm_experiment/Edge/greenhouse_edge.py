import csv
import datetime
import json
import logging
import math
import sys
import time
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib

import paho.mqtt.client as mqtt
import pandas as pd
import pytz


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
MODEL_BASE_DIR = PROJECT_ROOT / "Tool" / "Model"
for module_dir in (
    BASE_DIR,
    MODEL_BASE_DIR / "SPtoPPFD",
    MODEL_BASE_DIR / "EnvtoPN",
    MODEL_BASE_DIR / "PWMtoPPFD",
):
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)


# Import local ML modules after adjusting sys.path.
import predict_sp_to_ppfd as sp_model
import predict_env_to_pn as pn_model
import predict_pwm_from_ppfd as pwm_model


try:
    from src.shelly_controller import DEVICES, rpc
except ImportError:
    DEVICES = {"Red": "dev_red", "Blue": "dev_blue", "Heater": "dev_heat"}

    def rpc(dev, cmd, params):
        logger.debug("[MOCK] dev=%s cmd=%s params=%s", dev, cmd, params)


TIMEZONE = pytz.timezone("Australia/Sydney")

CONFIG_PATH = BASE_DIR / "edge_config.toml"
DEFAULT_SENSOR_FIELD_MAP = {
    "temperature": "Tleaf",
    "co2_ppm": "Ci",
    "humidity": "RH",
    "sp_415": "sp_415_mean",
    "sp_445": "sp_445_mean",
    "sp_480": "sp_480_mean",
    "sp_515": "sp_515_mean",
    "sp_555": "sp_555_mean",
    "sp_590": "sp_590_mean",
    "sp_630": "sp_630_mean",
    "sp_680": "sp_680_mean",
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
        "max_heater_pwm": int(runtime_cfg.get("max_heater_pwm", 100)),
        "topic_map": topic_map,
        "co2_topic": sensor_topics["co2_ppm"],
        "sensor_topics": sensor_topics,
    }


EDGE_CONFIG = load_edge_config()
TIMEZONE = pytz.timezone(EDGE_CONFIG["timezone"])

# --- 1. DIRECTORY & LOGGING SETUP ---
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

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

# --- 2. CSV SETUP ---
CSV_FILE = LOG_DIR / "greenhouse_data.csv"
CSV_HEADERS = [
    "Timestamp",
    "Tleaf",
    "Ci",
    "RH",
    "PPFD_Predicted",
    "Pn_Predicted",
    "Target_PPFD",
    "Red_PWM",
    "Blue_PWM",
    "Heater_PWM",
    "Is_Offline",
    "Data_Valid",
    "Missing_Fields",
    "Event",
    "MQTT_Connected",
]


def ensure_csv_file():
    current_header = None
    if CSV_FILE.exists():
        try:
            with CSV_FILE.open(mode="r", newline="", encoding="utf-8") as f:
                current_header = next(csv.reader(f), None)
        except Exception as exc:
            logger.warning("Failed to inspect existing CSV header: %s", exc)

    if current_header == CSV_HEADERS:
        return

    if CSV_FILE.exists():
        legacy_stamp = datetime.datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")
        backup_file = LOG_DIR / f"{CSV_FILE.stem}_legacy_{legacy_stamp}.csv"
        CSV_FILE.rename(backup_file)
        logger.warning(
            "CSV header changed; moved previous file to %s",
            backup_file,
        )

    with CSV_FILE.open(mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_HEADERS)


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

SPECTRAL_FIELDS = [
    "sp_415_mean",
    "sp_445_mean",
    "sp_480_mean",
    "sp_515_mean",
    "sp_555_mean",
    "sp_590_mean",
    "sp_630_mean",
    "sp_680_mean",
]
MODEL_REQUIRED_FIELDS = ["Tleaf", "Ci", *SPECTRAL_FIELDS]
ALL_SENSOR_FIELDS = list(TOPIC_MAP.values())
CO2_TOPIC = EDGE_CONFIG["co2_topic"]

# Global State Tracker
env_state = {k: None for k in ALL_SENSOR_FIELDS}
env_state.update(
    {
        "Ci": FIXED_CO2_PPM if USE_FIXED_CO2 else None,
        "current_red_pwm": 0,
        "current_blue_pwm": 0,
        "current_heater_pwm": 0,
        "target_ppfd": 0.0,
        "PPFD_pred": 0.0,
        "Pn_pred": 0.0,
    }
)

# Runtime status
OFFLINE_TIMEOUT_SECONDS = EDGE_CONFIG["offline_timeout_seconds"]
PUBLISH_INTERVAL = EDGE_CONFIG["publish_interval_seconds"]
MAX_HEATER_PWM = EDGE_CONFIG["max_heater_pwm"]
last_command_time = time.time()
is_offline_mode = False
mqtt_connected = False


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


def current_data_valid():
    return not get_missing_fields(MODEL_REQUIRED_FIELDS)


def safe_csv_value(value):
    return "" if value is None else value


def log_data_to_csv(event, data_valid, missing_fields=None):
    missing_fields = missing_fields or []
    try:
        ts = datetime.datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
        with CSV_FILE.open(mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ts,
                    safe_csv_value(env_state["Tleaf"]),
                    safe_csv_value(env_state["Ci"]),
                    safe_csv_value(env_state["RH"]),
                    env_state["PPFD_pred"],
                    env_state["Pn_pred"],
                    env_state["target_ppfd"],
                    env_state["current_red_pwm"],
                    env_state["current_blue_pwm"],
                    env_state["current_heater_pwm"],
                    int(is_offline_mode),
                    int(data_valid),
                    format_field_list(missing_fields),
                    event,
                    int(mqtt_connected),
                ]
            )
    except Exception:
        logger.exception("Failed to write CSV row for event=%s", event)


def apply_device_command(device_key, command, params):
    if device_key not in DEVICES:
        logger.debug("Skipping %s command because the device is not configured.", device_key)
        return

    try:
        rpc(DEVICES[device_key], command, params)
    except Exception:
        logger.exception(
            "Failed to control device=%s command=%s params=%s",
            device_key,
            command,
            params,
        )


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


def compute_env_to_pn_rb_feature(red_pwm, blue_pwm):
    total_pwm = red_pwm + blue_pwm
    if total_pwm > 0:
        return red_pwm / total_pwm

    recommended_ratio = pwm_pkg["recommended_rb_ratio"]
    return recommended_ratio / (recommended_ratio + 1.0)


ensure_csv_file()

# Load ML Models
logger.info("Loading ML models...")
try:
    sp_pkg = sp_model.load_package(sp_model.DEFAULT_MODEL_PACKAGE)
    pn_pkg = pn_model.load_package(pn_model.DEFAULT_MODEL_PACKAGE)
    pwm_pkg = pwm_model.load_model(None)
except Exception:
    logger.exception("Failed to load one or more ML model packages.")
    raise


# --- 4. CORE FUNCTIONS ---
def calculate_and_publish_state(client):
    if USE_FIXED_CO2:
        env_state["Ci"] = FIXED_CO2_PPM

    missing_model_fields = get_missing_fields(MODEL_REQUIRED_FIELDS)
    missing_sensor_fields = get_missing_fields(ALL_SENSOR_FIELDS)

    if missing_model_fields:
        logger.warning(
            "Skipping state evaluation because required sensor data is missing: %s",
            format_field_list(missing_model_fields),
        )
        log_data_to_csv("state_skipped_missing_data", False, missing_sensor_fields)
        return

    try:
        sp_df = pd.DataFrame([env_state])
        X_sp = sp_model.prepare_features(sp_df, sp_pkg["feature_columns"])
        env_state["PPFD_pred"] = float(sp_pkg["pipeline"].predict(X_sp)[0])

        red_pwm = env_state["current_red_pwm"]
        blue_pwm = env_state["current_blue_pwm"]
        rb_ratio = compute_env_to_pn_rb_feature(red_pwm, blue_pwm)

        pn_input = {
            "T": env_state["Tleaf"],
            "CO2": env_state["Ci"],
            "R:B": rb_ratio,
            "PPFD": env_state["PPFD_pred"],
        }
        pn_df = pd.DataFrame([pn_input])
        X_pn = pn_model.prepare_features(pn_df, pn_pkg["feature_columns"])
        env_state["Pn_pred"] = float(pn_pkg["pipeline"].predict(X_pn)[0])
    except Exception:
        logger.exception("State evaluation failed during local model inference.")
        log_data_to_csv("state_error", False, get_missing_fields(ALL_SENSOR_FIELDS))
        return

    now = datetime.datetime.now(TIMEZONE)
    payload = {
        "Tleaf": env_state["Tleaf"],
        "Ci": env_state["Ci"],
        "PPFD_current": env_state["PPFD_pred"],
        "Pn_current": env_state["Pn_pred"],
        "local_time": now.strftime("%H:%M"),
        "isDay": 1 if 6 <= now.hour < 18 else 0,
    }

    event = "state_computed_not_published"
    if mqtt_connected:
        try:
            publish_info = client.publish(TOPIC_STATE, json.dumps(payload))
            if publish_info.rc == mqtt.MQTT_ERR_SUCCESS:
                event = "state_published"
                logger.info(
                    "Published greenhouse state: ppfd=%.2f pn=%.2f target_ppfd=%.2f offline=%s missing=%s",
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
        logger.warning("MQTT disconnected; computed greenhouse state locally without publishing.")

    log_data_to_csv(event, True, missing_sensor_fields)


def run_fallback_control():
    global env_state, is_offline_mode

    now = datetime.datetime.now(TIMEZONE)
    is_day = 6 <= now.hour < 18

    safe_ppfd = 250.0 if is_day else 0.0
    tleaf = env_state["Tleaf"]
    safe_heater_pwm = 60 if (tleaf is not None and tleaf < 18.0) else 0

    logger.warning(
        "Applying fallback control: is_day=%s safe_ppfd=%.1f safe_heater_pwm=%d tleaf=%s",
        int(is_day),
        safe_ppfd,
        safe_heater_pwm,
        "missing" if tleaf is None else f"{tleaf:.2f}",
    )

    if safe_ppfd > 0:
        try:
            res = pwm_model.build_result(safe_ppfd, pwm_pkg["recommended_rb_ratio"], pwm_pkg)
            red_pwm = res["red_pwm"]
            blue_pwm = res["blue_pwm"]
        except Exception:
            logger.exception("Fallback PPFD-to-PWM conversion failed; forcing LEDs off.")
            red_pwm, blue_pwm = 0, 0
    else:
        red_pwm, blue_pwm = 0, 0

    env_state["target_ppfd"] = safe_ppfd
    env_state["current_red_pwm"] = red_pwm
    env_state["current_blue_pwm"] = blue_pwm
    env_state["current_heater_pwm"] = safe_heater_pwm

    apply_device_command(
        "Red",
        "Light.Set",
        {"id": 0, "on": red_pwm > 0, "brightness": red_pwm},
    )
    apply_device_command(
        "Blue",
        "Light.Set",
        {"id": 0, "on": blue_pwm > 0, "brightness": blue_pwm},
    )
    apply_device_command(
        "Heater",
        "Switch.Set",
        {"id": 0, "on": safe_heater_pwm > 0},
    )

    log_data_to_csv("fallback_applied", current_data_valid(), get_missing_fields(ALL_SENSOR_FIELDS))


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
            heater_pwm = int(
                round(
                    parse_nonnegative_number(
                        data.get("heater_pwm", 0),
                        "heater_pwm",
                        upper_bound=MAX_HEATER_PWM,
                    )
                )
            )
        except ValueError as exc:
            logger.warning("Ignoring command with invalid values: %s payload=%s", exc, payload)
            return

        if target_ppfd > 0:
            try:
                res = pwm_model.build_result(target_ppfd, pwm_pkg["recommended_rb_ratio"], pwm_pkg)
                red_pwm = res["red_pwm"]
                blue_pwm = res["blue_pwm"]
            except Exception:
                logger.exception(
                    "Failed to convert target_ppfd=%.2f into LED PWM values; command ignored.",
                    target_ppfd,
                )
                return
        else:
            red_pwm, blue_pwm = 0, 0

        env_state["target_ppfd"] = target_ppfd
        env_state["current_red_pwm"] = red_pwm
        env_state["current_blue_pwm"] = blue_pwm
        env_state["current_heater_pwm"] = heater_pwm

        logger.info(
            "Applying cloud command: target_ppfd=%.2f red_pwm=%d blue_pwm=%d heater_pwm=%d",
            target_ppfd,
            red_pwm,
            blue_pwm,
            heater_pwm,
        )

        apply_device_command(
            "Red",
            "Light.Set",
            {"id": 0, "on": red_pwm > 0, "brightness": red_pwm},
        )
        apply_device_command(
            "Blue",
            "Light.Set",
            {"id": 0, "on": blue_pwm > 0, "brightness": blue_pwm},
        )
        apply_device_command(
            "Heater",
            "Switch.Set",
            {"id": 0, "on": heater_pwm > 0},
        )
        log_data_to_csv(
            "cloud_command_applied",
            current_data_valid(),
            get_missing_fields(ALL_SENSOR_FIELDS),
        )
        return

    if topic not in TOPIC_MAP:
        logger.debug("Ignoring unmapped MQTT topic: %s", topic)
        return

    if USE_FIXED_CO2 and topic == CO2_TOPIC:
        env_state["Ci"] = FIXED_CO2_PPM
        logger.debug("Ignoring MQTT CO2 payload because fixed CO2 mode is enabled.")
        return

    try:
        value = float(payload)
    except ValueError:
        logger.warning("Ignoring non-numeric sensor payload on topic=%s payload=%r", topic, payload)
        return

    if not math.isfinite(value):
        logger.warning("Ignoring non-finite sensor payload on topic=%s payload=%r", topic, payload)
        return

    env_state[TOPIC_MAP[topic]] = value


def on_connect(client, userdata, flags, reason_code, properties):
    global mqtt_connected

    mqtt_connected = True
    logger.info(
        "Connected to MQTT broker at %s:%s with reason_code=%s",
        MQTT_BROKER_IP,
        MQTT_PORT,
        reason_code,
    )
    client.subscribe(TOPIC_CMD)
    for topic in TOPIC_MAP:
        client.subscribe(topic)


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

    last_publish_time = 0

    try:
        while True:
            current_time = time.time()

            if current_time - last_command_time > OFFLINE_TIMEOUT_SECONDS and not is_offline_mode:
                is_offline_mode = True
                run_fallback_control()

            if current_time - last_publish_time > PUBLISH_INTERVAL:
                calculate_and_publish_state(client)
                last_publish_time = current_time

            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Shutting down greenhouse edge node.")
    finally:
        client.loop_stop()
        client.disconnect()
