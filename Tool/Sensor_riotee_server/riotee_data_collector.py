# -*- coding: utf-8 -*-
"""
Riotee Data Collector - Simplified Version
Collects sensor data and outputs CSV
"""

import time, logging, csv, os, sys, argparse, struct, json
from pathlib import Path
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from datetime import datetime
from riotee_gateway import GatewayClient, base64_to_numpy


MODEL_BASE_DIR = Path(__file__).resolve().parent.parent / "Model"
for module_dir in (MODEL_BASE_DIR / "SPtoPPFD", MODEL_BASE_DIR / "EnvtoPN"):
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)

try:
    import predict_sp_to_ppfd as sp_model
    import predict_env_to_pn as pn_model
except Exception as exc:
    sp_model = None
    pn_model = None
    MODEL_IMPORT_ERROR = exc
else:
    MODEL_IMPORT_ERROR = None

# Command line args
parser = argparse.ArgumentParser(description='Riotee Data Collector')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--no-kill', action='store_true', help='Skip killing existing gateway')
parser.add_argument('--new-file', action='store_true', help='Create new timestamped file')
parser.add_argument('csv_name', nargs='?', default=None, help='CSV filename (optional)')
parser.add_argument('comment', nargs='?', default=None, help='Comment for CSV header')
args = parser.parse_args()

# Setup
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

CONFIG = {
    "mqtt_broker": "azure.nocolor.cc", "mqtt_port": 1883,
    "mqtt_username": "feiyue", "mqtt_password": "123456789",
    "gateway_host": "localhost", "gateway_port": 8000,
    "data_interval": 0.1
}

FIXED_CO2_PPM_FOR_PN = 400.0
DEFAULT_PN_RB_FEATURE = 0.75
DERIVED_MISSING_PAYLOAD = "None"  # Set to "" if you prefer Home Assistant to ignore updates.
RAW_CO2_MISSING_PAYLOAD = "unknown"

# Globals
mqtt_client = None
csv_file_all = csv_writer_all = csv_file_summary = csv_writer_summary = None
record_id = 0
session_start_time = None
device_last_state = {}
discovered_devices = set()
stats = {"packets_received": 0, "mqtt_success": 0, "errors": 0}
sp_pkg = None
pn_pkg = None
derived_models_ready = False

# Packet format constants
PKT_MAGIC = 0xAA
PKT_BASIC, PKT_SPECTRAL, PKT_FULL, PKT_BASIC_CO2 = 0x01, 0x02, 0x03, 0x04
CO2_STATE_INVALID, CO2_STATE_HOLD, CO2_STATE_FRESH = 0, 1, 2

# CSV fieldnames
FIELDNAMES = ['id', 'timestamp', 'device_id', 'update_type', 'temperature', 'humidity',
              'a1_raw', 'vcap_raw', 'co2_ppm', 'co2_state', 'sp_415', 'sp_445', 'sp_480', 'sp_515',
              'sp_555', 'sp_590', 'sp_630', 'sp_680', 'sp_clear', 'sp_nir',
              'spectral_gain', 'sleep_time']

SUMMARY_FIELDS = ['id', 'timestamp', 'device_id', 'update_type', 'temperature', 
                  'humidity', 'a1_raw', 'vcap_raw', 'co2_ppm', 'co2_state', 'spectral_gain', 'sleep_time']

STATE_FILE = "current_led_state.json"
PENDING_COMMENT_FILE = "pending_segment_comment.json"
state_cache = {"mtime_ns": None, "payload": {}}
pending_comment_cache = {"mtime_ns": None, "payload": {}}
last_comment_id = None

def gain_to_mult(g):
    """Convert gain code (0-10) to multiplier"""
    return {0:0.5,1:1,2:2,3:4,4:8,5:16,6:32,7:64,8:128,9:256,10:512}.get(int(g), 0)


def describe_co2_state(state_code):
    """Map compact-packet state code to a host-readable label."""
    return {
        CO2_STATE_INVALID: "INVALID",
        CO2_STATE_HOLD: "HOLD",
        CO2_STATE_FRESH: "FRESH",
    }.get(state_code, "UNKNOWN")


def infer_legacy_co2_state(pkt_type, co2):
    """Backward-compatible state mapping for older compact packets."""
    if pkt_type in (PKT_FULL, PKT_BASIC_CO2) and co2 >= 0:
        return "FRESH"
    if pkt_type in (PKT_BASIC, PKT_SPECTRAL):
        return "INVALID"
    return "UNKNOWN"


def load_derived_models():
    global sp_pkg, pn_pkg, derived_models_ready

    if sp_model is None or pn_model is None:
        logging.warning("Derived PPFD/Pn models unavailable: %s", MODEL_IMPORT_ERROR)
        return False

    try:
        sp_pkg = sp_model.load_package(sp_model.DEFAULT_MODEL_PACKAGE)
        pn_pkg = pn_model.load_package(pn_model.DEFAULT_MODEL_PACKAGE)
        derived_models_ready = True
        logging.info("Derived PPFD/Pn models loaded")
        return True
    except Exception as exc:
        derived_models_ready = False
        logging.warning(f"Derived PPFD/Pn models failed to load: {exc}")
        return False


def safe_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def compute_rb_feature_from_state():
    return DEFAULT_PN_RB_FEATURE


def compute_derived_metrics(temp, co2, spectrum, has_spectral_data):
    if not derived_models_ready or not has_spectral_data:
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
        X_sp = sp_model.prepare_features(sp_df, sp_pkg["feature_columns"])
        ppfd_pred = float(sp_pkg["pipeline"].predict(X_sp)[0])

        pn_input = {
            "T": temp,
            "CO2": co2 if co2 >= 0 else FIXED_CO2_PPM_FOR_PN,
            "R:B": compute_rb_feature_from_state(),
            "PPFD": ppfd_pred,
        }
        pn_df = pd.DataFrame([pn_input])
        X_pn = pn_model.prepare_features(pn_df, pn_pkg["feature_columns"])
        pn_pred = float(pn_pkg["pipeline"].predict(X_pn)[0])
        return ppfd_pred, pn_pred
    except Exception as exc:
        logging.warning(f"Derived PPFD/Pn inference failed: {exc}")
        return None, None


def publish_derived_metrics(dev, ppfd_pred, pn_pred):
    if not mqtt_client:
        return

    if ppfd_pred is None:
        publish_mqtt(f"riotee/{dev}/ppfd_pred", DERIVED_MISSING_PAYLOAD)
    else:
        publish_mqtt(f"riotee/{dev}/ppfd_pred", f"{ppfd_pred:.2f}")

    if pn_pred is None:
        publish_mqtt(f"riotee/{dev}/pn_pred", DERIVED_MISSING_PAYLOAD)
    else:
        publish_mqtt(f"riotee/{dev}/pn_pred", f"{pn_pred:.2f}")

# ============ MQTT ============
def setup_mqtt():
    try:
        client = mqtt.Client(client_id=f"riotee_{int(time.time())}", 
                            callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        client.username_pw_set(CONFIG["mqtt_username"], CONFIG["mqtt_password"])
        client.connect(CONFIG["mqtt_broker"], CONFIG["mqtt_port"], 60)
        client.loop_start()
        logging.info("MQTT connected")
        return client
    except Exception as e:
        logging.warning(f"MQTT failed: {e}")
        return None

def publish_mqtt(topic, value):
    if mqtt_client:
        try:
            mqtt_client.publish(topic, str(value), qos=0)
            stats["mqtt_success"] += 1
        except: pass


def get_device_topic_id(device_id):
    """Build a compact MQTT-safe topic suffix from the raw device id."""
    return device_id.replace("==", "").replace("=", "").replace("/", "_")[:8]


def get_device_display_name(device_id):
    """Expose devices in HA as GrowPod_Riotee_<first four alnum chars>."""
    clean_chars = ''.join(c for c in device_id if c.isalnum())
    suffix = clean_chars[:4] if clean_chars else "unkn"
    return f"GrowPod_Riotee_{suffix}"


def publish_ha_discovery(device_id, sensor_type, topic_suffix, unit, icon):
    if not mqtt_client:
        return False

    device_topic_id = get_device_topic_id(device_id)
    device_name = get_device_display_name(device_id)
    unique_id = f"growpod_{device_topic_id}_{sensor_type}"
    sensor_name = f"{device_name} {sensor_type.replace('_', ' ').title()}"
    config_topic = f"homeassistant/sensor/{unique_id}/config"
    payload = {
        "name": sensor_name,
        "unique_id": unique_id,
        "state_topic": f"riotee/{device_topic_id}/{topic_suffix}",
        "icon": icon,
        "device": {
            "identifiers": [f"growpod_{device_topic_id}"],
            "name": device_name,
            "manufacturer": "UNSW",
            "model": "Riotee Sensor",
        },
    }
    if unit:
        payload["unit_of_measurement"] = unit

    try:
        result = mqtt_client.publish(config_topic, json.dumps(payload), qos=0, retain=True)
        return result.rc == mqtt.MQTT_ERR_SUCCESS
    except Exception as e:
        logging.warning(f"HA discovery failed for {unique_id}: {e}")
        return False


def ensure_ha_discovery(device_id):
    device_topic_id = get_device_topic_id(device_id)
    if device_topic_id in discovered_devices:
        return

    sensor_configs = [
        ("temperature", "temperature", "degC", "mdi:thermometer"),
        ("humidity", "humidity", "%", "mdi:water-percent"),
        ("a1_raw", "a1_raw", "V", "mdi:current-ac"),
        ("vcap_raw", "vcap_raw", "V", "mdi:flash"),
        ("co2_ppm", "co2_ppm", "ppm", "mdi:molecule-co2"),
        ("co2_state", "co2_state", "", "mdi:update"),
        ("spectral_gain", "spectral_gain", "", "mdi:tune-vertical-variant"),
        ("sleep_time", "sleep_time", "s", "mdi:sleep"),
        ("sp_415", "sp_415", "count", "mdi:chart-bell-curve"),
        ("sp_445", "sp_445", "count", "mdi:chart-bell-curve"),
        ("sp_480", "sp_480", "count", "mdi:chart-bell-curve"),
        ("sp_515", "sp_515", "count", "mdi:chart-bell-curve"),
        ("sp_555", "sp_555", "count", "mdi:chart-bell-curve"),
        ("sp_590", "sp_590", "count", "mdi:chart-bell-curve"),
        ("sp_630", "sp_630", "count", "mdi:chart-bell-curve"),
        ("sp_680", "sp_680", "count", "mdi:chart-bell-curve"),
        ("sp_clear", "sp_clear", "count", "mdi:brightness-7"),
        ("sp_nir", "sp_nir", "count", "mdi:weather-sunset-up"),
        ("ppfd_pred", "ppfd_pred", "umol/m2/s", "mdi:white-balance-sunny"),
        ("pn_pred", "pn_pred", "umol CO2/m2/s", "mdi:leaf"),
    ]
    for sensor_type, topic_suffix, unit, icon in sensor_configs:
        publish_ha_discovery(device_id, sensor_type, topic_suffix, unit, icon)

    discovered_devices.add(device_topic_id)
    logging.info(f"HA discovery published for {get_device_display_name(device_id)}")

# ============ Gateway ============
def start_gateway():
    import subprocess, signal
    gateway_cmd = Path(sys.executable).resolve().parent / "riotee-gateway"
    gateway_exec = str(gateway_cmd) if gateway_cmd.exists() else "riotee-gateway"
    if not args.no_kill:
        try:
            result = subprocess.run(["pgrep", "-f", gateway_exec], capture_output=True, text=True)
            if result.returncode == 0:
                for pid in result.stdout.strip().split('\n'):
                    if pid.strip():
                        try: os.kill(int(pid), signal.SIGTERM)
                        except: pass
                time.sleep(2)
        except: pass
    
    try:
        proc = subprocess.Popen([gateway_exec, "server", "-p", str(CONFIG["gateway_port"]), 
                                "-h", CONFIG["gateway_host"]], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info(f"Gateway started, PID: {proc.pid}")
        time.sleep(3)
        return proc
    except Exception as e:
        logging.error(f"Gateway start failed: {e}")
        return None

def setup_gateway():
    try:
        client = GatewayClient(host=CONFIG["gateway_host"], port=CONFIG["gateway_port"])
        logging.info("Gateway connected")
        return client
    except Exception as e:
        logging.error(f"Gateway connection failed: {e}")
        return None

# ============ CSV & JSON ============
def setup_csv():
    global csv_file_all, csv_writer_all, csv_file_summary, csv_writer_summary, session_start_time
    session_start_time = time.time()
    
    base = args.csv_name or "riotee_data"
    if args.new_file:
        base = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    note = args.comment or base
    start_line = f"# Start @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {note}\n"
    
    # All data file
    path_all = os.path.join(LOGS_DIR, f"{base}_all.csv")
    mode = 'a' if not args.new_file and os.path.exists(path_all) else 'w'
    csv_file_all = open(path_all, mode, newline='')
    csv_file_all.write(start_line)
    csv_writer_all = csv.DictWriter(csv_file_all, fieldnames=FIELDNAMES)
    if mode == 'w': csv_writer_all.writeheader()
    
    # Summary file
    path_sum = os.path.join(LOGS_DIR, f"{base}_summary.csv")
    mode = 'a' if not args.new_file and os.path.exists(path_sum) else 'w'
    csv_file_summary = open(path_sum, mode, newline='')
    csv_file_summary.write(start_line)
    csv_writer_summary = csv.DictWriter(csv_file_summary, fieldnames=SUMMARY_FIELDS)
    if mode == 'w': csv_writer_summary.writeheader()
    
    logging.info(f"Files: {path_all}, {path_sum}")
    return path_all, path_sum


def read_control_state():
    """Read latest run control state emitted by run_led_calibration.py."""
    try:
        stat = os.stat(STATE_FILE)
    except OSError:
        state_cache["mtime_ns"] = None
        state_cache["payload"] = {}
        return {}

    if state_cache["mtime_ns"] == stat.st_mtime_ns:
        return state_cache["payload"]

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
            if not isinstance(payload, dict):
                payload = {}
    except Exception:
        payload = {}

    state_cache["mtime_ns"] = stat.st_mtime_ns
    state_cache["payload"] = payload
    return payload


def read_pending_segment_comment():
    try:
        stat = os.stat(PENDING_COMMENT_FILE)
    except OSError:
        pending_comment_cache["mtime_ns"] = None
        pending_comment_cache["payload"] = {}
        return {}

    if pending_comment_cache["mtime_ns"] == stat.st_mtime_ns:
        return pending_comment_cache["payload"]

    try:
        with open(PENDING_COMMENT_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
            if not isinstance(payload, dict):
                payload = {}
    except Exception:
        payload = {}

    pending_comment_cache["mtime_ns"] = stat.st_mtime_ns
    pending_comment_cache["payload"] = payload
    return payload


def maybe_write_control_comment():
    global last_comment_id
    comment_payload = read_pending_segment_comment()
    if not comment_payload:
        return
    comment_id = comment_payload.get("comment_id")
    if not comment_id or comment_id == last_comment_id:
        return

    comment = (
        "# SEGMENT @ {ts} segment_id={segment_id} pwm_r={pwm_r} pwm_b={pwm_b} rb_ratio_pwm={rb_ratio_pwm} PPFD_spec_mean={ppfd} ppfd_red_mean={ppfd_red} ppfd_blue_mean={ppfd_blue}\n".format(
            ts=datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            segment_id=comment_payload.get("segment_id", ""),
            pwm_r=comment_payload.get("pwm_r", ""),
            pwm_b=comment_payload.get("pwm_b", ""),
            rb_ratio_pwm=comment_payload.get("rb_ratio_pwm", ""),
            ppfd=comment_payload.get("PPFD_spec_mean", ""),
            ppfd_red=comment_payload.get("ppfd_red_mean", ""),
            ppfd_blue=comment_payload.get("ppfd_blue_mean", ""),
        )
    )
    try:
        csv_file_all.write(comment)
        csv_file_all.flush()
        last_comment_id = comment_id
    except Exception as e:
        logging.error(f"CSV comment write error: {e}")

# ============ Data Processing ============
def write_record(device_id, update_type, temp, hum, a1, vcap, co2, co2_state, spectrum, gain, sleep, has_spectral_data=False):
    """Write data to CSV and JSON"""
    global record_id
    record_id += 1
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    gain_mult = gain_to_mult(gain)
    last = device_last_state.get(device_id, {})
    last_valid_co2 = last.get('last_valid_co2', -1)
    current_co2 = co2 if co2 >= 0 else -1
    
    read_control_state()
    maybe_write_control_comment()

    # CSV row
    row = {
        'id': record_id, 'timestamp': ts, 'device_id': device_id, 'update_type': update_type,
        'temperature': f"{temp:.2f}", 'humidity': f"{hum:.2f}",
        'a1_raw': f"{a1:.3f}", 'vcap_raw': f"{vcap:.3f}",
        'co2_ppm': co2,
        'co2_state': co2_state,
        'spectral_gain': gain_mult, 'sleep_time': sleep,
    }
    row.update({k: f"{v:.2f}" for k, v in spectrum.items()})
    
    try:
        csv_writer_all.writerow(row)
        csv_file_all.flush()
    except Exception as e:
        logging.error(f"CSV write error: {e}")
    
    # Summary (on significant change)
    if not last or abs(temp - last.get('temp', temp)) > 0.5 or sleep != last.get('sleep', sleep):
        summary_row = {k: row[k] for k in SUMMARY_FIELDS if k in row}
        try:
            csv_writer_summary.writerow(summary_row)
            csv_file_summary.flush()
        except: pass

    ppfd_pred, pn_pred = compute_derived_metrics(temp, co2, spectrum, has_spectral_data)
    
    # MQTT
    if mqtt_client:
        dev = get_device_topic_id(device_id)
        ensure_ha_discovery(device_id)
        publish_mqtt(f"riotee/{dev}/temperature", f"{temp:.2f}")
        publish_mqtt(f"riotee/{dev}/humidity", f"{hum:.2f}")
        publish_mqtt(f"riotee/{dev}/a1_raw", f"{a1:.3f}")
        publish_mqtt(f"riotee/{dev}/vcap_raw", f"{vcap:.3f}")
        publish_mqtt(f"riotee/{dev}/spectral_gain", gain_mult)
        publish_mqtt(f"riotee/{dev}/sleep_time", sleep)
        for channel, value in spectrum.items():
            publish_mqtt(f"riotee/{dev}/{channel}", f"{value:.2f}")
        co2_payload = current_co2 if current_co2 >= 0 else RAW_CO2_MISSING_PAYLOAD
        publish_mqtt(f"riotee/{dev}/co2_ppm", co2_payload)
        publish_mqtt(f"riotee/{dev}/co2_state", co2_state)
        publish_derived_metrics(dev, ppfd_pred, pn_pred)
    
    co2_value = f"{co2}ppm" if co2 >= 0 else "-1"
    last_co2_value = f"{current_co2}ppm" if current_co2 >= 0 else "N/A"
    derived_summary = (
        f" PPFD={ppfd_pred:.2f} Pn={pn_pred:.2f}"
        if ppfd_pred is not None and pn_pred is not None
        else " PPFD/Pn=unknown"
    )
    logging.info(
        f"{device_id}: T={temp:.1f}C H={hum:.1f}% CO2={co2_value} ({co2_state}, last_valid={last_co2_value}) Sleep={sleep}s{derived_summary}"
    )

    device_last_state[device_id] = {
        'temp': temp,
        'sleep': sleep,
        'last_valid_co2': co2 if co2 >= 0 else last_valid_co2,
    }

def process_compact(device_id, raw):
    """Process compact binary packet (0xAA magic)"""
    if len(raw) < 12 or raw[0] != PKT_MAGIC: return
    
    pkt_type = raw[1]
    base_len = 12
    co2_state = "UNKNOWN"
    try:
        expected_new_len = {
            PKT_BASIC: 13,
            PKT_SPECTRAL: 33,
            PKT_FULL: 35,
            PKT_BASIC_CO2: 15,
        }.get(pkt_type)
        if expected_new_len is not None and len(raw) >= expected_new_len:
            temp_x100, hum_x100, a1_mv, vcap_mv, sleep, gain, co2_state_code = struct.unpack('<hHHHBBB', raw[2:13].tobytes())
            base_len = 13
            co2_state = describe_co2_state(co2_state_code)
        else:
            temp_x100, hum_x100, a1_mv, vcap_mv, sleep, gain = struct.unpack('<hHHHBB', raw[2:12].tobytes())
    except:
        return
    
    temp, hum = temp_x100/100.0, hum_x100/100.0
    a1, vcap = a1_mv/1000.0, vcap_mv/1000.0
    
    spectrum = {f'sp_{wl}': 0.0 for wl in [415,445,480,515,555,590,630,680]}
    spectrum.update({'sp_clear': 0.0, 'sp_nir': 0.0})
    co2 = -1
    
    has_spectral_data = pkt_type in (PKT_SPECTRAL, PKT_FULL) and len(raw) >= (base_len + 20)

    if has_spectral_data:
        try:
            arr = struct.unpack('<10H', raw[base_len:base_len + 20].tobytes())
            for i, wl in enumerate([415,445,480,515,555,590,630,680]):
                spectrum[f'sp_{wl}'] = float(arr[i])
            spectrum['sp_clear'], spectrum['sp_nir'] = float(arr[8]), float(arr[9])
        except: pass
    
    if pkt_type == PKT_FULL and len(raw) >= (base_len + 22):
        try: co2 = struct.unpack('<h', raw[base_len + 20:base_len + 22].tobytes())[0]
        except: pass
    elif pkt_type == PKT_BASIC_CO2 and len(raw) >= (base_len + 2):
        try: co2 = struct.unpack('<h', raw[base_len:base_len + 2].tobytes())[0]
        except: pass
    
    if base_len == 12:
        co2_state = infer_legacy_co2_state(pkt_type, co2)
    write_record(device_id, "COMPACT", temp, hum, a1, vcap, co2, co2_state, spectrum, gain, sleep, has_spectral_data=has_spectral_data)

def process_float(device_id, data):
    """Process float32 array packet (17 floats: 4 basic + 2 config + 10 spectrum + 1 co2)"""
    if len(data) < 4: return
    
    temp, hum, a1, vcap = data[0], data[1], data[2], data[3]
    
    if len(data) >= 17:
        sleep, gain = int(data[4]), int(data[5])
        spectrum = {
            'sp_415': data[6], 'sp_445': data[7], 'sp_480': data[8], 'sp_515': data[9],
            'sp_555': data[10], 'sp_590': data[11], 'sp_630': data[12], 'sp_680': data[13],
            'sp_clear': data[14], 'sp_nir': data[15]
        }
        co2 = int(data[16]) if data[16] >= 0 else -1
        co2_state = "FRESH" if co2 >= 0 else "INVALID"
        has_spectral_data = True
    elif len(data) >= 16:
        sleep, gain = int(data[4]), int(data[5])
        spectrum = {
            'sp_415': data[6], 'sp_445': data[7], 'sp_480': data[8], 'sp_515': data[9],
            'sp_555': data[10], 'sp_590': data[11], 'sp_630': data[12], 'sp_680': data[13],
            'sp_clear': data[14], 'sp_nir': data[15]
        }
        co2 = -1
        co2_state = "INVALID"
        has_spectral_data = True
    else:
        sleep, gain, co2 = 0, 255, -1
        spectrum = {f'sp_{wl}': 0.0 for wl in [415,445,480,515,555,590,630,680]}
        spectrum.update({'sp_clear': 0.0, 'sp_nir': 0.0})
        co2_state = "INVALID"
        has_spectral_data = False
    
    write_record(device_id, "FLOAT", temp, hum, a1, vcap, co2, co2_state, spectrum, gain, sleep, has_spectral_data=has_spectral_data)

# ============ Main ============
def main():
    global mqtt_client
    print("=" * 60)
    print("Riotee Data Collector Started")
    print("=" * 60)
    
    load_derived_models()
    mqtt_client = setup_mqtt()
    gateway_proc = start_gateway()
    
    # Connect to gateway with retry
    gateway = None
    for i in range(5):
        gateway = setup_gateway()
        if gateway: break
        time.sleep(2)
    
    if not gateway:
        logging.error("Gateway connection failed")
        if gateway_proc: gateway_proc.terminate()
        return
    
    setup_csv()
    logging.info("Listening for devices...")
    
    try:
        while True:
            try:
                devices = set(gateway.get_devices())
                if not devices:
                    time.sleep(CONFIG["data_interval"])
                    continue
                
                for dev_id in devices:
                    try:
                        packets = list(gateway.pops(dev_id))
                        for pkt in packets:
                            stats["packets_received"] += 1
                            raw = base64_to_numpy(pkt.data, np.uint8)
                            
                            if len(raw) >= 2 and raw[0] == PKT_MAGIC:
                                process_compact(dev_id, raw)
                            else:
                                data = base64_to_numpy(pkt.data, np.float32)
                                process_float(dev_id, data)
                    except Exception as e:
                        stats["errors"] += 1
                        logging.error(f"Packet error: {e}")
                
                time.sleep(CONFIG["data_interval"])
            except Exception as e:
                stats["errors"] += 1
                logging.error(f"Loop error: {e}")
                time.sleep(1)
                
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        # Cleanup
        dur = int(time.time() - session_start_time) if session_start_time else 0
        stop_line = f"# Stop @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Duration: {dur//60}m{dur%60}s)\n"
        
        if csv_file_all:
            csv_file_all.write(stop_line)
            csv_file_all.close()
        if csv_file_summary:
            csv_file_summary.write(stop_line)
            csv_file_summary.close()
        
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        if gateway_proc:
            gateway_proc.terminate()
        
        print("=" * 60)
        print(f"Stats: packets={stats['packets_received']}, mqtt={stats['mqtt_success']}, errors={stats['errors']}")
        print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("Riotee采集器异常退出:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
