import os
import csv
import json
import time
import datetime
import logging
import pytz
import pandas as pd
import paho.mqtt.client as mqtt

# Import local ML modules
import predict_sp_to_ppfd as sp_model
import predict_env_to_pn as pn_model
import predict_pwm_from_ppfd as pwm_model

try:
    from src.shelly_controller import rpc, DEVICES
except ImportError:
    DEVICES = {"Red": "dev_red", "Blue": "dev_blue", "Heater": "dev_heat"}
    def rpc(dev, cmd, params): logger.debug(f"[MOCK] {cmd} {params}")

# --- 1. DIRECTORY & LOGGING SETUP ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

timestamp_str = datetime.datetime.now().strftime("%Y%m%d")
log_filename = os.path.join(LOG_DIR, f'edge_node_{timestamp_str}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler() # Keeps printing to console
    ]
)
logger = logging.getLogger("GreenhouseEdge")

# --- 2. CSV SETUP ---
CSV_FILE = os.path.join(LOG_DIR, "greenhouse_data.csv")
CSV_HEADERS = [
    "Timestamp", "Tleaf", "Ci", "RH", 
    "PPFD_Predicted", "Pn_Predicted", 
    "Target_PPFD", "Red_PWM", "Blue_PWM", "Heater_PWM", 
    "Is_Offline"
]

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)

# --- 3. CONFIGURATION ---
TIMEZONE = pytz.timezone("Australia/Sydney")
MQTT_BROKER_IP = "azure.nocolor.cc"
MQTT_PORT = 1883
MQTT_USER = "feiyue"
MQTT_PASS = "123456789"

TOPIC_CMD = "growbox/commands/setpoints"
TOPIC_STATE = "growbox/state/aggregated"

TOPIC_MAP = {
    "homeassistant/sensor/chamber2_layer1_nano1_temperature/state": "Tleaf",
    "homeassistant/sensor/chamber2_co2_co2/state": "Ci",
    "homeassistant/sensor/chamber2_layer1_nano1_humidity/state": "RH", # Added RH for CSV
    "riotee/chamber_l6vs/sp_415": "sp_415_mean",
    "riotee/chamber_l6vs/sp_445": "sp_445_mean",
    "riotee/chamber_l6vs/sp_480": "sp_480_mean",
    "riotee/chamber_l6vs/sp_515": "sp_515_mean",
    "riotee/chamber_l6vs/sp_555": "sp_555_mean",
    "riotee/chamber_l6vs/sp_590": "sp_590_mean",
    "riotee/chamber_l6vs/sp_630": "sp_630_mean",
    "riotee/chamber_l6vs/sp_680": "sp_680_mean",
}

# Global State Tracker
env_state = {k: 0.0 for k in TOPIC_MAP.values()}
env_state.update({
    "current_red_pwm": 0, "current_blue_pwm": 0, "current_heater_pwm": 0,
    "target_ppfd": 0.0, "PPFD_pred": 0.0, "Pn_pred": 0.0
})

# Fallback Configuration
OFFLINE_TIMEOUT_SECONDS = 1800  # 30 minutes
last_command_time = time.time()
is_offline_mode = False

# Load ML Models
logger.info("Loading ML Models...")
sp_pkg = sp_model.load_package(sp_model.DEFAULT_MODEL_PACKAGE)
pn_pkg = pn_model.load_package(pn_model.DEFAULT_MODEL_PACKAGE)
pwm_pkg = pwm_model.load_model(None)

# --- 4. CORE FUNCTIONS ---

def log_data_to_csv():
    """Appends the current global state to the CSV file."""
    try:
        ts = datetime.datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                ts,
                env_state["Tleaf"],
                env_state["Ci"],
                env_state["RH"],
                env_state["PPFD_pred"],
                env_state["Pn_pred"],
                env_state["target_ppfd"],
                env_state["current_red_pwm"],
                env_state["current_blue_pwm"],
                env_state["current_heater_pwm"],
                int(is_offline_mode)
            ])
    except Exception as e:
        logger.error(f"Failed to write to CSV: {e}")

def calculate_and_publish_state(client):
    """Runs local ML models, sends data to cloud, and logs to CSV."""
    sp_df = pd.DataFrame([env_state])
    X_sp = sp_model.prepare_features(sp_df, sp_pkg["feature_columns"])
    env_state["PPFD_pred"] = float(sp_pkg["pipeline"].predict(X_sp)[0])
    
    r, b = env_state["current_red_pwm"], env_state["current_blue_pwm"]
    rb_ratio = (r / b) if b > 0 else 5.0 

    pn_input = {
        "T": env_state["Tleaf"], "CO2": env_state["Ci"], 
        "R:B": rb_ratio, "PPFD": env_state["PPFD_pred"]
    }
    pn_df = pd.DataFrame([pn_input])
    X_pn = pn_model.prepare_features(pn_df, pn_pkg["feature_columns"])
    env_state["Pn_pred"] = float(pn_pkg["pipeline"].predict(X_pn)[0])

    now = datetime.datetime.now(TIMEZONE)
    payload = {
        "Tleaf": env_state["Tleaf"],
        "Ci": env_state["Ci"],
        "PPFD_current": env_state["PPFD_pred"],
        "Pn_current": env_state["Pn_pred"],
        "local_time": now.strftime("%H:%M"),
        "isDay": 1 if 6 <= now.hour < 18 else 0
    }
    
    client.publish(TOPIC_STATE, json.dumps(payload))
    logger.info(f"📊 Published State & Logged Data: PPFD={env_state['PPFD_pred']:.1f}, Pn={env_state['Pn_pred']:.2f}")
    
    # Log to CSV every time we evaluate the state
    log_data_to_csv()

def run_fallback_control():
    global env_state, is_offline_mode
    now = datetime.datetime.now(TIMEZONE)
    is_day = 6 <= now.hour < 18

    safe_ppfd = 250.0 if is_day else 0.0
    safe_heater_pwm = 60 if env_state["Tleaf"] < 18.0 else 0
        
    logger.warning(f"⚠️ NETWORK TIMEOUT! Forcing safe state: PPFD={safe_ppfd}, Heat={safe_heater_pwm}%")

    if safe_ppfd > 0:
        res = pwm_model.build_result(safe_ppfd, pwm_pkg["recommended_rb_ratio"], pwm_pkg)
        red_pwm, blue_pwm = res["red_pwm"], res["blue_pwm"]
    else:
        red_pwm, blue_pwm = 0, 0

    env_state["target_ppfd"] = safe_ppfd
    env_state["current_red_pwm"] = red_pwm
    env_state["current_blue_pwm"] = blue_pwm
    env_state["current_heater_pwm"] = safe_heater_pwm

    if "Red" in DEVICES: rpc(DEVICES["Red"], "Light.Set", {"id": 0, "on": red_pwm > 0, "brightness": red_pwm})
    if "Blue" in DEVICES: rpc(DEVICES["Blue"], "Light.Set", {"id": 0, "on": blue_pwm > 0, "brightness": blue_pwm})
    if "Heater" in DEVICES: rpc(DEVICES["Heater"], "Switch.Set", {"id": 0, "on": safe_heater_pwm > 0})
    
    log_data_to_csv()

def on_message(client, userdata, msg):
    global last_command_time, is_offline_mode, env_state
    topic = msg.topic
    payload = msg.payload.decode()
    
    if topic == TOPIC_CMD:
        last_command_time = time.time()
        if is_offline_mode:
            logger.info("✅ Cloud connection resumed! Returning control to AI Agronomist.")
            is_offline_mode = False

        data = json.loads(payload)
        target_ppfd = data.get("target_ppfd", 0)
        heater_pwm = data.get("heater_pwm", 0)
        
        if target_ppfd > 0:
            res = pwm_model.build_result(target_ppfd, pwm_pkg["recommended_rb_ratio"], pwm_pkg)
            red_pwm, blue_pwm = res["red_pwm"], res["blue_pwm"]
        else:
            red_pwm, blue_pwm = 0, 0
            
        env_state["target_ppfd"] = target_ppfd
        env_state["current_red_pwm"] = red_pwm
        env_state["current_blue_pwm"] = blue_pwm
        env_state["current_heater_pwm"] = heater_pwm
        
        logger.info(f"⚙️ Actuating: PPFD({target_ppfd}) -> Red:{red_pwm}%, Blue:{blue_pwm}%, Heat:{heater_pwm}%")
        
        if "Red" in DEVICES: rpc(DEVICES["Red"], "Light.Set", {"id": 0, "on": red_pwm > 0, "brightness": red_pwm})
        if "Blue" in DEVICES: rpc(DEVICES["Blue"], "Light.Set", {"id": 0, "on": blue_pwm > 0, "brightness": blue_pwm})
        if "Heater" in DEVICES: rpc(DEVICES["Heater"], "Switch.Set", {"id": 0, "on": heater_pwm > 0})
        return

    try:
        val = float(payload)
        if topic in TOPIC_MAP:
            env_state[TOPIC_MAP[topic]] = val
    except ValueError:
        pass

def on_connect(client, userdata, flags, reason_code, properties):
    logger.info("✅ Connected to MQTT Broker!")
    client.subscribe(TOPIC_CMD)
    for t in TOPIC_MAP.keys():
        client.subscribe(t)

if __name__ == "__main__":
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.on_connect = on_connect
    client.on_message = on_message

    logger.info(f"Connecting to MQTT Broker at {MQTT_BROKER_IP}...")
    client.connect(MQTT_BROKER_IP, MQTT_PORT, 60)
    client.loop_start()

    last_publish_time = 0
    PUBLISH_INTERVAL = 900 # 15 minutes

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
        logger.info("🛑 Shutting down manually.")