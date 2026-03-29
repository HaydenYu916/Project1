import os
import json
import time
import datetime
import logging
import threading
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib

import paho.mqtt.client as mqtt

# Import your LLM controller modules
from controllers.llm_led import LLMLEDController, LLMLEDPolicy, LLMLoggerConfig

# --- 1. SYSTEM LOGGING SETUP ---
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "cloud_config.toml"


def load_cloud_config():
    with CONFIG_PATH.open("rb") as f:
        raw_config = tomllib.load(f)

    mqtt_cfg = raw_config.get("mqtt", {})
    topic_cfg = raw_config.get("topics", {})
    watchdog_cfg = raw_config.get("watchdog", {})
    co2_cfg = raw_config.get("co2", {})
    llm_cfg = raw_config.get("llm", {})

    return {
        "log_dir": raw_config.get("paths", {}).get("log_dir", "logs/cloud"),
        "mqtt_broker_ip": mqtt_cfg.get("broker", "azure.nocolor.cc"),
        "mqtt_port": int(mqtt_cfg.get("port", 1883)),
        "mqtt_user": mqtt_cfg.get("username", ""),
        "mqtt_pass": mqtt_cfg.get("password", ""),
        "model_name": llm_cfg.get("model_name", "gpt-oss:120b"),
        "use_fixed_co2": bool(co2_cfg.get("use_fixed", True)),
        "fixed_co2_ppm": float(co2_cfg.get("fixed_ppm", 400.0)),
        "topic_state": topic_cfg.get("state", "growbox/state/aggregated"),
        "topic_cmd": topic_cfg.get("command", "growbox/commands/setpoints"),
        "edge_timeout_seconds": int(watchdog_cfg.get("edge_timeout_seconds", 2000)),
    }


CLOUD_CONFIG = load_cloud_config()
LOG_DIR = CLOUD_CONFIG["log_dir"]
os.makedirs(LOG_DIR, exist_ok=True)

timestamp_str = datetime.datetime.now().strftime("%Y%m%d")
log_filename = os.path.join(LOG_DIR, f'cloud_agronomist_{timestamp_str}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CloudController")

# --- 2. CONFIGURATION ---
MQTT_BROKER_IP = CLOUD_CONFIG["mqtt_broker_ip"]
MQTT_PORT = CLOUD_CONFIG["mqtt_port"]
MQTT_USER = CLOUD_CONFIG["mqtt_user"]
MQTT_PASS = CLOUD_CONFIG["mqtt_pass"]
MODEL_NAME = CLOUD_CONFIG["model_name"]
USE_FIXED_CO2 = CLOUD_CONFIG["use_fixed_co2"]
FIXED_CO2_PPM = CLOUD_CONFIG["fixed_co2_ppm"]

TOPIC_STATE = CLOUD_CONFIG["topic_state"]
TOPIC_CMD = CLOUD_CONFIG["topic_cmd"]

# Watchdog config
EDGE_TIMEOUT_SECONDS = CLOUD_CONFIG["edge_timeout_seconds"]
last_edge_contact = time.time()
edge_is_offline = False

# --- 3. LLM SETUP ---
policy = LLMLEDPolicy(temp_min=20.0, temp_max=30.0)
llm_logger = LLMLoggerConfig(
    enabled=True, 
    log_dir=os.path.join(LOG_DIR, "llm_decisions"), 
    log_full_context=True
)
controller = LLMLEDController(policy=policy, model_name=MODEL_NAME, logger=llm_logger)

# --- 4. CORE FUNCTIONS ---

def on_connect(client, userdata, flags, reason_code, properties):
    logger.info("✅ Cloud LLM Connected to MQTT! Subscribing to Edge State...")
    client.subscribe(TOPIC_STATE)

def on_message(client, userdata, msg):
    global last_edge_contact, edge_is_offline
    
    try:
        # Reset the watchdog timer
        last_edge_contact = time.time()
        if edge_is_offline:
            logger.info("📡 Edge Node connection restored! Resuming LLM control.")
            edge_is_offline = False

        state = json.loads(msg.payload.decode())
        logger.info(f"📩 Received State: PPFD={state.get('PPFD_current', 0):.1f}, Pn={state.get('Pn_current', 0):.2f}")
        
        # Format for LLM Prompt
        co2_air = FIXED_CO2_PPM if USE_FIXED_CO2 else state.get("Ci", FIXED_CO2_PPM)

        obs = {
            "temp_air": state.get("Tleaf", 25.0),
            "co2_air": co2_air,
            "local_time": state.get("local_time", "12:00"),
            "isDay": state.get("isDay", 1)
        }
        physics_ctx = {
            "Pn_current_rate": state.get("Pn_current", 0.0),
            "current_ppfd": state.get("PPFD_current", 0.0),
            "electricity_price_$per_kWh": 0.20
        }
        forecast = {"next_1h_temp": state.get("Tleaf", 25.0)}

        # Run LLM
        logger.info("🧠 Consulting AI Agronomist...")
        decision = controller.decide(obs, physics_ctx, forecast)
        logger.info(f"💡 LLM Rationale: {decision['rationale']}")
        
        # Send command back to Edge
        cmd_payload = json.dumps({
            "target_ppfd": decision["target_ppfd"],
            "heater_pwm": int(decision["uHeat_frac"] * 100)
        })
        client.publish(TOPIC_CMD, cmd_payload, retain=True)
        logger.info(f"📤 Sent Setpoints: {cmd_payload}")

    except Exception as e:
        logger.error(f"❌ Error processing message: {e}")

def edge_watchdog():
    """Background thread to monitor if the edge node stops sending data."""
    global edge_is_offline
    while True:
        time.sleep(60)
        if time.time() - last_edge_contact > EDGE_TIMEOUT_SECONDS and not edge_is_offline:
            logger.warning("⚠️ CRITICAL: No data received from Edge Node for over 30 minutes. It may have reverted to fallback mode.")
            edge_is_offline = True

# --- MAIN ---
if __name__ == "__main__":
    # Start the watchdog thread
    watchdog = threading.Thread(target=edge_watchdog, daemon=True)
    watchdog.start()

    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    
    logger.info(f"Connecting to MQTT Broker at {MQTT_BROKER_IP}...")
    try:
        mqtt_client.connect(MQTT_BROKER_IP, MQTT_PORT, 60)
        logger.info("🚀 Cloud Controller Online. Waiting for Edge...")
        mqtt_client.loop_forever()
    except Exception as e:
        logger.critical(f"Failed to connect to MQTT broker: {e}")
