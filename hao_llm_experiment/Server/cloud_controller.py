import os
import json
import time
import datetime
import logging
import threading
from logging.handlers import TimedRotatingFileHandler
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

# Import your LLM controller modules
from controllers.llm_led import LLMLEDController, LLMLEDPolicy, LLMLoggerConfig

# --- 1. SYSTEM LOGGING SETUP ---
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "cloud_config.toml"


def load_cloud_config():
    with CONFIG_PATH.open("rb") as f:
        raw_config = tomllib.load(f)

    timezone_cfg = raw_config.get("timezone", {})
    mqtt_cfg = raw_config.get("mqtt", {})
    topic_cfg = raw_config.get("topics", {})
    watchdog_cfg = raw_config.get("watchdog", {})
    co2_cfg = raw_config.get("co2", {})
    llm_cfg = raw_config.get("llm", {})

    return {
        "log_dir": raw_config.get("paths", {}).get("log_dir", "logs/cloud"),
        "timezone": timezone_cfg.get("name", "Australia/Sydney"),
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
LOG_DIR_PATH = Path(CLOUD_CONFIG["log_dir"])
if not LOG_DIR_PATH.is_absolute():
    LOG_DIR_PATH = BASE_DIR / LOG_DIR_PATH
LOG_DIR = str(LOG_DIR_PATH)
TIMEZONE = ZoneInfo(CLOUD_CONFIG["timezone"])
os.makedirs(LOG_DIR, exist_ok=True)

class TimezoneFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.tz = tz

    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=self.tz)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


class DailyTimezoneRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, log_dir, filename_prefix, tz, encoding="utf-8"):
        self.log_dir = Path(log_dir)
        self.filename_prefix = filename_prefix
        self.tz = tz
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_date = self._date_str(time.time())
        initial_filename = self._build_filename(self.current_log_date)
        super().__init__(
            filename=initial_filename,
            when="midnight",
            interval=1,
            backupCount=0,
            encoding=encoding,
            delay=True,
        )
        self.rolloverAt = self.computeRollover(time.time())

    def _date_str(self, epoch_seconds):
        dt = datetime.datetime.fromtimestamp(epoch_seconds, tz=self.tz)
        return dt.strftime("%Y%m%d")

    def _build_filename(self, date_str):
        return str(self.log_dir / f"{self.filename_prefix}_{date_str}.log")

    def computeRollover(self, current_time):
        current_dt = datetime.datetime.fromtimestamp(current_time, tz=self.tz)
        next_date = current_dt.date() + datetime.timedelta(days=1)
        next_midnight = datetime.datetime.combine(next_date, datetime.time.min, tzinfo=self.tz)
        return int(next_midnight.timestamp())

    def shouldRollover(self, record):
        record_date = self._date_str(record.created)
        return int(record.created >= self.rolloverAt or record_date != self.current_log_date)

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None

        current_time = time.time()
        self.current_log_date = self._date_str(current_time)
        self.baseFilename = os.path.abspath(self._build_filename(self.current_log_date))
        self.rolloverAt = self.computeRollover(current_time)

        if not self.delay:
            self.stream = self._open()


formatter = TimezoneFormatter(
    fmt='%(asctime)s - %(levelname)s - %(message)s',
    tz=TIMEZONE,
)
file_handler = DailyTimezoneRotatingFileHandler(
    log_dir=LOG_DIR,
    filename_prefix="cloud_agronomist",
    tz=TIMEZONE,
)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
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
cloud_start_time = time.time()
last_edge_contact = time.time()
first_edge_contact = None
edge_message_count = 0
edge_is_offline = False

# --- 3. LLM SETUP ---
policy = LLMLEDPolicy(temp_min=20.0, temp_max=30.0)
llm_logger = LLMLoggerConfig(
    enabled=True, 
    log_dir=os.path.join(LOG_DIR, "llm_decisions"), 
    log_full_context=True,
    timezone_name=CLOUD_CONFIG["timezone"],
)
controller = LLMLEDController(policy=policy, model_name=MODEL_NAME, logger=llm_logger)

# --- 4. CORE FUNCTIONS ---


def _format_wall_time(epoch_seconds):
    if epoch_seconds is None:
        return "never"
    return datetime.datetime.fromtimestamp(epoch_seconds, tz=TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


def _format_duration(seconds):
    total_seconds = max(0, int(seconds))
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    return f"{minutes}m{sec:02d}s"

def on_connect(client, userdata, flags, reason_code, properties):
    logger.info(
        "✅ Cloud LLM Connected to MQTT! Subscribing to Edge State topic=%s",
        TOPIC_STATE,
    )
    client.subscribe(TOPIC_STATE)

def on_message(client, userdata, msg):
    global last_edge_contact, first_edge_contact, edge_message_count, edge_is_offline
    
    try:
        # Reset the watchdog timer
        previous_contact = last_edge_contact
        last_edge_contact = time.time()
        if first_edge_contact is None:
            first_edge_contact = last_edge_contact
        edge_message_count += 1
        if edge_is_offline:
            logger.info(
                "📡 Edge Node connection restored after %s without messages. total_messages=%s",
                _format_duration(last_edge_contact - previous_contact),
                edge_message_count,
            )
            edge_is_offline = False

        state = json.loads(msg.payload.decode())
        logger.info(
            "📩 Received Edge message #%s topic=%s: local_time=%s tz=%s ppfd_now=%.1f pn_now=%.2f tleaf_now=%.2f last_target_ppfd=%.1f red_pwm=%s blue_pwm=%s saturated=%s valid=%s",
            edge_message_count,
            msg.topic,
            state.get("local_time", "--:--"),
            state.get("timezone", "unknown"),
            state.get("ppfd_now", 0.0),
            state.get("pn_now", 0.0),
            state.get("tleaf_now", 0.0),
            state.get("last_target_ppfd", 0.0),
            state.get("current_red_pwm", 0),
            state.get("current_blue_pwm", 0),
            state.get("pwm_saturation_label", "none"),
            int(bool(state.get("sensor_data_valid", False))),
        )

        if not state.get("sensor_data_valid", False):
            logger.warning(
                "Skipping LLM call because sensor_data_valid is false. missing_fields=%s",
                state.get("missing_fields", []),
            )
            return

        # Format core fields for the LLM prompt.
        co2_air = FIXED_CO2_PPM if USE_FIXED_CO2 else state.get("co2_now", FIXED_CO2_PPM)

        obs = {
            "local_time": state.get("local_time", "12:00"),
            "is_day": state.get("is_day", 1),
            "tleaf_now": state.get("tleaf_now", 25.0),
            "co2_now": co2_air,
            "ppfd_now": state.get("ppfd_now", 0.0),
            "pn_now": state.get("pn_now", 0.0),
            "power_now_w": state.get("power_now_w", 0.0),
            "current_red_pwm": state.get("current_red_pwm", 0),
            "current_blue_pwm": state.get("current_blue_pwm", 0),
            "pwm_saturated": state.get("pwm_saturated", 0),
            "pwm_saturation_label": state.get("pwm_saturation_label", "none"),
        }
        physics_ctx = {
            "tleaf_avg_3min": state.get("tleaf_avg_3min", state.get("tleaf_now", 25.0)),
            "pn_avg_3min": state.get("pn_avg_3min", state.get("pn_now", 0.0)),
            "tleaf_delta_15min": state.get("tleaf_delta_15min", 0.0),
            "pn_delta_15min": state.get("pn_delta_15min", 0.0),
            "last_target_ppfd": state.get("last_target_ppfd", 0.0),
            "current_red_pwm": state.get("current_red_pwm", 0),
            "current_blue_pwm": state.get("current_blue_pwm", 0),
            "pwm_saturated": state.get("pwm_saturated", 0),
            "pwm_saturation_label": state.get("pwm_saturation_label", "none"),
            "electricity_price_$per_kWh": 0.20,
        }
        forecast = {
            "next_1h_temp_estimate": state.get("tleaf_avg_3min", state.get("tleaf_now", 25.0)),
            "trend_hint": {
                "tleaf_delta_15min": state.get("tleaf_delta_15min", 0.0),
                "pn_delta_15min": state.get("pn_delta_15min", 0.0),
            },
        }

        # Run LLM
        logger.info("🧠 Consulting AI Agronomist...")
        decision = controller.decide(obs, physics_ctx, forecast)
        logger.info(f"💡 LLM Rationale: {decision['rationale']}")
        
        # Send command back to Edge
        cmd_payload = json.dumps({
            "target_ppfd": decision["target_ppfd"]
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
            elapsed = time.time() - last_edge_contact
            if edge_message_count == 0:
                logger.warning(
                    "⚠️ CRITICAL: No Edge messages received since cloud startup. topic=%s startup_at=%s elapsed=%s. Edge may be offline or publishing elsewhere.",
                    TOPIC_STATE,
                    _format_wall_time(cloud_start_time),
                    _format_duration(elapsed),
                )
            else:
                logger.warning(
                    "⚠️ CRITICAL: Edge message stream stalled. topic=%s last_message_at=%s first_message_at=%s elapsed=%s total_messages=%s. Edge may have reverted to fallback mode.",
                    TOPIC_STATE,
                    _format_wall_time(last_edge_contact),
                    _format_wall_time(first_edge_contact),
                    _format_duration(elapsed),
                    edge_message_count,
                )
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
