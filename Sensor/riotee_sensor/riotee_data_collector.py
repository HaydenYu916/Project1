# -*- coding: utf-8 -*-
"""
Riotee Data Collector - Simplified Version
Collects sensor data and outputs CSV + JSON (LLM-friendly format)
"""

import time, json, logging, csv, os, sys, argparse, struct
import numpy as np
import paho.mqtt.client as mqtt
from datetime import datetime
from riotee_gateway import GatewayClient, base64_to_numpy

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
    "mqtt_broker": "azure.nocolor.pw", "mqtt_port": 1883,
    "mqtt_username": "feiyue", "mqtt_password": "123456789",
    "gateway_host": "localhost", "gateway_port": 8000,
    "data_interval": 0.1
}

# Globals
mqtt_client = None
csv_file_all = csv_writer_all = csv_file_summary = csv_writer_summary = None
record_id = 0
session_start_time = None
json_filepath = None
json_records = []
device_last_state = {}
stats = {"packets_received": 0, "mqtt_success": 0, "errors": 0}

# Packet format constants
PKT_MAGIC = 0xAA
PKT_BASIC, PKT_SPECTRAL, PKT_FULL, PKT_BASIC_CO2 = 0x01, 0x02, 0x03, 0x04

# CSV fieldnames
FIELDNAMES = ['id', 'timestamp', 'device_id', 'update_type', 'temperature', 'humidity',
              'a1_raw', 'vcap_raw', 'co2_ppm', 'sp_415', 'sp_445', 'sp_480', 'sp_515',
              'sp_555', 'sp_590', 'sp_630', 'sp_680', 'sp_clear', 'sp_nir', 
              'spectral_gain', 'sleep_time']

SUMMARY_FIELDS = ['id', 'timestamp', 'device_id', 'update_type', 'temperature', 
                  'humidity', 'a1_raw', 'vcap_raw', 'co2_ppm', 'spectral_gain', 'sleep_time']

def gain_to_mult(g):
    """Convert gain code (0-10) to multiplier"""
    return {0:0.5,1:1,2:2,3:4,4:8,5:16,6:32,7:64,8:128,9:256,10:512}.get(int(g), 0)

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

# ============ Gateway ============
def start_gateway():
    import subprocess, signal
    if not args.no_kill:
        try:
            result = subprocess.run(["pgrep", "-f", "riotee-gateway"], capture_output=True, text=True)
            if result.returncode == 0:
                for pid in result.stdout.strip().split('\n'):
                    if pid.strip():
                        try: os.kill(int(pid), signal.SIGTERM)
                        except: pass
                time.sleep(2)
        except: pass
    
    try:
        proc = subprocess.Popen(["riotee-gateway", "server", "-p", str(CONFIG["gateway_port"]), 
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
    
    # JSON file
    init_json(base, note)
    
    logging.info(f"Files: {path_all}, {path_sum}")
    return path_all, path_sum

def init_json(base, note):
    global json_filepath, json_records
    json_records = []
    json_filepath = os.path.join(LOGS_DIR, f"{base}_all.json")

def save_json():
    if not json_filepath or not json_records: return
    
    temps = [r["temperature"] for r in json_records if r.get("temperature")]
    hums = [r["humidity"] for r in json_records if r.get("humidity")]
    co2s = [r["co2_ppm"] for r in json_records if r.get("co2_ppm")]
    
    dur = int(time.time() - session_start_time) if session_start_time else 0
    
    output = {
        "prompt": "Analyze the following Riotee IoT sensor data.",
        "metadata": {
            "start_time": datetime.fromtimestamp(session_start_time).strftime("%Y-%m-%d %H:%M:%S") if session_start_time else "",
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration": f"{dur//3600}h{(dur%3600)//60}m{dur%60}s",
            "total_records": len(json_records),
            "devices": list(set(r.get("device_id","") for r in json_records))
        },
        "schema": {
            "temperature": "celsius", "humidity": "%RH", "a1_raw": "V", "vcap_raw": "V",
            "co2_ppm": "ppm", "spectral_data": "AS7341 10-channel counts",
            "spectral_gain": "multiplier", "sleep_time": "seconds"
        },
        "records": json_records,
        "summary": {
            "temperature": {"min": round(min(temps),2), "max": round(max(temps),2), "avg": round(sum(temps)/len(temps),2)} if temps else {},
            "humidity": {"min": round(min(hums),2), "max": round(max(hums),2), "avg": round(sum(hums)/len(hums),2)} if hums else {},
            "co2_ppm": {"min": min(co2s), "max": max(co2s), "avg": round(sum(co2s)/len(co2s))} if co2s else {}
        }
    }
    
    try:
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        logging.info(f"JSON saved: {len(json_records)} records")
    except Exception as e:
        logging.error(f"JSON save error: {e}")

# ============ Data Processing ============
def write_record(device_id, update_type, temp, hum, a1, vcap, co2, spectrum, gain, sleep):
    """Write data to CSV and JSON"""
    global record_id
    record_id += 1
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    gain_mult = gain_to_mult(gain)
    
    # CSV row
    row = {
        'id': record_id, 'timestamp': ts, 'device_id': device_id, 'update_type': update_type,
        'temperature': f"{temp:.2f}", 'humidity': f"{hum:.2f}",
        'a1_raw': f"{a1:.3f}", 'vcap_raw': f"{vcap:.3f}",
        'co2_ppm': co2 if co2 >= 0 else '',
        'spectral_gain': gain_mult, 'sleep_time': sleep
    }
    row.update({k: f"{v:.2f}" for k, v in spectrum.items()})
    
    try:
        csv_writer_all.writerow(row)
        csv_file_all.flush()
    except Exception as e:
        logging.error(f"CSV write error: {e}")
    
    # Summary (on significant change)
    last = device_last_state.get(device_id, {})
    if not last or abs(temp - last.get('temp', temp)) > 0.5 or sleep != last.get('sleep', sleep):
        summary_row = {k: row[k] for k in SUMMARY_FIELDS if k in row}
        try:
            csv_writer_summary.writerow(summary_row)
            csv_file_summary.flush()
        except: pass
    
    device_last_state[device_id] = {'temp': temp, 'sleep': sleep}
    
    # JSON - save every record
    json_records.append({
        "id": record_id, "timestamp": ts, "device_id": device_id, "update_type": update_type,
        "temperature": round(temp, 2), "humidity": round(hum, 2),
        "a1_raw": round(a1, 3), "vcap_raw": round(vcap, 3),
        "co2_ppm": co2 if co2 >= 0 else None,
        "spectral_data": {k: round(v, 2) for k, v in spectrum.items()},
        "spectral_gain": gain_mult, "sleep_time": sleep
    })
    save_json()
    
    # MQTT
    if mqtt_client:
        dev = device_id.replace("==","").replace("=","").replace("/","_")[:8]
        publish_mqtt(f"riotee/{dev}/temperature", f"{temp:.2f}")
        publish_mqtt(f"riotee/{dev}/humidity", f"{hum:.2f}")
        publish_mqtt(f"riotee/{dev}/vcap_raw", f"{vcap:.3f}")
        if co2 >= 0: publish_mqtt(f"riotee/{dev}/co2_ppm", co2)
    
    logging.info(f"{device_id}: T={temp:.1f}C H={hum:.1f}% CO2={co2 if co2>=0 else 'N/A'}ppm Sleep={sleep}s")

def process_compact(device_id, raw):
    """Process compact binary packet (0xAA magic)"""
    if len(raw) < 12 or raw[0] != PKT_MAGIC: return
    
    pkt_type = raw[1]
    try:
        temp_x100, hum_x100, a1_mv, vcap_mv, sleep, gain = struct.unpack('<hHHHBB', raw[2:12].tobytes())
    except: return
    
    temp, hum = temp_x100/100.0, hum_x100/100.0
    a1, vcap = a1_mv/1000.0, vcap_mv/1000.0
    
    spectrum = {f'sp_{wl}': 0.0 for wl in [415,445,480,515,555,590,630,680]}
    spectrum.update({'sp_clear': 0.0, 'sp_nir': 0.0})
    co2 = -1
    
    if pkt_type in (PKT_SPECTRAL, PKT_FULL) and len(raw) >= 32:
        try:
            arr = struct.unpack('<10H', raw[12:32].tobytes())
            for i, wl in enumerate([415,445,480,515,555,590,630,680]):
                spectrum[f'sp_{wl}'] = float(arr[i])
            spectrum['sp_clear'], spectrum['sp_nir'] = float(arr[8]), float(arr[9])
        except: pass
    
    if pkt_type == PKT_FULL and len(raw) >= 34:
        try: co2 = struct.unpack('<h', raw[32:34].tobytes())[0]
        except: pass
    elif pkt_type == PKT_BASIC_CO2 and len(raw) >= 14:
        try: co2 = struct.unpack('<h', raw[12:14].tobytes())[0]
        except: pass
    
    write_record(device_id, "COMPACT", temp, hum, a1, vcap, co2, spectrum, gain, sleep)

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
    elif len(data) >= 16:
        sleep, gain = int(data[4]), int(data[5])
        spectrum = {
            'sp_415': data[6], 'sp_445': data[7], 'sp_480': data[8], 'sp_515': data[9],
            'sp_555': data[10], 'sp_590': data[11], 'sp_630': data[12], 'sp_680': data[13],
            'sp_clear': data[14], 'sp_nir': data[15]
        }
        co2 = -1
    else:
        sleep, gain, co2 = 0, 255, -1
        spectrum = {f'sp_{wl}': 0.0 for wl in [415,445,480,515,555,590,630,680]}
        spectrum.update({'sp_clear': 0.0, 'sp_nir': 0.0})
    
    write_record(device_id, "FLOAT", temp, hum, a1, vcap, co2, spectrum, gain, sleep)

# ============ Main ============
def main():
    global mqtt_client
    print("=" * 60)
    print("Riotee Data Collector Started")
    print("=" * 60)
    
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
        
        save_json()
        
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        if gateway_proc:
            gateway_proc.terminate()
        
        print("=" * 60)
        print(f"Stats: packets={stats['packets_received']}, mqtt={stats['mqtt_success']}, errors={stats['errors']}")
        print("=" * 60)

if __name__ == "__main__":
    main()
