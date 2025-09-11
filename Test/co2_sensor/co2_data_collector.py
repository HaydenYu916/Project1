import serial
import time
import csv
import os
import json
import logging
import atexit
import sys
from datetime import datetime
from pathlib import Path
import paho.mqtt.client as mqtt

SERIAL_PORT = '/dev/Chamber2_Co2'  # 更新为新的设备名
BAUDRATE = 115200
CSV_PATH = '../logs/co2_data.csv'

# 当前目录文件配置
CURRENT_DIR = Path(__file__).parent
LOG_FILE = CURRENT_DIR / "co2_collector.log"
PID_FILE = CURRENT_DIR / "co2_collector.pid"
PRINT_OUTPUT_FILE = CURRENT_DIR / "co2_print_output.txt"  # print输出记录

# 全局变量
original_print = print
output_file_handle = None

# print输出记录功能
def setup_print_recording():
    """设置print输出记录"""
    global output_file_handle
    try:
        output_file_handle = open(PRINT_OUTPUT_FILE, 'w', encoding='utf-8')
        
        def custom_print(*args, **kwargs):
            # 调用原始print函数输出到控制台
            original_print(*args, **kwargs)
            # 同时写入文件
            if output_file_handle:
                original_print(*args, **kwargs, file=output_file_handle)
                output_file_handle.flush()
        
        # 替换全局print函数
        import builtins
        builtins.print = custom_print
        
        return True
    except Exception as e:
        original_print(f"❌ 设置print记录失败: {e}")
        return False

def close_print_recording():
    """关闭print输出记录"""
    global output_file_handle
    try:
        if output_file_handle:
            output_file_handle.close()
            output_file_handle = None
        
        # 恢复原始print函数
        import builtins
        builtins.print = original_print
    except Exception as e:
        original_print(f"⚠️ 关闭print记录时出错: {e}")

# MQTT配置
MQTT_CONFIG = {
    "broker": "azure.nocolor.pw",
    "port": 1883,
    "username": "feiyue",
    "password": "123456789",
    "device_name": "chamber2_co2",  # HA设备名称
}

# 全局变量
mqtt_client = None
ha_discovery_sent = False  # 防止重复发送HA发现配置

# 清理函数
def cleanup_logs(clean_log_file=False, clean_print_output=True):
    """程序退出时清理文件"""
    try:
        # 关闭print记录
        close_print_recording()
        
        # 总是清理PID文件
        if PID_FILE.exists():
            PID_FILE.unlink()
            print(f"✅ PID文件已清理: {PID_FILE}")
        
        # 根据参数决定是否清理print输出文件
        if clean_print_output and PRINT_OUTPUT_FILE.exists():
            PRINT_OUTPUT_FILE.unlink()
            print(f"✅ Print输出文件已清理: {PRINT_OUTPUT_FILE}")
        elif not clean_print_output and PRINT_OUTPUT_FILE.exists():
            print(f"📄 Print输出记录保存在: {PRINT_OUTPUT_FILE}")
        
        # 只在明确要求时清理日志文件
        if clean_log_file and LOG_FILE.exists():
            LOG_FILE.unlink()
            print(f"✅ 日志文件已清理: {LOG_FILE}")
        elif not clean_log_file and LOG_FILE.exists():
            print(f"📄 日志文件保存在: {LOG_FILE}")
    except Exception as e:
        print(f"⚠️ 清理文件时出错: {e}")

# 注册退出清理函数 - 只清理PID文件和print输出文件，保留日志文件
atexit.register(lambda: cleanup_logs(clean_log_file=False, clean_print_output=True))

# 配置日志 - 同时输出到控制台和文件
def setup_logging():
    """设置日志配置"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除默认处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建格式器
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# 初始化日志
logger = setup_logging()

def setup_mqtt():
    """设置MQTT连接"""
    global mqtt_client
    
    def on_connect(client, userdata, flags, rc, properties=None):
        # 兼容旧版本参数名（rc而不是reason_code）
        if rc == 0:
            logging.info("MQTT连接成功")
            # 发布测试消息验证连接
            test_topic = f"co2/{MQTT_CONFIG['device_name']}/status"
            client.publish(test_topic, "online", qos=0, retain=True)
            logging.info(f"MQTT测试消息已发布到: {test_topic}")
        else:
            logging.error(f"MQTT连接失败: {rc}")
    
    try:
        # 兼容新旧版本的paho-mqtt
        try:
            # 新版本 (paho-mqtt >= 2.0.0)
            client = mqtt.Client(
                client_id=f"co2_collector_{int(time.time())}",
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2
            )
        except AttributeError:
            # 旧版本 (paho-mqtt < 2.0.0)
            client = mqtt.Client(client_id=f"co2_collector_{int(time.time())}")
        
        client.username_pw_set(MQTT_CONFIG["username"], MQTT_CONFIG["password"])
        client.on_connect = on_connect
        
        client.connect(MQTT_CONFIG["broker"], MQTT_CONFIG["port"], 60)
        client.loop_start()
        return client
    except Exception as e:
        logging.error(f"MQTT连接失败: {e}")
        return None

def publish_mqtt(topic, value):
    """发布MQTT消息"""
    if mqtt_client:
        try:
            result = mqtt_client.publish(topic, str(value), qos=0, retain=False)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logging.debug(f"MQTT发布成功: {topic} = {value}")
                return True
            else:
                logging.warning(f"MQTT发布失败: {topic} = {value}, 错误码: {result.rc}")
                return False
        except Exception as e:
            logging.error(f"MQTT发布异常: {e}, 主题: {topic}, 值: {value}")
            return False
    else:
        logging.debug(f"MQTT不可用，跳过发布: {topic} = {value}")
        return False

def publish_ha_discovery():
    """发布Home Assistant自动发现配置"""
    global ha_discovery_sent
    
    if not mqtt_client or ha_discovery_sent:
        return False
    
    device_name = MQTT_CONFIG['device_name']
    
    # 先清理可能存在的旧配置
    old_config_topic = f"homeassistant/sensor/{device_name}_co2/config"
    mqtt_client.publish(old_config_topic, "", qos=0, retain=True)
    
    # 使用更标准的HA自动发现配置
    config = {
        "name": "CO2 Sensor",
        "object_id": "chamber2_co2_sensor",
        "unique_id": f"chamber2_co2_sensor",
        "state_topic": f"co2/{device_name}/value",
        "availability": {
            "topic": f"co2/{device_name}/status",
            "payload_available": "online",
            "payload_not_available": "offline"
        },
        "unit_of_measurement": "ppm",
        "icon": "mdi:molecule-co2",
        "device_class": "carbon_dioxide",
        "state_class": "measurement",
        "device": {
            "identifiers": ["chamber2_co2"],
            "name": "Chamber2 CO2",
            "manufacturer": "UNSW",
            "model": "CO2 Sensor",
            "sw_version": "1.0"
        }
    }
    
    # 新的配置主题 - 使用更清晰的命名
    config_topic = f"homeassistant/sensor/chamber2_co2_sensor/config"
    
    try:
        # 1. 发布可用性状态
        mqtt_client.publish(f"co2/{device_name}/status", "online", qos=0, retain=True)
        logging.info("设备状态已发布: online")
        
        # 2. 发布初始CO2值
        mqtt_client.publish(f"co2/{device_name}/value", "400", qos=0, retain=True)
        logging.info("初始CO2值已发布: 400 ppm")
        
        # 3. 发布HA配置（最后发布以确保状态主题已有数据）
        result = mqtt_client.publish(config_topic, json.dumps(config, indent=2), qos=0, retain=True)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logging.info(f"HA自动发现配置已发布到: {config_topic}")
            logging.info("配置内容:")
            logging.info(json.dumps(config, indent=2))
            
            ha_discovery_sent = True
            return True
        else:
            logging.warning(f"HA配置发布失败，错误码: {result.rc}")
            return False
    except Exception as e:
        logging.error(f"HA配置发布异常: {e}")
        return False

# 检查文件是否存在，决定是否写表头
def ensure_csv_header(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)  # 确保目录存在
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'co2'])

def cleanup_co2_files():
    """清理CO2系统相关文件"""
    print("🧹 清理CO2系统文件...")
    cleaned_count = 0
    
    # 定义CO2系统要清理的文件
    co2_files = [
        LOG_FILE,
        PID_FILE,
        PRINT_OUTPUT_FILE,
    ]
    
    for file_path in co2_files:
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"✅ 已清理: {file_path.name}")
                cleaned_count += 1
            except Exception as e:
                print(f"❌ 清理失败: {file_path.name} - {e}")
    
    print(f"🎉 CO2清理完成！共清理了 {cleaned_count} 个文件")
    return cleaned_count > 0

def main():
    global mqtt_client
    
    # 检查是否是stop命令
    if len(sys.argv) > 1 and sys.argv[1] == 'stop':
        cleanup_co2_files()
        return
    
    # 设置print输出记录
    if not setup_print_recording():
        return
    
    print("=" * 60)
    print("🚀 CO2传感器数据采集器启动")
    print("=" * 60)
    
    # 创建PID文件
    try:
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
        logging.info(f"PID文件已创建: {PID_FILE} (PID: {os.getpid()})")
        print(f"📄 日志文件: {LOG_FILE}")
        print(f"📄 PID文件: {PID_FILE}")
        print(f"📄 Print输出记录: {PRINT_OUTPUT_FILE}")
    except Exception as e:
        logging.error(f"创建PID文件失败: {e}")
        return
    
    # 初始化MQTT
    logging.info("初始化MQTT连接...")
    mqtt_client = setup_mqtt()
    if not mqtt_client:
        logging.warning("MQTT连接失败，将跳过MQTT上传功能，继续运行...")
    else:
        logging.info("MQTT连接成功")
        # 等待MQTT连接稳定
        time.sleep(2)
        # 重置HA发现状态以强制重新配置
        global ha_discovery_sent
        ha_discovery_sent = False
        # 发布HA自动发现配置
        if publish_ha_discovery():
            logging.info("Home Assistant自动发现配置完成")
        else:
            logging.warning("Home Assistant自动发现配置失败")
    
    ensure_csv_header(CSV_PATH)
    print(f"正在连接CO2传感器: {SERIAL_PORT}")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        print(f"✅ 开始监听串口 {SERIAL_PORT}，数据保存到 {CSV_PATH}")
        
        with open(CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            while True:
                # 监听 CO2 数据
                line = ser.readline().decode(errors='ignore').strip()
                if line:
                    try:
                        co2 = float(line)
                        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        writer.writerow([ts, co2])
                        f.flush()
                        
                        # 发布CO2数据到MQTT
                        device_name = MQTT_CONFIG['device_name']
                        publish_mqtt(f"co2/{device_name}/value", co2)
                        
                        # 控制台输出
                        mqtt_status = "已发布" if mqtt_client else "跳过(MQTT离线)"
                        print(f"{ts}, CO2: {co2} ppm [{mqtt_status}]")
                        
                    except ValueError:
                        # 非法数据行，跳过
                        pass
                
                time.sleep(0.1)
                
    except serial.SerialException as e:
        error_msg = f"❌ 串口错误: {e}"
        print(error_msg)
        logging.error(error_msg)
        print("请检查设备连接和权限")
    except KeyboardInterrupt:
        stop_msg = "\n⏹️  用户停止监听"
        print(stop_msg)
        logging.info("用户中断程序")
    except Exception as e:
        error_msg = f"❌ 意外错误: {e}"
        print(error_msg)
        logging.error(error_msg)
    finally:
        logging.info("开始清理资源...")
        
        try:
            ser.close()
            msg = "🔌 串口已关闭"
            print(msg)
            logging.info("串口连接已关闭")
        except:
            pass
        
        # 关闭MQTT连接
        if mqtt_client:
            try:
                # 发布离线状态
                device_name = MQTT_CONFIG['device_name']
                mqtt_client.publish(f"co2/{device_name}/status", "offline", qos=0, retain=True)
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
                logging.info("MQTT连接已关闭")
            except Exception as e:
                logging.error(f"关闭MQTT连接时出错: {e}")
        
        print("=" * 60)
        print("🛑 CO2传感器数据采集器已关闭")
        logging.info("CO2传感器数据采集器程序结束")
        print("=" * 60)
        
        # 手动调用清理函数确保资源清理 - 清理PID文件和print输出文件
        cleanup_logs(clean_log_file=False, clean_print_output=True)

if __name__ == '__main__':
    main()
