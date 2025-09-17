# -*- coding: utf-8 -*-
"""
Riotee Gateway 简化版 V7 - 适用于Chamber测试
===========================================

版本更新说明 (相比V5版本):
1. 新增光谱传感器增益字段支持 - 记录和传输设备增益设置(0-10对应0.5X-512X)
2. 新增设备休眠时间字段支持 - 记录设备断电间隔时间信息  
3. 增强数据兼容性 - 支持带增益和休眠时间的新格式数据
4. 改进CSV输出 - 自动包含spectral_gain和sleep_time列
5. 扩展MQTT发布 - 新增增益和休眠时间主题
6. 完善Home Assistant集成 - 自动发现增益和休眠时间传感器
7. 增强日志输出 - 显示增益倍数和休眠时间信息
8. 向后兼容性 - 自动处理老版本数据格式

核心功能:
- Riotee设备数据接收（支持新旧格式）
- 实时CSV数据记录（包含增益和休眠时间）
- MQTT数据发布（完整设备状态信息）
- 增益值智能转换显示
- 基础日志记录

数据格式支持:
- 基础传感器: 温度、湿度、电压
- 光谱数据: F1-F8, Clear, NIR (10通道)
- 设备配置: 增益设置、休眠时间
- 兼容模式: 自动处理缺失字段

适用场景: Chamber环境测试、光谱数据分析、设备状态监控
"""

import time
import json
import logging
import paho.mqtt.client as mqtt
import sys
import os
import argparse
from datetime import datetime
from riotee_gateway import GatewayClient, base64_to_numpy
import csv
import numpy as np

# ============================
# 命令行参数配置
# ============================
parser = argparse.ArgumentParser(description='Riotee Gateway简化版')
parser.add_argument('--debug', action='store_true', help='启用调试模式')
parser.add_argument('--force-overwrite', action='store_true', help='强制覆写已存在的CSV文件（默认会添加序号避免覆写）')
parser.add_argument('--no-timestamp', action='store_true', help='不使用时间戳命名（仅当指定csv_name时有效）')
parser.add_argument('--no-kill', action='store_true', help='不自动清理现有Gateway进程')
parser.add_argument('--new-file', action='store_true', help='新文件模式：强制创建新的带时间戳的CSV文件（默认为追加模式）')
# parser.add_argument('--csv_name', type=str, default=None, help='自定义CSV文件名（不含扩展名）')
parser.add_argument('csv_name', type=str, nargs='?', default=None, help='csv文件名（不带扩展名，可选）')
parser.add_argument('comment', type=str, nargs='?', default=None, help='写入csv第一行的注释（可选）')
args = parser.parse_args()

# ============================
# 基础环境设置  
# ============================
# 创建logs目录 - 使用本地logs目录，避免与其他传感器冲突
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# ============================
# 系统配置常量
# ============================
# 基础配置
CONFIG = {
    "mqtt_broker": "azure.nocolor.pw",
    "mqtt_port": 1883,
    "mqtt_username": "feiyue", 
    "mqtt_password": "123456789",
    "gateway_host": "localhost",
    "gateway_port": 8000,
    "data_interval": 0.1,  # 数据处理间隔(秒)
    "reconnect_interval": 5,  # 重连间隔(秒)
}

# 传感器类型定义
SENSOR_TYPES = {
    "temp": {"unit": "℃", "precision": 2},
    "hum": {"unit": "%", "precision": 2}, 
    "a1_raw": {"unit": "V", "precision": 3},
    "vcap_raw": {"unit": "V", "precision": 3},
    "sp_415": {"unit": "count", "precision": 1},
    "sp_445": {"unit": "count", "precision": 1},
    "sp_480": {"unit": "count", "precision": 1},
    "sp_515": {"unit": "count", "precision": 1},
    "sp_555": {"unit": "count", "precision": 1},
    "sp_590": {"unit": "count", "precision": 1},
    "sp_630": {"unit": "count", "precision": 1},
    "sp_680": {"unit": "count", "precision": 1},
    "sp_clear": {"unit": "count", "precision": 1},  # Clear通道
    "sp_nir": {"unit": "count", "precision": 1},    # NIR通道
}

# ============================
# 全局变量
# ============================
mqtt_client = None
gateway_client = None
csv_writer_all = None
csv_writer_summary = None
csv_file_all = None
csv_file_summary = None
record_id = 0
session_start_time = None  # 会话开始时间

# 添加统计信息（参考V4版本）
stats = {
    "packets_received": 0,
    "mqtt_publish_success": 0,
    "mqtt_publish_failed": 0,
    "connection_errors": 0,
    "data_processing_errors": 0
}

# 已发现的设备集合，用于HA自动发现
discovered_devices = set()

# ============================
# MQTT上传相关函数
# ============================
def setup_mqtt():
    """
    [MQTT上传] 设置MQTT连接
    功能: 初始化MQTT客户端并建立连接
    """
    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            logging.info("MQTT连接成功")
            # 发布测试消息验证连接
            test_topic = "riotee/system/status"
            client.publish(test_topic, "online", qos=0, retain=True)
            logging.info(f"MQTT测试消息已发布到: {test_topic}")
        else:
            logging.error(f"MQTT连接失败: {reason_code}")
    
    client = mqtt.Client(
        client_id=f"riotee_simple_{int(time.time())}",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2
    )
    
    client.username_pw_set(CONFIG["mqtt_username"], CONFIG["mqtt_password"])
    client.on_connect = on_connect
    
    try:
        client.connect(CONFIG["mqtt_broker"], CONFIG["mqtt_port"], 60)
        client.loop_start()
        return client
    except Exception as e:
        logging.error(f"MQTT连接失败: {e}")
        return None

def publish_mqtt(topic, value):
    """
    [MQTT上传] 发布MQTT消息
    功能: 将传感器数据发布到指定MQTT主题
    """
    if mqtt_client:
        try:
            result = mqtt_client.publish(topic, str(value), qos=0, retain=False)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                stats["mqtt_publish_success"] += 1
                # 调试模式下显示发布的消息
                if args.debug:
                    logging.debug(f"MQTT发布成功: {topic} = {value}")
                return True
            else:
                stats["mqtt_publish_failed"] += 1
                logging.warning(f"MQTT发布失败: {topic} = {value}, 错误码: {result.rc}")
                return False
        except Exception as e:
            logging.error(f"MQTT发布失败: {e}, 主题: {topic}, 值: {value}")
            stats["mqtt_publish_failed"] += 1
            return False
    else:
        # MQTT客户端不可用
        logging.warning(f"MQTT不可用，跳过发布: {topic} = {value}")
        return False

# ============================
# Riotee数据接收相关函数
# ============================
def start_gateway_server():
    """
    [Riotee接收] 启动Gateway服务器
    功能: 直接启动Riotee Gateway服务器，每次运行前先kill掉现有进程
    """
    import subprocess
    import time
    import signal
    import os
    
    def kill_existing_gateway():
        """
        强制kill掉现有的Gateway进程
        避免端口冲突和进程残留
        """
        try:
            # 查找并kill掉所有riotee-gateway进程
            result = subprocess.run(["pgrep", "-f", "riotee-gateway"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                logging.info(f"发现 {len(pids)} 个现有Gateway进程，正在终止...")
                
                for pid in pids:
                    if pid.strip():
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                            logging.info(f"已终止进程 PID: {pid}")
                        except (ValueError, ProcessLookupError) as e:
                            logging.debug(f"终止进程 {pid} 失败: {e}")
                
                # 等待进程完全终止
                time.sleep(2)
                
                # 再次检查，如果还有进程，强制kill
                result = subprocess.run(["pgrep", "-f", "riotee-gateway"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    remaining_pids = result.stdout.strip().split('\n')
                    logging.warning(f"仍有 {len(remaining_pids)} 个进程未终止，强制kill...")
                    
                    for pid in remaining_pids:
                        if pid.strip():
                            try:
                                os.kill(int(pid), signal.SIGKILL)
                                logging.info(f"强制终止进程 PID: {pid}")
                            except (ValueError, ProcessLookupError) as e:
                                logging.debug(f"强制终止进程 {pid} 失败: {e}")
                    
                    time.sleep(1)
                
                logging.info("现有Gateway进程清理完成")
            else:
                logging.info("未发现现有Gateway进程")
                
        except Exception as e:
            logging.warning(f"清理现有进程时出错: {e}")
    
    try:
        # 根据用户选择决定是否清理现有进程
        if not args.no_kill:
            kill_existing_gateway()
        else:
            logging.info("用户选择不自动清理进程，跳过进程清理")
        
        # 直接使用自动检测模式启动Gateway
        logging.info("启动Gateway服务器（自动检测模式）...")
        gateway_process = subprocess.Popen(
            ["riotee-gateway", "server", "-p", str(CONFIG["gateway_port"]), "-h", CONFIG["gateway_host"]],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logging.info(f"Riotee Gateway服务器已启动，PID: {gateway_process.pid}，端口: {CONFIG['gateway_port']}")
        
        # 等待服务器启动，给更多时间建立连接
        time.sleep(5)
        return gateway_process
        
    except Exception as e:
        logging.error(f"启动Gateway服务器失败: {e}")
        logging.info("请手动运行: riotee-gateway server -p 8000 -h localhost")
        return None

def setup_gateway():
    """
    [Riotee接收] 设置Gateway连接
    功能: 初始化Riotee Gateway客户端连接
    """
    try:
        client = GatewayClient(host=CONFIG["gateway_host"], port=CONFIG["gateway_port"])
        logging.info("Gateway连接成功")
        
        # 测试连接是否真的可用
        try:
            test_devices = list(client.get_devices())
            logging.info(f"Gateway连接测试成功，当前设备数量: {len(test_devices)}")
            if test_devices:
                logging.info(f"检测到的设备: {test_devices}")
        except Exception as e:
            logging.warning(f"Gateway连接测试失败: {e}")
            logging.info("Gateway可能还在启动中，继续尝试...")
        
        return client
    except Exception as e:
        logging.error(f"Gateway连接失败: {e}")
        return None

# ============================
# CSV数据记录相关函数
# ============================
def init_csv_session(experiment_note=""):
    """
    [CSV会话] 初始化CSV会话，写入启动标记
    功能: 类似aioshelly的会话管理，记录实验开始时间和备注
    """
    global session_start_time
    session_start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if experiment_note.strip():
        start_line = f"# Start @ {timestamp} - {experiment_note}\n"
    else:
        start_line = f"# Start @ {timestamp}\n"
    
    return start_line

def stop_csv_session():
    """
    [CSV会话] 结束CSV会话，写入结束标记和持续时间
    功能: 类似aioshelly的会话结束处理
    """
    global session_start_time
    end_time = time.time()
    duration_sec = int(end_time - session_start_time) if session_start_time else 0
    h, m, s = duration_sec // 3600, (duration_sec % 3600) // 60, duration_sec % 60
    duration_str = f"{h}h{m}m{s}s" if h else f"{m}m{s}s"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stop_line = f"# Stop @ {timestamp} (Duration: {duration_str})\n"
    
    return stop_line

def setup_csv():
    """
    [CSV记录] 设置双CSV文件系统
    功能: 创建全量数据文件和摘要文件，类似aioshelly的多文件日志
    - 全量文件: 记录所有数据包
    - 摘要文件: 记录设备首次发现和重要状态变化
    """
    global csv_file_all, csv_writer_all, csv_file_summary, csv_writer_summary
    
    def generate_unique_filename(base_name, extension=".csv"):
        """生成唯一文件名，避免覆写（仅在非追加模式下使用）"""
        if not base_name.endswith(extension):
            base_name += extension
        
        filepath = os.path.join(LOGS_DIR, base_name)
        
        # 默认追加模式：直接返回固定文件名（除非指定--new-file）
        if not args.new_file:
            return filepath
        
        # 非追加模式：使用原有逻辑
        if args.force_overwrite or not os.path.exists(filepath):
            if args.force_overwrite and os.path.exists(filepath):
                logging.warning(f"强制覆写已存在的文件: {filepath}")
            return filepath
        
        name_without_ext = base_name[:-len(extension)]
        counter = 1
        
        while True:
            new_name = f"{name_without_ext}_{counter}{extension}"
            new_filepath = os.path.join(LOGS_DIR, new_name)
            
            if not os.path.exists(new_filepath):
                logging.info(f"文件 {base_name} 已存在，使用新名称: {new_name}")
                return new_filepath
            
            counter += 1
    
    # 生成基础文件名
    if args.csv_name:
        if args.no_timestamp or not args.new_file:  # 默认追加模式不使用时间戳
            base_filename = args.csv_name
            experiment_note = args.comment if args.comment else args.csv_name
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"{args.csv_name}_{timestamp}"
            experiment_note = args.comment if args.comment else f"{args.csv_name}_{timestamp}"
    else:
        if not args.new_file:
            # 默认追加模式文件名
            base_filename = "riotee_data"
            experiment_note = args.comment if args.comment else "riotee_data"
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"riotee_data_{timestamp}"
            experiment_note = args.comment if args.comment else f"riotee_data_{timestamp}"
    
    # 定义字段名
    fieldnames = ['id', 'timestamp', 'device_id', 'update_type', 'temperature', 'humidity', 
                  'a1_raw', 'vcap_raw'] + [f'sp_{i}' for i in [415, 445, 480, 515, 555, 590, 630, 680]] + \
                 ['sp_clear', 'sp_nir', 'spectral_gain', 'sleep_time']
    
    # 摘要文件字段（简化版）
    summary_fieldnames = ['id', 'timestamp', 'device_id', 'update_type', 'temperature', 'humidity', 
                         'a1_raw', 'vcap_raw', 'spectral_gain', 'sleep_time']
    
    # 生成会话开始标记
    start_line = init_csv_session(experiment_note)
    
    # 1. 处理全量数据文件
    filepath_all = generate_unique_filename(f"{base_filename}_all.csv")
    
    if not args.new_file and os.path.exists(filepath_all):
        # 默认追加模式且文件存在
        logging.info(f"追加模式：追加到现有全量文件 {filepath_all}")
        csv_file_all = open(filepath_all, 'a', newline='')
        csv_file_all.write(start_line)  # 只写会话开始标记
        csv_writer_all = csv.DictWriter(csv_file_all, fieldnames=fieldnames)
        # 不重写表头
    else:
        # 新建文件模式
        if not args.new_file:
            logging.info(f"追加模式：创建新全量文件 {filepath_all}")
        else:
            logging.info(f"新文件模式：创建全量文件 {filepath_all}")
        csv_file_all = open(filepath_all, 'w', newline='')
        csv_file_all.write(start_line)
        csv_writer_all = csv.DictWriter(csv_file_all, fieldnames=fieldnames)
        csv_writer_all.writeheader()
    
    # 2. 处理摘要文件
    filepath_summary = generate_unique_filename(f"{base_filename}_summary.csv")
    
    if not args.new_file and os.path.exists(filepath_summary):
        # 默认追加模式且文件存在
        logging.info(f"追加模式：追加到现有摘要文件 {filepath_summary}")
        csv_file_summary = open(filepath_summary, 'a', newline='')
        csv_file_summary.write(start_line)  # 只写会话开始标记
        csv_writer_summary = csv.DictWriter(csv_file_summary, fieldnames=summary_fieldnames)
        # 不重写表头
    else:
        # 新建文件模式
        if not args.new_file:
            logging.info(f"追加模式：创建新摘要文件 {filepath_summary}")
        else:
            logging.info(f"新文件模式：创建摘要文件 {filepath_summary}")
        csv_file_summary = open(filepath_summary, 'w', newline='')
        csv_file_summary.write(start_line)
        csv_writer_summary = csv.DictWriter(csv_file_summary, fieldnames=summary_fieldnames)
        csv_writer_summary.writeheader()
    
    logging.info(f"全量数据文件: {filepath_all}")
    logging.info(f"摘要数据文件: {filepath_summary}")
    logging.info(f"实验备注: {experiment_note}")
    
    return filepath_all, filepath_summary

# ============================
# Home Assistant自动发现相关函数
# ============================
def publish_ha_discovery(device_clean, sensor_type, unit, icon="mdi:gauge"):
    """
    [HA集成] 发布Home Assistant自动发现配置
    功能: 让HA自动识别和配置传感器
    """
    if not mqtt_client:
        return False
    
    # 创建唯一ID
    unique_id = f"riotee_{device_clean}_{sensor_type}"
    
    # HA发现配置
    config = {
        "name": f"Riotee {device_clean.replace('_', ' ').title()} {sensor_type.replace('_', ' ').title()}",
        "unique_id": unique_id,
        "state_topic": f"riotee/{device_clean}/{sensor_type}",
        "unit_of_measurement": unit,
        "icon": icon,
        "device": {
            "identifiers": [f"riotee_{device_clean}"],
            "name": f"Riotee {device_clean.replace('_', ' ').title()}",
            "manufacturer": "UNSW",
            "model": "Riotee Sensor"
        }
    }
    
    # 发布配置到HA发现主题
    config_topic = f"homeassistant/sensor/riotee_{device_clean}_{sensor_type}/config"
    
    try:
        import json
        result = mqtt_client.publish(config_topic, json.dumps(config), qos=0, retain=True)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logging.info(f"HA自动发现配置已发布: {unique_id}")
            return True
        else:
            logging.warning(f"HA配置发布失败: {unique_id}")
            return False
    except Exception as e:
        logging.error(f"HA配置发布异常: {e}")
        return False

# ============================
# 统计信息相关函数
# ============================
def print_stats():
    """打印统计信息（参考V4版本）"""
    logging.info("===== 运行统计 =====")
    for key, value in stats.items():
        logging.info(f"{key}: {value}")
    logging.info("====================")

# ============================
# 数据处理核心函数
# ============================

def gain_value_to_string(gain_value):
    """
    [增益转换] 将增益数值转换为可读字符串
    功能: 将Riotee设备发送的增益数值转换为增益倍数描述
    参数: gain_value - 增益数值（0-10对应0.5X-512X，255表示无效）
    返回: 增益倍数字符串
    """
    gain_map = {
        0: "0.5X", 1: "1X", 2: "2X", 3: "4X", 4: "8X",
        5: "16X", 6: "32X", 7: "64X", 8: "128X", 9: "256X", 10: "512X"
    }
    return gain_map.get(gain_value, "Unknown" if gain_value == 255 else f"Invalid({gain_value})")

def gain_value_to_multiplier(gain_value):
    """
    将增益编码(0-10, 255无效)转换为数值倍率
    0->0.5, 1->1, 2->2, 3->4, 4->8, 5->16, 6->32, 7->64, 8->128, 9->256, 10->512
    无效值返回0
    """
    gain_mult_map = {
        0: 0.5, 1: 1, 2: 2, 3: 4, 4: 8,
        5: 16, 6: 32, 7: 64, 8: 128, 9: 256, 10: 512
    }
    return gain_mult_map.get(int(gain_value), 0)

# 设备状态跟踪（用于摘要文件）
device_last_state = {}

def process_device_data(device_id, data_arr, update_type="DATA"):
    """
    [数据处理] 处理设备数据 - 增强版双文件记录
    功能: 解析Riotee原始数据，同时保存到全量CSV和摘要CSV，并发布到MQTT
    包含: 数据解析 + 双CSV记录 + MQTT上传 + 智能摘要
    支持: 二进制数据格式和JSON格式
    新增: 光谱传感器增益值和设备休眠时间字段
    特性: 类似aioshelly的多文件日志系统
    """
    global record_id, csv_writer_all, csv_writer_summary, device_last_state
    
    record_id += 1
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    # 检查数据格式：如果是JSON格式，直接解析
    if len(data_arr) == 1 and isinstance(data_arr[0], str):
        try:
            # 尝试解析JSON数据
            import json
            json_data = json.loads(data_arr[0])
            
            # 提取JSON中的基础传感器数据
            temp = json_data.get('temperature', 0.0)
            hum = json_data.get('humidity', 0.0)
            a1_raw = json_data.get('a1_raw', 0.0)
            vcap_raw = json_data.get('v_raw', 0.0)  # 注意：JSON中是v_raw，不是vcap_raw
            
            # 提取新增的设备配置数据
            spectral_gain = json_data.get('spectral_gain', 255)  # 默认255表示无效值
            sleep_time = json_data.get('sleep_time', 0)  # 默认0表示未知
            
            # 提取光谱数据（如果存在）
            spectrum_data = {}
            wavelengths = [415, 445, 480, 515, 555, 590, 630, 680]
            for i, wl in enumerate(wavelengths):
                spectrum_data[f'sp_{wl}'] = json_data.get(f'sp_{wl}', 0.0)
            
            # 提取Clear和NIR通道数据
            spectrum_data['sp_clear'] = json_data.get('sp_clear', 0.0)
            spectrum_data['sp_nir'] = json_data.get('sp_nir', 0.0)
            
            logging.debug(f"成功解析JSON数据: {json_data}")
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logging.warning(f"JSON解析失败: {e}，使用默认值")
            # 解析失败时使用默认值
            temp = hum = a1_raw = vcap_raw = 0.0
            spectral_gain = 255  # 错误值
            sleep_time = 0  # 未知值
            spectrum_data = {f'sp_{wl}': 0.0 for wl in [415, 445, 480, 515, 555, 590, 630, 680]}
            spectrum_data.update({'sp_clear': 0.0, 'sp_nir': 0.0})
    else:
        # 二进制数据格式 - 智能识别数据包类型
        # [调试] 显示数组长度和内容
        logging.info(f"[调试] 设备{device_id}: 数据长度={len(data_arr)}, 数据内容={data_arr}")
        
        # [数据解析] 从原始数组提取基础传感器数据
        temp = data_arr[0] if len(data_arr) > 0 else 0
        hum = data_arr[1] if len(data_arr) > 1 else 0
        a1_raw = data_arr[2] if len(data_arr) > 2 else 0
        vcap_raw = data_arr[3] if len(data_arr) > 3 else 0
        
        # 根据数据长度判断数据包类型 - 修复为16个float的统一数据结构
        # C端 sensor_data_t 结构: temp, hum, a1_raw, v_raw, sleep_time_f, spectral_gain_f, spectrum[10]
        if len(data_arr) == 16:  # 统一数据包: 16个float (基础4 + 增益1 + 休眠1 + 光谱10 = 16)
            logging.info(f"[调试] 设备{device_id}: 检测到统一数据包（包含所有字段：温度、湿度、A1、VCAP、休眠时间、增益、光谱数据）")
            
            # [数据解析] 提取增益和休眠时间（位置：4和5）
            sleep_time = int(data_arr[4]) if len(data_arr) > 4 else 0
            spectral_gain = int(data_arr[5]) if len(data_arr) > 5 else 255
            
            # [数据解析] 提取光谱数据（位置：6-15，共10个float）
            # C端发送顺序: spectrum[0] 到 spectrum[9] 对应 F1-F8, Clear, NIR
            spectrum_data = {}
            # F1..F8 -> 415..680 (按顺序映射)
            spectrum_data['sp_415'] = data_arr[6]  if len(data_arr) > 6  else 0  # F1
            spectrum_data['sp_445'] = data_arr[7]  if len(data_arr) > 7  else 0  # F2
            spectrum_data['sp_480'] = data_arr[8]  if len(data_arr) > 8  else 0  # F3
            spectrum_data['sp_515'] = data_arr[9]  if len(data_arr) > 9  else 0  # F4
            spectrum_data['sp_555'] = data_arr[10] if len(data_arr) > 10 else 0  # F5
            spectrum_data['sp_590'] = data_arr[11] if len(data_arr) > 11 else 0  # F6
            spectrum_data['sp_630'] = data_arr[12] if len(data_arr) > 12 else 0  # F7
            spectrum_data['sp_680'] = data_arr[13] if len(data_arr) > 13 else 0  # F8
            # Clear, NIR
            spectrum_data['sp_clear'] = data_arr[14] if len(data_arr) > 14 else 0  # Clear
            spectrum_data['sp_nir']   = data_arr[15] if len(data_arr) > 15 else 0  # NIR
            
        elif len(data_arr) == 13:  # 兼容旧格式：完整数据包 (基础4 + 增益1 + 休眠1 + 部分光谱7 = 13)
            logging.info(f"[调试] 设备{device_id}: 检测到旧格式完整数据包（兼容模式）")
            
            # [数据解析] 提取增益和休眠时间（位置：4和5）
            spectral_gain = int(data_arr[4]) if len(data_arr) > 4 else 255
            sleep_time = int(data_arr[5]) if len(data_arr) > 5 else 0
            
            # [数据解析] 提取光谱数据（位置：6-12，共7个float）
            spectrum_data = {}
            # F1..F4 -> 415..515
            spectrum_data['sp_415'] = data_arr[6]  if len(data_arr) > 6  else 0
            spectrum_data['sp_445'] = data_arr[7]  if len(data_arr) > 7  else 0
            spectrum_data['sp_480'] = data_arr[8]  if len(data_arr) > 8  else 0
            spectrum_data['sp_515'] = data_arr[9]  if len(data_arr) > 9  else 0
            # Clear, NIR
            spectrum_data['sp_clear'] = data_arr[10] if len(data_arr) > 10 else 0
            spectrum_data['sp_nir']   = data_arr[11] if len(data_arr) > 11 else 0
            # F5 (仅一个)
            spectrum_data['sp_555'] = data_arr[12] if len(data_arr) > 12 else 0
            # F6,F7,F8超出范围，设为0
            spectrum_data['sp_590'] = 0.0
            spectrum_data['sp_630'] = 0.0
            spectrum_data['sp_680'] = 0.0
            
        elif len(data_arr) == 6:  # 扩展基础数据包 (基础4 + 休眠1 + 增益1 = 6)
            logging.info(f"[调试] 设备{device_id}: 检测到扩展基础数据包（包含休眠时间和增益）")
            # 没有光谱数据
            spectrum_data = {f'sp_{wl}': 0.0 for wl in [415, 445, 480, 515, 555, 590, 630, 680]}
            spectrum_data.update({'sp_clear': 0.0, 'sp_nir': 0.0})
            
            sleep_time = int(data_arr[4]) if len(data_arr) > 4 else 0  # 休眠时间在位置4
            spectral_gain = int(data_arr[5]) if len(data_arr) > 5 else 255  # 增益在位置5
            
        elif len(data_arr) == 5:  # 基础数据包 (基础4 + 休眠1 = 5)
            logging.info(f"[调试] 设备{device_id}: 检测到基础数据包（包含休眠时间）")
            # 没有光谱数据
            spectrum_data = {f'sp_{wl}': 0.0 for wl in [415, 445, 480, 515, 555, 590, 630, 680]}
            spectrum_data.update({'sp_clear': 0.0, 'sp_nir': 0.0})
            
            spectral_gain = 255  # 无光谱传感器
            sleep_time = int(data_arr[4]) if len(data_arr) > 4 else 0  # 休眠时间在位置4
            
        elif len(data_arr) == 4:  # 旧版基础数据包 (只有基础4个数据)
            logging.info(f"[调试] 设备{device_id}: 检测到旧版基础数据包")
            # 没有光谱数据和休眠时间
            spectrum_data = {f'sp_{wl}': 0.0 for wl in [415, 445, 480, 515, 555, 590, 630, 680]}
            spectrum_data.update({'sp_clear': 0.0, 'sp_nir': 0.0})
            
            spectral_gain = 255  # 无光谱传感器
            sleep_time = 0  # 无休眠时间信息
            
        else:  # 其他长度 - 尝试智能解析
            logging.info(f"[调试] 设备{device_id}: 未知数据包格式，尝试智能解析")
            # 部分光谱数据
            spectrum_data = {}
            wavelengths = [415, 445, 480, 515, 555, 590, 630, 680]
            for i, wl in enumerate(wavelengths):
                spectrum_data[f'sp_{wl}'] = data_arr[4+i] if len(data_arr) > 4+i else 0
            spectrum_data['sp_clear'] = data_arr[12] if len(data_arr) > 12 else 0
            spectrum_data['sp_nir'] = data_arr[13] if len(data_arr) > 13 else 0
            
            # 搜索增益和休眠时间
            spectral_gain = 255
            sleep_time = 0
            for i in range(len(data_arr)):
                val = int(data_arr[i])
                if 0 <= val <= 10:  # 可能的增益值
                    spectral_gain = val
                elif 1 <= val <= 3600:  # 可能的休眠时间
                    sleep_time = val
        
        # [调试] 显示解析结果
        logging.info(f"[调试] 设备{device_id}: 数据包长度={len(data_arr)}, 增益={spectral_gain}, 休眠={sleep_time}s")
    
    # [CSV记录] 写入双CSV文件系统 - 增益写入实际倍率而非编码
    spectral_gain_multiplier = gain_value_to_multiplier(spectral_gain)
    
    # 全量数据行（包含所有字段）
    row_all = {
        'id': record_id,
        'timestamp': timestamp,
        'device_id': device_id,
        'update_type': update_type,
        'temperature': f"{temp:.2f}",
        'humidity': f"{hum:.2f}",
        'a1_raw': f"{a1_raw:.3f}",
        'vcap_raw': f"{vcap_raw:.3f}",
        'spectral_gain': spectral_gain_multiplier,  # 以倍率写入（例如16或0.5）
        'sleep_time': sleep_time,  # 休眠时间（秒）
    }
    row_all.update({k: f"{v:.2f}" for k, v in spectrum_data.items()})
    
    # 摘要数据行（仅关键字段）
    row_summary = {
        'id': record_id,
        'timestamp': timestamp,
        'device_id': device_id,
        'update_type': update_type,
        'temperature': f"{temp:.2f}",
        'humidity': f"{hum:.2f}",
        'a1_raw': f"{a1_raw:.3f}",
        'vcap_raw': f"{vcap_raw:.3f}",
        'spectral_gain': spectral_gain_multiplier,
        'sleep_time': sleep_time,
    }
    
    # 写入全量文件（所有数据）
    csv_writer_all.writerow(row_all)
    csv_file_all.flush()
    
    # 智能摘要记录：首次发现设备或重要参数变化时写入摘要文件
    should_write_summary = False
    
    # 检查是否是首次发现设备
    if device_id not in device_last_state:
        should_write_summary = True
        update_type = "FIRST_SEEN"
        row_summary['update_type'] = update_type
        logging.info(f"[摘要] 首次发现设备: {device_id}")
    else:
        # 检查关键参数是否发生变化
        last_state = device_last_state[device_id]
        
        # 检查温度变化（超过1度）
        if abs(temp - last_state.get('temperature', temp)) > 1.0:
            should_write_summary = True
            update_type = "TEMP_CHANGE"
            row_summary['update_type'] = update_type
            
        # 检查增益变化
        elif spectral_gain != last_state.get('spectral_gain', spectral_gain):
            should_write_summary = True
            update_type = "GAIN_CHANGE"
            row_summary['update_type'] = update_type
            
        # 检查休眠时间变化
        elif sleep_time != last_state.get('sleep_time', sleep_time):
            should_write_summary = True
            update_type = "SLEEP_CHANGE"
            row_summary['update_type'] = update_type
            
        # 检查电压大幅变化（超过0.1V）
        elif abs(vcap_raw - last_state.get('vcap_raw', vcap_raw)) > 0.1:
            should_write_summary = True
            update_type = "VOLTAGE_CHANGE"
            row_summary['update_type'] = update_type
    
    # 写入摘要文件（仅重要变化）
    if should_write_summary:
        csv_writer_summary.writerow(row_summary)
        csv_file_summary.flush()
        logging.info(f"[摘要] 记录重要变化: {device_id} - {update_type}")
    
    # 更新设备状态
    device_last_state[device_id] = {
        'temperature': temp,
        'humidity': hum,
        'a1_raw': a1_raw,
        'vcap_raw': vcap_raw,
        'spectral_gain': spectral_gain,
        'sleep_time': sleep_time,
    }
    
    # [MQTT上传] 发布传感器数据到MQTT
    # 提取设备ID中的字母数字字符，跳过特殊字符如 _, ==
    clean_chars = ''.join(c for c in device_id if c.isalnum())  # 只保留字母数字
    device_short = clean_chars[:4] if len(clean_chars) >= 4 else clean_chars  # 取前4位
    device_name = f"Chamber_{device_short}"
    device_clean = device_name.lower()
    
    # [HA集成] 首次发现设备时发布自动发现配置
    if device_clean not in discovered_devices:
        logging.info(f"首次发现设备 {device_id}，发布HA自动发现配置...")
        
        # 发布所有传感器的HA配置 - 包含新增的增益和休眠时间字段
        sensor_configs = [
            ("temp", "°C", "mdi:thermometer"),
            ("hum", "%", "mdi:water-percent"),
            ("a1_raw", "V", "mdi:current-ac"),
            ("vcap_raw", "V", "mdi:flash"),
            ("sp_415", "count", "mdi:gradient-vertical"),
            ("sp_445", "count", "mdi:gradient-vertical"),
            ("sp_480", "count", "mdi:gradient-vertical"),
            ("sp_515", "count", "mdi:gradient-vertical"),
            ("sp_555", "count", "mdi:gradient-vertical"),
            ("sp_590", "count", "mdi:gradient-vertical"),
            ("sp_630", "count", "mdi:gradient-vertical"),
            ("sp_680", "count", "mdi:gradient-vertical"),
            ("sp_clear", "count", "mdi:brightness-7"),  # Clear通道
            ("sp_nir", "count", "mdi:heat-wave"),      # NIR通道
            ("spectral_gain", "", "mdi:tune-vertical-variant"),  # 光谱增益（0-10）
            ("sleep_time", "s", "mdi:sleep"),          # 休眠时间（秒）
        ]
        
        for sensor_type, unit, icon in sensor_configs:
            publish_ha_discovery(device_clean, sensor_type, unit, icon)
        
        discovered_devices.add(device_clean)
        logging.info(f"设备 {device_clean} HA配置完成")
    
    # 基础传感器数据上传
    publish_mqtt(f"riotee/{device_clean}/temp", f"{temp:.2f}")
    publish_mqtt(f"riotee/{device_clean}/hum", f"{hum:.2f}")
    publish_mqtt(f"riotee/{device_clean}/a1_raw", f"{a1_raw:.3f}")
    publish_mqtt(f"riotee/{device_clean}/vcap_raw", f"{vcap_raw:.3f}")
    
    # 光谱数据上传
    for wl, value in spectrum_data.items():
        publish_mqtt(f"riotee/{device_clean}/{wl}", f"{value:.2f}")
    
    # 新增配置数据上传
    publish_mqtt(f"riotee/{device_clean}/spectral_gain", str(spectral_gain_multiplier))
    publish_mqtt(f"riotee/{device_clean}/sleep_time", str(sleep_time))
    
    # 日志输出 - 包含新字段信息
    mqtt_status = "已发布" if mqtt_client else "跳过(MQTT离线)"
    gain_str = gain_value_to_string(spectral_gain)
    logging.info(f"设备 {device_id} (MQTT名称: {device_clean}): T={temp:.1f}°C, H={hum:.1f}%, A1={a1_raw:.2f}V, VCAP={vcap_raw:.2f}V, 增益={gain_str}, 休眠={sleep_time}s [{mqtt_status}]")
    
    return True

# ============================
# 主程序控制
# ============================
def main():
    """
    [主程序] 程序入口和主循环控制
    功能: 协调所有模块，实现完整数据流程
    流程: Riotee接收 -> 数据处理 -> CSV记录 + MQTT上传
    """
    # ============================
    # 程序启动分隔线
    # ============================
    print("=" * 80)
    print("🚀 Riotee数据接收网关启动")
    print("=" * 80)
    
    global mqtt_client, gateway_client
    gateway_process = None  # 初始化Gateway进程变量
    
    logging.info("启动Riotee Gateway简化版")
    logging.info(f"配置信息:")
    logging.info(f"  - MQTT Broker: {CONFIG['mqtt_broker']}:{CONFIG['mqtt_port']}")
    logging.info(f"  - Gateway: {CONFIG['gateway_host']}:{CONFIG['gateway_port']}")
    logging.info(f"  - 数据间隔: {CONFIG['data_interval']}秒")
    logging.info(f"  - 调试模式: {'开启' if args.debug else '关闭'}")
    logging.info(f"  - 自动清理进程: {'开启' if not args.no_kill else '关闭'}")
    logging.info(f"  - MQTT用户名: {CONFIG['mqtt_username']}")
    logging.info(f"  - 重连间隔: {CONFIG['reconnect_interval']}秒")
    logging.info(f"  - 文件模式: {'新文件模式' if args.new_file else '追加模式（默认）'}")
    logging.info(f"  - 防覆写模式: {'强制覆写' if args.force_overwrite else '自动序号'}")
    if args.csv_name:
        logging.info(f"  - 自定义文件名: {args.csv_name}")
        logging.info(f"  - 时间戳命名: {'关闭' if (args.no_timestamp or not args.new_file) else '开启'}")
    if not args.new_file:
        logging.info(f"  - 追加说明: 如果CSV文件存在将追加新会话，否则创建新文件")
    
    # [初始化] MQTT上传模块 - 允许失败但继续运行
    mqtt_client = setup_mqtt()
    if not mqtt_client:
        logging.warning("MQTT连接失败，将跳过MQTT上传功能，继续运行...")
    else:
        logging.info("MQTT连接成功")
    
    # [初始化] 先启动Gateway服务器
    logging.info("启动Riotee Gateway服务器...")
    if not args.no_kill:
        logging.info("注意：程序将自动清理现有Gateway进程，确保端口8000可用")
    else:
        logging.info("注意：用户选择不自动清理进程，请确保端口8000未被占用")
    gateway_process = start_gateway_server()
    
    # [初始化] Riotee接收模块 - 必须成功才能继续
    max_retries = 5
    retry_count = 0
    gateway_client = None
    
    while retry_count < max_retries and not gateway_client:
        logging.info(f"尝试连接Gateway... (第{retry_count + 1}次)")
        gateway_client = setup_gateway()
        if not gateway_client:
            retry_count += 1
            if retry_count < max_retries:
                logging.warning(f"Gateway连接失败，{CONFIG['data_interval']}秒后重试...")
                time.sleep(CONFIG["data_interval"])
            else:
                logging.error("Gateway连接失败次数过多，程序退出")
                if gateway_process:
                    gateway_process.terminate()
                return
    
    logging.info("Gateway连接成功")
    
    # [初始化] CSV记录模块 - 双文件系统
    csv_path_all, csv_path_summary = setup_csv()
    logging.info(f"全量数据文件: {csv_path_all}")
    logging.info(f"摘要数据文件: {csv_path_summary}")
    
    packet_count = 0
    no_data_count = 0  # 无数据计数器
    stats_print_count = 0  # 统计信息输出计数器
    
    # [主循环] 持续数据处理
    print("-" * 60)
    logging.info("开始监听Riotee设备数据...")
    print("-" * 60)
    try:
        while True:
            try:
                # [Riotee接收] 获取设备列表（先转换为列表再去重）
                try:
                    devices_list = list(gateway_client.get_devices())
                    devices = set(devices_list)
                    
                    if not devices:
                        no_data_count += 1
                        if no_data_count % 10 == 0:  # 每10次无数据时提示
                            logging.info(f"等待设备连接... (已等待 {no_data_count} 次)")
                        elif args.debug:
                            logging.debug(f"无设备连接，等待中... (计数: {no_data_count})")
                        time.sleep(CONFIG["data_interval"])
                        continue
                    else:
                        # 有设备时重置无数据计数器
                        if no_data_count > 0:
                            print("-" * 40)
                            logging.info(f"检测到 {len(devices)} 个设备: {list(devices)}")
                            print("-" * 40)
                            no_data_count = 0
                        elif args.debug:
                            # 调试模式下显示设备信息
                            logging.debug(f"当前连接设备: {list(devices)}")
                except Exception as e:
                    logging.error(f"获取设备列表失败: {e}")
                    time.sleep(CONFIG["data_interval"])
                    continue
                
                # [Riotee接收] 处理每个设备的数据包（参考V4版本逻辑）
                for dev_id in devices:
                    try:
                        packets_gen = gateway_client.pops(dev_id)
                        packets = list(packets_gen)  # 转换为列表
                        
                        # 如果没有数据包，直接跳过（参考V4版本）
                        if not packets:
                            if args.debug:
                                logging.debug(f"设备 {dev_id} 暂无数据包")
                            continue
                        
                        # 调试信息：显示收到的数据包数量
                        if args.debug:
                            logging.debug(f"设备 {dev_id} 收到 {len(packets)} 个数据包")
                        else:
                            logging.info(f"设备 {dev_id} 收到 {len(packets)} 个数据包")
                        
                        # 添加数据包分隔线
                        print("┌" + "─" * 48 + "┐")
                        print(f"│ 设备 {dev_id} 收到 {len(packets)} 个数据包")
                        print("└" + "─" * 48 + "┘")
                    except Exception as e:
                        logging.error(f"获取设备 {dev_id} 数据包失败: {e}")
                        continue
                    
                    for pkt in packets:
                        try:
                            # 统计信息：接收到的数据包数量（参考V4版本）
                            stats["packets_received"] += 1
                            
                            # [数据处理] 解码并处理数据
                            data_arr = base64_to_numpy(pkt.data, np.float32)
                            process_device_data(dev_id, data_arr, "PACKET_DATA")
                            
                            packet_count += 1
                            
                            # 每个数据包后添加分隔线
                            print("─" * 50)
                            
                            # 每50个包输出一次统计（参考V4版本）
                            if stats["packets_received"] % 50 == 0:
                                logging.info(f"已处理 {stats['packets_received']} 个数据包")
                                print_stats()
                                
                        except Exception as e:
                            # 统计信息：数据处理错误（参考V4版本）
                            stats["data_processing_errors"] += 1
                            logging.error(f"处理数据包失败: {e}")
                            continue
                
                # 数据处理间隔
                time.sleep(CONFIG["data_interval"])
                
            except Exception as e:
                # 统计信息：连接错误（参考V4版本）
                stats["connection_errors"] += 1
                logging.error(f"主循环异常: {e}")
                time.sleep(CONFIG.get("reconnect_interval", 5))
                
                # 尝试重新连接（参考V4版本的重连逻辑）
                if not mqtt_client or not hasattr(mqtt_client, 'is_connected') or not mqtt_client.is_connected():
                    logging.info("尝试重新连接MQTT...")
                    mqtt_client = setup_mqtt()
                
                if not gateway_client:
                    logging.info("尝试重新连接Gateway...")
                    gateway_client = setup_gateway()
                
                continue
            
    except KeyboardInterrupt:
        logging.info("接收到中断信号，正在退出...")
        print_stats()  # 退出时显示最终统计
    except Exception as e:
        logging.error(f"程序异常: {e}")
        stats["connection_errors"] += 1
    finally:
        # [清理资源] 写入会话结束标记并关闭所有连接
        stop_line = stop_csv_session()
        
        # 写入会话结束标记到两个CSV文件
        if csv_file_all:
            csv_file_all.write(stop_line)
            csv_file_all.write("# ----\n")  # 实验分隔线
            csv_file_all.close()
            logging.info("全量数据文件已关闭")
            
        if csv_file_summary:
            csv_file_summary.write(stop_line)
            csv_file_summary.write("# ----\n")  # 实验分隔线
            csv_file_summary.close()
            logging.info("摘要数据文件已关闭")
            
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            logging.info("MQTT连接已关闭")
            
        # 清理Gateway进程
        if gateway_process:
            gateway_process.terminate()
            logging.info("Gateway服务器已关闭")
            
        print_stats()  # 最终统计信息
        
        # ============================
        # 程序结束分隔线
        # ============================
        print("=" * 80)
        print("🛑 Riotee数据接收网关已关闭")
        print(f"📊 最终统计: 总包数={stats.get('packets_received', 0)}, 有效包数={stats.get('valid_packets', 0)}")
        print("=" * 80)
        
        logging.info("程序已退出")

if __name__ == "__main__":
    main() 