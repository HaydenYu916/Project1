#!/usr/bin/env python3
"""
房间温湿度传感器数据采集器
从串口读取真实温湿度传感器数据，只使用 Chamber2_TempHumidity 设备
如果串口不可用，自动切换到模拟数据模式
"""

import time
import csv
import os
import sys
import signal
import random
import logging
import paho.mqtt.client as mqtt
import serial
from datetime import datetime
from pathlib import Path

# 设置输出缓冲，确保print语句立即输出
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

class RoomTempHumidityCollector:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.logs_path = self.base_dir / "logs"  # 使用本地logs目录
        self.csv_file = self.logs_path / "temp_humidity_data.csv"
        self.running = True
        
        # 串口配置 - 只使用 Chamber2_TempHumidity 设备
        self.serial_config = {
            "port": "/dev/Chamber2_TempHumidity",  # 固定使用 Chamber2_TempHumidity 设备
            "baudrate": 9600,
            "timeout": 5,
            "bytesize": serial.EIGHTBITS,
            "parity": serial.PARITY_NONE,
            "stopbits": serial.STOPBITS_ONE
        }
        self.serial_connection = None
        self.serial_enabled = True  # 启用串口读取
        
        # MQTT配置
        self.mqtt_config = {
            "mqtt_broker": "azure.nocolor.pw",
            "mqtt_port": 1883,
            "mqtt_username": "feiyue", 
            "mqtt_password": "123456789",
        }
        self.mqtt_client = None
        
        # 确保logs目录存在
        self.logs_path.mkdir(exist_ok=True)
        
        # 注册信号处理器
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # 初始化CSV文件
        self.init_csv_file()
        
        # 设置串口连接（如果启用）
        if self.serial_enabled:
            self.setup_serial()
        
        # 初始化MQTT连接
        self.setup_mqtt()
    
    def signal_handler(self, signum, frame):
        """处理系统信号"""
        print(f"\n收到信号 {signum}，正在停止采集器...")
        self.running = False
    
    def setup_serial(self):
        """设置串口连接 - 只使用 Chamber2_TempHumidity 设备"""
        try:
            # 检查 Chamber2_TempHumidity 设备是否存在
            if not os.path.exists(self.serial_config["port"]):
                print(f"❌ Chamber2_TempHumidity 设备不存在: {self.serial_config['port']}")
                print("🔄 将使用模拟数据模式")
                self.serial_enabled = False
                return
            
            # 创建串口连接
            self.serial_connection = serial.Serial(**self.serial_config)
            print(f"🔗 串口连接成功: {self.serial_config['port']} @ {self.serial_config['baudrate']} bps")
            
        except serial.SerialException as e:
            print(f"❌ 串口连接失败: {e}")
            print("🔄 将使用模拟数据模式")
            self.serial_enabled = False
            self.serial_connection = None
        except Exception as e:
            print(f"❌ 串口设置异常: {e}")
            print("🔄 将使用模拟数据模式")
            self.serial_enabled = False
            self.serial_connection = None
    
    def setup_mqtt(self):
        """设置MQTT连接"""
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("✅ MQTT连接成功")
                # 发布Home Assistant自动发现配置
                self.publish_ha_discovery()
            else:
                print(f"❌ MQTT连接失败: {rc}")
        
        try:
            client = mqtt.Client(client_id=f"temp_humidity_{int(time.time())}")
            
            client.username_pw_set(self.mqtt_config["mqtt_username"], self.mqtt_config["mqtt_password"])
            client.on_connect = on_connect
            
            client.connect(self.mqtt_config["mqtt_broker"], self.mqtt_config["mqtt_port"], 60)
            client.loop_start()
            self.mqtt_client = client
            print("🔗 MQTT连接已建立")
            
        except Exception as e:
            print(f"❌ MQTT连接失败: {e}")
            self.mqtt_client = None
    
    def publish_ha_discovery(self):
        """发布Home Assistant自动发现配置"""
        if not self.mqtt_client:
            return
        
        # 温湿度传感器配置
        sensors = [
            {
                "name": "Temperature",
                "unique_id": "chamber2_room_temperature",
                "state_topic": "temp_humidity/chamber2/temperature",
                "unit": "°C",
                "icon": "mdi:thermometer",
                "device_class": "temperature"
            },
            {
                "name": "Humidity", 
                "unique_id": "chamber2_room_humidity",
                "state_topic": "temp_humidity/chamber2/humidity",
                "unit": "%",
                "icon": "mdi:water-percent",
                "device_class": "humidity"
            }
        ]
        
        device_info = {
            "identifiers": ["chamber2_room_sensor"],
            "name": "Chamber2 Room",
            "manufacturer": "Custom",
            "model": "Multi Sensor Hub"
        }
        
        for sensor in sensors:
            config = {
                "name": sensor["name"],
                "unique_id": sensor["unique_id"],
                "state_topic": sensor["state_topic"],
                "unit_of_measurement": sensor["unit"],
                "icon": sensor["icon"],
                "device": device_info
            }
            
            if "device_class" in sensor:
                config["device_class"] = sensor["device_class"]
            
            config_topic = f"homeassistant/sensor/{sensor['unique_id']}/config"
            
            try:
                import json
                result = self.mqtt_client.publish(config_topic, json.dumps(config), qos=0, retain=True)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    print(f"📡 HA自动发现配置已发布: {sensor['name']}")
                else:
                    print(f"⚠️ HA配置发布失败: {sensor['name']}")
            except Exception as e:
                print(f"❌ HA配置发布异常: {e}")
    
    def publish_mqtt(self, topic, value):
        """发布MQTT消息"""
        if self.mqtt_client:
            try:
                result = self.mqtt_client.publish(topic, str(value), qos=0, retain=False)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    return True
                else:
                    print(f"⚠️ MQTT发布失败: {topic} = {value}")
                    return False
            except Exception as e:
                print(f"❌ MQTT发布失败: {e}")
                return False
        else:
            print(f"⚠️ MQTT不可用，跳过发布: {topic} = {value}")
            return False
    
    def init_csv_file(self):
        """初始化CSV文件"""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'temperature', 'humidity'])
            print(f"✅ 创建CSV文件: {self.csv_file}")
    
    def read_sensor_data(self):
        """从串口读取温湿度数据，如果串口不可用则使用模拟数据"""
        if self.serial_enabled and self.serial_connection and self.serial_connection.is_open:
            try:
                # 检查是否有新数据
                if self.serial_connection.in_waiting > 0:
                    # 读取一行数据
                    line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                    print(f"📡 串口原始数据: {line}")
                    
                    # 解析温湿度数据
                    temp_humidity = self.parse_temp_humidity_data(line)
                    if temp_humidity is not None:
                        return temp_humidity
                    else:
                        print(f"⚠️ 无法解析串口数据: {line}")
                
                # 如果没有新数据，尝试发送查询命令
                self.serial_connection.write(b'READ\n')
                time.sleep(0.1)  # 等待响应
                
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                    print(f"📡 查询响应: {line}")
                    temp_humidity = self.parse_temp_humidity_data(line)
                    if temp_humidity is not None:
                        return temp_humidity
                
                # 没有新数据，返回None（不停止采集器）
                return None
                
            except serial.SerialException as e:
                print(f"❌ 串口读取错误: {e}")
                print("🔄 切换到模拟数据模式")
                self.serial_enabled = False
                return self.generate_simulated_data()
            except Exception as e:
                print(f"❌ 数据读取异常: {e}")
                return self.generate_simulated_data()
        else:
            # 串口不可用，使用模拟数据
            return self.generate_simulated_data()
    
    def parse_temp_humidity_data(self, data_line):
        """解析串口数据中的温湿度值"""
        try:
            # 移除空白字符
            data_line = data_line.strip()
            
            # 解析CSV格式: "温度,湿度,其他"
            if ',' in data_line:
                parts = data_line.split(',')
                if len(parts) >= 2:
                    try:
                        temperature = float(parts[0].strip())
                        humidity = float(parts[1].strip())
                        return temperature, humidity
                    except ValueError:
                        pass
            
            return None
            
        except Exception as e:
            print(f"❌ 数据解析错误: {e}")
            return None
    
    def generate_simulated_data(self):
        """生成模拟温湿度数据（备用方案）"""
        # 模拟温度传感器读数 (18-28°C)
        base_temp = 22.0
        temp_variation = random.uniform(-2.0, 3.0)
        temperature = round(max(18.0, min(28.0, base_temp + temp_variation)), 1)
        
        # 模拟湿度传感器读数 (30-80%)
        base_humidity = 55.0
        humidity_variation = random.uniform(-10.0, 15.0)
        humidity = round(max(30.0, min(80.0, base_humidity + humidity_variation)), 1)
        
        return temperature, humidity
    
    def save_data(self, temperature, humidity):
        """保存数据到CSV文件并发布到MQTT"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            # 保存到CSV文件
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, temperature, humidity])
            
            # 发布到MQTT
            mqtt_status = ""
            if self.publish_mqtt("temp_humidity/chamber2/temperature", f"{temperature:.1f}"):
                mqtt_status += "🌡️"
            if self.publish_mqtt("temp_humidity/chamber2/humidity", f"{humidity:.1f}"):
                mqtt_status += "💧"
            
            if mqtt_status:
                print(f"📊 {timestamp} | 温度: {temperature}°C | 湿度: {humidity}% {mqtt_status}")
            else:
                print(f"📊 {timestamp} | 温度: {temperature}°C | 湿度: {humidity}% (MQTT离线)")
            
            return True
        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
            return False
    
    def run(self):
        """运行采集器主循环"""
        print("🌡️ 房间温湿度传感器数据采集器启动")
        print(f"📄 数据文件: {self.csv_file}")
        
        # 显示串口状态
        if self.serial_enabled and self.serial_connection and self.serial_connection.is_open:
            print(f"🔌 串口: {self.serial_config['port']} @ {self.serial_config['baudrate']} bps (已连接)")
        elif self.serial_enabled:
            print(f"🔌 串口: {self.serial_config['port']} (连接失败，使用模拟数据)")
        else:
            print("🔌 串口: 已禁用，使用模拟数据")
        
        print(f"🔗 MQTT Broker: {self.mqtt_config['mqtt_broker']}:{self.mqtt_config['mqtt_port']}")
        print("📊 开始采集温湿度数据...")
        print("=" * 50)
        
        try:
            while self.running:
                # 读取温湿度数据
                sensor_data = self.read_sensor_data()
                
                # 只有在成功读取到数据时才保存和上传
                if sensor_data is not None:
                    temperature, humidity = sensor_data
                    if self.save_data(temperature, humidity):
                        pass  # 数据保存成功
                    else:
                        print("⚠️ 数据保存失败")
                # 如果没有数据，不输出任何信息，继续扫描
                
                # 等待下次扫描
                time.sleep(1)  # 每1秒扫描一次
                
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断采集")
        except Exception as e:
            print(f"❌ 采集器异常: {e}")
        finally:
            # 清理串口连接
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
                print("🔌 串口连接已关闭")
            
            # 清理MQTT连接
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                print("🔗 MQTT连接已关闭")
            print("👋 房间温湿度数据采集器已停止")

def main():
    collector = RoomTempHumidityCollector()
    collector.run()

if __name__ == '__main__':
    main()