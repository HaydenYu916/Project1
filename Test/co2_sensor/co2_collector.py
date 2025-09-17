#!/usr/bin/env python3
"""
CO2传感器数据采集器
从串口读取真实CO2传感器数据，只使用 Chamber2_CO2 设备
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

class CO2Collector:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.logs_path = self.base_dir / "logs"  # 使用本地logs目录
        self.csv_file = self.logs_path / "co2_data.csv"
        self.running = True
        
        # 串口配置 - 只使用 Chamber2_CO2 设备
        self.serial_config = {
            "port": "/dev/Chamber2_CO2",  # 固定使用 Chamber2_CO2 设备
            "baudrate": 115200,  # 使用正确的波特率
            "timeout": 5,
            "bytesize": serial.EIGHTBITS,
            "parity": serial.PARITY_NONE,
            "stopbits": serial.STOPBITS_ONE
        }
        self.serial_connection = None
        self.serial_enabled = True  # 启用串口读取
        
        # MQTT配置 - 使用与riotee传感器相同的服务器
        self.mqtt_config = {
            "mqtt_broker": "azure.nocolor.pw",
            "mqtt_port": 1883,
            "mqtt_username": "feiyue",
            "mqtt_password": "123456789"
        }
        self.mqtt_client = None
        self.mqtt_enabled = True  # 启用MQTT
        
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
        
        # 设置MQTT连接（如果启用）
        if self.mqtt_enabled:
            self.setup_mqtt()
    
    def signal_handler(self, signum, frame):
        """处理系统信号"""
        print(f"\n收到信号 {signum}，正在停止采集器...")
        self.running = False
    
    def setup_serial(self):
        """设置串口连接 - 只使用 Chamber2_CO2 设备"""
        try:
            # 检查 Chamber2_CO2 设备是否存在
            if not os.path.exists(self.serial_config["port"]):
                print(f"❌ Chamber2_CO2 设备不存在: {self.serial_config['port']}")
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
            client = mqtt.Client(client_id=f"co2_{int(time.time())}")
            client.username_pw_set(self.mqtt_config["mqtt_username"], self.mqtt_config["mqtt_password"])
            client.on_connect = on_connect
            
            client.connect(self.mqtt_config["mqtt_broker"], self.mqtt_config["mqtt_port"], 60)
            client.loop_start()
            
            self.mqtt_client = client
            print("🔗 MQTT客户端初始化完成")
            
        except Exception as e:
            print(f"❌ MQTT设置失败: {e}")
            self.mqtt_client = None
    
    def publish_ha_discovery(self):
        """发布Home Assistant自动发现配置"""
        if not self.mqtt_client:
            return
        
        # CO2传感器配置
        co2_config = {
            "name": "CO2",
            "unique_id": "chamber2_room_co2",
            "device_class": "carbon_dioxide",
            "unit_of_measurement": "ppm",
            "state_topic": "co2/chamber2/co2",
            "device": {
                "identifiers": ["chamber2_room_sensor"],
                "name": "Chamber2 Room",
                "model": "Multi Sensor Hub",
                "manufacturer": "Custom"
            }
        }
        
        # 发布配置
        config_topic = "homeassistant/sensor/chamber2_room_co2/config"
        try:
            import json
            result = self.mqtt_client.publish(config_topic, json.dumps(co2_config), qos=0, retain=True)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print("📡 已发布CO2传感器HA自动发现配置")
            else:
                print("⚠️ CO2传感器HA配置发布失败")
        except Exception as e:
            print(f"❌ CO2传感器HA配置发布异常: {e}")
    
    def init_csv_file(self):
        """初始化CSV文件"""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'co2'])
            print(f"✅ 创建CSV文件: {self.csv_file}")
    
    def read_co2_data(self):
        """从串口读取CO2数据，如果有数据就返回，没有数据返回None"""
        if self.serial_enabled and self.serial_connection and self.serial_connection.is_open:
            try:
                # 检查是否有新数据
                if self.serial_connection.in_waiting > 0:
                    # 读取一行数据
                    line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                    print(f"📡 串口原始数据: {line}")
                    
                    # 解析CO2数据 - 支持多种格式
                    co2_value = self.parse_co2_data(line)
                    if co2_value is not None:
                        return co2_value
                    else:
                        print(f"⚠️ 无法解析串口数据: {line}")
                
                # 如果没有新数据，尝试发送查询命令
                self.serial_connection.write(b'READ\n')
                time.sleep(0.1)  # 等待响应
                
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                    print(f"📡 查询响应: {line}")
                    co2_value = self.parse_co2_data(line)
                    if co2_value is not None:
                        return co2_value
                
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
    
    def parse_co2_data(self, data_line):
        """解析串口数据中的CO2值"""
        try:
            # 移除空白字符
            data_line = data_line.strip()
            
            # 尝试多种解析格式
            # 格式1: "CO2: 450 ppm"
            if 'CO2:' in data_line and 'ppm' in data_line:
                parts = data_line.split('CO2:')
                if len(parts) > 1:
                    co2_part = parts[1].split('ppm')[0].strip()
                    return int(float(co2_part))
            
            # 格式2: "450" (纯数字)
            if data_line.isdigit():
                return int(data_line)
            
            # 格式3: "450.5" (带小数点的数字)
            try:
                return int(float(data_line))
            except ValueError:
                pass
            
            # 格式4: JSON格式 {"co2": 450}
            if '{' in data_line and '}' in data_line:
                import json
                data = json.loads(data_line)
                if 'co2' in data:
                    return int(data['co2'])
            
            # 格式5: CSV格式 "timestamp,co2" 或 "temp,co2,other"
            if ',' in data_line:
                parts = data_line.split(',')
                if len(parts) >= 2:
                    try:
                        # 尝试解析第二个字段作为CO2值
                        return int(float(parts[1].strip()))
                    except ValueError:
                        pass
            
            return None
            
        except Exception as e:
            print(f"❌ 数据解析错误: {e}")
            return None
    
    def generate_simulated_data(self):
        """生成模拟CO2数据（备用方案）"""
        # 模拟CO2传感器读数 (400-2000 ppm)
        # 在正常范围内随机波动
        base_co2 = 500
        variation = random.randint(-50, 100)
        co2_value = max(400, min(2000, base_co2 + variation))
        return co2_value
    
    def save_data(self, co2_value):
        """保存数据到CSV文件并发布到MQTT"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            # 保存到CSV文件
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, co2_value])
            
            # 发布到MQTT（如果启用）
            if self.mqtt_enabled and self.mqtt_client:
                mqtt_topic = "co2/chamber2/co2"
                result = self.mqtt_client.publish(mqtt_topic, str(co2_value), qos=0, retain=False)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    print(f"📊 {timestamp} | CO2: {co2_value} ppm | MQTT: {mqtt_topic}")
                else:
                    print(f"📊 {timestamp} | CO2: {co2_value} ppm | MQTT发布失败")
            else:
                print(f"📊 {timestamp} | CO2: {co2_value} ppm")
            
            return True
        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
            return False
    
    def run(self):
        """运行采集器主循环"""
        print("🌬️ CO2传感器数据采集器启动")
        print(f"📄 数据文件: {self.csv_file}")
        
        # 显示串口状态
        if self.serial_enabled and self.serial_connection and self.serial_connection.is_open:
            print(f"🔌 串口: {self.serial_config['port']} @ {self.serial_config['baudrate']} bps (已连接)")
        elif self.serial_enabled:
            print(f"🔌 串口: {self.serial_config['port']} (连接失败，使用模拟数据)")
        else:
            print("🔌 串口: 已禁用，使用模拟数据")
        
        # 显示MQTT状态
        if self.mqtt_enabled:
            print(f"🔗 MQTT Broker: {self.mqtt_config['mqtt_broker']}:{self.mqtt_config['mqtt_port']}")
        else:
            print("🔗 MQTT: 已禁用")
        
        print("📊 开始采集CO2数据...")
        print("=" * 50)
        
        try:
            while self.running:
                # 读取CO2数据
                co2_value = self.read_co2_data()
                
                # 只有在成功读取到数据时才保存和上传
                if co2_value is not None:
                    if self.save_data(co2_value):
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
            print("👋 CO2数据采集器已停止")

def main():
    collector = CO2Collector()
    collector.run()

if __name__ == '__main__':
    main()