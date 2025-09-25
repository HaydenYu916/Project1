#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPPI控制循环集成脚本
每分钟运行一次，读取温度数据，运行MPPI控制，发送PWM命令到设备
"""

import sys
import os
import time
import json
import csv
from datetime import datetime
import numpy as np
import pandas as pd

# ==================== 配置宏定义 ====================
# 温度传感器设备ID配置（修改此处即可切换设备）
TEMPERATURE_DEVICE_ID = None  # None=自动选择, "T6ncwg=="=指定设备1, "L_6vSQ=="=指定设备2

# 控制循环间隔（分钟）
CONTROL_INTERVAL_MINUTES = 1


# 红蓝比例键
RB_RATIO_KEY = "5:1"

# =====================================================

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 日志文件路径
LOG_FILE = os.path.join(current_dir, "..", "..", "logs", "control_simulate_log.csv")
project_root = os.path.join(current_dir, '..', '..')
src_dir = os.path.join(project_root, 'src')
riotee_sensor_dir = os.path.join(project_root, '..', 'Sensor', 'riotee_sensor')
controller_dir = os.path.join(project_root, '..', 'Shelly', 'src')

# CO2数据文件路径
CO2_FILE = "/data/csv/co2_sensor.csv"

# 确保项目目录在路径中
sys.path.insert(0, src_dir)
sys.path.insert(0, riotee_sensor_dir)
sys.path.insert(0, controller_dir)

try:
    # 导入配置
    config_dir = os.path.join(current_dir, '..', '..', 'config')
    sys.path.insert(0, config_dir)
    from app_config import DEFAULT_MODEL_NAME
    
    # 导入riotee函数
    riotee_init_path = os.path.join(riotee_sensor_dir, '__init__.py')
    if os.path.exists(riotee_init_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("riotee_init", riotee_init_path)
        riotee_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(riotee_module)
        get_current_riotee = riotee_module.get_current_riotee
        get_riotee_devices = riotee_module.get_riotee_devices
        get_device_avg_a1_raw = getattr(riotee_module, 'get_device_avg_a1_raw', None)
    else:
        raise ImportError(f"riotee __init__.py not found at {riotee_init_path}")
    
    from mppi import LEDPlant, LEDMPPIController
    from shelly_controller import rpc, DEVICES
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

class MPPIControlLoop:
    def __init__(self):
        """初始化MPPI控制循环"""
        print("🚀 初始化MPPI控制循环...")
        
        # 使用宏定义配置
        self.temperature_device_id = TEMPERATURE_DEVICE_ID
        self.log_file = LOG_FILE
        
        # 初始化上一次的控制结果
        self.last_r_pwm = 0.0
        self.last_b_pwm = 0.0
        self.last_cost = None
        
        # 初始化LED植物模型
        self.plant = LEDPlant(
            model_key=RB_RATIO_KEY,  # 使用宏定义的红蓝比例
            use_efficiency=False,  # 暂时关闭效率模型
            heat_scale=1.0,
            model_name=DEFAULT_MODEL_NAME  # 使用配置文件中的模型名称
        )
        
        # 初始化MPPI控制器
        self.controller = LEDMPPIController(
            plant=self.plant,
            horizon=10,           # 预测时域
            num_samples=1000,     # 采样数量
            dt=0.1,              # 时间步长
            temperature=1.0,      # 温度参数
            maintain_rb_ratio=True,  # 维持红蓝比例
            rb_ratio_key="5:1"    # 红蓝比例键
        )
        
        # 使用MPPI控制器的默认参数，不进行覆盖设置
        
        # 设备IP地址
        self.devices = DEVICES
        
        print("✅ MPPI控制循环初始化完成")
        print(f"   温度设备: {self.temperature_device_id or '自动选择'}")
        print(f"   LED设备列表: {list(self.devices.keys())}")
        print(f"   红蓝比例: {RB_RATIO_KEY}")
        print(f"   控制间隔: {CONTROL_INTERVAL_MINUTES}分钟")
        print(f"   使用模型: {DEFAULT_MODEL_NAME}")
        
        # 初始化日志文件
        self.init_log_file()
    
    def init_log_file(self):
        """初始化日志文件"""
        try:
            # 确保日志目录存在
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['时间戳', '输入温度', 'CO2值', '红光PWM', '蓝光PWM', '成功状态', '成本', '备注'])
        except Exception as e:
            print(f"⚠️  日志文件初始化失败: {e}")
    
    def log_control_cycle(self, timestamp, input_temp, co2_value, output_r_pwm, output_b_pwm, success, cost=None, note=""):
        """记录控制循环日志"""
        try:
            cost_str = f"{cost:.2f}" if cost is not None else "N/A"
            co2_str = f"{co2_value:.1f}" if co2_value is not None else "N/A"
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, f"{input_temp:.2f}", co2_str, f"{output_r_pwm:.2f}", f"{output_b_pwm:.2f}", success, cost_str, note])
        except Exception as e:
            print(f"⚠️  日志记录失败: {e}")
    
    def read_temperature(self):
        """读取当前温度数据"""
        try:
            # 使用指定的设备ID或自动选择
            data = get_current_riotee(
                device_id=self.temperature_device_id, 
                max_age_seconds=86400
            )
            
            if data and data.get('temperature') is not None:
                temp = data['temperature']
                device_id = data.get('device_id', 'Unknown')
                age = data.get('_data_age_seconds', 0)
                
                # 数据新鲜度检查
                if age < 120:  # 2分钟内
                    status = "🟢"
                elif age < 300:  # 2-5分钟
                    status = "🟡"
                else:  # 超过5分钟
                    status = "🔴"
                
                # 如果模型为 solar_vol，尝试读取10分钟窗口内的A1_Raw均值
                self.last_a1_avg = None
                if 'solar_vol' in str(DEFAULT_MODEL_NAME).lower() and get_device_avg_a1_raw and device_id and device_id != 'Unknown':
                    try:
                        avg_info = get_device_avg_a1_raw(device_id, window_minutes=10)
                        self.last_a1_avg = avg_info.get('avg')
                        cnt = avg_info.get('count', 0)
                        if self.last_a1_avg is not None and cnt > 0:
                            print(f"🔆 A1_Raw(10min均值): {self.last_a1_avg:.2f} (n={cnt})")
                        else:
                            print("⚠️  A1_Raw近10分钟无有效数据，略过均值计算")
                    except Exception as _:
                        print("⚠️  A1_Raw均值读取失败，略过")

                print(f"🌡️  {status} 温度读取: {temp:.2f}°C (设备: {device_id}, {age:.0f}秒前)")
                return temp, True
            else:
                if self.temperature_device_id:
                    print(f"⚠️  指定设备 {self.temperature_device_id} 无有效温度数据")
                else:
                    print("⚠️  无有效温度数据")
                return None, False
                
        except Exception as e:
            print(f"❌ 温度读取错误: {e}")
            return None, False
    
    def read_co2(self):
        """读取当前CO2数据"""
        try:
            if not os.path.exists(CO2_FILE):
                print("⚠️  CO2文件不存在，使用模拟CO2值 420 ppm")
                return 420.0, True
            
            # 读取CO2数据文件
            df = pd.read_csv(CO2_FILE, header=None, names=['timestamp', 'co2'])
            
            if df.empty:
                print("⚠️  CO2文件为空，使用模拟CO2值 420 ppm")
                return 420.0, True
            
            # 获取最新的有效CO2值
            latest_row = df.iloc[-1]
            latest_timestamp = latest_row['timestamp']
            latest_co2 = latest_row['co2']
            
            # 检查CO2值是否有效
            if pd.isna(latest_co2) or latest_co2 is None:
                print("⚠️  最新CO2值无效，使用模拟CO2值 420 ppm")
                return 420.0, True
            
            # 计算数据年龄（秒）
            current_time = time.time()
            age_seconds = current_time - latest_timestamp
            
            # 数据新鲜度检查
            if age_seconds < 120:  # 2分钟内
                status = "🟢"
            elif age_seconds < 300:  # 2-5分钟
                status = "🟡"
            else:  # 超过5分钟
                status = "🔴"
            
            print(f"🌬️  {status} CO2读取: {latest_co2:.1f} ppm ({age_seconds:.0f}秒前)")
            return latest_co2, True
            
        except Exception as e:
            print(f"❌ CO2读取错误: {e}")
            print("⚠️  使用模拟CO2值 420 ppm")
            return 420.0, True
    
    def run_mppi_control(self, current_temp):
        """运行MPPI控制算法"""
        try:
            print(f"🎯 运行MPPI控制 (当前温度: {current_temp:.2f}°C)")
            
            # 运行MPPI求解
            optimal_action, optimal_sequence, success, cost, weights = self.controller.solve(
                current_temp=current_temp
            )
            
            if success:
                r_pwm = optimal_action[0]
                b_pwm = optimal_action[1]
                
                print(f"📊 MPPI结果:")
                print(f"   红光PWM: {r_pwm:.2f}")
                print(f"   蓝光PWM: {b_pwm:.2f}")
                print(f"   总PWM: {r_pwm + b_pwm:.2f}")
                print(f"   成本: {cost:.2f}")
                
                return r_pwm, b_pwm, True, cost
            else:
                print("❌ MPPI求解失败")
                return None, None, False, None
                
        except Exception as e:
            print(f"❌ MPPI控制错误: {e}")
            return None, None, False, None
    
    def send_pwm_commands(self, r_pwm, b_pwm):
        """发送PWM命令到设备"""
        try:
            print(f"📡 发送PWM命令到设备...")
            
            # 转换PWM值到亮度值 (PWM值直接作为亮度值，四舍五入)
            r_brightness = int(np.round(np.clip(r_pwm, 0, 100)))
            b_brightness = int(np.round(np.clip(b_pwm, 0, 100)))
            
            commands = []
            
            # 发送红光设备命令
            if "Red" in self.devices:
                red_ip = self.devices["Red"]
                red_cmd = {
                    "id": 0,
                    "on": True,
                    "brightness": r_brightness,
                    "transition": 1000  # 1秒过渡
                }
                
                print(f"🔴 红光设备 ({red_ip}): brightness={r_brightness}")
                print(f"   命令: {json.dumps(red_cmd, indent=2)}")
                
                # 这里只打印命令，不实际发送
                # response = rpc(red_ip, "Light.Set", red_cmd)
                # print(f"   响应: {response}")
                
                commands.append(("Red", red_cmd))
            
            # 发送蓝光设备命令
            if "Blue" in self.devices:
                blue_ip = self.devices["Blue"]
                blue_cmd = {
                    "id": 0,
                    "on": True,
                    "brightness": b_brightness,
                    "transition": 1000  # 1秒过渡
                }
                
                print(f"🔵 蓝光设备 ({blue_ip}): brightness={b_brightness}")
                print(f"   命令: {json.dumps(blue_cmd, indent=2)}")
                
                # 这里只打印命令，不实际发送
                # response = rpc(blue_ip, "Light.Set", blue_cmd)
                # print(f"   响应: {response}")
                
                commands.append(("Blue", blue_cmd))
            
            return commands, True
            
        except Exception as e:
            print(f"❌ 发送命令错误: {e}")
            return [], False
    
    def run_control_cycle(self):
        """运行一次完整的控制循环"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'='*60}")
        print(f"🔄 控制循环开始 - {timestamp}")
        print(f"{'='*60}")
        
        # 1. 读取温度（失败则重试5次，每次间隔1分钟；最终失败则跳过本次发送）
        current_temp, temp_ok = self.read_temperature()
        retry_count = 0
        while not temp_ok and retry_count < 5:
            retry_count += 1
            print(f"⏳ 温度读取失败，{retry_count}/5 次重试，1分钟后重试...")
            time.sleep(60)
            current_temp, temp_ok = self.read_temperature()

        if not temp_ok:
            print("❌ 温度读取连续失败(5次)，维持上次PWM，跳过本次控制发送")
            # 读取CO2用于日志记录（允许使用模拟CO2）
            current_co2, _ = self.read_co2()
            r_pwm = self.last_r_pwm
            b_pwm = self.last_b_pwm
            cost = self.last_cost
            note = "温度读取失败，已重试5次，维持上次PWM并跳过发送"
            # 记录失败日志并结束本次循环（不发送命令）
            self.log_control_cycle(timestamp, current_temp if current_temp is not None else float('nan'), current_co2, r_pwm, b_pwm, False, cost, note)
            return False
        else:
            # 2. 读取CO2
            current_co2, co2_ok = self.read_co2()
            if not co2_ok:
                print("❌ CO2读取失败，使用上一次PWM控制结果")
                # 使用上一次的控制结果
                r_pwm = self.last_r_pwm
                b_pwm = self.last_b_pwm
                cost = self.last_cost
                current_co2 = 420.0  # 使用默认CO2值
                note = "CO2读取失败，使用上次PWM"
            else:
                # 3. 运行MPPI控制
                r_pwm, b_pwm, control_ok, cost = self.run_mppi_control(current_temp)
                if not control_ok:
                    print("❌ MPPI控制失败，使用上一次PWM控制结果")
                    # 使用上一次的控制结果
                    r_pwm = self.last_r_pwm
                    b_pwm = self.last_b_pwm
                    cost = self.last_cost
                    note = "MPPI控制失败，使用上次PWM"
                else:
                    # 更新上一次的控制结果
                    self.last_r_pwm = r_pwm
                    self.last_b_pwm = b_pwm
                    self.last_cost = cost
                    note = ""
        
        # 4. 发送PWM命令
        commands, send_ok = self.send_pwm_commands(r_pwm, b_pwm)
        if not send_ok:
            print("❌ 命令发送失败")
            self.log_control_cycle(timestamp, current_temp, current_co2, r_pwm, b_pwm, False, cost, note)
            return False
        
        # 5. 记录成功日志
        if 'solar_vol' in str(DEFAULT_MODEL_NAME).lower():
            if getattr(self, 'last_a1_avg', None) is not None:
                a1_note = f"A1_Raw_10min_avg={self.last_a1_avg:.2f}"
            else:
                a1_note = "A1_Raw_10min_avg=N/A"
            if note:
                note = f"{note}; {a1_note}"
            else:
                note = a1_note
        
        self.log_control_cycle(timestamp, current_temp, current_co2, r_pwm, b_pwm, True, cost, note)
        print(f"✅ 控制循环完成")
        return True
    
    def run_continuous(self, interval_minutes=1):
        """连续运行控制循环"""
        print(f"🚀 开始连续控制循环 (间隔: {interval_minutes}分钟)")
        print("按 Ctrl+C 停止")
        
        try:
            while True:
                self.run_control_cycle()
                
                # 等待下次循环
                print(f"⏰ 等待 {interval_minutes} 分钟...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n🛑 控制循环已停止")
        except Exception as e:
            print(f"❌ 控制循环错误: {e}")

def main():
    """主函数"""
    print("🌱 MPPI LED控制循环系统")
    print("=" * 50)
    print(f"📱 配置信息:")
    print(f"   温度设备: {TEMPERATURE_DEVICE_ID or '自动选择'}")
    print(f"   红蓝比例: {RB_RATIO_KEY}")
    print(f"   控制间隔: {CONTROL_INTERVAL_MINUTES}分钟")
    print(f"   使用模型: {DEFAULT_MODEL_NAME}")
    print("=" * 50)
    
    # 创建控制循环实例
    control_loop = MPPIControlLoop()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "once":
            # 运行一次
            print("🔄 运行单次控制循环...")
            control_loop.run_control_cycle()
        elif sys.argv[1] == "continuous":
            # 连续运行
            print(f"🔄 开始连续控制循环...")
            control_loop.run_continuous(CONTROL_INTERVAL_MINUTES)
        elif sys.argv[1] == "list-devices":
            # 列出可用设备
            print("📱 可用温度设备:")
            try:
                from __init__ import get_riotee_devices
                devices = get_riotee_devices()
                if devices:
                    for device in devices:
                        print(f"   - {device}")
                else:
                    print("   无可用设备")
            except Exception as e:
                print(f"❌ 获取设备列表失败: {e}")
        else:
            print("❌ 无效参数")
            print("用法:")
            print("  python mppi_control_loop.py once")
            print("  python mppi_control_loop.py continuous")
            print("  python mppi_control_loop.py list-devices")
            print("")
            print("💡 提示: 修改代码顶部的宏定义来配置设备ID和其他参数")
    else:
        # 默认运行一次
        print("🔄 运行单次控制循环...")
        control_loop.run_control_cycle()

if __name__ == "__main__":
    main()
