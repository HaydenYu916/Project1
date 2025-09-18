#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPPI控制执行脚本
实际发送PWM命令到设备并检查状态
"""

import sys
import os
import time
import json
import csv
from datetime import datetime
import numpy as np

# ==================== 配置宏定义 ====================
# 温度传感器设备ID配置（修改此处即可切换设备）
TEMPERATURE_DEVICE_ID = None  # None=自动选择, "T6ncwg=="=指定设备1, "L_6vSQ=="=指定设备2

# 控制循环间隔（分钟）
CONTROL_INTERVAL_MINUTES = 1

# 红蓝比例键
RB_RATIO_KEY = "5:1"

# 日志文件路径
LOG_FILE = "mppi_control_execute_log.csv"

# 状态检查延迟（秒）
STATUS_CHECK_DELAY = 3
# =====================================================

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
riotee_sensor_dir = os.path.join(current_dir, '..', 'Test', 'riotee_sensor')
mppi_dir = os.path.join(current_dir, '..', 'AA_Test_9_16')
controller_dir = os.path.join(current_dir, '..', 'aioshelly', 'my_src')

# 确保MPPI目录在路径最前面，以便导入numpy等依赖
sys.path.insert(0, mppi_dir)
sys.path.insert(0, riotee_sensor_dir)
sys.path.insert(0, controller_dir)

try:
    from __init__ import get_current_riotee
    from mppi import LEDPlant, LEDMPPIController
    from controller import rpc, DEVICES
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

class MPPIControlExecute:
    def __init__(self):
        """初始化MPPI控制执行器"""
        print("🚀 初始化MPPI控制执行器...")
        
        # 使用宏定义配置
        self.temperature_device_id = TEMPERATURE_DEVICE_ID
        self.log_file = LOG_FILE
        
        # 初始化LED植物模型
        self.plant = LEDPlant(
            model_key=RB_RATIO_KEY,
            use_efficiency=False,
            heat_scale=1.0
        )
        
        # 初始化MPPI控制器
        self.controller = LEDMPPIController(
            plant=self.plant,
            horizon=10,
            num_samples=1000,
            dt=0.1,
            temperature=1.0,
            maintain_rb_ratio=True,
            rb_ratio_key="5:1"
        )
        
        # 设置控制器参数
        self.controller.set_weights(
            Q_photo=10.0,
            R_pwm=0.001,
            R_dpwm=0.05,
            R_power=0.01
        )
        
        self.controller.set_constraints(
            pwm_min=0.0,
            pwm_max=80.0,
            temp_min=20.0,
            temp_max=29.0
        )
        
        # 设备IP地址
        self.devices = DEVICES
        
        print("✅ MPPI控制执行器初始化完成")
        print(f"   温度设备: {self.temperature_device_id or '自动选择'}")
        print(f"   LED设备列表: {list(self.devices.keys())}")
        print(f"   红蓝比例: {RB_RATIO_KEY}")
        print(f"   控制间隔: {CONTROL_INTERVAL_MINUTES}分钟")
        print(f"   状态检查延迟: {STATUS_CHECK_DELAY}秒")
        
        # 初始化日志文件
        self.init_log_file()
    
    def init_log_file(self):
        """初始化日志文件"""
        try:
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['时间戳', '输入温度', '红光PWM', '蓝光PWM', '成功状态', '成本', '红光状态', '蓝光状态', '备注'])
        except Exception as e:
            print(f"⚠️  日志文件初始化失败: {e}")
    
    def log_control_cycle(self, timestamp, input_temp, output_r_pwm, output_b_pwm, success, cost=None, red_status=None, blue_status=None, note=""):
        """记录控制循环日志"""
        try:
            cost_str = f"{cost:.2f}" if cost is not None else "N/A"
            red_status_str = str(red_status) if red_status is not None else "N/A"
            blue_status_str = str(blue_status) if blue_status is not None else "N/A"
            
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, f"{input_temp:.2f}", f"{output_r_pwm:.2f}", f"{output_b_pwm:.2f}", 
                               success, cost_str, red_status_str, blue_status_str, note])
        except Exception as e:
            print(f"⚠️  日志记录失败: {e}")
    
    def read_temperature(self):
        """读取当前温度数据"""
        try:
            data = get_current_riotee(
                device_id=self.temperature_device_id, 
                max_age_seconds=86400
            )
            
            if data and data.get('temperature') is not None:
                temp = data['temperature']
                device_id = data.get('device_id', 'Unknown')
                age = data.get('_data_age_seconds', 0)
                
                if age < 120:
                    status = "🟢"
                elif age < 300:
                    status = "🟡"
                else:
                    status = "🔴"
                
                print(f"🌡️  {status} 温度读取: {temp:.2f}°C (设备: {device_id}, {age:.0f}秒前)")
                return temp, True
            else:
                if self.temperature_device_id:
                    print(f"⚠️  指定设备 {self.temperature_device_id} 无有效温度数据，使用模拟温度 24.5°C")
                else:
                    print("⚠️  无有效温度数据，使用模拟温度 24.5°C")
                return 24.5, True
                
        except Exception as e:
            print(f"❌ 温度读取错误: {e}")
            print("⚠️  使用模拟温度 24.5°C")
            return 24.5, True
    
    def run_mppi_control(self, current_temp):
        """运行MPPI控制算法"""
        try:
            print(f"🎯 运行MPPI控制 (当前温度: {current_temp:.2f}°C)")
            
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
    
    def get_device_status(self, device_ip, device_name):
        """获取设备状态"""
        try:
            response = rpc(device_ip, "Light.GetStatus", {"id": 0})
            if response and 'brightness' in response:
                brightness = response.get('brightness', 0)
                is_on = response.get('ison', False)
                print(f"📱 {device_name}状态: brightness={brightness}, on={is_on}")
                return {'brightness': brightness, 'on': is_on, 'success': True}
            else:
                print(f"❌ {device_name}状态获取失败")
                return {'brightness': 0, 'on': False, 'success': False}
        except Exception as e:
            print(f"❌ {device_name}状态检查错误: {e}")
            return {'brightness': 0, 'on': False, 'success': False}
    
    def send_pwm_commands(self, r_pwm, b_pwm):
        """发送PWM命令到设备并检查状态"""
        try:
            print(f"📡 发送PWM命令到设备...")
            
            # 转换PWM值到亮度值 (0-100)
            r_brightness = int(np.clip(r_pwm * 100 / 80, 0, 100))
            b_brightness = int(np.clip(b_pwm * 100 / 80, 0, 100))
            
            commands = []
            red_status = None
            blue_status = None
            
            # 发送红光设备命令
            if "Red" in self.devices:
                red_ip = self.devices["Red"]
                red_cmd = {
                    "id": 0,
                    "on": True,
                    "brightness": r_brightness,
                    "transition": 1000
                }
                
                print(f"🔴 发送红光命令 ({red_ip}): brightness={r_brightness}")
                try:
                    response = rpc(red_ip, "Light.Set", red_cmd)
                    print(f"   响应: {response}")
                    commands.append(("Red", red_cmd))
                except Exception as e:
                    print(f"❌ 红光命令发送失败: {e}")
                    return [], False, None, None
            
            # 发送蓝光设备命令
            if "Blue" in self.devices:
                blue_ip = self.devices["Blue"]
                blue_cmd = {
                    "id": 0,
                    "on": True,
                    "brightness": b_brightness,
                    "transition": 1000
                }
                
                print(f"🔵 发送蓝光命令 ({blue_ip}): brightness={b_brightness}")
                try:
                    response = rpc(blue_ip, "Light.Set", blue_cmd)
                    print(f"   响应: {response}")
                    commands.append(("Blue", blue_cmd))
                except Exception as e:
                    print(f"❌ 蓝光命令发送失败: {e}")
                    return [], False, None, None
            
            # 等待设备响应
            print(f"⏰ 等待 {STATUS_CHECK_DELAY} 秒后检查设备状态...")
            time.sleep(STATUS_CHECK_DELAY)
            
            # 检查设备状态
            if "Red" in self.devices:
                red_status = self.get_device_status(self.devices["Red"], "红光设备")
            
            if "Blue" in self.devices:
                blue_status = self.get_device_status(self.devices["Blue"], "蓝光设备")
            
            return commands, True, red_status, blue_status
            
        except Exception as e:
            print(f"❌ 发送命令错误: {e}")
            return [], False, None, None
    
    def run_control_cycle(self):
        """运行一次完整的控制循环"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'='*60}")
        print(f"🔄 控制循环开始 - {timestamp}")
        print(f"{'='*60}")
        
        # 1. 读取温度
        current_temp, temp_ok = self.read_temperature()
        if not temp_ok:
            print("❌ 温度读取失败，跳过本次控制循环")
            self.log_control_cycle(timestamp, 0.0, 0.0, 0.0, False, note="温度读取失败")
            return False
        
        # 2. 运行MPPI控制
        r_pwm, b_pwm, control_ok, cost = self.run_mppi_control(current_temp)
        if not control_ok:
            print("❌ MPPI控制失败，跳过本次控制循环")
            self.log_control_cycle(timestamp, current_temp, 0.0, 0.0, False, note="MPPI控制失败")
            return False
        
        # 3. 发送PWM命令并检查状态
        commands, send_ok, red_status, blue_status = self.send_pwm_commands(r_pwm, b_pwm)
        if not send_ok:
            print("❌ 命令发送失败")
            self.log_control_cycle(timestamp, current_temp, r_pwm, b_pwm, False, cost, red_status, blue_status, "命令发送失败")
            return False
        
        # 4. 记录成功日志
        self.log_control_cycle(timestamp, current_temp, r_pwm, b_pwm, True, cost, red_status, blue_status, "控制成功")
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
    print("🌱 MPPI LED控制执行系统")
    print("=" * 50)
    print(f"📱 配置信息:")
    print(f"   温度设备: {TEMPERATURE_DEVICE_ID or '自动选择'}")
    print(f"   红蓝比例: {RB_RATIO_KEY}")
    print(f"   控制间隔: {CONTROL_INTERVAL_MINUTES}分钟")
    print(f"   状态检查延迟: {STATUS_CHECK_DELAY}秒")
    print("=" * 50)
    
    # 创建控制执行器实例
    control_execute = MPPIControlExecute()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "once":
            # 运行一次
            print("🔄 运行单次控制循环...")
            control_execute.run_control_cycle()
        elif sys.argv[1] == "continuous":
            # 连续运行
            print(f"🔄 开始连续控制循环...")
            control_execute.run_continuous(CONTROL_INTERVAL_MINUTES)
        else:
            print("❌ 无效参数")
            print("用法:")
            print("  python mppi_control_execute.py once")
            print("  python mppi_control_execute.py continuous")
            print("")
            print("💡 提示: 修改代码顶部的宏定义来配置设备ID和其他参数")
    else:
        # 默认运行一次
        print("🔄 运行单次控制循环...")
        control_execute.run_control_cycle()

if __name__ == "__main__":
    main()
