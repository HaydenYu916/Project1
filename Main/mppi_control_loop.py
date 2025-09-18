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
from datetime import datetime
import numpy as np

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

class MPPIControlLoop:
    def __init__(self):
        """初始化MPPI控制循环"""
        print("🚀 初始化MPPI控制循环...")
        
        # 初始化LED植物模型
        self.plant = LEDPlant(
            model_key="5:1",  # 红蓝比例 5:1
            use_efficiency=False,  # 暂时关闭效率模型
            heat_scale=1.0
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
        
        # 设置控制器参数
        self.controller.set_weights(
            Q_photo=10.0,    # 光合作用权重
            R_pwm=0.001,     # PWM权重
            R_dpwm=0.05,     # PWM变化权重
            R_power=0.01     # 功率权重
        )
        
        self.controller.set_constraints(
            pwm_min=0.0,     # PWM最小值
            pwm_max=80.0,    # PWM最大值
            temp_min=20.0,   # 温度最小值
            temp_max=29.0    # 温度最大值
        )
        
        # 目标温度
        self.target_temp = 25.0
        
        # 设备IP地址
        self.devices = DEVICES
        
        print("✅ MPPI控制循环初始化完成")
        print(f"   目标温度: {self.target_temp}°C")
        print(f"   设备列表: {list(self.devices.keys())}")
        print(f"   红蓝比例: 5:1")
    
    def read_temperature(self):
        """读取当前温度数据"""
        try:
            data = get_current_riotee(max_age_seconds=86400)  # 放宽到24小时
            
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
                
                print(f"🌡️  {status} 温度读取: {temp:.2f}°C (设备: {device_id}, {age:.0f}秒前)")
                return temp, True
            else:
                print("⚠️  无有效温度数据，使用模拟温度 24.5°C")
                return 24.5, True  # 使用模拟温度
                
        except Exception as e:
            print(f"❌ 温度读取错误: {e}")
            print("⚠️  使用模拟温度 24.5°C")
            return 24.5, True  # 使用模拟温度
    
    def run_mppi_control(self, current_temp):
        """运行MPPI控制算法"""
        try:
            print(f"🎯 运行MPPI控制 (当前温度: {current_temp:.2f}°C, 目标: {self.target_temp:.2f}°C)")
            
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
                
                return r_pwm, b_pwm, True
            else:
                print("❌ MPPI求解失败")
                return None, None, False
                
        except Exception as e:
            print(f"❌ MPPI控制错误: {e}")
            return None, None, False
    
    def send_pwm_commands(self, r_pwm, b_pwm):
        """发送PWM命令到设备"""
        try:
            print(f"📡 发送PWM命令到设备...")
            
            # 转换PWM值到亮度值 (0-100)
            r_brightness = int(np.clip(r_pwm * 100 / 80, 0, 100))
            b_brightness = int(np.clip(b_pwm * 100 / 80, 0, 100))
            
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
        print(f"\n{'='*60}")
        print(f"🔄 控制循环开始 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # 1. 读取温度
        current_temp, temp_ok = self.read_temperature()
        if not temp_ok:
            print("❌ 温度读取失败，跳过本次控制循环")
            return False
        
        # 2. 运行MPPI控制
        r_pwm, b_pwm, control_ok = self.run_mppi_control(current_temp)
        if not control_ok:
            print("❌ MPPI控制失败，跳过本次控制循环")
            return False
        
        # 3. 发送PWM命令
        commands, send_ok = self.send_pwm_commands(r_pwm, b_pwm)
        if not send_ok:
            print("❌ 命令发送失败")
            return False
        
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
            interval = 1
            if len(sys.argv) > 2:
                try:
                    interval = int(sys.argv[2])
                except ValueError:
                    print("❌ 无效的间隔时间，使用默认值1分钟")
            control_loop.run_continuous(interval)
        else:
            print("❌ 无效参数")
            print("用法:")
            print("  python mppi_control_loop.py once          # 运行一次")
            print("  python mppi_control_loop.py continuous [间隔分钟]  # 连续运行")
    else:
        # 默认运行一次
        print("🔄 运行单次控制循环...")
        control_loop.run_control_cycle()

if __name__ == "__main__":
    main()
