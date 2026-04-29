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
import signal
import subprocess
import logging
import atexit
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# ==================== 配置宏定义 ====================
# 温度传感器设备ID配置（修改此处即可切换设备）
TEMPERATURE_DEVICE_ID = None  # None=自动选择, "T6ncwg=="=指定设备1, "L_6vSQ=="=指定设备2

# 控制循环间隔（分钟）
CONTROL_INTERVAL_MINUTES = 15

# 红蓝比例键
RB_RATIO_KEY = "5:1"

# 状态检查延迟（秒）
STATUS_CHECK_DELAY = 3

# 夜间休眠时间（24小时制）
NIGHT_START_HOUR = 23  # 23:00
NIGHT_END_HOUR = 7     # 07:00

# 后台运行相关配置
PID_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "mppi_control.pid")
BACKGROUND_LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "mppi_control_background.log")
# =====================================================

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 日志文件路径
LOG_FILE = os.path.join(current_dir, "..", "..", "logs", "control_real_log.csv")
project_root = os.path.join(current_dir, '..', '..')
src_dir = os.path.join(project_root, 'src')
tool_root = os.path.join(project_root, '..', 'Tool')
riotee_sensor_dir = os.path.join(tool_root, 'Sensor_riotee_server')
controller_dir = os.path.join(tool_root, 'LED_Shelly', 'src')
config_dir = os.path.join(current_dir, '..', '..', 'config')

# 确保项目目录在路径中
sys.path.insert(0, src_dir)
sys.path.insert(0, riotee_sensor_dir)
sys.path.insert(0, controller_dir)
sys.path.insert(0, config_dir)

try:
    # 动态加载 riotee_sensor 的 __init__.py，避免包相对导入问题
    riotee_init_path = os.path.join(riotee_sensor_dir, '__init__.py')
    if os.path.exists(riotee_init_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("riotee_init", riotee_init_path)
        riotee_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(riotee_module)
        get_current_riotee = riotee_module.get_current_riotee
        get_device_avg_a1_raw = getattr(riotee_module, 'get_device_avg_a1_raw', None)
        get_riotee_devices = getattr(riotee_module, 'get_riotee_devices', None)
    else:
        raise ImportError(f"riotee __init__.py not found at {riotee_init_path}")

    from mppi import LEDPlant, LEDMPPIController
    from shelly_controller import rpc, DEVICES
    from app_config import DEFAULT_MODEL_NAME
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

class MPPIControlExecute:
    def __init__(self, background_mode=False):
        """初始化MPPI控制执行器"""
        self.background_mode = background_mode
        
        if not background_mode:
            print("🚀 初始化MPPI控制执行器...")
        
        # 使用宏定义配置
        self.temperature_device_id = TEMPERATURE_DEVICE_ID
        self.log_file = LOG_FILE
        
        # 后台模式设置
        if background_mode:
            self.setup_background_logging()
        
        # 初始化LED植物模型
        self.plant = LEDPlant(
            model_key=RB_RATIO_KEY,
            use_efficiency=False,
            heat_scale=1.0,
            model_name=DEFAULT_MODEL_NAME
        )
        
        # 初始化MPPI控制器
        # 参数档位配置（与仿真一致，dt 使用控制间隔）
        PARAM_PROFILES = {
            'low': {
                'horizon': 5,
                'num_samples': 600,
                'temperature': 1.2,
                'constraints': dict(pwm_min=5.0, pwm_max=85.0, temp_min=20.0, temp_max=29.0),
                'penalties': dict(temp_penalty=200000.0),
                'weights': dict(Q_photo=8.0, R_pwm=0.002, R_dpwm=0.08, R_power=0.02),
                'pwm_std': np.array([8.0, 8.0], dtype=float),
            },
            'mid': {
                'horizon': 5,
                'num_samples': 700,
                'temperature': 1.1,
                'constraints': dict(pwm_min=5.0, pwm_max=90.0, temp_min=20.0, temp_max=30.0),
                'penalties': dict(temp_penalty=120000.0),
                'weights': dict(Q_photo=12.0, R_pwm=0.001, R_dpwm=0.06, R_power=0.01),
                'pwm_std': np.array([10.0, 10.0], dtype=float),
            },
            'high': {
                'horizon': 5,
                'num_samples': 800,
                'temperature': 1.2,
                'constraints': dict(pwm_min=5.0, pwm_max=95.0, temp_min=20.0, temp_max=31.0),
                'penalties': dict(temp_penalty=80000.0),
                'weights': dict(Q_photo=20.0, R_pwm=0.0005, R_dpwm=0.04, R_power=0.004),
                'pwm_std': np.array([14.0, 14.0], dtype=float),
            },
        }

        PROFILE = 'high'  # 可切换 'mid'/'low'
        prof = PARAM_PROFILES[PROFILE]

        self.controller = LEDMPPIController(
            plant=self.plant,
            horizon=prof['horizon'],
            num_samples=prof['num_samples'],
            dt=CONTROL_INTERVAL_MINUTES * 60,
            temperature=prof['temperature'],
            maintain_rb_ratio=True,
            rb_ratio_key="5:1"
        )
        
        # 应用档位
        self.controller.set_constraints(**prof['constraints'])
        for k, v in prof['penalties'].items():
            self.controller.penalties[k] = v
        self.controller.set_weights(**prof['weights'])
        self.controller.pwm_std = prof['pwm_std']
        
        # 设备IP地址
        self.devices = DEVICES
        
        print("✅ MPPI控制执行器初始化完成")
        print(f"   温度设备: {self.temperature_device_id or '自动选择'}")
        print(f"   LED设备列表: {list(self.devices.keys())}")
        print(f"   红蓝比例: {RB_RATIO_KEY}")
        print(f"   控制间隔: {CONTROL_INTERVAL_MINUTES}分钟")
        print(f"   使用模型: {DEFAULT_MODEL_NAME}")
        print(f"   状态检查延迟: {STATUS_CHECK_DELAY}秒")
        
        # 初始化日志文件
        self.init_log_file()
        
        # CO2 数据文件（与仿真一致路径）
        self.co2_file = "/data/csv/co2_sensor.csv"
        
        # 后台模式信号处理
        if background_mode:
            signal.signal(signal.SIGTERM, self.signal_handler)
            signal.signal(signal.SIGINT, self.signal_handler)
            atexit.register(self.cleanup_on_exit)
    
    def init_log_file(self):
        """初始化日志文件"""
        try:
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'input_temp', 'co2_value', 'solar_vol', 'photosynthesis_rate', 'red_pwm', 'blue_pwm', 'total_pwm', 'success', 'cost', 'red_status', 'blue_status', 'note', 'ppfd'])
        except Exception as e:
            print(f"⚠️  日志文件初始化失败: {e}")
    
    def log_control_cycle(self, timestamp, input_temp, co2_value, solar_vol, ppfd, photo_rate, output_r_pwm, output_b_pwm, success, cost=None, red_status=None, blue_status=None, note=""):
        """记录控制循环日志"""
        try:
            cost_str = f"{cost:.2f}" if cost is not None else "N/A"
            red_status_str = str(red_status) if red_status is not None else "N/A"
            blue_status_str = str(blue_status) if blue_status is not None else "N/A"
            total_pwm = output_r_pwm + output_b_pwm
            
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, 
                    f"{input_temp:.2f}", 
                    f"{co2_value:.1f}" if co2_value is not None else "N/A",
                    f"{solar_vol:.2f}" if solar_vol is not None else "N/A",
                    f"{photo_rate:.4f}" if photo_rate is not None else "N/A",
                    f"{output_r_pwm:.2f}", 
                    f"{output_b_pwm:.2f}",
                    f"{total_pwm:.2f}",
                    success, 
                    cost_str, 
                    red_status_str, 
                    blue_status_str, 
                    note,
                    f"{ppfd:.2f}" if ppfd is not None else "N/A"
                ])
        except Exception as e:
            print(f"⚠️  日志记录失败: {e}")
    
    def log_simple_run(self, action, details=""):
        """记录简单运行日志"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] {action}"
            if details:
                log_entry += f" - {details}"
            
            # 写入简单日志文件
            simple_log_file = os.path.join(os.path.dirname(self.log_file), "control_simple.log")
            with open(simple_log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + "\n")
            
            print(f"📝 {log_entry}")
        except Exception as e:
            print(f"⚠️  简单日志记录失败: {e}")
    
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
                
                # 如果模型为 solar_vol，尝试读取A1_Raw数据
                self.last_a1_avg = None
                if 'solar_vol' in str(DEFAULT_MODEL_NAME).lower():
                    self.last_a1_avg = self.get_solar_vol_data(device_id)

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

    def get_solar_vol_data(self, device_id):
        """获取Solar_Vol数据（A1_Raw均值）
        
        Args:
            device_id: 设备ID
            
        Returns:
            float: Solar_Vol值，如果获取失败返回None
        """
        try:
            # 检查函数是否可用
            if not get_device_avg_a1_raw or not device_id or device_id == 'Unknown':
                print("⚠️  Solar_Vol获取条件不满足：函数不可用或设备ID无效")
                return None
            
            # 尝试多个时间窗口获取数据
            window_minutes_list = [5, 10, 15, 30]  # 从短到长的时间窗口
            
            for window_minutes in window_minutes_list:
                try:
                    avg_info = get_device_avg_a1_raw(device_id, window_minutes=window_minutes)
                    avg_val = avg_info.get('avg')
                    cnt = avg_info.get('count', 0)
                    
                    if avg_val is not None and cnt > 0:
                        # 检查数据质量
                        if cnt >= 3:  # 至少需要3个数据点
                            print(f"🔆 Solar_Vol({window_minutes}min均值): {avg_val:.2f} (n={cnt})")
                            return float(avg_val)
                        else:
                            print(f"⚠️  Solar_Vol({window_minutes}min)数据点不足: {cnt} < 3")
                    else:
                        print(f"⚠️  Solar_Vol({window_minutes}min)无有效数据")
                        
                except Exception as e:
                    print(f"⚠️  Solar_Vol({window_minutes}min)读取失败: {e}")
                    continue
            
            # 所有时间窗口都失败，尝试获取最新单点数据
            print("⚠️  所有时间窗口Solar_Vol获取失败，尝试获取最新单点数据")
            try:
                current_data = get_current_riotee(device_id=device_id, max_age_seconds=300)  # 5分钟内的数据
                if current_data and 'a1_raw' in current_data:
                    a1_raw = current_data['a1_raw']
                    if a1_raw is not None:
                        print(f"🔆 Solar_Vol(最新单点): {a1_raw:.2f}")
                        return float(a1_raw)
            except Exception as e:
                print(f"⚠️  Solar_Vol最新单点获取失败: {e}")
            
            print("❌ Solar_Vol数据获取完全失败")
            return None
            
        except Exception as e:
            print(f"❌ Solar_Vol获取异常: {e}")
            return None

    def get_current_solar_vol(self, fallback_ppfd=None):
        """获取当前Solar_Vol值
        
        Args:
            fallback_ppfd: 当Solar_Vol获取失败时的备用PPFD值
            
        Returns:
            float: Solar_Vol值，如果获取失败返回备用值或None
        """
        try:
            # 检查是否使用solar_vol模型
            if not (hasattr(self.plant, 'model_name') and 'solar_vol' in str(self.plant.model_name).lower()):
                print("ℹ️  非solar_vol模型，跳过Solar_Vol获取")
                return fallback_ppfd
            
            # 尝试使用已缓存的Solar_Vol数据
            if hasattr(self, 'last_a1_avg') and self.last_a1_avg is not None:
                print(f"🔆 使用缓存的Solar_Vol: {self.last_a1_avg:.2f}")
                return float(self.last_a1_avg)
            
            # 如果没有缓存数据，尝试重新获取
            print("⚠️  无缓存的Solar_Vol数据，尝试重新获取")
            if hasattr(self, 'temperature_device_id') and self.temperature_device_id:
                # 使用指定的温度设备ID
                solar_vol = self.get_solar_vol_data(self.temperature_device_id)
                if solar_vol is not None:
                    return solar_vol
            
            # 如果指定设备获取失败，尝试自动选择设备
            print("⚠️  指定设备Solar_Vol获取失败，尝试自动选择设备")
            try:
                devices = get_riotee_devices()
                if devices:
                    for device_id in devices:
                        solar_vol = self.get_solar_vol_data(device_id)
                        if solar_vol is not None:
                            print(f"✅ 使用设备 {device_id} 的Solar_Vol: {solar_vol:.2f}")
                            return solar_vol
            except Exception as e:
                print(f"⚠️  自动选择设备Solar_Vol获取失败: {e}")
            
            # 所有方法都失败，使用备用值
            if fallback_ppfd is not None:
                print(f"⚠️  Solar_Vol获取失败，使用备用PPFD值: {fallback_ppfd:.2f}")
                return float(fallback_ppfd)
            else:
                print("❌ Solar_Vol获取失败且无备用值")
                return None
                
        except Exception as e:
            print(f"❌ Solar_Vol获取异常: {e}")
            return fallback_ppfd if fallback_ppfd is not None else None

    def read_co2(self):
        """读取当前CO2数据（与仿真同源文件）。若不可用，返回默认420ppm。"""
        try:
            if not os.path.exists(self.co2_file):
                print("⚠️  CO2文件不存在，使用模拟CO2值 420 ppm")
                return 420.0, True
            df = pd.read_csv(self.co2_file, header=None, names=['timestamp', 'co2'])
            if df.empty:
                print("⚠️  CO2文件为空，使用模拟CO2值 420 ppm")
                return 420.0, True
            latest_row = df.iloc[-1]
            latest_co2 = latest_row['co2']
            if pd.isna(latest_co2) or latest_co2 is None:
                print("⚠️  最新CO2值无效，使用模拟CO2值 420 ppm")
                return 420.0, True
            print(f"🌬️  CO2读取: {latest_co2:.1f} ppm")
            return float(latest_co2), True
        except Exception as e:
            print(f"❌ CO2读取错误: {e}")
            return 420.0, True
    
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
                
                # 获取Solar_Vol（如果使用solar_vol模型，需要从传感器数据获取）
                solar_vol = self.get_current_solar_vol()
                
                # 计算当前条件下的预测值
                ppfd_pred, temp_pred, power_pred, photo_pred = self.plant.predict(
                    np.array([[r_pwm, b_pwm]]), current_temp, self.controller.dt, solar_vol=solar_vol
                )
                
                current_ppfd = ppfd_pred[0]
                current_photo_rate = photo_pred[0]
                
                print(f"📊 MPPI结果:")
                print(f"   红光PWM: {r_pwm:.2f}")
                print(f"   蓝光PWM: {b_pwm:.2f}")
                print(f"   总PWM: {r_pwm + b_pwm:.2f}")
                print(f"   预测PPFD: {current_ppfd:.2f}")
                print(f"   预测光合作用速率: {current_photo_rate:.4f}")
                if solar_vol is not None:
                    print(f"   预测Solar_Vol: {solar_vol:.2f}")
                print(f"   成本: {cost:.2f}")
                
                return r_pwm, b_pwm, True, cost, solar_vol, current_ppfd, current_photo_rate
            else:
                print("❌ MPPI求解失败")
                return None, None, False, None, None, None, None
                
        except Exception as e:
            print(f"❌ MPPI控制错误: {e}")
            return None, None, False, None, None, None, None
    
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
            
            # 转换PWM值到亮度值 (PWM值直接作为亮度值，四舍五入)
            r_brightness = int(np.round(np.clip(r_pwm, 0, 100)))
            b_brightness = int(np.round(np.clip(b_pwm, 0, 100)))
            
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
        
        # 记录控制循环开始
        self.log_simple_run("控制循环开始")
        
        # 1. 读取温度
        current_temp, temp_ok = self.read_temperature()
        if not temp_ok:
            print("❌ 温度读取失败，跳过本次控制循环")
            self.log_simple_run("控制循环失败", "温度读取失败")
            self.log_control_cycle(timestamp, 0.0, None, None, None, None, None, 0.0, 0.0, False, note="温度读取失败")
            return False
        
        # 2. 读取CO2并注入到 plant
        current_co2, co2_ok = self.read_co2()
        try:
            self.plant.set_env_co2(float(current_co2))
        except Exception:
            pass
        
        # 3. 运行MPPI控制
        r_pwm, b_pwm, control_ok, cost, solar_vol, ppfd, photo_rate = self.run_mppi_control(current_temp)
        if not control_ok:
            print("❌ MPPI控制失败，跳过本次控制循环")
            self.log_simple_run("控制循环失败", "MPPI控制失败")
            self.log_control_cycle(timestamp, current_temp, current_co2, None, None, None, None, 0.0, 0.0, False, note="MPPI控制失败")
            return False
        
        # 4. 发送PWM命令并检查状态
        commands, send_ok, red_status, blue_status = self.send_pwm_commands(r_pwm, b_pwm)
        if not send_ok:
            print("❌ 命令发送失败")
            self.log_simple_run("控制循环失败", "命令发送失败")
            self.log_control_cycle(timestamp, current_temp, current_co2, solar_vol, ppfd, photo_rate, r_pwm, b_pwm, False, cost, red_status, blue_status, "命令发送失败")
            return False
        
        # 5. 记录成功日志
        self.log_simple_run("控制循环成功", f"温度:{current_temp:.1f}°C, PWM:R{r_pwm:.1f}/B{b_pwm:.1f}, 成本:{cost:.1f}")
        self.log_control_cycle(timestamp, current_temp, current_co2, solar_vol, ppfd, photo_rate, r_pwm, b_pwm, True, cost, red_status, blue_status, "控制成功")
        print(f"✅ 控制循环完成")
        return True
    
    def run_continuous(self, interval_minutes=1):
        """连续运行控制循环 - 在每小时的0,15,30,45分运行"""
        print(f"🚀 开始连续控制循环 (运行时间: 每小时的0,15,30,45分)")
        print("按 Ctrl+C 停止")
        print(f"🌙 夜间休眠时间: {NIGHT_START_HOUR:02d}:00 - {NIGHT_END_HOUR:02d}:00")
        
        # 记录连续运行开始
        self.log_simple_run("连续控制循环启动", "运行时间:每小时的0,15,30,45分")
        
        try:
            while True:
                now = datetime.now()
                current_hour = now.hour
                current_minute = now.minute
                
                # 检查是否在夜间休眠时间
                if self.is_night_time(current_hour):
                    print(f"🌙 当前时间 {current_hour:02d}:{current_minute:02d} 在夜间休眠时间 ({NIGHT_START_HOUR:02d}:00-{NIGHT_END_HOUR:02d}:00)，跳过控制循环")
                    # 计算到下一个运行时间点的等待时间
                    wait_seconds = self.calculate_wait_time(now)
                    print(f"⏰ 等待 {wait_seconds//60} 分钟 {wait_seconds%60} 秒...")
                    time.sleep(wait_seconds)
                    continue
                
                # 检查是否在运行时间点 (0, 15, 30, 45分)
                if current_minute in [0, 15, 30, 45]:
                    print(f"⏰ 当前时间 {current_hour:02d}:{current_minute:02d} - 运行控制循环")
                    self.run_control_cycle()
                else:
                    print(f"⏰ 当前时间 {current_hour:02d}:{current_minute:02d} - 非运行时间点，等待...")
                
                # 计算到下一个运行时间点的等待时间
                wait_seconds = self.calculate_wait_time(now)
                print(f"⏰ 等待 {wait_seconds//60} 分钟 {wait_seconds%60} 秒...")
                time.sleep(wait_seconds)
                
        except KeyboardInterrupt:
            print("\n🛑 控制循环已停止")
            self.log_simple_run("连续控制循环停止", "用户中断")
        except Exception as e:
            print(f"❌ 控制循环错误: {e}")
            self.log_simple_run("连续控制循环错误", str(e))

    def is_night_time(self, current_hour):
        """检查当前时间是否在夜间休眠时间"""
        if NIGHT_START_HOUR <= 23 and NIGHT_END_HOUR >= 0:
            # 跨午夜的情况 (23:00-07:00)
            return current_hour >= NIGHT_START_HOUR or current_hour < NIGHT_END_HOUR
        else:
            # 不跨午夜的情况 (例如 22:00-06:00)
            return NIGHT_START_HOUR <= current_hour < NIGHT_END_HOUR
    
    def calculate_wait_time(self, now):
        """计算到下一个运行时间点的等待时间（秒）"""
        current_minute = now.minute
        current_second = now.second
        
        # 运行时间点：0, 15, 30, 45分
        run_minutes = [0, 15, 30, 45]
        
        # 找到下一个运行时间点
        next_run_minute = None
        for minute in run_minutes:
            if minute > current_minute:
                next_run_minute = minute
                break
        
        if next_run_minute is None:
            # 如果当前时间已过所有运行点，等待到下一小时的0分
            next_run_minute = 0
            # 计算到下一小时0分的秒数
            wait_seconds = (60 - current_minute) * 60 - current_second
        else:
            # 计算到下一个运行时间点的秒数
            wait_seconds = (next_run_minute - current_minute) * 60 - current_second
        
        return max(1, wait_seconds)  # 至少等待1秒

    def setup_background_logging(self):
        """设置后台模式日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(BACKGROUND_LOG_FILE, mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        logging.info("MPPI控制后台模式启动")

    def signal_handler(self, signum, frame):
        """信号处理器"""
        logging.info(f"收到信号 {signum}，准备退出...")
        self.cleanup_on_exit()
        sys.exit(0)

    def cleanup_on_exit(self):
        """退出时清理"""
        try:
            if os.path.exists(PID_FILE):
                os.unlink(PID_FILE)
                logging.info("PID文件已清理")
        except Exception as e:
            logging.error(f"清理PID文件失败: {e}")

    def log_simple_run(self, action, details=""):
        """记录简单运行日志（后台模式兼容）"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] {action}"
            if details:
                log_entry += f" - {details}"
            
            # 写入简单日志文件
            simple_log_file = os.path.join(os.path.dirname(self.log_file), "control_simple.log")
            with open(simple_log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + "\n")
            
            if self.background_mode:
                logging.info(log_entry)
            else:
                print(f"📝 {log_entry}")
        except Exception as e:
            if self.background_mode:
                logging.error(f"简单日志记录失败: {e}")
            else:
                print(f"⚠️  简单日志记录失败: {e}")

def is_running():
    """检查后台进程是否运行"""
    if not os.path.exists(PID_FILE):
        return False
    
    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        os.unlink(PID_FILE)
        return False

def start_background():
    """启动后台进程"""
    if is_running():
        print("✅ MPPI控制已在后台运行")
        return True
    
    print("🚀 启动MPPI控制后台进程...")
    
    try:
        # 启动后台进程
        cmd = [sys.executable, __file__, "background"]
        with open(BACKGROUND_LOG_FILE, 'a', encoding='utf-8') as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(__file__),
                text=True
            )
        
        # 保存PID
        with open(PID_FILE, 'w') as f:
            f.write(str(process.pid))
        
        print(f"✅ MPPI控制后台进程已启动 (PID: {process.pid})")
        print(f"📄 后台日志: {BACKGROUND_LOG_FILE}")
        return True
        
    except Exception as e:
        print(f"❌ 启动后台进程失败: {e}")
        return False

def stop_background():
    """停止后台进程"""
    if not is_running():
        print("⏹️  MPPI控制后台进程未运行")
        return True
    
    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
        
        print(f"⏹️  停止MPPI控制后台进程 (PID: {pid})...")
        os.kill(pid, signal.SIGTERM)
        
        # 等待进程结束
        for i in range(10):
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except OSError:
                break
        
        # 清理PID文件
        if os.path.exists(PID_FILE):
            os.unlink(PID_FILE)
        
        print("✅ MPPI控制后台进程已停止")
        return True
        
    except Exception as e:
        print(f"❌ 停止后台进程失败: {e}")
        return False

def show_status():
    """显示状态"""
    print("📊 MPPI控制系统状态")
    print("=" * 30)
    
    if is_running():
        with open(PID_FILE, 'r') as f:
            pid = f.read().strip()
        print(f"🟢 后台进程: 运行中 (PID: {pid})")
    else:
        print("🔴 后台进程: 未运行")
    
    # 检查日志文件
    if os.path.exists(BACKGROUND_LOG_FILE):
        size = os.path.getsize(BACKGROUND_LOG_FILE)
        mtime = os.path.getmtime(BACKGROUND_LOG_FILE)
        last_modified = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        print(f"📄 后台日志: {BACKGROUND_LOG_FILE}")
        print(f"   大小: {size} bytes, 更新: {last_modified}")
    else:
        print("📄 后台日志: 无")

def main():
    """主函数"""
    print("🌱 MPPI LED控制执行系统")
    print("=" * 50)
    print(f"📱 配置信息:")
    print(f"   温度设备: {TEMPERATURE_DEVICE_ID or '自动选择'}")
    print(f"   红蓝比例: {RB_RATIO_KEY}")
    print(f"   运行时间: 每小时的0,15,30,45分")
    print(f"   状态检查延迟: {STATUS_CHECK_DELAY}秒")
    print(f"   夜间休眠: {NIGHT_START_HOUR:02d}:00-{NIGHT_END_HOUR:02d}:00")
    print("=" * 50)
    
    # 创建控制执行器实例
    control_execute = MPPIControlExecute()
    
    # 记录程序启动
    control_execute.log_simple_run("程序启动", f"模式:{sys.argv[1] if len(sys.argv) > 1 else '默认'}")
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "once":
            # 运行一次
            print("🔄 运行单次控制循环...")
            control_execute.run_control_cycle()
        elif command == "continuous":
            # 连续运行
            print(f"🔄 开始连续控制循环...")
            control_execute.run_continuous(CONTROL_INTERVAL_MINUTES)
        elif command == "background":
            # 后台模式
            control_execute = MPPIControlExecute(background_mode=True)
            control_execute.log_simple_run("后台模式启动")
            control_execute.run_continuous(CONTROL_INTERVAL_MINUTES)
        elif command == "start":
            # 启动后台进程
            start_background()
        elif command == "stop":
            # 停止后台进程
            stop_background()
        elif command == "restart":
            # 重启后台进程
            stop_background()
            time.sleep(1)
            start_background()
        elif command == "status":
            # 显示状态
            show_status()
        else:
            print("❌ 无效参数")
            print("用法:")
            print("  python mppi_control_real.py once")
            print("  python mppi_control_real.py continuous")
            print("  python mppi_control_real.py start      # 启动后台进程")
            print("  python mppi_control_real.py stop       # 停止后台进程")
            print("  python mppi_control_real.py restart    # 重启后台进程")
            print("  python mppi_control_real.py status     # 查看状态")
            print("")
            print("💡 提示: 修改代码顶部的宏定义来配置设备ID和其他参数")
    else:
        # 默认运行一次
        print("🔄 运行单次控制循环...")
        control_execute.run_control_cycle()

if __name__ == "__main__":
    main()
