#!/usr/bin/env python3
"""
集成MPPI控制系统
整合传感器数据获取和MPPI控制算法的主控制器
"""

import time
import sys
import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import threading
import json

# 添加路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'Test'))
sys.path.insert(0, os.path.join(project_root, 'mpc_farming_organized', 'core'))

# 导入传感器模块
try:
    from sensor_hub import get_simple_all_data, get_data_status, check_systems_ready
    SENSOR_HUB_AVAILABLE = True
    print("✅ 传感器模块加载成功")
except ImportError as e:
    SENSOR_HUB_AVAILABLE = False
    print(f"⚠️ 传感器模块不可用: {e}")

# 导入MPPI模块
try:
    from mppi_api import mppi_next_ppfd
    MPPI_AVAILABLE = True
    print("✅ MPPI模块加载成功")
except ImportError as e:
    MPPI_AVAILABLE = False
    print(f"⚠️ MPPI模块不可用: {e}")


class SimulatedDataReader:
    """仿真数据读取器 - 从现有logs读取数据模拟实时传感器"""
    
    def __init__(self, logs_dir="/Users/z5540822/Desktop/Project1/Test/logs"):
        self.logs_dir = logs_dir
        self.co2_data = None
        self.riotee_data = None
        self.current_co2_index = 0
        self.current_riotee_index = 0
        self.load_data()
    
    def load_data(self):
        """加载现有的传感器数据"""
        # 加载CO2数据
        co2_file = os.path.join(self.logs_dir, "co2_data.csv")
        if os.path.exists(co2_file):
            self.co2_data = pd.read_csv(co2_file)
            print(f"✅ 加载CO2数据: {len(self.co2_data)} 条记录")
        else:
            print(f"❌ 未找到CO2数据文件: {co2_file}")
        
        # 加载Riotee数据 - 使用30s数据文件
        riotee_file = os.path.join(self.logs_dir, "30s_20250910_083928.csv")
        if os.path.exists(riotee_file):
            # 读取时跳过第一行注释
            self.riotee_data = pd.read_csv(riotee_file, skiprows=1)
            print(f"✅ 加载Riotee数据: {len(self.riotee_data)} 条记录")
        else:
            print(f"❌ 未找到Riotee数据文件: {riotee_file}")
    
    def get_next_sensor_data(self):
        """获取下一组模拟传感器数据"""
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'co2_value': None,
            'temperature': None,
            'humidity': None,
            'data_available': False
        }
        
        # 获取CO2数据
        if self.co2_data is not None and self.current_co2_index < len(self.co2_data):
            co2_row = self.co2_data.iloc[self.current_co2_index]
            result['co2_value'] = float(co2_row['co2'])
            self.current_co2_index = (self.current_co2_index + 1) % len(self.co2_data)
        
        # 获取Riotee数据
        if self.riotee_data is not None and self.current_riotee_index < len(self.riotee_data):
            riotee_row = self.riotee_data.iloc[self.current_riotee_index]
            result['temperature'] = float(riotee_row['temperature'])
            result['humidity'] = float(riotee_row['humidity'])
            result['riotee_device_id'] = str(riotee_row['device_id'])
            
            # 添加其他Riotee数据
            result['a1_raw'] = float(riotee_row['a1_raw'])
            result['vcap_raw'] = float(riotee_row['vcap_raw'])
            
            self.current_riotee_index = (self.current_riotee_index + 1) % len(self.riotee_data)
        
        # 检查数据可用性
        result['data_available'] = (result['co2_value'] is not None and 
                                  result['temperature'] is not None and 
                                  result['humidity'] is not None)
        
        return result


class IntegratedMPPIController:
    """集成MPPI控制器"""
    
    def __init__(self, 
                 output_csv="/Users/z5540822/Desktop/Project1/mppi_control_log.csv",
                 sampling_interval=60,  # 采样间隔（秒）
                 simulation_mode=True):
        
        self.output_csv = output_csv
        self.sampling_interval = sampling_interval
        self.simulation_mode = simulation_mode
        self.running = False
        
        # 初始化数据读取器
        if simulation_mode:
            self.data_reader = SimulatedDataReader()
            print("🎯 启动仿真模式")
        else:
            self.data_reader = None
            if not SENSOR_HUB_AVAILABLE:
                raise RuntimeError("实时模式需要sensor_hub模块")
            print("🔴 启动实时模式")
        
        # MPPI状态
        self.current_ppfd = 100.0  # 初始PPFD值
        self.control_history = []
        
        # 创建CSV文件头
        self.init_csv_file()
    
    def init_csv_file(self):
        """初始化CSV输出文件"""
        headers = [
            'timestamp', 'co2_ppm', 'temperature_c', 'humidity_percent',
            'current_ppfd', 'target_ppfd', 'control_action', 'mppi_result',
            'data_source', 'control_quality'
        ]
        
        # 如果文件不存在，创建并写入头部
        if not os.path.exists(self.output_csv):
            with open(self.output_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        
        print(f"📝 CSV日志文件: {self.output_csv}")
    
    def get_sensor_data(self):
        """获取传感器数据"""
        if self.simulation_mode:
            return self.data_reader.get_next_sensor_data()
        else:
            # 实时模式
            try:
                data = get_simple_all_data(max_age_seconds=120)
                if data:
                    return {
                        'timestamp': data.get('timestamp'),
                        'co2_value': data.get('co2_value'),
                        'temperature': data.get('temperature'),
                        'humidity': data.get('humidity'),
                        'riotee_device_id': data.get('riotee_device_id'),
                        'a1_raw': data.get('a1_raw'),
                        'vcap_raw': data.get('vcap_raw'),
                        'data_available': True
                    }
                else:
                    return {'data_available': False}
            except Exception as e:
                print(f"❌ 获取实时传感器数据失败: {e}")
                return {'data_available': False}
    
    def run_mppi_control(self, sensor_data):
        """运行MPPI控制算法"""
        if not MPPI_AVAILABLE:
            return {
                'success': False,
                'error': 'MPPI模块不可用',
                'target_ppfd': self.current_ppfd
            }
        
        try:
            # 提取必要的传感器数据
            current_ppfd = self.current_ppfd
            temperature = sensor_data.get('temperature', 25.0)
            co2 = sensor_data.get('co2_value', 400.0)
            humidity = sensor_data.get('humidity', 50.0)
            
            # 调用MPPI算法
            target_ppfd = mppi_next_ppfd(
                current_ppfd=current_ppfd,
                temperature=temperature,
                co2=co2,
                humidity=humidity
            )
            
            # 更新当前PPFD状态
            self.current_ppfd = target_ppfd
            
            return {
                'success': True,
                'target_ppfd': target_ppfd,
                'control_action': 'ppfd_adjustment',
                'inputs': {
                    'current_ppfd': current_ppfd,
                    'temperature': temperature,
                    'co2': co2,
                    'humidity': humidity
                }
            }
            
        except Exception as e:
            print(f"❌ MPPI控制计算失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'target_ppfd': self.current_ppfd
            }
    
    def log_data(self, sensor_data, mppi_result):
        """记录数据到CSV"""
        try:
            row = [
                sensor_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                sensor_data.get('co2_value', ''),
                sensor_data.get('temperature', ''),
                sensor_data.get('humidity', ''),
                self.current_ppfd,
                mppi_result.get('target_ppfd', ''),
                mppi_result.get('control_action', ''),
                json.dumps(mppi_result) if mppi_result.get('success') else mppi_result.get('error', ''),
                'simulation' if self.simulation_mode else 'real',
                'good' if mppi_result.get('success') else 'error'
            ]
            
            with open(self.output_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
        except Exception as e:
            print(f"❌ 数据记录失败: {e}")
    
    def control_step(self):
        """执行一次完整的控制步骤"""
        print(f"\n🔄 执行控制步骤 - {datetime.now().strftime('%H:%M:%S')}")
        
        # 获取传感器数据
        sensor_data = self.get_sensor_data()
        
        if not sensor_data.get('data_available', False):
            print("❌ 传感器数据不可用")
            return False
        
        # 显示传感器数据
        print(f"📊 传感器数据:")
        print(f"   CO2: {sensor_data.get('co2_value', 'N/A')} ppm")
        print(f"   温度: {sensor_data.get('temperature', 'N/A')}°C")
        print(f"   湿度: {sensor_data.get('humidity', 'N/A')}%")
        
        # 运行MPPI控制
        mppi_result = self.run_mppi_control(sensor_data)
        
        if mppi_result.get('success'):
            print(f"✅ MPPI控制成功:")
            print(f"   当前PPFD: {self.current_ppfd:.1f} µmol/m²/s")
            print(f"   目标PPFD: {mppi_result['target_ppfd']:.1f} µmol/m²/s")
        else:
            print(f"❌ MPPI控制失败: {mppi_result.get('error', '未知错误')}")
        
        # 记录数据
        self.log_data(sensor_data, mppi_result)
        
        return True
    
    def start_control_loop(self, duration_minutes=None):
        """启动控制循环"""
        print(f"🚀 启动MPPI控制循环")
        print(f"📅 采样间隔: {self.sampling_interval} 秒")
        print(f"📁 输出文件: {self.output_csv}")
        
        if duration_minutes:
            print(f"⏰ 运行时长: {duration_minutes} 分钟")
        
        self.running = True
        start_time = time.time()
        step_count = 0
        
        try:
            while self.running:
                # 检查运行时长
                if duration_minutes:
                    elapsed_minutes = (time.time() - start_time) / 60
                    if elapsed_minutes >= duration_minutes:
                        print(f"⏰ 达到运行时长 {duration_minutes} 分钟，停止控制循环")
                        break
                
                # 执行控制步骤
                success = self.control_step()
                step_count += 1
                
                if success:
                    print(f"✅ 完成第 {step_count} 次控制步骤")
                else:
                    print(f"❌ 第 {step_count} 次控制步骤失败")
                
                # 等待下一个采样周期
                if self.running:
                    print(f"⏳ 等待 {self.sampling_interval} 秒...")
                    time.sleep(self.sampling_interval)
                    
        except KeyboardInterrupt:
            print("\n🛑 用户中断控制循环")
        except Exception as e:
            print(f"\n❌ 控制循环出错: {e}")
        finally:
            self.running = False
            print(f"🏁 控制循环结束，共执行 {step_count} 次控制步骤")
    
    def stop(self):
        """停止控制循环"""
        self.running = False
        print("🛑 正在停止控制循环...")


def main():
    """主函数"""
    print("🏭 集成MPPI控制系统")
    print("=" * 60)
    
    # 配置参数
    SAMPLING_INTERVAL = 2   # 每2秒采样一次（测试用）
    SIMULATION_MODE = True   # 仿真模式
    DURATION_MINUTES = 1     # 运行1分钟用于测试
    
    try:
        # 创建控制器
        controller = IntegratedMPPIController(
            sampling_interval=SAMPLING_INTERVAL,
            simulation_mode=SIMULATION_MODE
        )
        
        # 启动控制循环
        controller.start_control_loop(duration_minutes=DURATION_MINUTES)
        
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        return 1
    
    print("✅ 系统正常退出")
    return 0


if __name__ == '__main__':
    exit(main())
