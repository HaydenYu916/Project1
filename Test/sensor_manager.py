#!/usr/bin/env python3
"""
统一传感器管理系统
用于管理所有传感器（CO2、温度湿度、Riotee）
"""

import os
import sys
import time
import subprocess
from pathlib import Path

class SensorManager:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.sensors = {
            'co2': {
                'name': 'CO2传感器',
                'path': self.base_dir / 'co2_sensor',
                'manager': 'co2_system_manager.py',
                'icon': '🌬️'
            },
            'temp_humidity': {
                'name': '温度湿度传感器',
                'path': self.base_dir / 'temp_humidity_sensor',
                'manager': 'temp_humidity_system_manager.py',
                'icon': '🌡️'
            }
        }
    
    def run_command(self, sensor_key, command):
        """运行指定传感器的命令"""
        if sensor_key not in self.sensors:
            print(f"❌ 未知传感器: {sensor_key}")
            return False
        
        sensor = self.sensors[sensor_key]
        manager_path = sensor['path'] / sensor['manager']
        
        if not manager_path.exists():
            print(f"❌ 系统管理器不存在: {manager_path}")
            return False
        
        try:
            cmd = ['python3', str(manager_path), command]
            result = subprocess.run(cmd, cwd=str(sensor['path']), 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✅ {sensor['icon']} {sensor['name']} - {command} 成功")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                print(f"❌ {sensor['icon']} {sensor['name']} - {command} 失败")
                if result.stderr:
                    print(result.stderr)
                return False
        except subprocess.TimeoutExpired:
            print(f"⏰ {sensor['icon']} {sensor['name']} - {command} 超时")
            return False
        except Exception as e:
            print(f"❌ {sensor['icon']} {sensor['name']} - {command} 异常: {e}")
            return False
    
    def start_all(self):
        """启动所有传感器"""
        print("🚀 启动所有传感器...")
        print("=" * 50)
        
        success_count = 0
        for sensor_key in self.sensors:
            if self.run_command(sensor_key, 'start'):
                success_count += 1
            time.sleep(1)  # 避免同时启动造成冲突
        
        print(f"\n🎉 启动完成！成功启动 {success_count}/{len(self.sensors)} 个传感器")
        return success_count == len(self.sensors)
    
    def stop_all(self):
        """停止所有传感器"""
        print("⏹️ 停止所有传感器...")
        print("=" * 50)
        
        success_count = 0
        for sensor_key in self.sensors:
            if self.run_command(sensor_key, 'stop'):
                success_count += 1
            time.sleep(1)  # 避免同时停止造成冲突
        
        print(f"\n🎉 停止完成！成功停止 {success_count}/{len(self.sensors)} 个传感器")
        return success_count == len(self.sensors)
    
    def restart_all(self):
        """重启所有传感器"""
        print("🔄 重启所有传感器...")
        print("=" * 50)
        
        self.stop_all()
        time.sleep(2)
        return self.start_all()
    
    def status_all(self):
        """查看所有传感器状态"""
        print("📊 所有传感器状态")
        print("=" * 50)
        
        for sensor_key, sensor in self.sensors.items():
            print(f"\n{sensor['icon']} {sensor['name']}:")
            print("-" * 30)
            self.run_command(sensor_key, 'status')
    
    def start_sensor(self, sensor_key):
        """启动指定传感器"""
        if sensor_key not in self.sensors:
            print(f"❌ 未知传感器: {sensor_key}")
            print(f"可用传感器: {', '.join(self.sensors.keys())}")
            return False
        
        return self.run_command(sensor_key, 'start')
    
    def stop_sensor(self, sensor_key):
        """停止指定传感器"""
        if sensor_key not in self.sensors:
            print(f"❌ 未知传感器: {sensor_key}")
            print(f"可用传感器: {', '.join(self.sensors.keys())}")
            return False
        
        return self.run_command(sensor_key, 'stop')
    
    def restart_sensor(self, sensor_key):
        """重启指定传感器"""
        if sensor_key not in self.sensors:
            print(f"❌ 未知传感器: {sensor_key}")
            print(f"可用传感器: {', '.join(self.sensors.keys())}")
            return False
        
        print(f"🔄 重启 {self.sensors[sensor_key]['name']}...")
        self.stop_sensor(sensor_key)
        time.sleep(1)
        return self.start_sensor(sensor_key)
    
    def status_sensor(self, sensor_key):
        """查看指定传感器状态"""
        if sensor_key not in self.sensors:
            print(f"❌ 未知传感器: {sensor_key}")
            print(f"可用传感器: {', '.join(self.sensors.keys())}")
            return False
        
        return self.run_command(sensor_key, 'status')
    
    def show_menu(self):
        """显示主菜单"""
        print("\n🏭 统一传感器管理系统")
        print("=" * 40)
        print("1. 启动所有传感器")
        print("2. 停止所有传感器")
        print("3. 重启所有传感器")
        print("4. 查看所有传感器状态")
        print("5. 管理单个传感器")
        print("6. 退出")
    
    def show_sensor_menu(self):
        """显示传感器选择菜单"""
        print("\n📡 选择传感器:")
        print("=" * 30)
        for i, (key, sensor) in enumerate(self.sensors.items(), 1):
            print(f"{i}. {sensor['icon']} {sensor['name']} ({key})")
        print(f"{len(self.sensors) + 1}. 返回主菜单")
    
    def run_interactive(self):
        """运行交互模式"""
        while True:
            self.show_menu()
            choice = input("请选择 (1-6): ").strip()
            
            if choice == '1':
                self.start_all()
            elif choice == '2':
                self.stop_all()
            elif choice == '3':
                self.restart_all()
            elif choice == '4':
                self.status_all()
            elif choice == '5':
                self.manage_single_sensor()
            elif choice == '6':
                print("👋 再见!")
                break
            else:
                print("❌ 无效选择")
    
    def manage_single_sensor(self):
        """管理单个传感器"""
        while True:
            self.show_sensor_menu()
            choice = input("请选择传感器 (1-{}): ".format(len(self.sensors) + 1)).strip()
            
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(self.sensors):
                    sensor_key = list(self.sensors.keys())[choice_num - 1]
                    self.sensor_operations(sensor_key)
                elif choice_num == len(self.sensors) + 1:
                    break
                else:
                    print("❌ 无效选择")
            except ValueError:
                print("❌ 请输入数字")
    
    def sensor_operations(self, sensor_key):
        """单个传感器操作菜单"""
        sensor = self.sensors[sensor_key]
        
        while True:
            print(f"\n{sensor['icon']} {sensor['name']} 操作:")
            print("-" * 30)
            print("1. 启动")
            print("2. 停止")
            print("3. 重启")
            print("4. 查看状态")
            print("5. 返回传感器选择")
            
            choice = input("请选择 (1-5): ").strip()
            
            if choice == '1':
                self.start_sensor(sensor_key)
            elif choice == '2':
                self.stop_sensor(sensor_key)
            elif choice == '3':
                self.restart_sensor(sensor_key)
            elif choice == '4':
                self.status_sensor(sensor_key)
            elif choice == '5':
                break
            else:
                print("❌ 无效选择")

def main():
    print("=" * 60)
    print("🏭 统一传感器管理系统启动")
    print("=" * 60)
    
    manager = SensorManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'start-all':
            manager.start_all()
        elif command == 'stop-all':
            manager.stop_all()
        elif command == 'restart-all':
            manager.restart_all()
        elif command == 'status-all':
            manager.status_all()
        elif command.startswith('start-'):
            sensor_key = command[6:]
            manager.start_sensor(sensor_key)
        elif command.startswith('stop-'):
            sensor_key = command[5:]
            manager.stop_sensor(sensor_key)
        elif command.startswith('restart-'):
            sensor_key = command[8:]
            manager.restart_sensor(sensor_key)
        elif command.startswith('status-'):
            sensor_key = command[7:]
            manager.status_sensor(sensor_key)
        else:
            print(f"❌ 未知命令: {command}")
            print("可用命令:")
            print("  start-all, stop-all, restart-all, status-all")
            print("  start-<sensor>, stop-<sensor>, restart-<sensor>, status-<sensor>")
            print("  传感器: co2, temp_humidity")
    else:
        # 交互模式
        manager.run_interactive()

if __name__ == '__main__':
    main()
