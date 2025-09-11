#!/usr/bin/env python3
"""
CO2数据系统管理工具
用于启动、停止和监控CO2数据采集系统
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

class CO2System:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.serial_script = self.base_dir / "co2_data_collector.py"
        self.csv_path = self.base_dir.parent / "logs" / "co2_data.csv"
        self.pid_file = self.base_dir / "co2_collector.pid"
        
    def is_running(self):
        """检查数据采集进程是否运行"""
        if not self.pid_file.exists():
            return False
            
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # 检查进程是否存在
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            # 进程不存在，删除过时的PID文件
            self.pid_file.unlink(missing_ok=True)
            return False
    
    def start_collector(self):
        """启动数据采集器"""
        if self.is_running():
            print("✅ 数据采集器已在运行")
            return True
            
        print("🚀 启动CO2数据采集器...")
        
        try:
            # 以后台进程启动
            process = subprocess.Popen([
                sys.executable, str(self.serial_script)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 保存PID
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))
            
            # 等待一下确认启动成功
            time.sleep(2)
            
            if process.poll() is None:  # 进程仍在运行
                print(f"✅ 数据采集器启动成功 (PID: {process.pid})")
                return True
            else:
                # 进程已退出，获取错误信息
                stdout, stderr = process.communicate()
                print(f"❌ 数据采集器启动失败:")
                if stderr:
                    print(f"错误: {stderr.decode()}")
                return False
                
        except Exception as e:
            print(f"❌ 启动数据采集器时出错: {e}")
            return False
    
    def stop_collector(self):
        """停止数据采集器"""
        if not self.is_running():
            print("⏹️  数据采集器未运行")
            return True
            
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            print(f"⏹️  停止数据采集器 (PID: {pid})...")
            os.kill(pid, signal.SIGTERM)
            
            # 等待进程结束
            for _ in range(10):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.5)
                except OSError:
                    break
            
            # 如果进程仍在运行，强制终止
            try:
                os.kill(pid, 0)
                print("进程未响应，强制终止...")
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            
            self.pid_file.unlink(missing_ok=True)
            print("✅ 数据采集器已停止")
            return True
            
        except Exception as e:
            print(f"❌ 停止数据采集器时出错: {e}")
            return False
    
    def show_status(self):
        """显示系统状态"""
        print("📊 CO2数据系统状态")
        print("=" * 30)
        
        # 检查采集器状态
        if self.is_running():
            with open(self.pid_file, 'r') as f:
                pid = f.read().strip()
            print(f"🟢 数据采集器: 运行中 (PID: {pid})")
        else:
            print("🔴 数据采集器: 未运行")
        
        # 检查设备状态
        device_path = "/dev/Chamber2_Co2"
        if os.path.exists(device_path):
            link_target = os.readlink(device_path)
            print(f"🟢 CO2设备: {device_path} -> {link_target}")
        else:
            print(f"🔴 CO2设备: {device_path} 不存在")
        
        # 检查数据文件状态
        if self.csv_path.exists():
            size = self.csv_path.stat().st_size
            mtime = self.csv_path.stat().st_mtime
            last_modified = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
            print(f"🟢 数据文件: {self.csv_path} ({size} bytes, 更新: {last_modified})")
            
            # 显示最新数据
            try:
                with open(self.csv_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip()
                        print(f"📊 最新数据: {last_line}")
            except Exception as e:
                print(f"❌ 读取数据文件出错: {e}")
        else:
            print(f"🔴 数据文件: {self.csv_path} 不存在")
    
    def view_live_data(self):
        """实时查看数据"""
        from . import CO2DataReader
        
        print("📈 实时CO2数据 (按Ctrl+C退出)")
        print("=" * 40)
        
        reader = CO2DataReader(str(self.csv_path))
        
        try:
            while True:
                data = reader.get_latest_value(max_age_seconds=30)
                if data:
                    status = "🟡 数据较旧" if data['is_stale'] else "🟢"
                    print(f"{time.strftime('%H:%M:%S')} {status} CO2: {data['value']} ppm ({data['age_seconds']}秒前)")
                else:
                    print(f"{time.strftime('%H:%M:%S')} 🔴 无数据")
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n⏹️  停止实时监控")

def main():
    system = CO2System()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'start':
            system.start_collector()
        elif command == 'stop':
            system.stop_collector()
        elif command == 'restart':
            system.stop_collector()
            time.sleep(1)
            system.start_collector()
        elif command == 'status':
            system.show_status()
        elif command == 'live':
            system.view_live_data()
        else:
            print(f"❌ 未知命令: {command}")
            print("可用命令: start, stop, restart, status, live")
    else:
        # 交互模式
        while True:
            print("\n🏭 CO2数据系统管理")
            print("1. 启动数据采集器")
            print("2. 停止数据采集器") 
            print("3. 重启数据采集器")
            print("4. 查看状态")
            print("5. 实时数据")
            print("6. 退出")
            
            choice = input("请选择 (1-6): ").strip()
            
            if choice == '1':
                system.start_collector()
            elif choice == '2':
                system.stop_collector()
            elif choice == '3':
                system.stop_collector()
                time.sleep(1)
                system.start_collector()
            elif choice == '4':
                system.show_status()
            elif choice == '5':
                system.view_live_data()
            elif choice == '6':
                print("👋 再见!")
                break
            else:
                print("❌ 无效选择")

if __name__ == '__main__':
    main()
