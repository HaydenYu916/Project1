#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PWM调度器服务管理脚本
提供简单的启动、停止、重启、状态检查功能
"""

import sys
import os
import subprocess
import time
import signal
from datetime import datetime

class PWMServiceManager:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.scheduler_script = os.path.join(self.script_dir, "pwm_scheduler.py")
        self.csv_file = os.path.join(self.script_dir, "src", "extended_schedule_20250919_071157_20250919_071157.csv")
        self.pid_file = os.path.join(self.script_dir, "pwm_scheduler.pid")
        self.log_file = os.path.join(self.script_dir, "pwm_scheduler.log")
        self.riotee_data_dir = "/home/pi/Desktop/Test/riotee_sensor/logs"
    
    def start(self, daemon=True):
        """启动调度器"""
        print("启动PWM调度器服务...")
        
        # 检查是否已经在运行
        if self.is_running():
            print("调度器已经在运行中")
            return False
        
        # 检查CSV文件
        if not os.path.exists(self.csv_file):
            print(f"错误: CSV文件不存在 - {self.csv_file}")
            return False
        
        # 构建命令
        cmd = [sys.executable, self.scheduler_script, self.csv_file, '-r', self.riotee_data_dir]
        if daemon:
            cmd.append('-d')
        
        try:
            if daemon:
                # 后台运行
                subprocess.Popen(cmd, cwd=self.script_dir)
                time.sleep(2)  # 等待启动
                
                if self.is_running():
                    print("✓ 调度器已成功启动 (后台模式)")
                    print(f"  PID文件: {self.pid_file}")
                    print(f"  日志文件: {self.log_file}")
                    return True
                else:
                    print("✗ 调度器启动失败")
                    return False
            else:
                # 前台运行
                print("前台运行模式...")
                subprocess.run(cmd, cwd=self.script_dir)
                return True
                
        except Exception as e:
            print(f"启动失败: {e}")
            return False
    
    def stop(self):
        """停止调度器"""
        print("停止PWM调度器服务...")
        
        if not self.is_running():
            print("调度器未运行")
            return True
        
        try:
            # 使用调度器脚本的停止功能
            cmd = [sys.executable, self.scheduler_script, "--stop"]
            result = subprocess.run(cmd, cwd=self.script_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ 调度器已停止")
                print(result.stdout)
                return True
            else:
                print(f"✗ 停止失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"停止失败: {e}")
            return False
    
    def restart(self, daemon=True):
        """重启调度器"""
        print("重启PWM调度器服务...")
        
        if self.is_running():
            self.stop()
            time.sleep(3)
        
        return self.start(daemon)
    
    def status(self):
        """检查调度器状态"""
        print("检查PWM调度器状态...")
        
        if not self.is_running():
            print("✗ 调度器未运行")
            return False
        
        try:
            # 使用调度器脚本的状态检查功能
            cmd = [sys.executable, self.scheduler_script, "--status"]
            result = subprocess.run(cmd, cwd=self.script_dir, capture_output=True, text=True)
            
            print("✓ 调度器状态:")
            print(result.stdout)
            
            if result.stderr:
                print("错误信息:")
                print(result.stderr)
            
            return True
            
        except Exception as e:
            print(f"检查状态失败: {e}")
            return False
    
    def is_running(self):
        """检查是否正在运行"""
        if not os.path.exists(self.pid_file):
            return False
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # 检查进程是否存在
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            # 进程不存在或PID文件无效
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
            return False
    
    def show_logs(self, lines=20):
        """显示日志"""
        if not os.path.exists(self.log_file):
            print("日志文件不存在")
            return
        
        print(f"显示最近 {lines} 行日志:")
        print("-" * 50)
        
        try:
            with open(self.log_file, 'r') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:]
                for line in recent_lines:
                    print(line.strip())
        except Exception as e:
            print(f"读取日志失败: {e}")
    
    def show_help(self):
        """显示帮助信息"""
        print("""
PWM调度器服务管理工具

使用方法:
  python3 pwm_service.py <command> [options]

命令:
  start [--foreground]  启动调度器 (默认后台运行)
  stop                 停止调度器
  restart [--foreground] 重启调度器
  status               检查运行状态
  logs [--lines N]     显示日志 (默认20行)
  help                 显示此帮助信息

选项:
  --foreground         前台运行模式
  --lines N           显示日志行数

示例:
  python3 pwm_service.py start              # 后台启动
  python3 pwm_service.py start --foreground # 前台启动
  python3 pwm_service.py status             # 检查状态
  python3 pwm_service.py logs --lines 50    # 显示最近50行日志
  python3 pwm_service.py restart            # 重启服务
        """)

def main():
    manager = PWMServiceManager()
    
    if len(sys.argv) < 2:
        manager.show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "start":
        daemon = "--foreground" not in sys.argv
        manager.start(daemon)
    
    elif command == "stop":
        manager.stop()
    
    elif command == "restart":
        daemon = "--foreground" not in sys.argv
        manager.restart(daemon)
    
    elif command == "status":
        manager.status()
    
    elif command == "logs":
        lines = 20
        if "--lines" in sys.argv:
            try:
                idx = sys.argv.index("--lines")
                lines = int(sys.argv[idx + 1])
            except (ValueError, IndexError):
                print("错误: --lines 参数需要指定数字")
                return
        
        manager.show_logs(lines)
    
    elif command == "help":
        manager.show_help()
    
    else:
        print(f"未知命令: {command}")
        manager.show_help()

if __name__ == "__main__":
    main()
