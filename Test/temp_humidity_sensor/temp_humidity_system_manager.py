#!/usr/bin/env python3
"""
温度湿度传感器数据系统管理工具
用于启动、停止和监控温度湿度数据采集系统
"""

import os
import sys
import time
import signal
import subprocess
import logging
import atexit
from pathlib import Path
from datetime import datetime

class TempHumiditySystem:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.collector_script = self.base_dir / "collector.py"
        self.logs_path = self.base_dir / "logs"  # 使用本地logs目录
        self.pid_file = self.base_dir / "temp_humidity_collector.pid"
        self.log_file = self.base_dir / "temp_humidity_system_manager.log"
        self.print_output_file = self.base_dir / "temp_humidity_collector_print_output.txt"  # 采集器print输出记录
        self.files_cleaned = False  # 标记是否已经清理过文件
        
        # 设置日志
        self.setup_logging()
        
        # 注册清理函数 - 默认保留日志文件（仅在程序异常退出时调用）
        atexit.register(self.atexit_cleanup)
    
    def atexit_cleanup(self):
        """程序退出时的清理函数，只在未手动清理时执行"""
        if not self.files_cleaned:
            self.cleanup_logs(clean_log_file=False, clean_print_output=False)
    
    def cleanup_logs(self, clean_log_file=True, clean_print_output=False):
        """程序退出时清理日志文件"""
        try:
            # 只在要求时清理print输出文件
            if clean_print_output and self.print_output_file.exists():
                self.print_output_file.unlink()
                print(f"✅ 采集器Print输出文件已清理: {self.print_output_file}")
            elif not clean_print_output and self.print_output_file.exists():
                print(f"📄 采集器Print输出记录保存在: {self.print_output_file}")
            
            if clean_log_file and self.log_file.exists():
                self.log_file.unlink()
                print(f"✅ 系统管理日志已清理: {self.log_file}")
            elif not clean_log_file:
                print(f"📄 系统管理日志保存在: {self.log_file}")
        except Exception as e:
            print(f"⚠️ 清理日志文件时出错: {e}")
    
    def setup_logging(self):
        """设置日志配置"""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # 清除默认处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 创建格式器
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logging.info(f"温度湿度系统管理器日志已初始化: {self.log_file}")
        return logger
        
    def is_running(self):
        """检查数据采集进程是否运行"""
        if not self.pid_file.exists():
            logging.debug("PID文件不存在，采集器未运行")
            return False
            
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # 检查进程是否存在
            os.kill(pid, 0)
            logging.debug(f"采集器正在运行 (PID: {pid})")
            return True
        except (OSError, ValueError) as e:
            # 进程不存在，删除过时的PID文件
            logging.debug(f"进程不存在或PID文件无效，清理PID文件: {e}")
            self.pid_file.unlink(missing_ok=True)
            return False
    
    def start_collector(self):
        """启动温度湿度数据采集器"""
        if self.is_running():
            msg = "✅ 温度湿度数据采集器已在运行"
            print(msg)
            logging.info("尝试启动采集器，但已在运行中")
            return True
            
        msg = "🚀 启动温度湿度数据采集器..."
        print(msg)
        logging.info("开始启动温度湿度数据采集器")
        
        try:
            # 构建启动命令
            cmd = ["python3", str(self.collector_script)]
            logging.info(f"启动命令: {' '.join(cmd)}")
            
            # 创建print输出文件，用于记录采集器的输出
            with open(self.print_output_file, 'w', encoding='utf-8') as output_file:
                output_file.write(f"# 温度湿度采集器启动 @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                output_file.write(f"# 启动命令: {' '.join(cmd)}\n")
                output_file.write("# " + "="*50 + "\n")
                output_file.flush()
                
                # 以后台进程启动，将stdout重定向到文件
                process = subprocess.Popen(
                    cmd,
                    stdout=output_file, 
                    stderr=subprocess.STDOUT,  # 将stderr也重定向到stdout
                    cwd=str(self.base_dir),
                    text=True
                )
            
            # 保存PID
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))
            logging.info(f"采集器进程已启动，PID: {process.pid}，PID文件: {self.pid_file}")
            print(f"📄 采集器Print输出记录: {self.print_output_file}")
            
            # 等待一下确认启动成功
            time.sleep(3)
            
            if process.poll() is None:  # 进程仍在运行
                success_msg = f"✅ 温度湿度数据采集器启动成功 (PID: {process.pid})"
                print(success_msg)
                logging.info(f"采集器启动成功，进程PID: {process.pid}")
                return True
            else:
                # 进程已退出，读取输出文件查看错误信息
                error_msg = "❌ 温度湿度数据采集器启动失败:"
                print(error_msg)
                logging.error("采集器启动失败，进程已退出")
                
                # 读取print输出文件中的错误信息
                try:
                    with open(self.print_output_file, 'r', encoding='utf-8') as f:
                        output_content = f.read()
                        if output_content.strip():
                            print("采集器输出:")
                            print(output_content)
                            logging.error(f"采集器输出: {output_content}")
                except Exception as e:
                    logging.error(f"读取采集器输出文件失败: {e}")
                
                return False
                
        except Exception as e:
            error_msg = f"❌ 启动温度湿度数据采集器时出错: {e}"
            print(error_msg)
            logging.error(f"启动采集器异常: {e}")
            return False
    
    def cleanup_temp_humidity_files(self):
        """清理温度湿度系统相关文件"""
        print("🧹 清理温度湿度系统文件...")
        cleaned_count = 0
        
        # 定义温度湿度系统要清理的文件
        temp_humidity_files = [
            self.log_file,
            self.pid_file,
            self.print_output_file,
        ]
        
        for file_path in temp_humidity_files:
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"✅ 已清理: {file_path.name}")
                    cleaned_count += 1
                except Exception as e:
                    print(f"❌ 清理失败: {file_path.name} - {e}")
        
        print(f"🎉 温度湿度清理完成！共清理了 {cleaned_count} 个文件")
        self.files_cleaned = True  # 标记已清理，防止atexit再次清理
        return cleaned_count > 0

    def stop_collector(self):
        """停止数据采集器并清理所有文件"""
        stopped = False
        
        if not self.is_running():
            msg = "⏹️  温度湿度数据采集器未运行"
            print(msg)
            logging.info("尝试停止采集器，但未运行")
            stopped = True
        else:
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                msg = f"⏹️  停止温度湿度数据采集器 (PID: {pid})..."
                print(msg)
                logging.info(f"开始停止采集器，PID: {pid}")
                os.kill(pid, signal.SIGTERM)
                
                # 等待进程结束
                for i in range(10):
                    try:
                        os.kill(pid, 0)
                        time.sleep(0.5)
                        logging.debug(f"等待进程结束... ({i+1}/10)")
                    except OSError:
                        logging.info("进程已正常结束")
                        break
                
                # 如果进程仍在运行，强制终止
                try:
                    os.kill(pid, 0)
                    print("进程未响应，强制终止...")
                    logging.warning("进程未响应SIGTERM，发送SIGKILL强制终止")
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                
                success_msg = "✅ 温度湿度数据采集器已停止"
                print(success_msg)
                logging.info("采集器已成功停止")
                stopped = True
                
            except Exception as e:
                error_msg = f"❌ 停止温度湿度数据采集器时出错: {e}"
                print(error_msg)
                logging.error(f"停止采集器异常: {e}")
        
        # 无论采集器是否在运行，都清理温度湿度系统文件
        self.cleanup_temp_humidity_files()
        return stopped
    
    def show_status(self):
        """显示系统状态"""
        print("📊 温度湿度数据系统状态")
        print("=" * 30)
        
        # 检查采集器状态
        if self.is_running():
            with open(self.pid_file, 'r') as f:
                pid = f.read().strip()
            print(f"🟢 数据采集器: 运行中 (PID: {pid})")
        else:
            print("🔴 数据采集器: 未运行")
        
        # 检查logs目录状态
        if self.logs_path.exists():
            # 查找最新的CSV文件
            import glob
            csv_files = list(glob.glob(str(self.logs_path / "*.csv")))
            if csv_files:
                # 按修改时间排序
                csv_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                latest_csv = csv_files[0]
                size = os.path.getsize(latest_csv)
                mtime = os.path.getmtime(latest_csv)
                last_modified = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                print(f"🟢 最新数据文件: {os.path.basename(latest_csv)}")
                print(f"   大小: {size} bytes, 更新: {last_modified}")
                
                # 显示最新数据
                try:
                    import csv
                    with open(latest_csv, 'r') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows:
                            last_row = rows[-1]
                            timestamp = last_row.get('timestamp', 'N/A')
                            temperature = last_row.get('temperature', 'N/A')
                            humidity = last_row.get('humidity', 'N/A')
                            print(f"📊 最新数据: 温度={temperature}°C, 湿度={humidity}%, 时间={timestamp}")
                except Exception as e:
                    print(f"❌ 读取数据文件出错: {e}")
            else:
                print(f"🔴 数据文件: {self.logs_path} 目录为空")
        else:
            print(f"🔴 数据目录: {self.logs_path} 不存在")
    
    def view_live_data(self):
        """实时查看数据"""
        print("📈 实时温度湿度数据 (按Ctrl+C退出)")
        print("=" * 40)
        
        try:
            while True:
                # 读取最新的CSV数据
                csv_file = self.logs_path / "temp_humidity_data.csv"
                if csv_file.exists():
                    try:
                        import csv
                        with open(csv_file, 'r') as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                            if rows:
                                last_row = rows[-1]
                                timestamp = last_row.get('timestamp', 'N/A')
                                temperature = last_row.get('temperature', 'N/A')
                                humidity = last_row.get('humidity', 'N/A')
                                print(f"{time.strftime('%H:%M:%S')} 🟢 温度={temperature}°C, 湿度={humidity}%, 时间={timestamp}")
                            else:
                                print(f"{time.strftime('%H:%M:%S')} 🔴 无数据")
                    except Exception as e:
                        print(f"{time.strftime('%H:%M:%S')} ❌ 读取数据出错: {e}")
                else:
                    print(f"{time.strftime('%H:%M:%S')} 🔴 数据文件不存在")
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n⏹️  停止实时监控")

def main():
    print("=" * 60)
    print("🌡️ 温度湿度传感器数据系统管理器启动")
    print("=" * 60)
    
    system = TempHumiditySystem()
    
    print(f"📄 系统管理日志: {system.log_file}")
    print(f"📄 采集器PID文件: {system.pid_file}")
    print(f"📄 采集器Print输出记录: {system.print_output_file}")
    print("=" * 60)
    
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
        
        # 命令行模式完成后记录
        if command != 'stop':  # stop命令的日志文件已被清理，不要再记录
            logging.info(f"命令 '{command}' 执行完成")
            system.cleanup_logs(clean_log_file=False, clean_print_output=False)
        return
    else:
        # 交互模式
        while True:
            print("\n🌡️ 温度湿度传感器数据系统管理")
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
                logging.info("用户退出系统管理器")
                break
            else:
                print("❌ 无效选择")
    
    # 交互模式退出时不额外清理，因为stop命令已经清理过了
    print("👋 程序退出")
    logging.info("交互模式正常退出")

if __name__ == '__main__':
    main()

