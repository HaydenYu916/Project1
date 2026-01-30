#!/usr/bin/env python3
"""
Riotee数据系统管理工具
用于启动、停止和监控Riotee数据采集系统
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

class RioteeSystem:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.collector_script = self.base_dir / "riotee_data_collector.py"
        self.logs_path = self.base_dir / "logs"  # 使用本地logs目录
        self.pid_file = self.base_dir / "riotee_collector.pid"
        self.log_file = self.base_dir / "riotee_system_manager.log"
        self.print_output_file = self.base_dir / "riotee_collector_print_output.txt"  # 采集器print输出记录
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
        
        logging.info(f"Riotee系统管理器日志已初始化: {self.log_file}")
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
    
    def start_collector(self, experiment_name=None):
        """启动数据采集器"""
        if self.is_running():
            msg = "✅ Riotee数据采集器已在运行"
            print(msg)
            logging.info("尝试启动采集器，但已在运行中")
            return True
            
        msg = "🚀 启动Riotee数据采集器..."
        print(msg)
        logging.info(f"开始启动Riotee数据采集器，实验名称: {experiment_name}")
        
        try:
            # 构建启动命令 - 优先使用 riotee 虚拟环境，否则用当前 Python
            riotee_python = "/home/pi/Desktop/riotee-env/bin/python3"
            if not os.path.exists(riotee_python):
                riotee_python = sys.executable
            cmd = [riotee_python, str(self.collector_script)]
            if experiment_name:
                cmd.append(experiment_name)
                cmd.append(f"Riotee数据采集实验-{experiment_name}")
                logging.info(f"启动命令: {' '.join(cmd)}")
            
            # 创建print输出文件，用于记录采集器的输出
            with open(self.print_output_file, 'w', encoding='utf-8') as output_file:
                output_file.write(f"# Riotee采集器启动 @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                output_file.write(f"# 启动命令: {' '.join(cmd)}\n")
                output_file.write("# " + "="*50 + "\n")
                output_file.flush()
                
                # 以后台进程启动，将stdout重定向到文件（无缓冲以便立即看到错误）
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                process = subprocess.Popen(
                    cmd,
                    stdout=output_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.base_dir),
                    text=True,
                    env=env
                )
            
            # 保存PID
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))
            logging.info(f"采集器进程已启动，PID: {process.pid}，PID文件: {self.pid_file}")
            print(f"📄 采集器Print输出记录: {self.print_output_file}")
            
            # 等待一下确认启动成功
            time.sleep(3)
            
            if process.poll() is None:  # 进程仍在运行
                success_msg = f"✅ Riotee数据采集器启动成功 (PID: {process.pid})"
                print(success_msg)
                logging.info(f"采集器启动成功，进程PID: {process.pid}")
                return True
            else:
                # 进程已退出，读取输出文件查看错误信息
                error_msg = "❌ Riotee数据采集器启动失败:"
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
            error_msg = f"❌ 启动Riotee数据采集器时出错: {e}"
            print(error_msg)
            logging.error(f"启动采集器异常: {e}")
            return False
    
    def daily_cleanup_print_output(self):
        """每日清理采集器 Print 输出文件。采集器未运行时才执行，可配合 cron 每日执行。"""
        if self.is_running():
            msg = "⏭️  采集器运行中，跳过每日清理（避免影响输出）。请先 stop 后再清理，或由 cron 在停机时段执行。"
            print(msg)
            logging.info("每日清理：采集器运行中，跳过")
            return False
        if not self.print_output_file.exists():
            msg = "📄 采集器 Print 输出文件不存在，无需清理"
            print(msg)
            logging.info("每日清理：文件不存在，跳过")
            return True
        try:
            self.print_output_file.write_text(
                f"# 每日清理 @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n# 已清空历史输出\n",
                encoding="utf-8",
            )
            msg = f"✅ 已每日清理: {self.print_output_file.name}"
            print(msg)
            logging.info(f"每日清理完成: {self.print_output_file}")
            return True
        except Exception as e:
            msg = f"❌ 每日清理失败: {e}"
            print(msg)
            logging.error(f"每日清理异常: {e}")
            return False

    def cleanup_riotee_files(self):
        """清理Riotee系统相关文件"""
        print("🧹 清理Riotee系统文件...")
        cleaned_count = 0
        
        # 定义Riotee系统要清理的文件
        riotee_files = [
            self.log_file,
            self.pid_file,
            self.print_output_file,
        ]
        
        for file_path in riotee_files:
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"✅ 已清理: {file_path.name}")
                    cleaned_count += 1
                except Exception as e:
                    print(f"❌ 清理失败: {file_path.name} - {e}")
        
        print(f"🎉 Riotee清理完成！共清理了 {cleaned_count} 个文件")
        self.files_cleaned = True  # 标记已清理，防止atexit再次清理
        return cleaned_count > 0

    def stop_collector(self):
        """停止数据采集器并清理所有文件"""
        stopped = False
        
        if not self.is_running():
            msg = "⏹️  Riotee数据采集器未运行"
            print(msg)
            logging.info("尝试停止采集器，但未运行")
            stopped = True
        else:
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                msg = f"⏹️  停止Riotee数据采集器 (PID: {pid})..."
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
                
                success_msg = "✅ Riotee数据采集器已停止"
                print(success_msg)
                logging.info("采集器已成功停止")
                stopped = True
                
            except Exception as e:
                error_msg = f"❌ 停止Riotee数据采集器时出错: {e}"
                print(error_msg)
                logging.error(f"停止采集器异常: {e}")
        
        # 无论采集器是否在运行，都清理Riotee系统文件
        self.cleanup_riotee_files()
        return stopped
    
    def show_status(self):
        """显示系统状态"""
        print("📊 Riotee数据系统状态")
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
                        # 跳过注释行
                        first_line = f.readline()
                        if first_line.startswith('#'):
                            reader = csv.DictReader(f)
                        else:
                            f.seek(0)
                            reader = csv.DictReader(f)
                        
                        rows = list(reader)
                        if rows:
                            last_row = rows[-1]
                            device_id = last_row.get('device_id', 'N/A')
                            timestamp = last_row.get('timestamp', 'N/A')
                            temp = last_row.get('temperature', 'N/A')
                            print(f"📊 最新数据: 设备{device_id}, T={temp}°C, 时间={timestamp}")
                except Exception as e:
                    print(f"❌ 读取数据文件出错: {e}")
            else:
                print(f"🔴 数据文件: {self.logs_path} 目录为空")
        else:
            print(f"🔴 数据目录: {self.logs_path} 不存在")
    
    def view_live_data(self):
        """实时查看数据"""
        from . import get_current_riotee
        
        print("📈 实时Riotee数据 (按Ctrl+C退出)")
        print("=" * 40)
        
        try:
            while True:
                data = get_current_riotee(max_age_seconds=60)
                if data:
                    device = data.get('device_id', 'N/A')
                    temp = data.get('temperature', 'N/A')
                    humidity = data.get('humidity', 'N/A')
                    age = data.get('_data_age_seconds', 0)
                    status = "🟢" if not data.get('_is_stale', True) else "🟡"
                    print(f"{time.strftime('%H:%M:%S')} {status} 设备{device}: T={temp}°C, H={humidity}%, ({age:.0f}秒前)")
                else:
                    print(f"{time.strftime('%H:%M:%S')} 🔴 无数据")
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n⏹️  停止实时监控")

def main():
    print("=" * 60)
    print("🏭 Riotee数据系统管理器启动")
    print("=" * 60)
    
    system = RioteeSystem()
    
    print(f"📄 系统管理日志: {system.log_file}")
    print(f"📄 采集器PID文件: {system.pid_file}")
    print(f"📄 采集器Print输出记录: {system.print_output_file}")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'start':
            experiment_name = sys.argv[2] if len(sys.argv) > 2 else None
            system.start_collector(experiment_name)
        elif command == 'stop':
            system.stop_collector()
        elif command == 'daily-cleanup':
            system.daily_cleanup_print_output()
        elif command == 'restart':
            system.stop_collector()
            time.sleep(1)
            experiment_name = sys.argv[2] if len(sys.argv) > 2 else None
            system.start_collector(experiment_name)
        elif command == 'status':
            system.show_status()
        elif command == 'live':
            system.view_live_data()
        else:
            print(f"❌ 未知命令: {command}")
            print("可用命令: start [experiment_name], stop, restart [experiment_name], status, live, daily-cleanup")
        
        # 命令行模式完成后记录
        if command != 'stop':  # stop命令的日志文件已被清理，不要再记录
            logging.info(f"命令 '{command}' 执行完成")
            system.cleanup_logs(clean_log_file=False, clean_print_output=False)
        return
    else:
        # 交互模式
        while True:
            print("\n🏭 Riotee数据系统管理")
            print("1. 启动数据采集器")
            print("2. 停止数据采集器") 
            print("3. 重启数据采集器")
            print("4. 查看状态")
            print("5. 实时数据")
            print("6. 退出")
            
            choice = input("请选择 (1-6): ").strip()
            
            if choice == '1':
                experiment_name = input("实验名称 (可选): ").strip() or None
                system.start_collector(experiment_name)
            elif choice == '2':
                system.stop_collector()
            elif choice == '3':
                system.stop_collector()
                time.sleep(1)
                experiment_name = input("实验名称 (可选): ").strip() or None
                system.start_collector(experiment_name)
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
