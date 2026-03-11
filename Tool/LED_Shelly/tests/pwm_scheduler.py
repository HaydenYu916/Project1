#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PWM调度器 - 根据CSV时间表控制Shelly设备的PWM值
支持前台和后台运行模式
"""

import sys
import os
import csv
import time
import subprocess
import threading
import signal
import logging
import argparse
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# 添加父目录到路径，以便导入shelly_controller
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PWMScheduler:
    def __init__(self, csv_file: str, daemon_mode: bool = False, log_file: str = None, riotee_data_dir: str = None):
        self.csv_file = csv_file
        self.schedule_data = []
        self.current_index = 0
        self.is_running = False
        self.daemon_mode = daemon_mode
        self.log_file = log_file or os.path.join(os.path.dirname(os.path.abspath(__file__)), "pwm_scheduler.log")
        self.pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pwm_scheduler.pid")
        self.controller_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shelly_controller.py")
        
        # Riotee数据收集相关
        self.riotee_data_dir = riotee_data_dir or "/home/pi/Desktop/Test/riotee_sensor/logs"
        self.current_ppfd_data = []  # 当前PPFD时间段的数据
        self.current_ppfd_value = None
        self.current_phase_name = None
        self.data_collection_start_time = None
        
        # 设置日志
        self.setup_logging()
        
        # 设置信号处理
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def setup_logging(self):
        """设置日志记录"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        if self.daemon_mode:
            # 后台模式：只记录到文件
            logging.basicConfig(
                filename=self.log_file,
                level=logging.INFO,
                format=log_format,
                filemode='a'
            )
        else:
            # 前台模式：同时输出到控制台和文件
            logging.basicConfig(
                level=logging.INFO,
                format=log_format,
                handlers=[
                    logging.FileHandler(self.log_file, mode='a'),
                    logging.StreamHandler(sys.stdout)
                ]
            )
    
    def signal_handler(self, signum, frame):
        """信号处理器"""
        self.log(f"收到信号 {signum}，正在停止调度器...")
        self.is_running = False
        self.cleanup()
    
    def log(self, message, level=logging.INFO):
        """记录日志"""
        if level == logging.INFO:
            logging.info(message)
        elif level == logging.ERROR:
            logging.error(message)
        elif level == logging.WARNING:
            logging.warning(message)
        elif level == logging.DEBUG:
            logging.debug(message)
    
    def write_pid(self):
        """写入PID文件"""
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            self.log(f"PID文件已写入: {self.pid_file}")
        except Exception as e:
            self.log(f"写入PID文件失败: {e}", logging.ERROR)
    
    def remove_pid(self):
        """删除PID文件"""
        try:
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
                self.log("PID文件已删除")
        except Exception as e:
            self.log(f"删除PID文件失败: {e}", logging.ERROR)
    
    def is_already_running(self):
        """检查是否已经在运行"""
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
            self.remove_pid()
            return False
    
    def cleanup(self):
        """清理资源"""
        self.remove_pid()
        self.log("调度器已清理完成")
        
    def load_schedule(self):
        """加载CSV时间表数据"""
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 解析时间
                    time_str = row['time']
                    dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                    
                    # 获取PWM值和PPFD值
                    r_pwm = int(row['r_pwm'])
                    b_pwm = int(row['b_pwm'])
                    ppfd = int(row['ppfd'])
                    
                    self.schedule_data.append({
                        'time': dt,
                        'time_str': time_str,
                        'ppfd': ppfd,
                        'r_pwm': r_pwm,
                        'b_pwm': b_pwm,
                        'phase': row['phase'],
                        'phase_name': row['phase_name'],
                        'cycle': int(row['cycle'])
                    })
            
            self.log(f"已加载 {len(self.schedule_data)} 个时间点")
            return True
            
        except Exception as e:
            self.log(f"加载CSV文件失败: {e}", logging.ERROR)
            return False
    
    def execute_controller_command(self, device: str, action: str, *args) -> Dict:
        """执行shelly_controller命令"""
        try:
            cmd = ['python3', self.controller_path, device, action] + list(map(str, args))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # 尝试解析JSON响应
                try:
                    import json
                    return json.loads(result.stdout.strip())
                except:
                    return {'status': 'success', 'output': result.stdout.strip()}
            else:
                return {'status': 'error', 'error': result.stderr}
                
        except subprocess.TimeoutExpired:
            return {'status': 'error', 'error': '命令执行超时'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def set_pwm_values(self, r_pwm: int, b_pwm: int) -> bool:
        """设置红蓝PWM值"""
        self.log(f"设置PWM值: Red={r_pwm}, Blue={b_pwm}")
        
        # 设置红色设备
        red_result = self.execute_controller_command("Red", "brightness", r_pwm)
        if red_result.get('status') == 'error':
            self.log(f"设置红色PWM失败: {red_result.get('error')}", logging.ERROR)
            return False
        
        # 设置蓝色设备
        blue_result = self.execute_controller_command("Blue", "brightness", b_pwm)
        if blue_result.get('status') == 'error':
            self.log(f"设置蓝色PWM失败: {blue_result.get('error')}", logging.ERROR)
            return False
        
        self.log("PWM设置成功")
        return True
    
    def check_device_status(self) -> bool:
        """检查设备状态"""
        self.log("检查设备状态...")
        
        # 检查红色设备
        red_status = self.execute_controller_command("Red", "get_status")
        if red_status.get('status') == 'error':
            self.log(f"检查红色设备状态失败: {red_status.get('error')}", logging.ERROR)
            return False
        
        # 检查蓝色设备
        blue_status = self.execute_controller_command("Blue", "get_status")
        if blue_status.get('status') == 'error':
            self.log(f"检查蓝色设备状态失败: {blue_status.get('error')}", logging.ERROR)
            return False
        
        self.log("设备状态检查完成")
        self.log(f"红色设备: {red_status}")
        self.log(f"蓝色设备: {blue_status}")
        return True
    
    def find_next_schedule_index(self) -> int:
        """找到下一个要执行的时间点索引"""
        now = datetime.now()
        
        for i, schedule in enumerate(self.schedule_data):
            if schedule['time'] > now:
                return i
        
        # 如果没有找到未来的时间点，返回-1表示结束
        return -1
    
    def wait_for_next_schedule(self, target_time: datetime) -> bool:
        """等待到下一个时间点，期间定期更新当前时间段数据"""
        now = datetime.now()
        wait_seconds = (target_time - now).total_seconds()
        
        if wait_seconds <= 0:
            return True  # 时间已到
        
        self.log(f"等待 {wait_seconds:.1f} 秒到 {target_time.strftime('%H:%M:%S')}")
        
        # 分段等待，每30秒更新一次当前时间段数据
        update_interval = 30  # 30秒更新一次
        last_update = now
        
        while wait_seconds > 0 and self.is_running:
            sleep_time = min(1.0, wait_seconds)
            time.sleep(sleep_time)
            wait_seconds -= sleep_time
            
            # 如果当前正在收集数据，定期更新数据文件
            if self.current_ppfd_value is not None:
                current_time = datetime.now()
                if (current_time - last_update).total_seconds() >= update_interval:
                    self.log("定期更新当前时间段数据...")
                    self.update_current_ppfd_data()
                    last_update = current_time
        
        return self.is_running
    
    def daemonize(self):
        """转换为守护进程"""
        try:
            # 第一次fork
            pid = os.fork()
            if pid > 0:
                # 父进程退出
                sys.exit(0)
        except OSError as e:
            self.log(f"第一次fork失败: {e}", logging.ERROR)
            sys.exit(1)
        
        # 脱离终端
        os.setsid()
        os.umask(0)
        
        try:
            # 第二次fork
            pid = os.fork()
            if pid > 0:
                # 父进程退出
                sys.exit(0)
        except OSError as e:
            self.log(f"第二次fork失败: {e}", logging.ERROR)
            sys.exit(1)
        
        # 重定向标准输入输出
        sys.stdout.flush()
        sys.stderr.flush()
        
        # 关闭文件描述符
        os.close(sys.stdin.fileno())
        os.close(sys.stdout.fileno())
        os.close(sys.stderr.fileno())
        
        self.log("已转换为守护进程")
    
    def handle_missed_schedules(self):
        """处理已经错过的时间点，收集历史数据"""
        now = datetime.now()
        self.log("检查是否有错过的时间点需要处理...")
        
        missed_schedules = []
        for i, schedule in enumerate(self.schedule_data):
            if schedule['time'] < now:
                missed_schedules.append((i, schedule))
        
        if not missed_schedules:
            self.log("没有错过的时间点")
            return
        
        self.log(f"发现 {len(missed_schedules)} 个错过的时间点")
        
        # 处理错过的时间点，收集历史数据
        for i, (index, schedule) in enumerate(missed_schedules):
            self.log(f"处理错过的时间点 {i+1}/{len(missed_schedules)}: {schedule['time_str']} ({schedule['phase_name']})")
            
            # 确定时间范围
            start_time = schedule['time']
            if i < len(missed_schedules) - 1:
                # 不是最后一个，结束时间是下一个时间点
                end_time = missed_schedules[i + 1][1]['time']
            else:
                # 是最后一个，结束时间是当前时间
                end_time = now
            
            # 收集该时间段的数据
            self.log(f"收集时间段数据: {start_time.strftime('%H:%M:%S')} 到 {end_time.strftime('%H:%M:%S')}")
            
            # 加载riotee数据
            riotee_data = self.load_riotee_data(start_time, end_time)
            
            # 保存到CSV文件
            output_path = self.save_ppfd_data_csv(
                schedule['ppfd'], 
                schedule['phase_name'], 
                riotee_data, 
                start_time, 
                end_time
            )
            
            if output_path:
                self.log(f"错过的时间点 {schedule['time_str']} 数据收集完成，保存了 {len(riotee_data)} 条记录")
    
    def start_current_ppfd_data_collection(self):
        """开始当前时间段的数据收集（如果当前时间在某个PPFD时间段内）"""
        now = datetime.now()
        self.log("检查是否在当前PPFD时间段内...")
        
        # 找到当前应该处于的时间段
        for i, schedule in enumerate(self.schedule_data):
            if schedule['time'] <= now:
                # 检查是否在下一个时间点之前
                if i < len(self.schedule_data) - 1:
                    next_schedule = self.schedule_data[i + 1]
                    if now < next_schedule['time']:
                        # 当前在某个时间段内，开始数据收集
                        self.log(f"当前在PPFD {schedule['ppfd']} ({schedule['phase_name']}) 时间段内")
                        self.start_ppfd_data_collection(
                            schedule['ppfd'], 
                            schedule['phase_name'], 
                            schedule['time']
                        )
                        return
        
        self.log("当前不在任何PPFD时间段内")
    
    def update_current_ppfd_data(self):
        """更新当前时间段的数据文件"""
        if self.current_ppfd_value is None or self.data_collection_start_time is None:
            return
        
        try:
            # 重新加载当前时间段的数据（从开始时间到当前时间）
            current_time = datetime.now()
            riotee_data = self.load_riotee_data(self.data_collection_start_time, current_time)
            
            # 更新CSV文件
            output_path = self.save_ppfd_data_csv(
                self.current_ppfd_value, 
                self.current_phase_name, 
                riotee_data, 
                self.data_collection_start_time, 
                current_time
            )
            
            if output_path:
                self.log(f"当前PPFD {self.current_ppfd_value} 数据已更新，包含 {len(riotee_data)} 条记录")
            
        except Exception as e:
            self.log(f"更新当前PPFD数据失败: {e}", logging.ERROR)
    
    def run_scheduler(self):
        """运行调度器"""
        # 检查是否已经在运行
        if self.is_already_running():
            self.log("调度器已经在运行中", logging.WARNING)
            return False
        
        if not self.load_schedule():
            return False
        
        # 如果是后台模式，转换为守护进程
        if self.daemon_mode:
            self.daemonize()
        
        # 写入PID文件
        self.write_pid()
        
        self.is_running = True
        self.log("PWM调度器开始运行...")
        
        # 处理已经错过的时间点
        self.handle_missed_schedules()
        
        # 开始当前时间段的数据收集（如果有的话）
        self.start_current_ppfd_data_collection()
        
        try:
            while self.is_running:
                # 找到下一个要执行的时间点
                next_index = self.find_next_schedule_index()
                
                if next_index == -1:
                    # 结束最后一个PPFD时间段的数据收集
                    if self.current_ppfd_value is not None:
                        self.end_ppfd_data_collection(datetime.now())
                    self.log("所有时间点已执行完毕")
                    break
                
                schedule = self.schedule_data[next_index]
                self.log(f"下一个执行时间: {schedule['time_str']} ({schedule['phase_name']})")
                
                # 等待到执行时间
                if not self.wait_for_next_schedule(schedule['time']):
                    break
                
                # 结束上一个PPFD时间段的数据收集（如果有的话）
                if self.current_ppfd_value is not None:
                    self.end_ppfd_data_collection(schedule['time'])
                
                # 开始新的PPFD时间段数据收集
                self.start_ppfd_data_collection(
                    schedule['ppfd'], 
                    schedule['phase_name'], 
                    schedule['time']
                )
                
                # 执行PWM设置
                self.log(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.log(f"阶段: {schedule['phase_name']} (周期 {schedule['cycle']})")
                self.log(f"PPFD值: {schedule['ppfd']}")
                
                if self.set_pwm_values(schedule['r_pwm'], schedule['b_pwm']):
                    # 等待5秒后检查状态
                    self.log("等待5秒后检查状态...")
                    time.sleep(5)
                    self.check_device_status()
                else:
                    self.log("PWM设置失败，跳过状态检查", logging.WARNING)
                
                self.current_index = next_index + 1
                
        except KeyboardInterrupt:
            self.log("用户中断，停止调度器")
        except Exception as e:
            self.log(f"调度器运行错误: {e}", logging.ERROR)
        finally:
            self.is_running = False
            self.cleanup()
            self.log("PWM调度器已停止")
        
        return True
    
    def load_riotee_data(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """加载指定时间范围内的riotee数据"""
        riotee_data = []
        
        try:
            # 查找所有riotee数据文件
            riotee_files = glob.glob(os.path.join(self.riotee_data_dir, "*riotee_data*.csv"))
            
            for file_path in riotee_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                        # 跳过注释行，找到数据行
                        for line in lines:
                            if line.strip().startswith('#') or not line.strip():
                                continue
                            
                            # 解析CSV行
                            parts = line.strip().split(',')
                            if len(parts) >= 20:  # 确保有足够的列（包含光谱数据）
                                try:
                                    # 解析时间戳
                                    timestamp_str = parts[1]  # timestamp列
                                    data_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                                    
                                    # 检查是否在时间范围内
                                    if start_time <= data_time <= end_time:
                                        # 构建完整的数据记录，包含所有光谱数据
                                        data_record = {
                                            'id': parts[0],
                                            'timestamp': timestamp_str,
                                            'device_id': parts[2],
                                            'update_type': parts[3],
                                            'temperature': float(parts[4]),
                                            'humidity': float(parts[5]),
                                            'a1_raw': float(parts[6]),
                                            'vcap_raw': float(parts[7]),
                                            'sp_415': float(parts[8]) if parts[8] else 0.0,
                                            'sp_445': float(parts[9]) if parts[9] else 0.0,
                                            'sp_480': float(parts[10]) if parts[10] else 0.0,
                                            'sp_515': float(parts[11]) if parts[11] else 0.0,
                                            'sp_555': float(parts[12]) if parts[12] else 0.0,
                                            'sp_590': float(parts[13]) if parts[13] else 0.0,
                                            'sp_630': float(parts[14]) if parts[14] else 0.0,
                                            'sp_680': float(parts[15]) if parts[15] else 0.0,
                                            'sp_clear': float(parts[16]) if parts[16] else 0.0,
                                            'sp_nir': float(parts[17]) if parts[17] else 0.0,
                                            'spectral_gain': int(parts[18]),
                                            'sleep_time': int(parts[19])
                                        }
                                        riotee_data.append(data_record)
                                except (ValueError, IndexError) as e:
                                    # 跳过格式错误的行
                                    continue
                                    
                except Exception as e:
                    self.log(f"读取riotee文件失败 {file_path}: {e}", logging.WARNING)
                    continue
            
            self.log(f"加载了 {len(riotee_data)} 条riotee数据记录")
            return riotee_data
            
        except Exception as e:
            self.log(f"加载riotee数据失败: {e}", logging.ERROR)
            return []
    
    def save_ppfd_data_csv(self, ppfd_value: int, phase_name: str, riotee_data: List[Dict], start_time: datetime, end_time: datetime):
        """保存PPFD时间段的数据到CSV文件"""
        try:
            # 创建输出文件名
            timestamp_str = start_time.strftime('%Y%m%d_%H%M%S')
            output_filename = f"ppfd_{ppfd_value}_{phase_name}_{timestamp_str}.csv"
            output_path = os.path.join(os.path.dirname(self.csv_file), "logs", output_filename)
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 写入CSV文件
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 写入头部信息
                writer.writerow([f"# PPFD时间段数据收集"])
                writer.writerow([f"# PPFD值: {ppfd_value}"])
                writer.writerow([f"# 阶段: {phase_name}"])
                writer.writerow([f"# 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"])
                writer.writerow([f"# 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}"])
                writer.writerow([f"# 数据条数: {len(riotee_data)}"])
                writer.writerow([])  # 空行
                
                if riotee_data:
                    # 写入数据头部（包含所有光谱数据列）
                    writer.writerow([
                        'id', 'timestamp', 'device_id', 'update_type', 'temperature', 'humidity', 
                        'a1_raw', 'vcap_raw', 'sp_415', 'sp_445', 'sp_480', 'sp_515', 'sp_555', 
                        'sp_590', 'sp_630', 'sp_680', 'sp_clear', 'sp_nir', 'spectral_gain', 'sleep_time'
                    ])
                    
                    # 写入数据行
                    for data in riotee_data:
                        writer.writerow([
                            data['id'],
                            data['timestamp'],
                            data['device_id'],
                            data['update_type'],
                            data['temperature'],
                            data['humidity'],
                            data['a1_raw'],
                            data['vcap_raw'],
                            data['sp_415'],
                            data['sp_445'],
                            data['sp_480'],
                            data['sp_515'],
                            data['sp_555'],
                            data['sp_590'],
                            data['sp_630'],
                            data['sp_680'],
                            data['sp_clear'],
                            data['sp_nir'],
                            data['spectral_gain'],
                            data['sleep_time']
                        ])
                else:
                    writer.writerow(['# 该时间段内无riotee数据'])
            
            self.log(f"PPFD {ppfd_value} 数据已保存到: {output_path}")
            return output_path
            
        except Exception as e:
            self.log(f"保存PPFD数据失败: {e}", logging.ERROR)
            return None
    
    def start_ppfd_data_collection(self, ppfd_value: int, phase_name: str, start_time: datetime):
        """开始PPFD数据收集"""
        self.current_ppfd_value = ppfd_value
        self.current_phase_name = phase_name
        self.data_collection_start_time = start_time
        self.current_ppfd_data = []
        
        self.log(f"开始收集PPFD {ppfd_value} ({phase_name}) 的数据")
    
    def end_ppfd_data_collection(self, end_time: datetime):
        """结束PPFD数据收集并保存数据"""
        if self.current_ppfd_value is None or self.data_collection_start_time is None:
            return
        
        try:
            # 加载时间范围内的riotee数据
            riotee_data = self.load_riotee_data(self.data_collection_start_time, end_time)
            
            # 保存到CSV文件
            output_path = self.save_ppfd_data_csv(
                self.current_ppfd_value, 
                self.current_phase_name, 
                riotee_data, 
                self.data_collection_start_time, 
                end_time
            )
            
            if output_path:
                self.log(f"PPFD {self.current_ppfd_value} 数据收集完成，保存了 {len(riotee_data)} 条记录")
            
            # 重置收集状态
            self.current_ppfd_value = None
            self.current_phase_name = None
            self.data_collection_start_time = None
            self.current_ppfd_data = []
            
        except Exception as e:
            self.log(f"结束PPFD数据收集失败: {e}", logging.ERROR)

def main():
    parser = argparse.ArgumentParser(description='PWM调度器 - 根据CSV时间表控制Shelly设备')
    parser.add_argument('csv_file', nargs='?', help='CSV时间表文件路径')
    parser.add_argument('-d', '--daemon', action='store_true', help='后台运行模式')
    parser.add_argument('-l', '--log', help='日志文件路径 (默认: pwm_scheduler.log)')
    parser.add_argument('-r', '--riotee-dir', help='riotee数据目录路径 (默认: /home/pi/Desktop/Test/riotee_sensor/logs)')
    parser.add_argument('--stop', action='store_true', help='停止正在运行的调度器')
    parser.add_argument('--status', action='store_true', help='检查调度器运行状态')
    
    args = parser.parse_args()
    
    # 停止调度器
    if args.stop:
        stop_scheduler()
        return
    
    # 检查状态
    if args.status:
        check_status()
        return
    
    # 检查CSV文件
    if not args.csv_file:
        print("错误: 需要指定CSV文件路径")
        print("用法: python3 pwm_scheduler.py <csv_file> [options]")
        sys.exit(1)
        
    if not os.path.exists(args.csv_file):
        print(f"CSV文件不存在: {args.csv_file}")
        sys.exit(1)
    
    # 创建调度器
    scheduler = PWMScheduler(
        args.csv_file, 
        daemon_mode=args.daemon, 
        log_file=args.log,
        riotee_data_dir=args.riotee_dir
    )
    
    # 运行调度器
    success = scheduler.run_scheduler()
    sys.exit(0 if success else 1)

def stop_scheduler():
    """停止调度器"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pid_file = os.path.join(script_dir, "pwm_scheduler.pid")
    
    if not os.path.exists(pid_file):
        print("调度器未运行")
        return
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        os.kill(pid, signal.SIGTERM)
        print(f"已发送停止信号到进程 {pid}")
        
        # 等待进程结束
        time.sleep(2)
        
        if os.path.exists(pid_file):
            os.remove(pid_file)
            print("PID文件已清理")
        
        print("调度器已停止")
        
    except (OSError, ValueError) as e:
        print(f"停止调度器失败: {e}")
        if os.path.exists(pid_file):
            os.remove(pid_file)

def check_status():
    """检查调度器状态"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pid_file = os.path.join(script_dir, "pwm_scheduler.pid")
    
    if not os.path.exists(pid_file):
        print("调度器未运行")
        return
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        os.kill(pid, 0)
        print(f"调度器正在运行 (PID: {pid})")
        
        # 显示日志文件最后几行
        log_file = os.path.join(script_dir, "pwm_scheduler.log")
        if os.path.exists(log_file):
            print(f"\n最近日志:")
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-5:]:
                    print(f"  {line.strip()}")
        
    except (OSError, ValueError):
        print("调度器未运行或PID文件无效")
        if os.path.exists(pid_file):
            os.remove(pid_file)

if __name__ == "__main__":
    main()
