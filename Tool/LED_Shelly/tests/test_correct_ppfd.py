#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试正确的PPFD值对应关系
"""

import sys
import os
from datetime import datetime, timedelta
from pwm_scheduler import PWMScheduler

def test_correct_ppfd_mapping():
    """测试正确的PPFD值映射"""
    print("测试PPFD值与时间段的正确对应关系...")
    
    # 创建调度器实例
    csv_file = "src/extended_schedule_20250919_071157_20250919_071157.csv"
    riotee_data_dir = "/home/pi/Desktop/Test/riotee_sensor/logs"
    
    scheduler = PWMScheduler(csv_file, riotee_data_dir=riotee_data_dir)
    
    # 加载时间表
    if scheduler.load_schedule():
        print(f"✓ 成功加载 {len(scheduler.schedule_data)} 个时间点")
        
        # 显示前几个时间点的详细信息
        print("\n前5个时间点的详细信息:")
        for i, schedule in enumerate(scheduler.schedule_data[:5]):
            print(f"  {i+1}. 时间: {schedule['time_str']}")
            print(f"     PPFD: {schedule['ppfd']}")
            print(f"     PWM: R={schedule['r_pwm']}, B={schedule['b_pwm']}")
            print(f"     阶段: {schedule['phase_name']}")
            print()
        
        # 测试特定时间段的数据收集
        print("测试07:00-07:30时间段 (应该是PPFD=100):")
        start_time = datetime(2025, 9, 19, 7, 0, 0)
        end_time = datetime(2025, 9, 19, 7, 30, 0)
        
        # 开始数据收集
        scheduler.start_ppfd_data_collection(100, "Heating 1", start_time)
        print("✓ 开始收集PPFD=100的数据")
        
        # 结束数据收集
        scheduler.end_ppfd_data_collection(end_time)
        print("✓ 结束数据收集")
        
        # 检查生成的文件
        logs_dir = os.path.join(os.path.dirname(csv_file), "logs")
        if os.path.exists(logs_dir):
            files = os.listdir(logs_dir)
            csv_files = [f for f in files if f.endswith('.csv')]
            print(f"\n生成的CSV文件: {csv_files}")
            
            # 查找PPFD=100的文件
            ppfd_100_files = [f for f in csv_files if 'ppfd_100' in f]
            if ppfd_100_files:
                print(f"✓ 找到PPFD=100文件: {ppfd_100_files[0]}")
                
                # 显示文件内容头部
                file_path = os.path.join(logs_dir, ppfd_100_files[0])
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    print("文件头部信息:")
                    for i, line in enumerate(lines[:8]):
                        print(f"  {i+1}: {line.strip()}")
            else:
                print("✗ 未找到PPFD=100的文件")
        
        print("\n测试07:30-08:00时间段 (应该是PPFD=200):")
        start_time = datetime(2025, 9, 19, 7, 30, 0)
        end_time = datetime(2025, 9, 19, 8, 0, 0)
        
        # 开始数据收集
        scheduler.start_ppfd_data_collection(200, "Heating 1", start_time)
        print("✓ 开始收集PPFD=200的数据")
        
        # 结束数据收集
        scheduler.end_ppfd_data_collection(end_time)
        print("✓ 结束数据收集")
        
        # 检查生成的文件
        if os.path.exists(logs_dir):
            files = os.listdir(logs_dir)
            csv_files = [f for f in files if f.endswith('.csv')]
            print(f"\n生成的CSV文件: {csv_files}")
            
            # 查找PPFD=200的文件
            ppfd_200_files = [f for f in csv_files if 'ppfd_200' in f]
            if ppfd_200_files:
                print(f"✓ 找到PPFD=200文件: {ppfd_200_files[0]}")
                
                # 显示文件内容头部
                file_path = os.path.join(logs_dir, ppfd_200_files[0])
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    print("文件头部信息:")
                    for i, line in enumerate(lines[:8]):
                        print(f"  {i+1}: {line.strip()}")
            else:
                print("✗ 未找到PPFD=200的文件")
    else:
        print("✗ 加载时间表失败")

def main():
    test_correct_ppfd_mapping()

if __name__ == "__main__":
    main()
