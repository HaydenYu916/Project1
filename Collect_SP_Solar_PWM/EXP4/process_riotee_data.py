#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理riotee数据文件：
1. 去掉所有device_id为T6ncwg==的记录
2. 每15分钟采样一次数据
"""

import csv
from datetime import datetime, timedelta
import sys

def parse_timestamp(ts_str):
    """解析时间戳字符串"""
    try:
        # 处理带微秒的时间戳
        if '.' in ts_str:
            dt_str, micro = ts_str.split('.')
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
            # 处理微秒部分
            if len(micro) > 6:
                micro = micro[:6]
            elif len(micro) < 6:
                micro = micro.ljust(6, '0')
            dt = dt.replace(microsecond=int(micro))
            return dt
        else:
            return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"解析时间戳错误: {ts_str}, {e}")
        return None

def format_timestamp(dt):
    """格式化时间戳为字符串"""
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # 保留毫秒

def floor_15min(dt):
    """将时间向下取整到15分钟"""
    # 将分钟数向下取整到15的倍数
    minute = (dt.minute // 15) * 15
    return dt.replace(minute=minute, second=0, microsecond=0)

def process_riotee_data(input_file, output_file):
    """
    处理riotee数据文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"正在读取文件: {input_file}")
    
    # 读取所有行
    comment_lines = []
    header_line = None
    data_rows = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if line.startswith('#'):
                comment_lines.append(line)
            elif line.startswith('id,timestamp,device_id'):
                header_line = line
            elif line and not line.startswith('#'):
                # 检查是否是T6ncwg==设备
                if 'T6ncwg==' not in line:
                    # 解析CSV行
                    reader = csv.reader([line])
                    row = next(reader)
                    if len(row) > 2:
                        data_rows.append(row)
    
    if not header_line:
        print("错误: 未找到表头行")
        return
    
    print(f"原始数据行数: {len(data_rows)}")
    print(f"注释行数: {len(comment_lines)}")
    
    # 解析表头
    header_reader = csv.reader([header_line])
    headers = next(header_reader)
    timestamp_idx = headers.index('timestamp')
    device_id_idx = headers.index('device_id')
    
    # 解析数据并添加时间戳对象
    parsed_rows = []
    for row in data_rows:
        if len(row) > timestamp_idx:
            ts_str = row[timestamp_idx]
            dt = parse_timestamp(ts_str)
            if dt:
                parsed_rows.append((dt, row))
    
    # 按时间戳排序
    parsed_rows.sort(key=lambda x: x[0])
    
    print(f"有效数据行数: {len(parsed_rows)}")
    if parsed_rows:
        print(f"时间范围: {parsed_rows[0][0]} 到 {parsed_rows[-1][0]}")
    
    # 每15分钟采样一次
    sampled_rows = []
    current_bin = None
    current_row = None
    
    for dt, row in parsed_rows:
        time_bin = floor_15min(dt)
        if current_bin != time_bin:
            # 新的15分钟区间，保存上一个区间的数据
            if current_row is not None:
                sampled_rows.append(current_row)
            current_bin = time_bin
            current_row = (dt, row)
        else:
            # 同一区间，保留第一个数据点（已经按时间排序）
            if current_row is None:
                current_row = (dt, row)
    
    # 添加最后一个数据点
    if current_row is not None:
        sampled_rows.append(current_row)
    
    print(f"采样后数据行数: {len(sampled_rows)}")
    
    # 写入输出文件
    print(f"正在写入文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入注释行
        for comment in comment_lines:
            f.write(comment + '\n')
        
        # 写入表头
        f.write(header_line + '\n')
        
        # 写入数据
        for idx, (dt, row) in enumerate(sampled_rows, 1):
            # 更新id和时间戳
            row[0] = str(idx)  # id
            row[timestamp_idx] = format_timestamp(dt)
            
            # 写入CSV行
            writer = csv.writer(f)
            writer.writerow(row)
    
    print("处理完成！")
    print(f"输出文件: {output_file}")
    print(f"最终数据行数: {len(sampled_rows)}")

if __name__ == '__main__':
    input_file = '/home/pi/Desktop/Sensor/EXP4/riotee_data_all.csv'
    output_file = '/home/pi/Desktop/Sensor/EXP4/riotee_data_all_processed.csv'
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    process_riotee_data(input_file, output_file)
