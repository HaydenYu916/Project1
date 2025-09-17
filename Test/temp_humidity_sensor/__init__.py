"""
温湿度传感器数据读取模块
提供简单的温湿度数据读取接口
"""

import csv
import os
from datetime import datetime, timedelta

class TempHumidityDataReader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
    
    def get_latest_value(self, max_age_seconds=30):
        """获取最新的温湿度值"""
        if not os.path.exists(self.csv_path):
            return None
            
        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
            if len(rows) <= 1:  # 只有表头或空文件
                return None
                
            # 获取最后一行数据
            last_row = rows[-1]
            timestamp_str = last_row[0]
            temperature = float(last_row[1])
            humidity = float(last_row[2])
            
            # 计算数据年龄
            data_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            current_time = datetime.now()
            age_seconds = (current_time - data_time).total_seconds()
            
            # 检查数据是否太旧
            if age_seconds > max_age_seconds:
                return {
                    'temperature': temperature,
                    'humidity': humidity,
                    'timestamp': timestamp_str,
                    'age_seconds': int(age_seconds),
                    'is_stale': True
                }
            
            return {
                'temperature': temperature,
                'humidity': humidity,
                'timestamp': timestamp_str,
                'age_seconds': int(age_seconds),
                'is_stale': False
            }
            
        except Exception as e:
            return None
    
    def get_recent_values(self, count=10):
        """获取最近N个温湿度值"""
        if not os.path.exists(self.csv_path):
            return []
            
        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
            if len(rows) <= 1:
                return []
                
            # 获取最后count行数据（跳过表头）
            recent_rows = rows[-count:] if len(rows) > count else rows[1:]
            
            result = []
            for row in recent_rows:
                try:
                    result.append({
                        'timestamp': row[0],
                        'temperature': float(row[1]),
                        'humidity': float(row[2])
                    })
                except (ValueError, IndexError):
                    continue
                    
            return result
            
        except Exception as e:
            return []
    
    def get_average(self, minutes=5):
        """获取过去N分钟的平均温湿度值"""
        if not os.path.exists(self.csv_path):
            return None
            
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            temp_total = 0
            humidity_total = 0
            count = 0
            
            with open(self.csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                
                for row in reader:
                    try:
                        timestamp_str = row[0]
                        temperature = float(row[1])
                        humidity = float(row[2])
                        data_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        
                        if data_time >= cutoff_time:
                            temp_total += temperature
                            humidity_total += humidity
                            count += 1
                    except (ValueError, IndexError):
                        continue
            
            if count > 0:
                return {
                    'temperature': temp_total / count,
                    'humidity': humidity_total / count
                }
            return None
            
        except Exception as e:
            return None

# 创建全局读取器实例
_csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'temp_humidity_data.csv')
_reader = TempHumidityDataReader(_csv_path)

def get_current_temperature(max_age_seconds=30):
    """
    获取当前温度值（最常用的接口）
    
    Args:
        max_age_seconds: 数据最大时效性（秒）
        
    Returns:
        float: 温度值（°C），如果无数据则返回None
    """
    data = _reader.get_latest_value(max_age_seconds)
    if data and not data['is_stale']:
        return data['temperature']
    return None

def get_current_humidity(max_age_seconds=30):
    """
    获取当前湿度值（最常用的接口）
    
    Args:
        max_age_seconds: 数据最大时效性（秒）
        
    Returns:
        float: 湿度值（%），如果无数据则返回None
    """
    data = _reader.get_latest_value(max_age_seconds)
    if data and not data['is_stale']:
        return data['humidity']
    return None

def get_current_temp_humidity(max_age_seconds=30):
    """
    获取当前温湿度值
    
    Args:
        max_age_seconds: 数据最大时效性（秒）
        
    Returns:
        dict: 包含temperature和humidity的字典，如果无数据则返回None
    """
    data = _reader.get_latest_value(max_age_seconds)
    if data and not data['is_stale']:
        return {
            'temperature': data['temperature'],
            'humidity': data['humidity']
        }
    return None

def get_temp_humidity_data(max_age_seconds=30):
    """
    获取详细的温湿度数据
    
    Returns:
        dict: 包含temperature, humidity, timestamp, age_seconds, is_stale的完整数据
        或 None 如果无数据
    """
    return _reader.get_latest_value(max_age_seconds)

def get_temp_humidity_average(minutes=5):
    """
    获取指定时间内的平均温湿度值
    
    Args:
        minutes: 时间范围（分钟）
        
    Returns:
        dict: 包含temperature和humidity平均值的字典，如果无数据则返回None
    """
    return _reader.get_average(minutes)

def get_recent_temp_humidity(count=10):
    """
    获取最近的温湿度读数列表
    
    Args:
        count: 获取的数据点数量
        
    Returns:
        list: 包含timestamp, temperature和humidity的字典列表
    """
    return _reader.get_recent_values(count)

__all__ = [
    'get_current_temperature', 
    'get_current_humidity', 
    'get_current_temp_humidity',
    'get_temp_humidity_data', 
    'get_temp_humidity_average', 
    'get_recent_temp_humidity'
]
