"""
Riotee Reading Module
提供简单的Riotee数据读取接口
"""

import os
import csv
import glob
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

class RioteeDataReader:
    def __init__(self, logs_dir="../logs"):
        self.logs_dir = logs_dir
    
    def _find_latest_csv(self):
        """查找最新的Riotee CSV文件"""
        if not os.path.exists(self.logs_dir):
            return None
        
        # 优先查找riotee_data_all.csv
        preferred_file = os.path.join(self.logs_dir, "riotee_data_all.csv")
        if os.path.exists(preferred_file):
            return preferred_file
        
        # 如果没有，查找以riotee开头的CSV文件
        pattern = os.path.join(self.logs_dir, "riotee*.csv")
        files = glob.glob(pattern)
        if files:
            # 按修改时间排序，返回最新的
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return files[0]
        
        # 最后查找所有CSV文件中包含Riotee数据的
        pattern = os.path.join(self.logs_dir, "*.csv")
        files = glob.glob(pattern)
        for file in files:
            if 'riotee' in os.path.basename(file).lower():
                return file
        
        return None
    
    def _safe_float(self, value):
        """安全转换为浮点数"""
        if value is None or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _parse_timestamp(self, ts_str):
        """解析时间戳字符串"""
        if not ts_str:
            return None
        
        formats = [
            '%Y-%m-%d %H:%M:%S.%f',  # 完整格式
            '%Y-%m-%d %H:%M:%S',     # 无毫秒
            '%Y%m%d_%H%M%S',         # 紧凑格式
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(ts_str, fmt)
            except ValueError:
                continue
        return None
    
    def get_latest_data(self, max_age_seconds=120):
        """获取最新一条Riotee数据"""
        csv_path = self._find_latest_csv()
        if not csv_path:
            return None
        
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                # 跳过注释行
                first_line = f.readline()
                if not first_line.startswith('#'):
                    f.seek(0)
                
                reader = csv.DictReader(f)
                rows = list(reader)
                
            if not rows:
                return None
            
            row = rows[-1]  # 最后一行
            
            # 构建返回数据
            result = {
                '_csv_file': os.path.basename(csv_path),
                '_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'id': row.get('id', ''),
                'timestamp': row.get('timestamp', ''),
                'device_id': row.get('device_id', ''),
                'temperature': self._safe_float(row.get('temperature')),
                'humidity': self._safe_float(row.get('humidity')),
                'a1_raw': self._safe_float(row.get('a1_raw')),
                'vcap_raw': self._safe_float(row.get('vcap_raw')),
                'spectral_gain': self._safe_float(row.get('spectral_gain')),
                'sleep_time': self._safe_float(row.get('sleep_time')),
            }
            
            # 添加光谱数据
            spectral_data = {}
            wavelengths = [415, 445, 480, 515, 555, 590, 630, 680]
            for wl in wavelengths:
                spectral_data[f'sp_{wl}'] = self._safe_float(row.get(f'sp_{wl}'))
            spectral_data['sp_clear'] = self._safe_float(row.get('sp_clear'))
            spectral_data['sp_nir'] = self._safe_float(row.get('sp_nir'))
            result['spectral'] = spectral_data
            
            # 检查数据新鲜度
            data_time = self._parse_timestamp(row.get('timestamp', ''))
            if data_time:
                age_seconds = (datetime.now() - data_time).total_seconds()
                result['_data_age_seconds'] = age_seconds
                result['_is_fresh'] = age_seconds <= max_age_seconds
                result['_is_stale'] = age_seconds > max_age_seconds
            else:
                result['_data_age_seconds'] = None
                result['_is_fresh'] = False  
                result['_is_stale'] = True
            
            return result
            
        except Exception as e:
            return None
    
    def get_device_latest_data(self, device_id, max_age_seconds=120):
        """获取指定设备的最新数据"""
        csv_path = self._find_latest_csv()
        if not csv_path:
            return None
        
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                # 跳过注释行
                first_line = f.readline()
                if not first_line.startswith('#'):
                    f.seek(0)
                
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # 反向查找指定设备的最新记录
            for row in reversed(rows[-100:]):  # 只查找最近100条记录
                if row.get('device_id') == device_id:
                    result = {
                        '_csv_file': os.path.basename(csv_path),
                        '_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'id': row.get('id', ''),
                        'timestamp': row.get('timestamp', ''),
                        'device_id': row.get('device_id', ''),
                        'temperature': self._safe_float(row.get('temperature')),
                        'humidity': self._safe_float(row.get('humidity')),
                        'a1_raw': self._safe_float(row.get('a1_raw')),
                        'vcap_raw': self._safe_float(row.get('vcap_raw')),
                        'spectral_gain': self._safe_float(row.get('spectral_gain')),
                        'sleep_time': self._safe_float(row.get('sleep_time')),
                    }
                    
                    # 添加光谱数据
                    spectral_data = {}
                    wavelengths = [415, 445, 480, 515, 555, 590, 630, 680]
                    for wl in wavelengths:
                        spectral_data[f'sp_{wl}'] = self._safe_float(row.get(f'sp_{wl}'))
                    spectral_data['sp_clear'] = self._safe_float(row.get('sp_clear'))
                    spectral_data['sp_nir'] = self._safe_float(row.get('sp_nir'))
                    result['spectral'] = spectral_data
                    
                    # 数据新鲜度检查
                    data_time = self._parse_timestamp(row.get('timestamp', ''))
                    if data_time:
                        age_seconds = (datetime.now() - data_time).total_seconds()
                        result['_data_age_seconds'] = age_seconds
                        result['_is_fresh'] = age_seconds <= max_age_seconds
                        result['_is_stale'] = age_seconds > max_age_seconds
                    else:
                        result['_data_age_seconds'] = None
                        result['_is_fresh'] = False  
                        result['_is_stale'] = True
                    
                    return result
            
            return None
            
        except Exception as e:
            return None
    
    def get_recent_data(self, count=10):
        """获取最近N条记录"""
        csv_path = self._find_latest_csv()
        if not csv_path:
            return []
        
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                # 跳过注释行
                first_line = f.readline()
                if not first_line.startswith('#'):
                    f.seek(0)
                
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # 获取最后count行数据
            recent_rows = rows[-count:] if len(rows) > count else rows
            
            result = []
            for row in recent_rows:
                item = {
                    'id': row.get('id', ''),
                    'timestamp': row.get('timestamp', ''),
                    'device_id': row.get('device_id', ''),
                    'temperature': self._safe_float(row.get('temperature')),
                    'humidity': self._safe_float(row.get('humidity')),
                    'a1_raw': self._safe_float(row.get('a1_raw')),
                    'vcap_raw': self._safe_float(row.get('vcap_raw')),
                    'spectral_gain': self._safe_float(row.get('spectral_gain')),
                    'sleep_time': self._safe_float(row.get('sleep_time'))
                }
                result.append(item)
            
            return result
            
        except Exception as e:
            return []

    def get_device_window_avg(self, device_id: str, field: str, window_minutes: int = 10):
        """获取指定设备在时间窗口内某字段的平均值
        
        Args:
            device_id: 设备ID
            field: 字段名（例如 'a1_raw'）
            window_minutes: 时间窗口（分钟）
        Returns:
            dict: { 'device_id': str, 'field': str, 'avg': float|None, 'count': int, 'start': str, 'end': str }
        """
        csv_path = self._find_latest_csv()
        if not csv_path:
            return {
                'device_id': device_id,
                'field': field,
                'avg': None,
                'count': 0,
                'start': None,
                'end': None,
            }
        try:
            window_end = datetime.now()
            window_start = window_end - timedelta(minutes=window_minutes)
            values: List[float] = []
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                first_line = f.readline()
                if not first_line.startswith('#'):
                    f.seek(0)
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('device_id') != device_id:
                        continue
                    ts = self._parse_timestamp(row.get('timestamp', ''))
                    if not ts:
                        continue
                    if ts < window_start or ts > window_end:
                        continue
                    v = self._safe_float(row.get(field))
                    if v is not None:
                        values.append(v)
            avg_val = sum(values) / len(values) if values else None
            return {
                'device_id': device_id,
                'field': field,
                'avg': avg_val,
                'count': len(values),
                'start': window_start.strftime('%Y-%m-%d %H:%M:%S'),
                'end': window_end.strftime('%Y-%m-%d %H:%M:%S'),
            }
        except Exception:
            return {
                'device_id': device_id,
                'field': field,
                'avg': None,
                'count': 0,
                'start': None,
                'end': None,
            }

# 创建全局读取器实例
_logs_path = os.path.join(os.path.dirname(__file__), 'logs')
_reader = RioteeDataReader(_logs_path)

def get_current_riotee(device_id=None, max_age_seconds=120):
    """
    获取当前Riotee数据（最常用的接口）
    
    Args:
        device_id: 可选的设备ID，如果不指定则返回最新的任意设备数据
        max_age_seconds: 数据最大时效性（秒）
        
    Returns:
        dict: Riotee数据，如果无数据则返回None
    """
    if device_id:
        data = _reader.get_device_latest_data(device_id, max_age_seconds)
    else:
        data = _reader.get_latest_data(max_age_seconds)
    
    if data and not data.get('_is_stale', True):
        return data
    return None

def get_riotee_data(device_id=None, max_age_seconds=120):
    """
    获取详细的Riotee数据
    
    Returns:
        dict: 包含完整Riotee数据的字典，或 None 如果无数据
    """
    if device_id:
        return _reader.get_device_latest_data(device_id, max_age_seconds)
    else:
        return _reader.get_latest_data(max_age_seconds)

def get_riotee_devices():
    """
    获取活跃的Riotee设备列表
    
    Returns:
        list: 设备ID列表
    """
    recent_data = _reader.get_recent_data(50)  # 获取最近50条记录
    device_ids = set()
    for record in recent_data:
        if record.get('device_id'):
            device_ids.add(record['device_id'])
    return list(device_ids)

def get_recent_riotee(count=10):
    """
    获取最近的Riotee读数列表
    
    Args:
        count: 获取的数据点数量
        
    Returns:
        list: 包含timestamp、device_id和传感器数据的字典列表
    """
    return _reader.get_recent_data(count)

__all__ = ['get_current_riotee', 'get_riotee_data', 'get_riotee_devices', 'get_recent_riotee']

def get_device_avg_a1_raw(device_id: str, window_minutes: int = 10):
    """
    获取指定设备在给定时间窗口内的 A1_Raw 均值
    
    Returns:
        dict: { 'device_id', 'field', 'avg', 'count', 'start', 'end' }
    """
    return _reader.get_device_window_avg(device_id, 'a1_raw', window_minutes)

__all__.append('get_device_avg_a1_raw')
