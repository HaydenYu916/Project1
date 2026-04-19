#!/usr/bin/env python3
"""
实时数据获取 API - 适配 Master_Version_7_Chamber.py
=================================================

功能特性:
1. 获取最新一条记录 - get_latest_data()
2. 获取指定设备最新数据 - get_device_latest_data()
3. 获取最近N条记录 - get_recent_data()
4. 监控特定光谱通道 - get_spectral_data()
5. 获取增益和休眠时间信息 - get_device_config()
6. 实时状态检查 - check_system_status()

环境变量配置 (可选):
- LIVE_LOGS_DIR: 日志目录 (默认 'logs')
- LIVE_CSV_PATH: 指定CSV文件路径 (默认自动选择最新)
- LIVE_SPECTRAL: 默认光谱通道 (默认 415,445,480,515,555,590,630,680)
- LIVE_UPDATE_THRESHOLD: 数据更新阈值秒数 (默认 30，超过视为数据过期)

使用示例:
    from temp.live_data_api import get_latest_data, get_device_latest_data
    
    # 获取最新一条数据
    data = get_latest_data()
    
    # 获取指定设备最新数据
    device_data = get_device_latest_data("your_device_id")
    
    # 获取最近5条记录
    recent = get_recent_data(count=5)
"""

import os
import csv
import glob
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict


# 默认配置
DEFAULT_SPECTRAL = [415, 445, 480, 515, 555, 590, 630, 680]
DEFAULT_LOGS_DIR = "logs"
DEFAULT_UPDATE_THRESHOLD = 120  # 秒


def _get_config() -> Dict[str, Any]:
    """获取环境变量配置"""
    logs_dir = os.getenv('LIVE_LOGS_DIR', DEFAULT_LOGS_DIR)
    csv_path = os.getenv('LIVE_CSV_PATH')
    
    # 解析光谱通道配置
    spectral_env = os.getenv('LIVE_SPECTRAL')
    if spectral_env:
        try:
            spectral = [int(x.strip()) for x in spectral_env.replace(',', ' ').split() if x.strip().isdigit()]
        except ValueError:
            spectral = DEFAULT_SPECTRAL
    else:
        spectral = DEFAULT_SPECTRAL
    
    # 数据更新阈值
    try:
        update_threshold = int(os.getenv('LIVE_UPDATE_THRESHOLD', str(DEFAULT_UPDATE_THRESHOLD)))
    except ValueError:
        update_threshold = DEFAULT_UPDATE_THRESHOLD
    
    return {
        'logs_dir': logs_dir,
        'csv_path': csv_path,
        'spectral': spectral,
        'update_threshold': update_threshold
    }


def _find_latest_csv(logs_dir: str, explicit_csv: Optional[str] = None) -> Optional[str]:
    """查找最新的CSV文件"""
    if explicit_csv and os.path.exists(explicit_csv):
        return explicit_csv
    
    if not os.path.exists(logs_dir):
        return None
    
    pattern = os.path.join(logs_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    
    # 按修改时间排序，返回最新的
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """解析时间戳字符串"""
    if not ts_str:
        return None
    
    # 支持多种时间格式
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


def _read_csv_data(csv_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """读取CSV数据，返回行列表"""
    if not os.path.exists(csv_path):
        return []
    
    rows = []
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8', errors='replace') as f:
            sanitized_lines = (line.replace('\x00', '') for line in f)
            # 跳过注释行
            first_line = next(sanitized_lines, '')
            if not first_line.startswith('#'):
                reader = csv.DictReader([first_line, *sanitized_lines])
            else:
                reader = csv.DictReader(sanitized_lines)
            
            # 如果需要限制行数，并且文件较大，使用更高效的方法
            if limit and limit <= 100:
                # 对于小的limit（如获取最新数据），读取所有行然后取最后N行
                all_rows = list(reader)
                return all_rows[-limit:] if len(all_rows) > limit else all_rows
            else:
                # 对于大的limit或无限制，按原来的方式读取
                for row in reader:
                    rows.append(row)
                
                # 如果指定了限制，返回最后N行
                if limit and len(rows) > limit:
                    return rows[-limit:]
                
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return []
    
    return rows


def _resolve_co2_fields(rows: List[Dict[str, Any]], target_index: int) -> Tuple[Optional[float], Optional[float], str, bool]:
    """Return (effective_co2, raw_co2, co2_state, is_fresh) for the target row."""
    if target_index < 0 or target_index >= len(rows):
        return None, None, "UNKNOWN", False

    row = rows[target_index]
    device_id = row.get("device_id", "")
    row_state = str(row.get("co2_state", "")).strip().upper()
    raw_co2 = _safe_float(row.get("co2_ppm"))
    if row_state in {"INVALID", "HOLD", "FRESH"}:
        return raw_co2, raw_co2, row_state, row_state == "FRESH"
    if raw_co2 is not None:
        return raw_co2, raw_co2, "FRESH", True

    for idx in range(target_index - 1, -1, -1):
        prev = rows[idx]
        if prev.get("device_id", "") != device_id:
            continue
        prev_co2 = _safe_float(prev.get("co2_ppm"))
        if prev_co2 is not None:
            return prev_co2, None, "HOLD", False

    return None, None, "INVALID", False


def get_latest_data(include_spectral: bool = True, 
                   include_config: bool = True) -> Optional[Dict[str, Any]]:
    """
    获取最新一条记录
    
    参数:
    - include_spectral: 是否包含光谱数据
    - include_config: 是否包含增益和休眠时间配置
    
    返回:
    - dict: 包含最新数据的字典，或None如果无数据
    """
    config = _get_config()
    csv_path = _find_latest_csv(config['logs_dir'], config['csv_path'])
    
    if not csv_path:
        return None
    
    rows = _read_csv_data(csv_path, limit=200)
    if not rows:
        return None
    
    row_index = len(rows) - 1
    row = rows[row_index]  # 最后一行
    
    # 构建返回数据
    result = {
        '_csv_file': os.path.basename(csv_path),
        '_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'id': row.get('id', ''),
        'timestamp': row.get('timestamp', ''),
        'device_id': row.get('device_id', ''),
        'temperature': _safe_float(row.get('temperature')),
        'humidity': _safe_float(row.get('humidity')),
        'a1_raw': _safe_float(row.get('a1_raw')),
        'vcap_raw': _safe_float(row.get('vcap_raw')),
    }
    effective_co2, raw_co2, co2_state, co2_fresh = _resolve_co2_fields(rows, row_index)
    result['co2_ppm'] = effective_co2
    result['co2_raw_ppm'] = raw_co2
    result['co2_state'] = co2_state
    result['co2_fresh'] = co2_fresh
    
    # 添加光谱数据
    if include_spectral:
        spectral_data = {}
        for wl in config['spectral']:
            spectral_data[f'sp_{wl}'] = _safe_float(row.get(f'sp_{wl}'))
        
        # 添加Clear和NIR通道
        spectral_data['sp_clear'] = _safe_float(row.get('sp_clear'))
        spectral_data['sp_nir'] = _safe_float(row.get('sp_nir'))
        
        result['spectral'] = spectral_data
    
    # 添加设备配置信息
    if include_config:
        result['config'] = {
            'spectral_gain': _safe_float(row.get('spectral_gain')),
            'sleep_time': _safe_int(row.get('sleep_time'))
        }
    
    # 检查数据新鲜度
    data_time = _parse_timestamp(row.get('timestamp', ''))
    if data_time:
        age_seconds = (datetime.now() - data_time).total_seconds()
        result['_data_age_seconds'] = age_seconds
        result['_is_fresh'] = age_seconds <= config['update_threshold']
    
    return result


def get_device_latest_data(device_id: str, 
                          include_spectral: bool = True,
                          include_config: bool = True) -> Optional[Dict[str, Any]]:
    """
    获取指定设备的最新数据
    
    参数:
    - device_id: 设备ID
    - include_spectral: 是否包含光谱数据
    - include_config: 是否包含配置信息
    
    返回:
    - dict: 设备最新数据，或None如果未找到
    """
    config = _get_config()
    csv_path = _find_latest_csv(config['logs_dir'], config['csv_path'])
    
    if not csv_path:
        return None
    
    # 读取较多行以查找指定设备
    rows = _read_csv_data(csv_path, limit=100)
    
    # 反向查找指定设备的最新记录
    for idx in range(len(rows) - 1, -1, -1):
        row = rows[idx]
        if row.get('device_id') == device_id:
            result = {
                '_csv_file': os.path.basename(csv_path),
                '_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'id': row.get('id', ''),
                'timestamp': row.get('timestamp', ''),
                'device_id': row.get('device_id', ''),
                'temperature': _safe_float(row.get('temperature')),
                'humidity': _safe_float(row.get('humidity')),
                'a1_raw': _safe_float(row.get('a1_raw')),
                'vcap_raw': _safe_float(row.get('vcap_raw')),
            }
            effective_co2, raw_co2, co2_state, co2_fresh = _resolve_co2_fields(rows, idx)
            result['co2_ppm'] = effective_co2
            result['co2_raw_ppm'] = raw_co2
            result['co2_state'] = co2_state
            result['co2_fresh'] = co2_fresh
            
            if include_spectral:
                spectral_data = {}
                for wl in config['spectral']:
                    spectral_data[f'sp_{wl}'] = _safe_float(row.get(f'sp_{wl}'))
                spectral_data['sp_clear'] = _safe_float(row.get('sp_clear'))
                spectral_data['sp_nir'] = _safe_float(row.get('sp_nir'))
                result['spectral'] = spectral_data
            
            if include_config:
                result['config'] = {
                    'spectral_gain': _safe_float(row.get('spectral_gain')),
                    'sleep_time': _safe_int(row.get('sleep_time'))
                }
            
            # 数据新鲜度检查
            data_time = _parse_timestamp(row.get('timestamp', ''))
            if data_time:
                age_seconds = (datetime.now() - data_time).total_seconds()
                result['_data_age_seconds'] = age_seconds
                result['_is_fresh'] = age_seconds <= config['update_threshold']
            
            return result
    
    return None


def get_recent_data(count: int = 10, 
                   device_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    获取最近N条记录
    
    参数:
    - count: 返回记录数量
    - device_id: 可选的设备ID过滤
    
    返回:
    - list: 最近的记录列表
    """
    config = _get_config()
    csv_path = _find_latest_csv(config['logs_dir'], config['csv_path'])
    
    if not csv_path:
        return []
    
    rows = _read_csv_data(csv_path, limit=count * 3)  # 读取多一些以备过滤
    
    # 如果指定设备ID，进行过滤
    if device_id:
        filtered_rows = [row for row in rows if row.get('device_id') == device_id]
        rows = filtered_rows[-count:] if len(filtered_rows) > count else filtered_rows
    else:
        rows = rows[-count:] if len(rows) > count else rows
    
    result = []
    for idx, row in enumerate(rows):
        item = {
            'id': row.get('id', ''),
            'timestamp': row.get('timestamp', ''),
            'device_id': row.get('device_id', ''),
            'temperature': _safe_float(row.get('temperature')),
            'humidity': _safe_float(row.get('humidity')),
            'a1_raw': _safe_float(row.get('a1_raw')),
            'vcap_raw': _safe_float(row.get('vcap_raw')),
            'spectral_gain': _safe_float(row.get('spectral_gain')),
            'sleep_time': _safe_int(row.get('sleep_time'))
        }
        effective_co2, raw_co2, co2_state, co2_fresh = _resolve_co2_fields(rows, idx)
        item['co2_ppm'] = effective_co2
        item['co2_raw_ppm'] = raw_co2
        item['co2_state'] = co2_state
        item['co2_fresh'] = co2_fresh
        result.append(item)
    
    return result


def get_spectral_data(device_id: Optional[str] = None,
                     channels: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
    """
    获取光谱数据
    
    参数:
    - device_id: 可选的设备ID
    - channels: 指定光谱通道列表
    
    返回:
    - dict: 光谱数据
    """
    config = _get_config()
    channels = channels or config['spectral']
    
    data = get_device_latest_data(device_id) if device_id else get_latest_data()
    
    if not data or 'spectral' not in data:
        return None
    
    spectral = data['spectral']
    result = {
        'device_id': data['device_id'],
        'timestamp': data['timestamp'],
        'spectral_gain': data.get('config', {}).get('spectral_gain'),
        'channels': {}
    }
    
    for ch in channels:
        result['channels'][f'sp_{ch}'] = spectral.get(f'sp_{ch}')
    
    result['channels']['sp_clear'] = spectral.get('sp_clear')
    result['channels']['sp_nir'] = spectral.get('sp_nir')
    
    return result


def get_device_config(device_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    获取设备配置信息（增益和休眠时间）
    
    参数:
    - device_id: 可选的设备ID
    
    返回:
    - dict: 设备配置信息
    """
    data = get_device_latest_data(device_id, include_spectral=False) if device_id else get_latest_data(include_spectral=False)
    
    if not data or 'config' not in data:
        return None
    
    config = data['config']
    
    # 增益值转换为可读格式
    gain_value = config.get('spectral_gain', 0)
    gain_map = {
        0.5: "0.5X", 1: "1X", 2: "2X", 4: "4X", 8: "8X",
        16: "16X", 32: "32X", 64: "64X", 128: "128X", 256: "256X", 512: "512X"
    }
    gain_str = gain_map.get(gain_value, f"{gain_value}X" if gain_value > 0 else "Unknown")
    
    return {
        'device_id': data['device_id'],
        'timestamp': data['timestamp'],
        'spectral_gain_value': gain_value,
        'spectral_gain_text': gain_str,
        'sleep_time': config.get('sleep_time', 0),
        '_is_fresh': data.get('_is_fresh', False)
    }


def check_system_status() -> Dict[str, Any]:
    """
    检查系统状态
    
    返回:
    - dict: 系统状态信息
    """
    config = _get_config()
    csv_path = _find_latest_csv(config['logs_dir'], config['csv_path'])
    
    status = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'logs_dir': config['logs_dir'],
        'csv_file': os.path.basename(csv_path) if csv_path else None,
        'csv_exists': csv_path is not None and os.path.exists(csv_path),
        'data_available': False,
        'latest_data_time': None,
        'data_age_seconds': None,
        'is_data_fresh': False,
        'total_records': 0,
        'active_devices': []
    }
    
    if not csv_path or not os.path.exists(csv_path):
        return status
    
    # 读取数据并分析
    try:
        rows = _read_csv_data(csv_path, limit=50)  # 读取最近50条记录用于分析
        
        if rows:
            status['data_available'] = True
            status['total_records'] = len(rows)
            
            # 最新数据时间
            latest_row = rows[-1]
            latest_time_str = latest_row.get('timestamp', '')
            latest_time = _parse_timestamp(latest_time_str)
            
            if latest_time:
                status['latest_data_time'] = latest_time_str
                age_seconds = (datetime.now() - latest_time).total_seconds()
                status['data_age_seconds'] = age_seconds
                status['is_data_fresh'] = age_seconds <= config['update_threshold']
            
            # 活跃设备统计
            device_counts = defaultdict(int)
            for row in rows:
                device_id = row.get('device_id', '')
                if device_id:
                    device_counts[device_id] += 1
            
            status['active_devices'] = [
                {'device_id': dev_id, 'record_count': count}
                for dev_id, count in device_counts.items()
            ]
    
    except Exception as e:
        status['error'] = str(e)
    
    return status


def _safe_float(value: Any) -> Optional[float]:
    """安全转换为浮点数"""
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    """安全转换为整数"""
    if value is None or value == '':
        return None
    try:
        return int(float(value))  # 先转float再转int，处理"1.0"这样的字符串
    except (ValueError, TypeError):
        return None


# 便捷导入函数
def quick_status() -> str:
    """快速状态检查，返回简洁字符串"""
    status = check_system_status()
    if not status['data_available']:
        return "❌ 无数据可用"
    
    age = status.get('data_age_seconds', 0)
    fresh = "🟢" if status.get('is_data_fresh', False) else "🔴"
    devices = len(status.get('active_devices', []))
    
    return f"{fresh} 数据: {age:.0f}s前, {devices}个设备, {status['total_records']}条记录"


if __name__ == "__main__":
    # 简单测试
    print("Live Data API 测试")
    print("-" * 40)
    
    print("系统状态:")
    status = check_system_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))
    
    print("\n最新数据:")
    latest = get_latest_data()
    if latest:
        print(json.dumps(latest, indent=2, ensure_ascii=False))
    else:
        print("无数据")
    
    print(f"\n快速状态: {quick_status()}")
