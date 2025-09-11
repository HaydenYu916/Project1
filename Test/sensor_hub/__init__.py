"""
Unified Data Module
统一数据读取模块 - 同时获取CO2和Riotee数据并拼接
"""

import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

# 添加路径以便导入CO2和Riotee模块
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from co2_sensor import get_current_co2, get_co2_data
    CO2_AVAILABLE = True
except ImportError:
    CO2_AVAILABLE = False
    print("⚠️  co2_sensor模块不可用")

try:
    from riotee_sensor import get_current_riotee, get_riotee_data, get_riotee_devices
    RIOTEE_AVAILABLE = True
except ImportError:
    RIOTEE_AVAILABLE = False
    print("⚠️  riotee_sensor模块不可用")

class UnifiedDataReader:
    def __init__(self):
        self.co2_available = CO2_AVAILABLE
        self.riotee_available = RIOTEE_AVAILABLE
    
    def get_system_status(self):
        """获取系统状态"""
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'co2_available': self.co2_available,
            'riotee_available': self.riotee_available,
            'systems_ready': self.co2_available and self.riotee_available
        }
    
    def get_co2_only(self, max_age_seconds=120):
        """仅获取CO2数据"""
        if not self.co2_available:
            return None
        
        try:
            return get_co2_data(max_age_seconds)
        except Exception as e:
            print(f"❌ 获取CO2数据失败: {e}")
            return None
    
    def get_riotee_only(self, device_id=None, max_age_seconds=120):
        """仅获取Riotee数据"""
        if not self.riotee_available:
            return None
        
        try:
            return get_riotee_data(device_id, max_age_seconds)
        except Exception as e:
            print(f"❌ 获取Riotee数据失败: {e}")
            return None
    
    def get_unified_data(self, riotee_device_id=None, max_age_seconds=120, require_both=False):
        """
        获取统一的CO2和Riotee数据
        
        Args:
            riotee_device_id: 指定Riotee设备ID，如果不指定则使用最新的任意设备
            max_age_seconds: 数据最大时效性（秒）
            require_both: 是否要求两种数据都存在才返回结果
            
        Returns:
            dict: 包含CO2和Riotee数据的统一字典，或None如果无法获取数据
        """
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'co2_data': None,
            'riotee_data': None,
            'status': {
                'co2_available': False,
                'riotee_available': False,
                'both_fresh': False
            }
        }
        
        # 获取CO2数据
        if self.co2_available:
            try:
                co2_data = get_co2_data(max_age_seconds)
                if co2_data and not co2_data.get('is_stale', True):
                    result['co2_data'] = co2_data
                    result['status']['co2_available'] = True
            except Exception as e:
                print(f"❌ 获取CO2数据失败: {e}")
        
        # 获取Riotee数据
        if self.riotee_available:
            try:
                riotee_data = get_riotee_data(riotee_device_id, max_age_seconds)
                if riotee_data and not riotee_data.get('_is_stale', True):
                    result['riotee_data'] = riotee_data
                    result['status']['riotee_available'] = True
            except Exception as e:
                print(f"❌ 获取Riotee数据失败: {e}")
        
        # 检查数据完整性和时间差
        result['status']['both_fresh'] = result['status']['co2_available'] and result['status']['riotee_available']
        
        # 检查CO2和Riotee数据时间差不超过2分钟
        if result['status']['both_fresh']:
            try:
                # 解析CO2时间戳
                co2_time_str = result['co2_data']['timestamp']
                co2_time = datetime.strptime(co2_time_str, '%Y-%m-%d %H:%M:%S')
                
                # 解析Riotee时间戳（支持毫秒格式）
                riotee_time_str = result['riotee_data']['timestamp']
                try:
                    riotee_time = datetime.strptime(riotee_time_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # 尝试带毫秒的格式
                    riotee_time = datetime.strptime(riotee_time_str, '%Y-%m-%d %H:%M:%S.%f')
                
                time_diff = abs((co2_time - riotee_time).total_seconds())
                
                if time_diff > 120:  # 超过2分钟
                    result['status']['both_fresh'] = False
                    result['status']['time_diff_too_large'] = True
                    result['time_difference_seconds'] = time_diff
                else:
                    result['status']['time_diff_too_large'] = False
                    result['time_difference_seconds'] = time_diff
            except Exception as e:
                result['status']['both_fresh'] = False
                result['status']['time_check_error'] = str(e)
        
        # 如果要求两种数据都存在但不满足条件，返回None
        if require_both and not result['status']['both_fresh']:
            return None
        
        # 如果没有任何数据，返回None
        if not result['status']['co2_available'] and not result['status']['riotee_available']:
            return None
        
        return result
    
    def get_simple_data(self, riotee_device_id=None, max_age_seconds=120):
        """
        获取简化的数据格式，便于使用
        
        Returns:
            dict: 扁平化的数据字典，包含所有传感器数据
        """
        unified = self.get_unified_data(riotee_device_id, max_age_seconds, require_both=False)
        if not unified:
            return None
        
        result = {
            'timestamp': unified['timestamp'],
            'data_status': unified['status']
        }
        
        # 添加CO2数据
        if unified['co2_data']:
            result['co2_value'] = unified['co2_data'].get('value')
            result['co2_timestamp'] = unified['co2_data'].get('timestamp')
            result['co2_age_seconds'] = unified['co2_data'].get('age_seconds')
        
        # 添加Riotee数据
        if unified['riotee_data']:
            result['riotee_device_id'] = unified['riotee_data'].get('device_id')
            result['riotee_timestamp'] = unified['riotee_data'].get('timestamp')
            result['riotee_age_seconds'] = unified['riotee_data'].get('_data_age_seconds')
            result['temperature'] = unified['riotee_data'].get('temperature')
            result['humidity'] = unified['riotee_data'].get('humidity')
            result['a1_raw'] = unified['riotee_data'].get('a1_raw')
            result['vcap_raw'] = unified['riotee_data'].get('vcap_raw')
            result['spectral_gain'] = unified['riotee_data'].get('spectral_gain')
            result['sleep_time'] = unified['riotee_data'].get('sleep_time')
            
            # 添加光谱数据
            spectral = unified['riotee_data'].get('spectral', {})
            for key, value in spectral.items():
                result[f'riotee_{key}'] = value
        
        return result
    
    def get_data_summary(self, max_age_seconds=120):
        """获取数据摘要信息"""
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'systems': {
                'co2': {'available': self.co2_available, 'data': None},
                'riotee': {'available': self.riotee_available, 'data': None}
            },
            'combined_status': 'unknown'
        }
        
        # CO2摘要
        if self.co2_available:
            try:
                co2_data = get_co2_data(max_age_seconds)
                if co2_data:
                    summary['systems']['co2']['data'] = {
                        'value': co2_data.get('value'),
                        'age_seconds': co2_data.get('age_seconds'),
                        'is_fresh': not co2_data.get('is_stale', True)
                    }
            except Exception:
                pass
        
        # Riotee摘要
        if self.riotee_available:
            try:
                riotee_data = get_riotee_data(None, max_age_seconds)
                if riotee_data:
                    summary['systems']['riotee']['data'] = {
                        'device_id': riotee_data.get('device_id'),
                        'temperature': riotee_data.get('temperature'),
                        'humidity': riotee_data.get('humidity'),
                        'age_seconds': riotee_data.get('_data_age_seconds'),
                        'is_fresh': not riotee_data.get('_is_stale', True)
                    }
            except Exception:
                pass
        
        # 计算综合状态
        co2_ok = summary['systems']['co2']['data'] is not None
        riotee_ok = summary['systems']['riotee']['data'] is not None
        
        if co2_ok and riotee_ok:
            summary['combined_status'] = 'both_available'
        elif co2_ok or riotee_ok:
            summary['combined_status'] = 'partial_available'
        else:
            summary['combined_status'] = 'no_data'
        
        return summary

# 创建全局实例
_unified_reader = UnifiedDataReader()

def get_all_current_data(riotee_device_id=None, max_age_seconds=120, require_both=False):
    """
    获取当前所有传感器数据（最常用的接口）
    
    Args:
        riotee_device_id: 指定Riotee设备ID
        max_age_seconds: 数据最大时效性（秒）
        require_both: 是否要求CO2和Riotee数据都存在
        
    Returns:
        dict: 统一的传感器数据，包含CO2和Riotee信息
    """
    return _unified_reader.get_unified_data(riotee_device_id, max_age_seconds, require_both)

def get_simple_all_data(riotee_device_id=None, max_age_seconds=120):
    """
    获取简化格式的所有传感器数据
    
    Returns:
        dict: 扁平化的传感器数据字典
    """
    return _unified_reader.get_simple_data(riotee_device_id, max_age_seconds)

def get_co2_current():
    """获取当前CO2数据（兼容接口）"""
    return _unified_reader.get_co2_only()

def get_riotee_current(device_id=None):
    """获取当前Riotee数据（兼容接口）"""
    return _unified_reader.get_riotee_only(device_id)

def get_data_status():
    """获取数据系统状态"""
    return _unified_reader.get_system_status()

def get_quick_summary():
    """获取快速数据摘要"""
    return _unified_reader.get_data_summary()

def check_systems_ready():
    """检查所有系统是否就绪"""
    status = get_data_status()
    return status['systems_ready']

__all__ = [
    'get_all_current_data', 'get_simple_all_data', 
    'get_co2_current', 'get_riotee_current',
    'get_data_status', 'get_quick_summary', 'check_systems_ready'
]
