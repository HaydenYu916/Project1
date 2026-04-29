
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速CO2显示
"""

import sys
import os
import time

# ==================== 配置宏定义 ====================
# CO2数据文件路径
CO2_FILE = "/data/csv/co2_sensor.csv"
# =====================================================

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')

try:
    from .sensor_reader import DemoSensorReader
except ImportError:
    # 兼容直接运行
    sys.path.insert(0, os.path.dirname(__file__))
    from sensor_reader import DemoSensorReader  # type: ignore

def read_co2():
    """通过统一的 SensorReading 读取当前CO2数据"""
    try:
        reader = DemoSensorReader(co2_data_path=CO2_FILE)
        co2, ts = reader.read_latest_co2_with_timestamp()
        if co2 is None or ts is None:
            return None
        age_seconds = max(0.0, time.time() - float(ts))
        return {
            'co2': float(co2),
            'timestamp': float(ts),
            'age_seconds': age_seconds,
        }
    except Exception as e:
        print(f"❌ CO2读取错误: {e}")
        return None

def main():
    print(f"📱 配置信息: CO2文件 = {CO2_FILE}")
    print("=" * 50)
    
    # 读取CO2数据
    data = read_co2()
    
    if data:
        co2_value = data['co2']
        age = data['age_seconds']
        
        # 状态指示
        if age < 120:  # 2分钟内为新鲜
            status = "🟢"
        elif age < 300:  # 2-5分钟为较旧
            status = "🟡"
        else:  # 超过5分钟为过期
            status = "🔴"
        
        print(f"🌬️  {status} CO2: {co2_value:.1f} ppm ({age:.0f}秒前)")
    else:
        print("❌ CO2数据读取失败")

if __name__ == "__main__":
    main()
