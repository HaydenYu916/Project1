#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速温度显示
"""

import sys
import os
import time

# ==================== 配置宏定义 ====================
# 温度传感器设备ID配置（修改此处即可切换设备）
TEMPERATURE_DEVICE_ID = "T6ncwg=="  # None=自动选择, "T6ncwg=="=指定设备1, "L_6vSQ=="=指定设备2
# =====================================================

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')

try:
    from .sensor_reader import DemoSensorReader, DEFAULT_DEVICE_ID
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from sensor_reader import DemoSensorReader, DEFAULT_DEVICE_ID  # type: ignore

def main():
    print(f"📱 配置信息: 温度设备 = {TEMPERATURE_DEVICE_ID or '自动选择'}")
    print("=" * 50)
    
    # 读取温度（以及可选的 solar_vol、pn_avg）
    reader = DemoSensorReader(device_id=TEMPERATURE_DEVICE_ID or DEFAULT_DEVICE_ID)
    temp, _solar_vol, _pn, ts = reader.read_latest_riotee_data()

    if temp is None or ts is None:
        if TEMPERATURE_DEVICE_ID:
            print(f"❌ 指定设备 {TEMPERATURE_DEVICE_ID} 无数据")
        else:
            print("❌ 无数据")
        return

    age = max(0.0, time.time() - ts.timestamp())
    # 状态指示
    if age < 120:
        status = "🟢"
    elif age < 300:
        status = "🟡"
    else:
        status = "🔴"

    print(f"🌡️  {status} 设备 {TEMPERATURE_DEVICE_ID or DEFAULT_DEVICE_ID}: {float(temp):.2f}°C ({age:.0f}秒前)")

if __name__ == "__main__":
    main()
