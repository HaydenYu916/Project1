#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速温度显示
"""

import sys
import os

# 添加 riotee_sensor 目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
riotee_sensor_dir = os.path.join(current_dir, '..', 'Test', 'riotee_sensor')
riotee_sensor_path = os.path.abspath(riotee_sensor_dir)
sys.path.insert(0, riotee_sensor_path)

try:
    from __init__ import get_current_riotee
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def main():
    # 读取温度数据
    data = get_current_riotee(max_age_seconds=86400)
    
    if data:
        temp = data.get('temperature')
        device = data.get('device_id', 'Unknown')
        age = data.get('_data_age_seconds', 0)
        
        if temp is not None:
            # 状态指示
            if age < 120:  # 2分钟内为新鲜
                status = "🟢"
            elif age < 300:  # 2-5分钟为较旧
                status = "🟡"
            else:  # 超过5分钟为过期
                status = "🔴"
            
            print(f"🌡️  {status} 设备 {device}: {temp:.2f}°C ({age:.0f}秒前)")
        else:
            print(f"❌ 设备 {device}: 温度数据无效")
    else:
        print("❌ 无数据")

if __name__ == "__main__":
    main()
