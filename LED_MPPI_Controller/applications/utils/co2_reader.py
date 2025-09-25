
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速CO2显示
"""

import sys
import os
import pandas as pd
import time

# ==================== 配置宏定义 ====================
# CO2数据文件路径
CO2_FILE = "/data/csv/co2_sensor.csv"
# =====================================================

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')

def read_co2():
    """读取当前CO2数据"""
    try:
        if not os.path.exists(CO2_FILE):
            print("⚠️  CO2文件不存在")
            return None
        
        # 读取CO2数据文件
        df = pd.read_csv(CO2_FILE, header=None, names=['timestamp', 'co2'])
        
        if df.empty:
            print("⚠️  CO2文件为空")
            return None
        
        # 获取最新的有效CO2值
        latest_row = df.iloc[-1]
        latest_timestamp = latest_row['timestamp']
        latest_co2 = latest_row['co2']
        
        # 检查CO2值是否有效
        if pd.isna(latest_co2) or latest_co2 is None:
            print("⚠️  最新CO2值无效")
            return None
        
        # 计算数据年龄（秒）
        current_time = time.time()
        age_seconds = current_time - latest_timestamp
        
        return {
            'co2': latest_co2,
            'timestamp': latest_timestamp,
            'age_seconds': age_seconds
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

