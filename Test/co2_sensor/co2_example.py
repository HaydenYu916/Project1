#!/usr/bin/env python3
"""
CO2数据使用示例
演示如何从其他文件中简单获取CO2数据
"""

import time
import sys
import os

# 确保能找到CO2_Sensor模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CO2_Sensor import get_current_co2, get_co2_data, get_co2_average

def simple_example():
    """最简单的使用方式 - 只获取当前值"""
    print("🔍 简单获取当前CO2值:")
    
    co2_value = get_current_co2()
    if co2_value is not None:
        print(f"✅ 当前CO2: {co2_value} ppm")
    else:
        print("❌ 无法获取CO2数据（请确保数据采集器正在运行）")

def detailed_example():
    """详细使用方式 - 获取完整数据信息"""
    print("\n📊 获取详细CO2数据:")
    
    data = get_co2_data()
    if data:
        print(f"CO2值: {data['value']} ppm")
        print(f"时间戳: {data['timestamp']}")
        print(f"数据年龄: {data['age_seconds']} 秒")
        print(f"是否过时: {'是' if data['is_stale'] else '否'}")
    else:
        print("❌ 无法获取CO2数据")

def control_logic_example():
    """控制逻辑示例"""
    print("\n🎛️  控制逻辑示例:")
    
    # 设置目标值和阈值
    target_co2 = 400
    tolerance = 50
    
    current_co2 = get_current_co2()
    if current_co2 is not None:
        print(f"当前CO2: {current_co2} ppm (目标: {target_co2}±{tolerance})")
        
        if current_co2 > target_co2 + tolerance:
            print("🔴 CO2过高 - 建议增加通风")
            # 这里可以执行控制操作
            # control_ventilation("increase")
        elif current_co2 < target_co2 - tolerance:
            print("🔵 CO2过低 - 建议减少通风")
            # control_ventilation("decrease")
        else:
            print("✅ CO2水平正常")
    else:
        print("❌ 无法获取CO2数据进行控制判断")

def monitoring_loop_example():
    """监控循环示例"""
    print("\n🔄 监控循环示例 (运行10秒):")
    
    start_time = time.time()
    readings = []
    
    while time.time() - start_time < 10:
        co2_value = get_current_co2()
        if co2_value is not None:
            readings.append(co2_value)
            print(f"📊 {time.strftime('%H:%M:%S')} - CO2: {co2_value} ppm")
        else:
            print(f"❌ {time.strftime('%H:%M:%S')} - 无数据")
        
        time.sleep(2)
    
    if readings:
        avg = sum(readings) / len(readings)
        print(f"\n📈 监控期间平均CO2: {avg:.1f} ppm")
        print(f"📊 监控期间范围: {min(readings):.1f} - {max(readings):.1f} ppm")

def main():
    print("🏭 CO2数据获取使用示例")
    print("=" * 40)
    print("请确保CO2数据采集器正在运行:")
    print("cd CO2_Reading && python3 co2_system.py start")
    print("=" * 40)
    
    # 运行各种示例
    simple_example()
    detailed_example()
    control_logic_example()
    monitoring_loop_example()
    
    print("\n💡 在您的代码中使用:")
    print("```python")
    print("from CO2_Reading import get_current_co2")
    print("")
    print("# 获取当前CO2值")
    print("co2 = get_current_co2()")
    print("if co2 is not None:")
    print("    print(f'当前CO2: {co2} ppm')")
    print("```")

if __name__ == '__main__':
    main()
