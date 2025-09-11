#!/usr/bin/env python3
"""
统一数据读取使用示例
演示如何同时获取CO2和Riotee数据
"""

import time
import sys
import os

# 确保能找到所有模块 - 添加项目根目录到搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensor_hub import (
    get_all_current_data, get_simple_all_data,
    get_co2_current, get_riotee_current,
    get_data_status, get_quick_summary, check_systems_ready
)

def example_1_system_status():
    """示例1: 检查系统状态"""
    print("=" * 50)
    print("示例1: 检查系统状态")
    print("=" * 50)
    
    status = get_data_status()
    print(f"系统时间: {status['timestamp']}")
    print(f"CO2系统: {'✅ 可用' if status['co2_available'] else '❌ 不可用'}")
    print(f"Riotee系统: {'✅ 可用' if status['riotee_available'] else '❌ 不可用'}")
    print(f"系统就绪: {'✅ 是' if status['systems_ready'] else '❌ 否'}")
    
    if not status['systems_ready']:
        print("\n⚠️  提示:")
        if not status['co2_available']:
            print("   - 启动CO2系统: cd co2_sensor && python3 co2_system_manager.py start")
        if not status['riotee_available']:
            print("   - 启动Riotee系统: cd riotee_sensor && python3 riotee_system_manager.py start")

def example_2_get_unified_data():
    """示例2: 获取统一数据"""
    print("\n" + "=" * 50)
    print("示例2: 获取统一传感器数据")
    print("=" * 50)
    
    # 获取完整的统一数据
    data = get_all_current_data(require_both=False)
    
    if data:
        print(f"数据时间: {data['timestamp']}")
        print(f"状态: CO2={data['status']['co2_available']}, "
              f"Riotee={data['status']['riotee_available']}, "
              f"完整={data['status']['both_fresh']}")
        
        # 显示CO2数据
        if data['co2_data']:
            co2 = data['co2_data']
            print(f"\n🌬️  CO2数据:")
            print(f"   值: {co2.get('value')} ppm")
            print(f"   时间: {co2.get('timestamp')}")
            print(f"   年龄: {co2.get('age_seconds')} 秒")
        else:
            print("\n❌ CO2数据不可用")
        
        # 显示Riotee数据
        if data['riotee_data']:
            riotee = data['riotee_data']
            print(f"\n🌡️  Riotee数据:")
            print(f"   设备: {riotee.get('device_id')}")
            print(f"   温度: {riotee.get('temperature')}°C")
            print(f"   湿度: {riotee.get('humidity')}%")
            print(f"   A1电压: {riotee.get('a1_raw')}V")
            print(f"   VCAP电压: {riotee.get('vcap_raw')}V")
            print(f"   增益: {riotee.get('spectral_gain')}")
            print(f"   时间: {riotee.get('timestamp')}")
            print(f"   年龄: {riotee.get('_data_age_seconds')} 秒")
            
            # 显示光谱数据
            spectral = riotee.get('spectral', {})
            if spectral:
                print(f"   光谱数据:")
                for wl, value in spectral.items():
                    if value is not None and value > 0:
                        print(f"     {wl}: {value}")
        else:
            print("\n❌ Riotee数据不可用")
    else:
        print("❌ 无法获取统一数据")

def example_3_get_simple_data():
    """示例3: 获取简化数据格式"""
    print("\n" + "=" * 50)
    print("示例3: 获取简化数据格式")
    print("=" * 50)
    
    data = get_simple_all_data()
    
    if data:
        print(f"时间戳: {data['timestamp']}")
        
        # CO2数据
        if data.get('co2_value') is not None:
            print(f"CO2: {data['co2_value']} ppm ({data.get('co2_age_seconds', 0)}秒前)")
        
        # 环境数据
        if data.get('temperature') is not None:
            print(f"温度: {data['temperature']}°C")
        if data.get('humidity') is not None:
            print(f"湿度: {data['humidity']}%")
        
        # 电压数据
        if data.get('a1_raw') is not None:
            print(f"A1电压: {data['a1_raw']}V")
        if data.get('vcap_raw') is not None:
            print(f"VCAP电压: {data['vcap_raw']}V")
        
        # Riotee设备信息
        if data.get('riotee_device_id'):
            print(f"Riotee设备: {data['riotee_device_id']} ({data.get('riotee_age_seconds', 0)}秒前)")
    else:
        print("❌ 无法获取简化数据")

def example_4_monitoring_loop():
    """示例4: 监控循环"""
    print("\n" + "=" * 50)
    print("示例4: 数据监控循环 (运行10次)")
    print("=" * 50)
    
    for i in range(10):
        print(f"\n第 {i+1} 次检查:")
        
        # 获取简化数据
        data = get_simple_all_data()
        
        if data:
            # 构建状态字符串
            status_parts = []
            
            if data.get('co2_value') is not None:
                status_parts.append(f"CO2={data['co2_value']}ppm")
            
            if data.get('temperature') is not None:
                status_parts.append(f"T={data['temperature']:.1f}°C")
            
            if data.get('humidity') is not None:
                status_parts.append(f"H={data['humidity']:.1f}%")
            
            if data.get('riotee_device_id'):
                status_parts.append(f"设备={data['riotee_device_id']}")
            
            print(f"  📊 {', '.join(status_parts)}")
            
            # 简单的报警逻辑
            if data.get('co2_value') and data['co2_value'] > 1000:
                print("  ⚠️  CO2浓度过高警告!")
            
            if data.get('temperature') and data['temperature'] > 30:
                print("  🔥 温度过高警告!")
            elif data.get('temperature') and data['temperature'] < 15:
                print("  ❄️  温度较低提醒")
        else:
            print("  ❌ 无数据")
        
        if i < 9:  # 最后一次不等待
            time.sleep(2)

def example_5_data_summary():
    """示例5: 数据摘要"""
    print("\n" + "=" * 50)
    print("示例5: 数据摘要")
    print("=" * 50)
    
    summary = get_quick_summary()
    
    print(f"摘要时间: {summary['timestamp']}")
    print(f"综合状态: {summary['combined_status']}")
    
    # CO2系统摘要
    co2_sys = summary['systems']['co2']
    print(f"\nCO2系统:")
    print(f"  可用性: {'✅' if co2_sys['available'] else '❌'}")
    if co2_sys['data']:
        co2_data = co2_sys['data']
        fresh_icon = "🟢" if co2_data['is_fresh'] else "🟡"
        print(f"  数据: {fresh_icon} {co2_data['value']} ppm ({co2_data['age_seconds']}秒前)")
    else:
        print(f"  数据: ❌ 无数据")
    
    # Riotee系统摘要
    riotee_sys = summary['systems']['riotee']
    print(f"\nRiotee系统:")
    print(f"  可用性: {'✅' if riotee_sys['available'] else '❌'}")
    if riotee_sys['data']:
        riotee_data = riotee_sys['data']
        fresh_icon = "🟢" if riotee_data['is_fresh'] else "🟡"
        print(f"  设备: {riotee_data['device_id']}")
        print(f"  数据: {fresh_icon} T={riotee_data['temperature']}°C, H={riotee_data['humidity']}% ({riotee_data['age_seconds']}秒前)")
    else:
        print(f"  数据: ❌ 无数据")

def example_6_control_logic():
    """示例6: 控制逻辑示例"""
    print("\n" + "=" * 50)
    print("示例6: 智能控制逻辑")
    print("=" * 50)
    
    # 设定阈值
    CO2_TARGET = 400
    CO2_TOLERANCE = 50
    TEMP_MIN = 20
    TEMP_MAX = 25
    
    data = get_simple_all_data()
    
    if data:
        actions = []
        
        # CO2控制逻辑
        co2_value = data.get('co2_value')
        if co2_value:
            if co2_value > CO2_TARGET + CO2_TOLERANCE:
                actions.append("🌬️  增加通风 (CO2过高)")
            elif co2_value < CO2_TARGET - CO2_TOLERANCE:
                actions.append("🔒 减少通风 (CO2过低)")
            else:
                actions.append("✅ CO2正常")
        
        # 温度控制逻辑
        temperature = data.get('temperature')
        if temperature:
            if temperature > TEMP_MAX:
                actions.append("❄️  启动制冷 (温度过高)")
            elif temperature < TEMP_MIN:
                actions.append("🔥 启动加热 (温度过低)")
            else:
                actions.append("✅ 温度正常")
        
        # 湿度监控
        humidity = data.get('humidity')
        if humidity:
            if humidity > 70:
                actions.append("💨 启动除湿 (湿度过高)")
            elif humidity < 30:
                actions.append("💧 启动加湿 (湿度过低)")
            else:
                actions.append("✅ 湿度正常")
        
        # 电压监控
        vcap = data.get('vcap_raw')
        if vcap and vcap < 3.0:
            actions.append("⚡ 设备电压低，检查电源")
        
        print("控制建议:")
        for action in actions:
            print(f"  {action}")
        
        # 记录当前状态
        print(f"\n当前状态:")
        if co2_value: print(f"  CO2: {co2_value} ppm (目标: {CO2_TARGET}±{CO2_TOLERANCE})")
        if temperature: print(f"  温度: {temperature}°C (范围: {TEMP_MIN}-{TEMP_MAX})")
        if humidity: print(f"  湿度: {humidity}% (范围: 30-70)")
        if vcap: print(f"  设备电压: {vcap}V")
    else:
        print("❌ 无法获取数据进行控制决策")

def main():
    print("🏭 统一传感器数据系统使用示例")
    print("=" * 60)
    print("请确保相关数据采集器正在运行:")
    print("- CO2: cd co2_sensor && python3 co2_system_manager.py start")
    print("- Riotee: cd riotee_sensor && python3 riotee_system_manager.py start")
    print("=" * 60)
    
    try:
        # 运行所有示例
        example_1_system_status()
        example_2_get_unified_data()
        example_3_get_simple_data()
        example_4_monitoring_loop()
        example_5_data_summary()
        example_6_control_logic()
        
        print("\n" + "=" * 50)
        print("✅ 所有示例运行完成!")
        print("=" * 50)
        
        print("\n💡 在您的代码中使用:")
        print("```python")
        print("from Sensor_Hub import get_simple_all_data")
        print("")
        print("# 获取所有传感器数据")
        print("data = get_simple_all_data()")
        print("if data:")
        print("    co2 = data.get('co2_value')")
        print("    temp = data.get('temperature')")
        print("    humidity = data.get('humidity')")
        print("    print(f'CO2: {co2}ppm, 温度: {temp}°C, 湿度: {humidity}%')")
        print("```")
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断程序")
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")

if __name__ == '__main__':
    main()
