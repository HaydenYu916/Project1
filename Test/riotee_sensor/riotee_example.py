#!/usr/bin/env python3
"""
Live Data API 使用示例
=====================

展示如何在您的Python程序中调用live_data_api.py的各种API函数
"""

import sys
import json
from datetime import datetime

# 导入API函数
from live_data_api import (
    get_latest_data,
    get_device_latest_data, 
    get_recent_data,
    get_spectral_data,
    get_device_config,
    check_system_status,
    quick_status
)

def example_1_get_latest_data():
    """示例1: 获取最新一条数据"""
    print("=" * 50)
    print("示例1: 获取最新数据")
    print("=" * 50)
    
    # 获取完整的最新数据
    data = get_latest_data()
    if data:
        print(f"设备ID: {data['device_id']}")
        print(f"温度: {data['temperature']:.1f}°C")
        print(f"湿度: {data['humidity']:.1f}%")
        print(f"数据时间: {data['timestamp']}")
        print(f"数据年龄: {data.get('_data_age_seconds', 0):.1f}秒")
        print(f"数据新鲜: {'是' if data.get('_is_fresh') else '否'}")
        
        # 获取光谱数据
        if 'spectral' in data:
            spectral = data['spectral']
            print(f"555nm通道: {spectral.get('sp_555', 0):.1f}")
            print(f"Clear通道: {spectral.get('sp_clear', 0):.1f}")
        
        # 获取设备配置
        if 'config' in data:
            config = data['config']
            print(f"光谱增益: {config.get('spectral_gain', 0)}X")
            print(f"休眠时间: {config.get('sleep_time', 0)}秒")
    else:
        print("未获取到数据")

def example_2_get_device_data():
    """示例2: 获取指定设备数据"""
    print("\n" + "=" * 50)
    print("示例2: 获取指定设备数据")
    print("=" * 50)
    
    # 获取特定设备的最新数据
    device_id = "T6ncwg=="  # 替换为您的设备ID
    data = get_device_latest_data(device_id)
    
    if data:
        print(f"设备 {device_id} 的最新数据:")
        print(f"  温度: {data['temperature']:.1f}°C")
        print(f"  湿度: {data['humidity']:.1f}%")
        print(f"  A1电压: {data['a1_raw']:.3f}V")
        print(f"  VCAP电压: {data['vcap_raw']:.3f}V")
        print(f"  数据时间: {data['timestamp']}")
    else:
        print(f"设备 {device_id} 未找到数据")

def example_3_get_recent_data():
    """示例3: 获取最近N条记录"""
    print("\n" + "=" * 50)
    print("示例3: 获取最近5条记录")
    print("=" * 50)
    
    # 获取最近5条记录
    recent_data = get_recent_data(count=5)
    
    print(f"获取到 {len(recent_data)} 条记录:")
    for i, record in enumerate(recent_data, 1):
        print(f"  {i}. 设备:{record['device_id']} "
              f"温度:{record['temperature']:.1f}°C "
              f"时间:{record['timestamp']}")

def example_4_get_spectral_data():
    """示例4: 获取光谱数据"""
    print("\n" + "=" * 50)
    print("示例4: 获取光谱数据")
    print("=" * 50)
    
    # 获取光谱数据
    spectral_data = get_spectral_data()
    
    if spectral_data:
        print(f"设备: {spectral_data['device_id']}")
        print(f"增益: {spectral_data['spectral_gain']}X")
        print("光谱通道数据:")
        
        channels = spectral_data['channels']
        for channel, value in channels.items():
            if value is not None:
                print(f"  {channel}: {value:.1f}")
    else:
        print("未获取到光谱数据")

def example_5_check_system_status():
    """示例5: 检查系统状态"""
    print("\n" + "=" * 50)
    print("示例5: 系统状态检查")
    print("=" * 50)
    
    # 快速状态检查
    quick_st = quick_status()
    print(f"快速状态: {quick_st}")
    
    # 详细状态检查
    status = check_system_status()
    print(f"数据可用: {'是' if status['data_available'] else '否'}")
    print(f"CSV文件: {status['csv_file']}")
    print(f"总记录数: {status['total_records']}")
    print(f"活跃设备数: {len(status['active_devices'])}")
    
    if status['latest_data_time']:
        age = status.get('data_age_seconds', 0)
        print(f"最新数据: {status['latest_data_time']} ({age:.1f}秒前)")

def example_6_monitoring_loop():
    """示例6: 简单的监控循环"""
    print("\n" + "=" * 50)
    print("示例6: 监控循环 (运行5次)")
    print("=" * 50)
    
    import time
    
    for i in range(5):
        print(f"\n第 {i+1} 次检查:")
        
        # 获取最新数据
        data = get_latest_data()
        if data:
            temp = data['temperature']
            humidity = data['humidity']
            device = data['device_id']
            age = data.get('_data_age_seconds', 0)
            
            print(f"  设备 {device}: T={temp:.1f}°C, H={humidity:.1f}%, 数据{age:.1f}秒前")
            
            # 简单的温度报警
            if temp > 30:
                print("  ⚠️  温度过高警告!")
            elif temp < 20:
                print("  ❄️  温度较低提醒")
        else:
            print("  ❌ 无数据")
        
        if i < 4:  # 最后一次不等待
            time.sleep(2)

def example_7_custom_analysis():
    """示例7: 自定义数据分析"""
    print("\n" + "=" * 50)
    print("示例7: 自定义数据分析")
    print("=" * 50)
    
    # 获取最近10条记录进行分析
    recent_data = get_recent_data(count=10)
    
    if recent_data:
        # 计算平均温度
        temperatures = [r['temperature'] for r in recent_data if r['temperature'] is not None]
        if temperatures:
            avg_temp = sum(temperatures) / len(temperatures)
            max_temp = max(temperatures)
            min_temp = min(temperatures)
            
            print(f"最近10条记录温度分析:")
            print(f"  平均温度: {avg_temp:.2f}°C")
            print(f"  最高温度: {max_temp:.2f}°C")
            print(f"  最低温度: {min_temp:.2f}°C")
            print(f"  温度范围: {max_temp - min_temp:.2f}°C")
        
        # 统计设备数据量
        device_counts = {}
        for record in recent_data:
            device = record['device_id']
            device_counts[device] = device_counts.get(device, 0) + 1
        
        print(f"\n设备数据分布:")
        for device, count in device_counts.items():
            print(f"  {device}: {count} 条记录")
    else:
        print("无足够数据进行分析")

if __name__ == "__main__":
    print("🚀 Live Data API 使用示例")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 运行所有示例
        example_1_get_latest_data()
        example_2_get_device_data()
        example_3_get_recent_data()
        example_4_get_spectral_data()
        example_5_check_system_status()
        example_6_monitoring_loop()
        example_7_custom_analysis()
        
        print("\n" + "=" * 50)
        print("✅ 所有示例运行完成!")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断程序")
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")

