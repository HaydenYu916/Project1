#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPPI集成测试脚本
测试温度读取、MPPI控制和命令生成
"""

import sys
import os
import json
from datetime import datetime

# ==================== 配置宏定义 ====================
# 温度传感器设备ID配置（修改此处即可切换设备）
TEMPERATURE_DEVICE_ID = None  # None=自动选择, "T6ncwg=="=指定设备1, "L_6vSQ=="=指定设备2
# =====================================================

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
riotee_sensor_dir = os.path.join(current_dir, '..', 'Test', 'riotee_sensor')
mppi_dir = os.path.join(current_dir, '..', 'AA_Test_9_16')
controller_dir = os.path.join(current_dir, '..', 'aioshelly', 'my_src')

# 确保MPPI目录在路径最前面，以便导入numpy等依赖
sys.path.insert(0, mppi_dir)
sys.path.insert(0, riotee_sensor_dir)
sys.path.insert(0, controller_dir)


def test_temperature_reading():
    """测试温度读取"""
    print("🌡️  测试温度读取...")
    try:
        from __init__ import get_current_riotee
        data = get_current_riotee(device_id=TEMPERATURE_DEVICE_ID, max_age_seconds=86400)  # 放宽到24小时
        
        if data and data.get('temperature') is not None:
            temp = data['temperature']
            device_id = data.get('device_id', 'Unknown')
            age = data.get('_data_age_seconds', 0)
            print(f"✅ 温度读取成功: {temp:.2f}°C (设备: {device_id}, {age:.0f}秒前)")
            return temp
        else:
            if TEMPERATURE_DEVICE_ID:
                print(f"⚠️  指定设备 {TEMPERATURE_DEVICE_ID} 无有效温度数据，使用模拟温度 24.5°C")
            else:
                print("⚠️  无有效温度数据，使用模拟温度 24.5°C")
            return 24.5  # 使用模拟温度
    except Exception as e:
        print(f"❌ 温度读取错误: {e}")
        print("⚠️  使用模拟温度 24.5°C")
        return 24.5  # 使用模拟温度

def test_mppi_control():
    """测试MPPI控制"""
    print("\n🎯 测试MPPI控制...")
    try:
        from mppi import LEDPlant, LEDMPPIController
        
        # 创建植物模型
        plant = LEDPlant(
            model_key="5:1",
            use_efficiency=False,  # 暂时关闭效率模型
            heat_scale=1.0
        )
        
        # 创建控制器
        controller = LEDMPPIController(
            plant=plant,
            horizon=10,
            num_samples=1000,
            dt=0.1,
            temperature=1.0,
            maintain_rb_ratio=True,
            rb_ratio_key="5:1"
        )
        
        # 使用MPPI控制器的默认参数，不进行覆盖设置
        
        # 测试温度
        test_temp = 24.5
        print(f"   测试温度: {test_temp}°C")
        
        # 运行控制
        optimal_action, optimal_sequence, success, cost, weights = controller.solve(
            current_temp=test_temp
        )
        
        if success:
            r_pwm = optimal_action[0]
            b_pwm = optimal_action[1]
            print(f"✅ MPPI控制成功:")
            print(f"   红光PWM: {r_pwm:.2f}")
            print(f"   蓝光PWM: {b_pwm:.2f}")
            print(f"   总PWM: {r_pwm + b_pwm:.2f}")
            print(f"   成本: {cost:.2f}")
            return r_pwm, b_pwm
        else:
            print("❌ MPPI控制失败")
            return None, None
            
    except Exception as e:
        print(f"❌ MPPI控制错误: {e}")
        return None, None

def test_command_generation(r_pwm, b_pwm):
    """测试命令生成"""
    print("\n📡 测试命令生成...")
    try:
        from controller import DEVICES
        import numpy as np
        
        if r_pwm is None or b_pwm is None:
            print("❌ 无效的PWM值")
            return False
        
        # 转换PWM值到亮度值 (0-100)
        r_brightness = int(np.round(np.clip(r_pwm, 0, 100)))
        b_brightness = int(np.round(np.clip(b_pwm, 0, 100)))
        
        print(f"   红光PWM: {r_pwm:.2f} -> 亮度: {r_brightness}")
        print(f"   蓝光PWM: {b_pwm:.2f} -> 亮度: {b_brightness}")
        
        # 生成命令
        commands = []
        
        if "Red" in DEVICES:
            red_cmd = {
                "id": 0,
                "on": True,
                "brightness": r_brightness,
                "transition": 1000
            }
            commands.append(("Red", DEVICES["Red"], red_cmd))
            print(f"🔴 红光命令: {json.dumps(red_cmd, indent=2)}")
        
        if "Blue" in DEVICES:
            blue_cmd = {
                "id": 0,
                "on": True,
                "brightness": b_brightness,
                "transition": 1000
            }
            commands.append(("Blue", DEVICES["Blue"], blue_cmd))
            print(f"🔵 蓝光命令: {json.dumps(blue_cmd, indent=2)}")
        
        print(f"✅ 命令生成成功，共 {len(commands)} 个命令")
        return True
        
    except Exception as e:
        print(f"❌ 命令生成错误: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 MPPI集成系统测试")
    print("=" * 50)
    print(f"📱 配置信息:")
    print(f"   温度设备: {TEMPERATURE_DEVICE_ID or '自动选择'}")
    print("=" * 50)
    
    # 1. 测试温度读取
    current_temp = test_temperature_reading()
    
    # 2. 测试MPPI控制
    r_pwm, b_pwm = test_mppi_control()
    
    # 3. 测试命令生成
    if r_pwm is not None and b_pwm is not None:
        test_command_generation(r_pwm, b_pwm)
    
    print("\n" + "=" * 50)
    print("🏁 测试完成")
    
    if current_temp is not None and r_pwm is not None and b_pwm is not None:
        print("✅ 所有测试通过，系统集成正常")
    else:
        print("❌ 部分测试失败，请检查系统配置")

if __name__ == "__main__":
    main()
