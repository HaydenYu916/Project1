#!/usr/bin/env python3
"""
测试 solar_vol 模型集成的脚本
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mppi import LEDPlant, LEDMPPIController
import numpy as np

def test_solar_vol_model():
    """测试 solar_vol 模型的基本功能"""
    print("=== 测试 solar_vol 模型集成 ===")
    
    try:
        # 创建 LEDPlant
        plant = LEDPlant()
        print("✓ LEDPlant 创建成功")
        
        # 测试不同的输入条件
        test_cases = [
            (50.0, 10.0, 0.1),  # 高红低蓝
            (10.0, 50.0, 0.1),  # 低红高蓝
            (30.0, 30.0, 0.1),  # 平衡
            (70.0, 20.0, 0.1),  # 高功率
            (20.0, 10.0, 0.1),  # 低功率
        ]
        
        print("\n=== 单步预测测试 ===")
        for i, (r_pwm, b_pwm, dt) in enumerate(test_cases):
            ppfd, temp, power, photo = plant.step(r_pwm, b_pwm, dt)
            print(f"测试 {i+1}: R_PWM={r_pwm}, B_PWM={b_pwm}")
            print(f"  结果: PPFD={ppfd:.2f}, Temp={temp:.2f}°C, Power={power:.2f}W, Photo={photo:.2f}")
        
        # 测试 MPPI 控制器
        print("\n=== MPPI 控制器测试 ===")
        controller = LEDMPPIController(
            plant, 
            horizon=10, 
            num_samples=500,
            maintain_rb_ratio=True,
            rb_ratio_key="5:1"
        )
        
        # 测试不同温度下的控制
        test_temps = [22.0, 25.0, 28.0]
        for temp in test_temps:
            action, sequence, success, cost, weights = controller.solve(temp)
            print(f"当前温度 {temp}°C:")
            print(f"  控制动作: R_PWM={action[0]:.2f}, B_PWM={action[1]:.2f}")
            print(f"  成功: {success}, 成本: {cost:.2f}")
        
        # 测试序列预测
        print("\n=== 序列预测测试 ===")
        pwm_sequence = np.array([[50, 10], [45, 15], [40, 20], [35, 25], [30, 30]])
        ppfd_pred, temp_pred, power_pred, photo_pred = plant.predict(pwm_sequence, 25.0)
        
        print("PWM序列预测结果:")
        for i in range(len(pwm_sequence)):
            print(f"  步骤 {i+1}: R_PWM={pwm_sequence[i,0]}, B_PWM={pwm_sequence[i,1]}")
            print(f"    预测: PPFD={ppfd_pred[i]:.2f}, Temp={temp_pred[i]:.2f}°C, Power={power_pred[i]:.2f}W, Photo={photo_pred[i]:.2f}")
        
        print("\n✓ 所有测试完成！solar_vol 模型工作正常。")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_solar_vol_model()
