#!/usr/bin/env python3
"""
多模型使用示例脚本
演示如何在代码中选择使用不同的模型
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mppi import LEDPlant, LEDMPPIController
import numpy as np

def compare_models():
    """比较不同模型的预测结果"""
    print("=== 模型比较示例 ===")
    
    models = ['solar_vol', 'ppfd', 'sp']
    test_conditions = [
        (50.0, 10.0, "高红低蓝"),
        (10.0, 50.0, "低红高蓝"),
        (30.0, 30.0, "平衡"),
    ]
    
    print(f"{'模型':<12} {'条件':<8} {'PPFD':<8} {'Temp':<6} {'Power':<7} {'Photo':<8}")
    print("-" * 60)
    
    for model_name in models:
        plant = LEDPlant(model_name=model_name)
        
        for r_pwm, b_pwm, desc in test_conditions:
            ppfd, temp, power, photo = plant.step(r_pwm, b_pwm, 0.1)
            print(f"{model_name:<12} {desc:<8} {ppfd:<8.1f} {temp:<6.1f} {power:<7.1f} {photo:<8.2f}")

def mppi_control_example():
    """MPPI 控制示例"""
    print("\n=== MPPI 控制示例 ===")
    
    # 使用 solar_vol 模型进行控制
    plant = LEDPlant(model_name='solar_vol')
    controller = LEDMPPIController(
        plant, 
        horizon=10, 
        num_samples=500,
        maintain_rb_ratio=True,
        rb_ratio_key="5:1"
    )
    
    # 模拟控制循环
    current_temp = 25.0
    print(f"初始温度: {current_temp}°C")
    
    for step in range(5):
        action, sequence, success, cost, weights = controller.solve(current_temp)
        r_pwm, b_pwm = action
        
        # 模拟温度变化
        ppfd, new_temp, power, photo = plant.step(r_pwm, b_pwm, 0.1)
        current_temp = new_temp
        
        print(f"步骤 {step+1}: 控制动作 R_PWM={r_pwm:.1f}, B_PWM={b_pwm:.1f}")
        print(f"        结果: PPFD={ppfd:.1f}, Temp={current_temp:.1f}°C, Photo={photo:.2f}")
        print(f"        成本: {cost:.2f}")

def sequence_prediction_example():
    """序列预测示例"""
    print("\n=== 序列预测示例 ===")
    
    # 使用 ppfd 模型进行序列预测
    plant = LEDPlant(model_name='ppfd')
    
    # 定义控制序列：从高功率逐渐降低
    pwm_sequence = np.array([
        [60, 20],  # 高功率
        [50, 15],  # 中等功率
        [40, 10],  # 低功率
        [30, 8],   # 更低功率
        [20, 5],   # 最低功率
    ])
    
    print("PWM序列:")
    for i, (r_pwm, b_pwm) in enumerate(pwm_sequence):
        print(f"  步骤 {i+1}: R_PWM={r_pwm}, B_PWM={b_pwm}")
    
    # 预测整个序列
    ppfd_pred, temp_pred, power_pred, photo_pred = plant.predict(pwm_sequence, 25.0)
    
    print("\n预测结果:")
    for i in range(len(pwm_sequence)):
        print(f"  步骤 {i+1}: PPFD={ppfd_pred[i]:.1f}, Temp={temp_pred[i]:.1f}°C, "
              f"Power={power_pred[i]:.1f}W, Photo={photo_pred[i]:.2f}")

def model_switching_example():
    """模型切换示例"""
    print("\n=== 模型切换示例 ===")
    
    # 相同的输入条件
    r_pwm, b_pwm = 40.0, 15.0
    
    print(f"输入条件: R_PWM={r_pwm}, B_PWM={b_pwm}")
    print()
    
    models = ['solar_vol', 'ppfd', 'sp']
    
    for model_name in models:
        plant = LEDPlant(model_name=model_name)
        ppfd, temp, power, photo = plant.step(r_pwm, b_pwm, 0.1)
        
        print(f"{model_name.upper()} 模型:")
        print(f"  PPFD: {ppfd:.2f}")
        print(f"  温度: {temp:.2f}°C")
        print(f"  功率: {power:.2f}W")
        print(f"  光合作用速率: {photo:.2f}")
        print()

if __name__ == "__main__":
    print("多模型使用示例")
    print("=" * 50)
    
    try:
        compare_models()
        mppi_control_example()
        sequence_prediction_example()
        model_switching_example()
        
        print("\n✓ 所有示例运行成功！")
        print("\n使用提示:")
        print("1. 根据应用场景选择合适的模型")
        print("2. Solar_Vol 模型通常提供最稳定的预测")
        print("3. PPFD 模型可能提供更高的光合作用速率预测")
        print("4. SP 模型适合需要光谱信息的应用")
        
    except Exception as e:
        print(f"✗ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()
