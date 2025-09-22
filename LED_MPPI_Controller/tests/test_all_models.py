#!/usr/bin/env python3
"""
测试所有模型（solar_vol, ppfd, sp）集成的脚本
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mppi import LEDPlant, LEDMPPIController
import numpy as np

def test_all_models():
    """测试所有模型的基本功能"""
    print("=== 测试所有模型集成 ===")
    
    models_to_test = ['solar_vol', 'ppfd', 'sp']
    results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*50}")
        print(f"测试 {model_name.upper()} 模型")
        print(f"{'='*50}")
        
        try:
            # 创建 LEDPlant
            plant = LEDPlant(model_name=model_name)
            print(f"✓ LEDPlant ({model_name}) 创建成功")
            
            # 测试不同的输入条件
            test_cases = [
                (50.0, 10.0, 0.1, "高红低蓝"),
                (10.0, 50.0, 0.1, "低红高蓝"),
                (30.0, 30.0, 0.1, "平衡"),
                (70.0, 20.0, 0.1, "高功率"),
                (20.0, 10.0, 0.1, "低功率"),
            ]
            
            print(f"\n--- {model_name} 单步预测测试 ---")
            model_results = []
            for i, (r_pwm, b_pwm, dt, desc) in enumerate(test_cases):
                ppfd, temp, power, photo = plant.step(r_pwm, b_pwm, dt)
                result = {
                    'case': desc,
                    'r_pwm': r_pwm,
                    'b_pwm': b_pwm,
                    'ppfd': ppfd,
                    'temp': temp,
                    'power': power,
                    'photo': photo
                }
                model_results.append(result)
                print(f"测试 {i+1} ({desc}): R_PWM={r_pwm}, B_Pwm={b_pwm}")
                print(f"  结果: PPFD={ppfd:.2f}, Temp={temp:.2f}°C, Power={power:.2f}W, Photo={photo:.2f}")
            
            # 测试 MPPI 控制器
            print(f"\n--- {model_name} MPPI 控制器测试 ---")
            controller = LEDMPPIController(
                plant, 
                horizon=10, 
                num_samples=500,
                maintain_rb_ratio=True,
                rb_ratio_key="5:1"
            )
            
            # 测试不同温度下的控制
            test_temps = [22.0, 25.0, 28.0]
            controller_results = []
            for temp in test_temps:
                action, sequence, success, cost, weights = controller.solve(temp)
                result = {
                    'temp': temp,
                    'action': action,
                    'success': success,
                    'cost': cost
                }
                controller_results.append(result)
                print(f"当前温度 {temp}°C:")
                print(f"  控制动作: R_PWM={action[0]:.2f}, B_PWM={action[1]:.2f}")
                print(f"  成功: {success}, 成本: {cost:.2f}")
            
            # 测试序列预测
            print(f"\n--- {model_name} 序列预测测试 ---")
            pwm_sequence = np.array([[50, 10], [45, 15], [40, 20], [35, 25], [30, 30]])
            ppfd_pred, temp_pred, power_pred, photo_pred = plant.predict(pwm_sequence, 25.0)
            
            print("PWM序列预测结果:")
            sequence_results = []
            for i in range(len(pwm_sequence)):
                result = {
                    'step': i+1,
                    'r_pwm': pwm_sequence[i,0],
                    'b_pwm': pwm_sequence[i,1],
                    'ppfd': ppfd_pred[i],
                    'temp': temp_pred[i],
                    'power': power_pred[i],
                    'photo': photo_pred[i]
                }
                sequence_results.append(result)
                print(f"  步骤 {i+1}: R_PWM={pwm_sequence[i,0]}, B_PWM={pwm_sequence[i,1]}")
                print(f"    预测: PPFD={ppfd_pred[i]:.2f}, Temp={temp_pred[i]:.2f}°C, Power={power_pred[i]:.2f}W, Photo={photo_pred[i]:.2f}")
            
            # 保存结果
            results[model_name] = {
                'single_step': model_results,
                'controller': controller_results,
                'sequence': sequence_results,
                'status': 'success'
            }
            
            print(f"\n✓ {model_name} 模型测试完成")
            
        except Exception as e:
            print(f"✗ {model_name} 模型测试失败: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # 模型比较
    print(f"\n{'='*50}")
    print("模型性能比较")
    print(f"{'='*50}")
    
    if all(results[model]['status'] == 'success' for model in models_to_test):
        print("\n--- 相同输入下的预测结果比较 ---")
        test_case = (50.0, 10.0, 0.1)  # 使用第一个测试案例
        
        for model_name in models_to_test:
            plant = LEDPlant(model_name=model_name)
            ppfd, temp, power, photo = plant.step(*test_case)
            print(f"{model_name:>10}: PPFD={ppfd:.2f}, Temp={temp:.2f}°C, Power={power:.2f}W, Photo={photo:.2f}")
    
    print(f"\n{'='*50}")
    print("测试总结")
    print(f"{'='*50}")
    
    success_count = sum(1 for model in models_to_test if results[model]['status'] == 'success')
    total_count = len(models_to_test)
    
    print(f"成功测试的模型: {success_count}/{total_count}")
    for model_name in models_to_test:
        status = "✓" if results[model_name]['status'] == 'success' else "✗"
        print(f"  {status} {model_name}")
    
    if success_count == total_count:
        print("\n🎉 所有模型测试成功！")
        print("\n使用方法:")
        print("  # 使用 solar_vol 模型")
        print("  plant = LEDPlant(model_name='solar_vol')")
        print("  # 使用 ppfd 模型")
        print("  plant = LEDPlant(model_name='ppfd')")
        print("  # 使用 sp 模型")
        print("  plant = LEDPlant(model_name='sp')")
    else:
        print(f"\n⚠️  有 {total_count - success_count} 个模型测试失败")
    
    return results

if __name__ == "__main__":
    test_all_models()
