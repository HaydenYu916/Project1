#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证日志PN值Demo
读取control_real_log.csv中的实际控制数据，用模型验证PN预测结果

运行:
    /home/pi/Desktop/riotee-env/bin/python verify_log_pn_demo.py
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
src_dir = os.path.join(project_root, 'src')
config_dir = os.path.join(project_root, 'config')

sys.path.insert(0, src_dir)
sys.path.insert(0, config_dir)

try:
    from mppi import LEDPlant
    from app_config import DEFAULT_MODEL_NAME
    print(f"✅ 成功导入LEDPlant，使用模型: {DEFAULT_MODEL_NAME}")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

class LogPNVerificationDemo:
    def __init__(self):
        """初始化验证demo"""
        self.plant = None
        self.log_file = os.path.join(project_root, 'logs', 'control_real_log.csv')
        self.setup_plant()
        
    def setup_plant(self):
        """设置LEDPlant实例"""
        try:
            print("🔧 初始化LEDPlant...")
            self.plant = LEDPlant(
                model_key='5:1',
                use_efficiency=False,
                heat_scale=1.0,
                model_name=DEFAULT_MODEL_NAME
            )
            print(f"✅ LEDPlant初始化成功，模型类型: {self.plant.model_name}")
            print(f"   预测器类型: {self.plant.photo_predictor.model_name}")
        except Exception as e:
            print(f"❌ LEDPlant初始化失败: {e}")
            sys.exit(1)
    
    def read_log_data(self):
        """读取日志数据"""
        print(f"\n📊 读取日志文件: {self.log_file}")
        
        if not os.path.exists(self.log_file):
            print(f"❌ 日志文件不存在: {self.log_file}")
            return None
        
        try:
            df = pd.read_csv(self.log_file)
            if df.empty:
                print("⚠️  日志文件为空")
                return None
            
            print(f"✅ 成功读取日志数据，共 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"❌ 读取日志文件失败: {e}")
            return None
    
    def verify_single_record(self, row):
        """验证单条记录的PN值"""
        print(f"\n🔍 验证记录: {row['timestamp']}")
        
        # 提取日志中的值
        log_temp = float(row['input_temp'])
        log_co2 = float(row['co2_value'])
        log_solar_vol = float(row['solar_vol'])
        log_ppfd = float(row['ppfd'])
        log_pn = float(row['photosynthesis_rate'])
        log_r_pwm = float(row['red_pwm'])
        log_b_pwm = float(row['blue_pwm'])
        log_total_pwm = float(row['total_pwm'])
        
        print(f"📋 日志数据:")
        print(f"   温度: {log_temp}°C")
        print(f"   CO2: {log_co2} ppm")
        print(f"   Solar_Vol: {log_solar_vol}")
        print(f"   PPFD: {log_ppfd}")
        print(f"   红光PWM: {log_r_pwm}")
        print(f"   蓝光PWM: {log_b_pwm}")
        print(f"   总PWM: {log_total_pwm}")
        print(f"   日志PN: {log_pn}")
        
        # 计算R:B比例
        calculated_rb_ratio = log_r_pwm / log_total_pwm
        print(f"   计算R:B比例: {calculated_rb_ratio:.4f}")
        
        # 设置环境CO2
        self.plant.set_env_co2(log_co2)
        print(f"   ✅ 设置环境CO2: {self.plant.current_co2} ppm")
        
        # 验证PN模型预测
        print(f"\n🧮 模型验证:")
        
        # 方法1: 使用日志中的Solar_Vol直接预测
        if 'solar_vol' in str(DEFAULT_MODEL_NAME).lower():
            print(f"   方法1: 使用日志Solar_Vol ({log_solar_vol}) 直接预测PN")
            predicted_pn_direct = self.plant.photo_predictor.predict(
                log_solar_vol, log_co2, log_temp, calculated_rb_ratio
            )
            print(f"   直接预测PN: {predicted_pn_direct:.4f}")
            print(f"   日志PN: {log_pn:.4f}")
            print(f"   差异: {abs(predicted_pn_direct - log_pn):.4f}")
            print(f"   相对误差: {abs(predicted_pn_direct - log_pn) / log_pn * 100:.2f}%")
        
        # 方法2: 使用PWM计算PPFD，然后预测PN
        print(f"\n   方法2: 使用PWM计算PPFD，然后预测PN")
        ppfd_pred, temp_pred, power_pred, photo_pred = self.plant.predict(
            np.array([[log_r_pwm, log_b_pwm]]), log_temp, 0.1
        )
        
        calculated_ppfd = ppfd_pred[0]
        predicted_pn_from_pwm = photo_pred[0]
        
        print(f"   PWM计算PPFD: {calculated_ppfd:.2f}")
        print(f"   日志PPFD: {log_ppfd:.2f}")
        print(f"   PPFD差异: {abs(calculated_ppfd - log_ppfd):.2f}")
        print(f"   从PWM预测PN: {predicted_pn_from_pwm:.4f}")
        print(f"   日志PN: {log_pn:.4f}")
        print(f"   PN差异: {abs(predicted_pn_from_pwm - log_pn):.4f}")
        print(f"   相对误差: {abs(predicted_pn_from_pwm - log_pn) / log_pn * 100:.2f}%")
        
        # 方法3: 验证R:B比例是否为固定的0.83
        print(f"\n   方法3: 验证R:B比例")
        target_rb_ratio = 5.0 / (5.0 + 1.0)  # 0.8333
        print(f"   目标R:B比例: {target_rb_ratio:.4f}")
        print(f"   实际R:B比例: {calculated_rb_ratio:.4f}")
        print(f"   比例差异: {abs(calculated_rb_ratio - target_rb_ratio):.4f}")
        print(f"   ✅ 比例一致: {abs(calculated_rb_ratio - target_rb_ratio) < 0.001}")
        
        # 返回验证结果
        return {
            'timestamp': row['timestamp'],
            'log_pn': log_pn,
            'predicted_pn_direct': predicted_pn_direct if 'solar_vol' in str(DEFAULT_MODEL_NAME).lower() else None,
            'predicted_pn_from_pwm': predicted_pn_from_pwm,
            'log_ppfd': log_ppfd,
            'calculated_ppfd': calculated_ppfd,
            'log_rb_ratio': calculated_rb_ratio,
            'target_rb_ratio': target_rb_ratio,
            'rb_ratio_match': abs(calculated_rb_ratio - target_rb_ratio) < 0.001
        }
    
    def run_verification(self):
        """运行验证"""
        print("🚀 日志PN值验证Demo")
        print("=" * 60)
        print(f"📅 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 目标: 验证日志中的PN值是否与模型预测一致")
        print("=" * 60)
        
        # 读取日志数据
        df = self.read_log_data()
        if df is None:
            return
        
        # 验证每条记录
        results = []
        for index, row in df.iterrows():
            if pd.isna(row['photosynthesis_rate']) or row['photosynthesis_rate'] == '':
                print(f"⚠️  跳过记录 {index}: PN值为空")
                continue
            
            result = self.verify_single_record(row)
            if result:
                results.append(result)
        
        # 总结验证结果
        if results:
            print(f"\n" + "=" * 60)
            print("📊 验证结果总结")
            print("=" * 60)
            
            for result in results:
                print(f"\n📅 {result['timestamp']}:")
                print(f"   日志PN: {result['log_pn']:.4f}")
                
                if result['predicted_pn_direct'] is not None:
                    direct_error = abs(result['predicted_pn_direct'] - result['log_pn']) / result['log_pn'] * 100
                    print(f"   直接预测PN: {result['predicted_pn_direct']:.4f} (误差: {direct_error:.2f}%)")
                
                pwm_error = abs(result['predicted_pn_from_pwm'] - result['log_pn']) / result['log_pn'] * 100
                print(f"   PWM预测PN: {result['predicted_pn_from_pwm']:.4f} (误差: {pwm_error:.2f}%)")
                
                print(f"   R:B比例: {result['log_rb_ratio']:.4f} {'✅' if result['rb_ratio_match'] else '❌'}")
                
                # 判断验证结果
                if result['predicted_pn_direct'] is not None:
                    if direct_error < 1.0:
                        print(f"   ✅ 直接预测验证通过 (误差 < 1%)")
                    else:
                        print(f"   ⚠️  直接预测误差较大 (误差 ≥ 1%)")
                
                if pwm_error < 5.0:
                    print(f"   ✅ PWM预测验证通过 (误差 < 5%)")
                else:
                    print(f"   ⚠️  PWM预测误差较大 (误差 ≥ 5%)")
        else:
            print("❌ 没有有效的验证记录")
        
        print(f"\n🎉 验证完成！")

def main():
    """主函数"""
    try:
        demo = LogPNVerificationDemo()
        demo.run_verification()
        return 0
    except Exception as e:
        print(f"❌ Demo运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())




