#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basil光合速率预测器
使用训练好的机器学习模型预测光合速率

输入特征: PPFD (光量子密度, umol/m2/s) + CO2 (ppm) + 温度 (°C) + R:B (红蓝光比例)
输出: Pn (光合速率, umol/m2/s)
"""

import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ==================== 设置预测变量 ====================
PPFD = 500  # 光量子密度 (umol/m2/s)
CO2 = 400   # 二氧化碳浓度 (ppm)
TEMPERATURE = 25  # 温度 (°C)
RB_RATIO = 0.83   # 红蓝光比例
# =====================================================


class PhotosynthesisPredictor:
    """光合速率预测器"""

    def __init__(self):
        # PPFD模型路径 - 使用绝对路径或相对于当前文件的路径
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_base = os.path.join(current_dir, "..", "MODEL", "PPFD")
        
        self.model_path = os.path.join(model_base, "best_model.pkl")
        self.norm_path = os.path.join(model_base, "normalization_params.pkl")
        self.feature_info_path = os.path.join(model_base, "feature_info.pkl")

        # 模型组件
        self.model = None
        self.norm_params = None
        self.feature_info = None
        self.is_loaded = False

        self.load_model()

    def load_model(self):
        """加载模型"""
        try:
            # 加载训练好的模型
            self.model = joblib.load(self.model_path)
            self.norm_params = joblib.load(self.norm_path)
            self.feature_info = joblib.load(self.feature_info_path)

            self.is_loaded = True

        except FileNotFoundError as e:
            self.is_loaded = False
            # Instead of printing, let the caller handle the error
            raise e
        except Exception as e:
            self.is_loaded = False
            # Instead of printing, let the caller handle the error
            raise e

    def predict(self, ppfd, co2, temperature, rb_ratio, humidity: float | None = None):
        """
        预测光合速率
        
        参数:
        - ppfd: 光合光子通量密度 (μmol·m⁻²·s⁻¹)
        - co2: 二氧化碳浓度 (ppm)
        - temperature: 温度 (°C)
        - rb_ratio: 红蓝光比例
        - humidity: 相对湿度 (%, optional)
        
        返回:
        - 预测的Pn值
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        # Input validation can be done by the caller or be more sophisticated
        # For now, we'll trust the inputs are within a reasonable range.

        # 准备输入数据
        input_data = np.array([[ppfd, co2, temperature, rb_ratio]])
        
        # 标准化
        if self.norm_params:
            input_norm = (input_data - self.norm_params['feat_mean']) / self.norm_params['feat_std']
        else:
            input_norm = input_data
        
        # 预测
        pred_norm = self.model.predict(input_norm)
        
        # 反标准化
        if self.norm_params:
            prediction = pred_norm * self.norm_params['target_std'] + self.norm_params['target_mean']
        else:
            prediction = pred_norm
        
        return float(prediction[0])


def main():
    """主函数"""
    print("🌱 Basil光合速率预测器")
    print("=" * 40)

    predictor = PhotosynthesisPredictor()

    if not predictor.is_loaded:
        return

    print("\n📊 预测条件:")
    print(f"  PPFD = {PPFD} μmol·m⁻²·s⁻¹")
    print(f"  CO2 = {CO2} ppm")
    print(f"  温度 = {TEMPERATURE} °C")
    print(f"  R:B = {RB_RATIO}")

    result = predictor.predict(PPFD, CO2, TEMPERATURE, RB_RATIO)

    if result is not None:
        print(f"\n🎯 预测结果: Pn = {result:.3f} μmol·m⁻²·s⁻¹")


if __name__ == "__main__":
    main()
