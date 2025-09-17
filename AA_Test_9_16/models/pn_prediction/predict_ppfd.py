#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新的PPFD光合速率预测器
使用PPFD模型预测光合速率

支持多种特征输入: PPFD, CO2, 温度, R:B比例
"""

import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class PPFDPhotosynthesisPredictor:
    """基于PPFD模型的光合速率预测器"""

    def __init__(self, model_path="./MODEL/PPFD/best_model.pkl", norm_path="./MODEL/PPFD/normalization_params.pkl", feature_info_path="./MODEL/PPFD/feature_info.pkl"):
        # 新的PPFD模型路径
        self.model_path = model_path
        self.norm_path = norm_path
        self.feature_info_path = feature_info_path

        # 模型参数
        self.model = None
        self.norm_params = None
        self.feature_info = None
        self.is_loaded = False
        self.model_type = "unknown"

    def load_model(self):
        """加载训练好的模型和标准化器（兼容性加载）"""
        try:
            # 方法1: 尝试加载新的PPFD模型
            try:
                import sys
                import io
                
                # 临时重定向stderr来捕获警告
                old_stderr = sys.stderr
                sys.stderr = mystderr = io.StringIO()
                
                try:
                    # 设置兼容性参数
                    import numpy as np
                    np.random.MT19937 = np.random.mtrand.RandomState
                    
                    with open(self.model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    with open(self.norm_path, 'rb') as f:
                        self.norm_params = pickle.load(f)
                    with open(self.feature_info_path, 'rb') as f:
                        self.feature_info = pickle.load(f)
                finally:
                    sys.stderr = old_stderr
                
                self.is_loaded = True
                self.model_type = "PPFD"
                return
                
            except Exception as e1:
                # 方法2: 尝试使用joblib兼容模式
                try:
                    import joblib
                    self.model = joblib.load(self.model_path, mmap_mode=None)
                    self.norm_params = joblib.load(self.norm_path, mmap_mode=None)
                    self.feature_info = joblib.load(self.feature_info_path, mmap_mode=None)
                    self.is_loaded = True
                    self.model_type = "PPFD"
                    return
                except Exception as e2:
                    raise IOError(f"Failed to load PPFD model: {e1} / {e2}")

        except Exception as e:
            self.is_loaded = False
            raise IOError(f"All model loading methods failed: {e}")

    def predict(self, ppfd, co2=400, temperature=22, rb_ratio=0.75):
        """
        预测光合速率
        
        参数:
        - ppfd: 光合光子通量密度 (μmol·m⁻²·s⁻¹)
        - co2: 二氧化碳浓度 (ppm)，默认400
        - temperature: 温度 (°C)，默认22
        - rb_ratio: 红蓝光比例，默认0.75
        
        返回:
        - 预测的光合速率 (μmol·m⁻²·s⁻¹)
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Please call load_model() first.")

        try:
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
            
            return prediction[0]
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def batch_predict(self, input_data):
        """
        批量预测
        
        参数:
        - input_data: 形状为(n_samples, 4)的数组，每行为[PPFD, CO2, T, R:B]
        
        返回:
        - 预测结果数组
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Please call load_model() first.")

        try:
            input_data = np.array(input_data)
            
            # 标准化
            if self.norm_params:
                input_norm = (input_data - self.norm_params['feat_mean']) / self.norm_params['feat_std']
            else:
                input_norm = input_data
            
            # 预测
            pred_norm = self.model.predict(input_norm)
            
            # 反标准化
            if self.norm_params:
                predictions = pred_norm * self.norm_params['target_std'] + self.norm_params['target_mean']
            else:
                predictions = pred_norm
            
            return predictions
            
        except Exception as e:
            raise RuntimeError(f"Batch prediction failed: {e}")

    def get_model_info(self):
        """获取模型信息"""
        if not self.is_loaded:
            return {"status": "未加载"}
        
        info = {
            "status": "已加载",
            "model_type": self.model_type,
            "sklearn_model": type(self.model).__name__,
        }
        
        if self.feature_info:
            info["features"] = self.feature_info.get('feature_columns', 'N/A')
            info["target"] = self.feature_info.get('pn_column', 'N/A')
        
        return info
