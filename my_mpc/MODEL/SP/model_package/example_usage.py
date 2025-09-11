import joblib
import numpy as np
import pandas as pd

def load_sp_pn_model():
    """加载光谱Pn预测模型"""
    try:
        model = joblib.load('best_model.pkl')
        norm_params = joblib.load('normalization_params.pkl')
        feature_info = joblib.load('feature_info.pkl')
        print("✅ 光谱Pn预测模型加载成功!")
        return model, norm_params, feature_info
    except Exception as e:
        print(f"❌ 模型加载失败: {{e}}")
        return None, None, None

def predict_pn_from_spectrum(spectrum_values, co2, temperature, rb_ratio, model, norm_params):
    """
    从光谱数据预测Pn值
    
    参数:
    - spectrum_values: 8个光谱波段值列表 [sp_415, sp_445, sp_480, sp_515, sp_555, sp_590, sp_630, sp_680]
    - co2: 二氧化碳浓度 (ppm)
    - temperature: 温度 (°C)  
    - rb_ratio: 红蓝光比例
    - model: 训练好的模型
    - norm_params: 归一化参数
    
    返回:
    - 预测的Pn值
    """
    # 准备输入数据: 8个光谱 + 3个环境参数
    input_data = np.array([spectrum_values + [co2, temperature, rb_ratio]])
    
    # 标准化
    if norm_params:
        input_norm = (input_data - norm_params['feat_mean']) / norm_params['feat_std']
    else:
        input_norm = input_data
    
    # 预测
    pred_norm = model.predict(input_norm)
    
    # 反标准化
    if norm_params:
        prediction = pred_norm * norm_params['target_std'] + norm_params['target_mean']
    else:
        prediction = pred_norm
    
    return prediction[0]

def batch_predict_spectrum(input_data, model, norm_params):
    """
    批量光谱预测
    
    参数:
    - input_data: 形状为(n_samples, 11)的数组
    - model: 训练好的模型
    - norm_params: 归一化参数
    
    返回:
    - 预测结果数组
    """
    # 标准化
    if norm_params:
        input_norm = (input_data - norm_params['feat_mean']) / norm_params['feat_std']
    else:
        input_norm = input_data
    
    # 预测
    pred_norm = model.predict(input_norm)
    
    # 反标准化
    if norm_params:
        predictions = pred_norm * norm_params['target_std'] + norm_params['target_mean']
    else:
        predictions = pred_norm
    
    return predictions

def create_sample_spectrum_data():
    """创建示例光谱数据"""
    # 使用合理的训练数据范围内的光谱值
    sample_spectra = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # 低光谱值
        [54.2, 45.2, 47.0, 44.5, 1412.6, 115.3, 61.3, 70.1],  # 中等光谱值
        [212.0, 876.7, 480.2, 171.6, 5327.4, 402.9, 225.3, 250.4]  # 高光谱值
    ]
    return sample_spectra

# 使用示例
if __name__ == "__main__":
    # 1. 加载模型
    model, norm_params, feature_info = load_sp_pn_model()
    if model is None:
        exit(1)
    
    print(f"模型类型: {feature_info['model_name']}")
    print(f"特征列: {feature_info['feature_columns']}")
    print(f"目标列: {feature_info['pn_column']}")
    
    # 2. 单个预测示例
    print("\n单个预测示例:")
    spectrum = [54.2, 45.2, 47.0, 44.5, 1412.6, 115.3, 61.3, 70.1]
    pn_pred = predict_pn_from_spectrum(spectrum, 400, 22, 0.75, model, norm_params)
    print(f"光谱值: {spectrum}")
    print(f"环境参数: CO2=400, T=22, R:B=0.75")
    print(f"预测Pn: {pn_pred:.4f}")
    
    # 3. 批量预测示例
    print("\n批量预测示例:")
    sample_spectra = create_sample_spectrum_data()
    env_params = [400, 22, 0.75]  # CO2, T, R:B
    
    for i, spectrum in enumerate(sample_spectra):
        input_data = np.array([spectrum + env_params])
        prediction = batch_predict_spectrum(input_data, model, norm_params)
        print(f"样本{i+1}: 光谱{spectrum[:3]}... -> Pn={prediction[0]:.4f}")
    
    # 4. 特征重要性分析（如果模型支持）
    if hasattr(model, 'feature_importances_'):
        print("\n特征重要性:")
        for i, importance in enumerate(model.feature_importances_):
            print(f"  {feature_info['feature_columns'][i]}: {importance:.4f}")
    elif hasattr(model, 'coef_'):
        print("\n特征系数:")
        for i, coef in enumerate(model.coef_):
            print(f"  {feature_info['feature_columns'][i]}: {coef:.4f}")
