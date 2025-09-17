import joblib
import numpy as np
import pandas as pd

def load_solar_vol_model():
    """加载Solar_Vol Pn预测模型"""
    try:
        model = joblib.load('best_model.pkl')
        norm_params = joblib.load('normalization_params.pkl')
        feature_info = joblib.load('feature_info.pkl')
        print("✅ 模型加载成功!")
        return model, norm_params, feature_info
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None, None

def predict_pn(solar_vol, co2, temperature, rb_ratio, model, norm_params):
    """
    预测Pn值
    
    参数:
    - solar_vol: 太阳能电压 (0.0-1.8)
    - co2: 二氧化碳浓度 (400-800 ppm)
    - temperature: 温度 (18-30 °C)
    - rb_ratio: 红蓝光比例 (0.5-1.0)
    - model: 训练好的模型
    - norm_params: 归一化参数
    
    返回:
    - 预测的Pn值
    """
    # 准备输入数据
    input_data = np.array([[solar_vol, co2, temperature, rb_ratio]])
    
    print(f"🔍 调试信息:")
    print(f"  原始输入: {input_data[0]}")
    
    # 标准化
    if norm_params:
        input_norm = (input_data - norm_params['feat_mean']) / norm_params['feat_std']
        print(f"  标准化后: {input_norm[0]}")
        print(f"  特征均值: {norm_params['feat_mean']}")
        print(f"  特征标准差: {norm_params['feat_std']}")
    else:
        input_norm = input_data
        print(f"  未标准化")
    
    # 预测
    pred_norm = model.predict(input_norm)
    print(f"  标准化预测值: {pred_norm[0]}")
    
    # 反标准化
    if norm_params:
        prediction = pred_norm * norm_params['target_std'] + norm_params['target_mean']
        print(f"  目标均值: {norm_params['target_mean']}")
        print(f"  目标标准差: {norm_params['target_std']}")
        print(f"  反标准化后: {prediction[0]}")
    else:
        prediction = pred_norm
        print(f"  未反标准化")
    
    return prediction[0]

def batch_predict(input_data, model, norm_params):
    """
    批量预测
    
    参数:
    - input_data: 形状为(n_samples, 4)的数组
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

# 使用示例
if __name__ == "__main__":
    # 1. 加载模型
    model, norm_params, feature_info = load_solar_vol_model()
    if model is None:
        exit(1)
    
    print(f"模型类型: {feature_info['model_name']}")
    print(f"特征列: {feature_info['feature_columns']}")
    print(f"目标列: {feature_info['pn_column']}")
    
    # 2. 单个预测示例
    print("\n单个预测示例:")
    pn_pred = predict_pn(1.0, 400, 22, 0.75, model, norm_params)
    print(f"输入: Solar_Vol=1.0, CO2=400, T=22, R:B=0.75")
    print(f"预测Pn: {pn_pred:.4f}")
    
    # 3. 批量预测示例
    print("\n批量预测示例:")
    test_data = np.array([
        [0.5, 400, 20, 0.5],
        [1.0, 400, 22, 0.75],
        [1.5, 800, 24, 1.0]
    ])
    
    batch_predictions = batch_predict(test_data, model, norm_params)
    for i, (data, pred) in enumerate(zip(test_data, batch_predictions)):
        print(f"样本{i+1}: {data} -> Pn={pred:.4f}")
