import joblib
import numpy as np
import pandas as pd

def load_pn_model():
    """加载Pn预测模型"""
    try:
        model = joblib.load('best_model.pkl')
        norm_params = joblib.load('normalization_params.pkl')
        feature_info = joblib.load('feature_info.pkl')
        print("✅ Pn预测模型加载成功!")
        return model, norm_params, feature_info
    except Exception as e:
        print(f"❌ 模型加载失败: {{e}}")
        return None, None, None

def predict_pn(ppfd, co2, temperature, rb_ratio, model, norm_params):
    """
    预测Pn值
    
    参数:
    - ppfd: 光合光子通量密度 (μmol·m⁻²·s⁻¹)
    - co2: 二氧化碳浓度 (ppm)
    - temperature: 温度 (°C)
    - rb_ratio: 红蓝光比例
    - model: 训练好的模型
    - norm_params: 归一化参数
    
    返回:
    - 预测的Pn值
    """
    # 准备输入数据
    input_data = np.array([[ppfd, co2, temperature, rb_ratio]])
    
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

def create_sample_data():
    """创建示例数据"""
    # 模拟不同光照条件下的数据
    sample_data = [
        [50, 400, 20, 0.5],   # 低光照
        [200, 400, 22, 0.75], # 中等光照
        [500, 800, 24, 1.0]   # 高光照
    ]
    return sample_data

def validate_training_data(model, norm_params, feature_info):
    """
    验证训练数据，检查模型是否能正确预测已知样本
    """
    print("\n🔍 验证训练数据...")
    
    # 从averaged_data.csv中提取的已知样本
    known_samples = [
        [50.0, 400.0, 20.0, 0.5, 2.41071015809815],  # 实际Pn=2.4107
        [100.0, 400.0, 20.0, 0.5, 4.482195309391792], # 实际Pn=4.4822
        [200.0, 400.0, 20.0, 0.5, 7.813590918067845], # 实际Pn=7.8136
    ]
    
    print(f"验证样本:")
    for i, sample in enumerate(known_samples):
        input_data = np.array([sample[:4]])  # 前4个是特征
        actual_pn = sample[4]                # 第5个是实际Pn值
        
        # 预测
        pred_pn = predict_pn(input_data[0, 0], input_data[0, 1], 
                            input_data[0, 2], input_data[0, 3], 
                            model, norm_params)
        
        # 计算误差
        error = abs(pred_pn - actual_pn)
        error_percent = (error / actual_pn) * 100
        
        print(f"  样本{i+1}: 输入{sample[:4]} -> 实际Pn={actual_pn:.4f}, 预测Pn={pred_pn:.4f}")
        print(f"    绝对误差: {error:.4f}, 相对误差: {error_percent:.2f}%")
        print()

# 使用示例
if __name__ == "__main__":
    # 1. 加载模型
    model, norm_params, feature_info = load_pn_model()
    if model is None:
        exit(1)
    
    print(f"模型类型: {feature_info['model_name']}")
    print(f"特征列: {feature_info['feature_columns']}")
    print(f"目标列: {feature_info['pn_column']}")
    
    # 2. 验证训练数据
    validate_training_data(model, norm_params, feature_info)
    
    # 3. 单个预测示例
    print("单个预测示例:")
    pn_pred = predict_pn(200, 400, 22, 0.75, model, norm_params)
    print(f"输入: PPFD=200, CO2=400, T=22, R:B=0.75")
    print(f"预测Pn: {pn_pred:.4f}")
    
    # 4. 批量预测示例
    print("\n批量预测示例:")
    test_data = create_sample_data()
    batch_predictions = batch_predict(np.array(test_data), model, norm_params)
    for i, (data, pred) in enumerate(zip(test_data, batch_predictions)):
        print(f"样本{i+1}: {data} -> Pn={pred:.4f}")
    
    # 5. 特征重要性分析（如果模型支持）
    if hasattr(model, 'feature_importances_'):
        print("\n特征重要性:")
        for i, importance in enumerate(model.feature_importances_):
            print(f"  {feature_info['feature_columns'][i]}: {importance:.4f}")
    elif hasattr(model, 'coef_'):
        print("\n特征系数:")
        for i, coef in enumerate(model.coef_):
            print(f"  {feature_info['feature_columns'][i]}: {coef:.4f}")
