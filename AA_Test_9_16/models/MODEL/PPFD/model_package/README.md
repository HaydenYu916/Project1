# Pn预测模型包

## 📋 简介
这是一个基于环境特征的净光合速率(Pn)预测模型包，使用MLP Regressor算法训练。

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 基本使用
```python
import joblib

# 加载模型
model = joblib.load('best_model.pkl')
norm_params = joblib.load('normalization_params.pkl')

# 预测
input_data = [[200, 400, 22, 0.75]]  # [PPFD, CO2, T, R:B]
prediction = model.predict(input_data)
```

### 3. 运行示例
```bash
python example_usage.py
```

## 📊 模型信息
- **算法**: MLP Regressor
- **输入特征**: 4个参数 (PPFD, CO2, T, R:B)
- **输出**: Pn_avg
- **数据来源**: averaged_data.csv
- **特征类型**: 环境参数

## 🔬 特征说明

### 环境特征 (4个参数)
- **PPFD**: 光合光子通量密度 (μmol·m⁻²·s⁻¹)
- **CO2**: 二氧化碳浓度 (ppm)
- **T**: 温度 (°C)
- **R:B**: 红蓝光比例

## 📁 文件结构
```
model_package/
├── best_model.pkl          # 最佳训练模型
├── normalization_params.pkl # 归一化参数
├── feature_info.pkl        # 特征信息
├── all_trained_models.pkl  # 所有模型
├── 使用说明.md             # 详细使用说明
├── example_usage.py        # 示例代码
├── requirements.txt        # 依赖包列表
└── README.md              # 本文件
```

## 🔧 技术细节
- 使用Z-score标准化
- 训练/验证/测试集比例: 70%/15%/15%
- 包含超参数调优
- 支持鲁棒性测试
- 环境特征预测

## 📞 支持
如有问题，请查看`使用说明.md`或运行`example_usage.py`获取帮助。
