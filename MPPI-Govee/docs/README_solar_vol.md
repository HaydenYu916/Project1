# 多模型集成说明

## 概述
成功将三个模型（`solar_vol`、`ppfd`、`sp`）集成到 MPPI 控制器中，用于预测光合作用速率。用户可以在代码中选择使用哪个模型。

## 模型信息

### 1. Solar_Vol 模型
- **模型类型**: SVR (RBF) - 支持向量回归，径向基函数核
- **输入特征**: ['Solar_Vol', 'CO2', 'T', 'R:B']
- **输出**: Pn_avg (光合作用速率)
- **性能**: R² = 0.9866

### 2. PPFD 模型
- **模型类型**: SVR (RBF) - 支持向量回归，径向基函数核
- **输入特征**: ['PPFD', 'CO2', 'T', 'R:B']
- **输出**: Pn_avg (光合作用速率)
- **性能**: R² = 0.9807

### 3. SP 模型 (光谱模型)
- **模型类型**: SVR (RBF) - 支持向量回归，径向基函数核
- **输入特征**: ['sp_415', 'sp_445', 'sp_480', 'sp_515', 'sp_555', 'sp_590', 'sp_630', 'sp_680', 'CO2', 'T', 'R:B']
- **输出**: Pn_avg (光合作用速率)
- **性能**: R² = 0.9788

## 文件结构
```
Models/
├── solar_vol/
│   ├── best_model.pkl
│   ├── normalization_params.pkl
│   ├── feature_info.pkl
│   └── solar_vol_baseline_results.csv
├── ppfd/
│   ├── best_model.pkl
│   ├── normalization_params.pkl
│   ├── feature_info.pkl
│   └── ppfd_baseline_results.csv
└── sp/
    ├── best_model.pkl
    ├── normalization_params.pkl
    ├── feature_info.pkl
    └── sp_baseline_results.csv
```

## 使用方法

### 1. 模型选择
```python
from mppi import LEDPlant, LEDMPPIController

# 使用 solar_vol 模型 (默认)
plant = LEDPlant(model_name='solar_vol')

# 使用 ppfd 模型
plant = LEDPlant(model_name='ppfd')

# 使用 sp 模型 (光谱模型)
plant = LEDPlant(model_name='sp')
```

### 2. 基本使用
```python
# 单步预测
ppfd, temp, power, photo = plant.step(r_pwm=50.0, b_pwm=10.0, dt=0.1)
print(f"PPFD={ppfd:.2f}, Temp={temp:.2f}°C, Power={power:.2f}W, Photo={photo:.2f}")
```

### 3. MPPI 控制器
```python
# 创建控制器
controller = LEDMPPIController(
    plant, 
    horizon=10, 
    num_samples=500,
    maintain_rb_ratio=True,
    rb_ratio_key="5:1"
)

# 求解最优控制动作
action, sequence, success, cost, weights = controller.solve(current_temp=25.0)
print(f"控制动作: R_PWM={action[0]:.2f}, B_PWM={action[1]:.2f}")
```

### 4. 序列预测
```python
import numpy as np

# 定义 PWM 序列
pwm_sequence = np.array([
    [50, 10], [45, 15], [40, 20], [35, 25], [30, 30]
])

# 预测整个序列
ppfd_pred, temp_pred, power_pred, photo_pred = plant.predict(
    pwm_sequence, initial_temp=25.0
)
```

## 技术细节

### 模型加载
- 使用 `joblib` 加载模型文件（兼容性更好）
- 手动实现标准化和反标准化（使用保存的均值和标准差）
- 自动回退机制：如果 solar_vol 模型加载失败，会尝试加载原始的 PPFD 模型

### 输入特征映射
- `ppfd` 参数 → Solar_Vol
- `co2` 参数 → CO2
- `temperature` 参数 → T (温度)
- `rb_ratio` 参数 → R:B (红蓝比例)

### 标准化参数
```python
feat_mean = [1.19, 594.56, 23.52, 0.77]  # 特征均值
feat_std = [0.55, 199.93, 3.96, 0.17]    # 特征标准差
target_mean = 9.16                        # 目标均值
target_std = 5.98                         # 目标标准差
```

## 模型比较

### 性能表现
根据 `unified_comparison_wide_r2.csv`：
- **Solar_Vol 模型**: R² = 0.9866 (最佳)
- **PPFD 模型**: R² = 0.9807
- **SP 模型**: R² = 0.9788

### 预测结果差异
相同输入条件下 (R_PWM=50, B_PWM=10)：
- **Solar_Vol**: Photo = 5.83
- **PPFD**: Photo = 13.75 (最高)
- **SP**: Photo = 0.18 (最低)

## 测试结果
运行测试脚本验证所有模型功能：

```bash
conda activate ppfd_env
# 测试单个模型
python test_solar_vol.py

# 测试所有模型
python test_all_models.py
```

测试包括：
- 单步预测测试
- MPPI 控制器测试
- 序列预测测试
- 模型性能比较

## 注意事项
1. 确保 `ppfd_env` 环境已激活
2. 模型文件路径必须正确
3. 输入参数需要在合理范围内
4. 温度约束：20-30°C
5. PWM 约束：5-95%
