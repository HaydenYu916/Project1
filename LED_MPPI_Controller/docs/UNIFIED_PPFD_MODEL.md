# 统一PPFD热力学模型集成

## 概述

本次更新将基于实验数据拟合的统一PPFD温度差模型集成到LED控制系统中，提供更准确的温度预测能力。

## 模型特点

### 核心函数
```python
def unified_temp_diff_model(t, ppfd):
    K1 = 0.013198 * ppfd + 0.493192
    tau1 = 0.009952 * ppfd + 0.991167
    K2 = 0.013766 * ppfd^0.988656
    tau2 = 0.796845 * ppfd^-0.144439
    return K1 * (1 - exp(-t/tau1)) + K2 * (1 - exp(-t/tau2))
```

### 模型质量
- **R² = 0.9254** (92.54% 拟合度)
- **MAE = 0.4654°C** (平均绝对误差)
- **RMSE = 0.6077°C** (均方根误差)
- **数据点: 3,483个**

### 关键优势
✓ **单一函数预测任意PPFD值**  
✓ **PPFD作为直接输入参数**  
✓ **高拟合质量 (R² > 0.92)**  
✓ **物理意义明确**  
✓ **连续预测能力**  

## 代码修改

### 1. 新增模型类
- `UnifiedPPFDThermalModel`: 统一PPFD热力学模型
- 继承自 `BaseThermalModel` 抽象基类
- 支持PPFD直接输入的温度预测

### 2. 模型工厂更新
```python
# 支持新的模型类型
led = Led(model_type="unified_ppfd")  # 或 "ppfd", "unified", "3", "up"
```

### 3. Led类增强
- 新增 `step_with_ppfd()` 方法用于PPFD输入
- 新增 `is_ppfd_model` 属性判断模型类型
- 新增 `get_model_info()` 方法获取模型状态

### 4. 前向步进接口更新
- `forward_step()` 新增 `use_unified_ppfd` 参数
- `forward_step_batch()` 支持混合模型批量处理
- 自动检测模型类型并选择相应计算方式

### 5. PWMtoPPFDModel修复
- 新增 `predict()` 方法支持PWM→PPFD反向预测
- 修复了标定数据演示中的预测功能

## 使用示例

### 基本使用
```python
from led import Led, LedThermalParams

# 创建统一PPFD模型
led = Led(model_type="unified_ppfd", initial_temp=25.0)

# PPFD步进
temp = led.step_with_ppfd(ppfd=300.0, dt=1.0)

# 稳态温度预测
steady_temp = led.target_temperature(ppfd=300.0)
```

### 与前向步进接口集成
```python
from led import forward_step, PWMtoPPFDModel, PWMtoPowerModel

# 使用统一PPFD模型进行前向步进
output = forward_step(
    thermal_model=thermal_model,
    r_pwm=50.0,
    b_pwm=30.0,
    dt=1.0,
    power_model=power_model,
    ppfd_model=ppfd_model,
    use_unified_ppfd=True  # 启用统一PPFD模型
)
```

### 批量处理
```python
# 支持混合模型批量处理
outputs = forward_step_batch(
    thermal_models=[model1, model2],  # 可以是不同类型的模型
    r_pwms=[50.0, 60.0],
    b_pwms=[30.0, 40.0],
    dt=1.0,
    power_model=power_model,
    ppfd_model=ppfd_model,
    use_unified_ppfd=True
)
```

## 模型状态信息

统一PPFD模型提供详细的状态信息：
```python
model_info = led.get_model_info()
# 返回:
# {
#     'time_elapsed': 61.0,      # 累计时间
#     'current_ppfd': 300.0,     # 当前PPFD值
#     'ambient_temp': 33.32,     # 当前温度
#     'base_ambient_temp': 25.0   # 环境基准温度
# }
```

## 兼容性

### 向后兼容
- 所有现有代码继续正常工作
- 传统热学模型 (`FirstOrderThermalModel`, `SecondOrderThermalModel`) 保持不变
- 默认模型类型仍为 `"first_order"`

### 模型选择
```python
# 传统热学模型
led_traditional = Led(model_type="first_order")

# 统一PPFD模型  
led_ppfd = Led(model_type="unified_ppfd")

# 检查模型类型
if led.is_ppfd_model:
    temp = led.step_with_ppfd(ppfd=300.0, dt=1.0)
else:
    temp = led.step_with_heat(power=10.0, dt=1.0)
```

## 演示和测试

### 运行演示
```bash
cd LED_MPPI_Controller
python examples/unified_ppfd_demo.py
```

### 演示内容
1. **基本功能演示**: 稳态温度预测、时间序列仿真
2. **阶跃变化演示**: PPFD阶跃变化的温度响应
3. **标定数据集成**: PWM→PPFD→温度完整工作流程
4. **可视化结果**: 自动生成温度响应图表

## 性能特点

### 计算效率
- **单步计算**: O(1) 时间复杂度
- **批量处理**: 支持NumPy向量化
- **内存占用**: 最小化状态存储

### 数值稳定性
- **时间常数保护**: 防止除零错误
- **PPFD范围检查**: 自动处理边界条件
- **温度限制**: 合理的物理约束

## 未来扩展

### 可能的改进方向
1. **多PPFD模型**: 支持不同光谱组合的独立模型
2. **自适应参数**: 根据环境条件动态调整模型参数
3. **机器学习集成**: 结合深度学习进一步提升预测精度
4. **实时校准**: 在线更新模型参数

### 接口扩展
- 支持更多PPFD相关参数
- 集成光谱分析功能
- 添加模型验证和诊断工具

## 总结

统一PPFD热力学模型的集成显著提升了LED控制系统的温度预测能力：

- **精度提升**: R² = 0.9254 的高拟合质量
- **使用简化**: PPFD直接输入，无需复杂的功率转换
- **兼容性好**: 与现有系统无缝集成
- **扩展性强**: 为未来功能扩展奠定基础

该模型为LED控制系统的精确温度管理提供了强有力的工具，特别适用于需要高精度PPFD-温度映射的应用场景。
