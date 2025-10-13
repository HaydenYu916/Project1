# MPPI控制器自动调参指南

## 概述

本项目提供了多种自动调参方法，用于优化MPPI控制器的参数，提升控制性能。主要包括：

1. **离线调参**: 贝叶斯优化、差分进化、网格搜索
2. **在线调参**: 自适应参数调整
3. **集成调参**: 直接集成到控制脚本中

## 方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 贝叶斯优化 | 高效，收敛快 | 需要skopt库 | 离线调参，参数空间较大 |
| 差分进化 | 全局优化，无需梯度 | 计算量大 | 离线调参，复杂优化问题 |
| 网格搜索 | 简单可靠，全覆盖 | 计算量巨大 | 离线调参，参数空间较小 |
| 在线自适应 | 实时调整，适应环境变化 | 收敛较慢 | 在线控制，环境变化大 |

## 使用方法

### 1. 离线自动调参

#### 贝叶斯优化
```bash
cd LED_MPPI_Controller
python applications/tuning/auto_tune_mppi.py --method bayesian --iterations 20
```

#### 差分进化
```bash
python applications/tuning/auto_tune_mppi.py --method evolutionary --iterations 30
```

#### 网格搜索
```bash
python applications/tuning/auto_tune_mppi.py --method grid --grid-size 3
```

### 2. 在线自适应调参

#### 启用自适应控制
```bash
python applications/control/mppi_control_adaptive.py once
python applications/control/mppi_control_adaptive.py continuous
python applications/control/mppi_control_adaptive.py background
```

#### 查看自适应状态
```bash
python applications/control/mppi_control_adaptive.py adaptive_status
```

#### 禁用自适应调参
```bash
python applications/control/mppi_control_adaptive.py once --disable-adaptive
```

### 3. 手动应用最优参数

调参完成后，可以手动将最优参数应用到控制脚本：

```python
# 在 mppi_control_real_v2.py 中修改参数
self.controller.set_weights(
    Q_photo=32.5,    # 调优后的值
    R_du=0.015,      # 调优后的值
    R_power=0.003,   # 调优后的值
    Q_ref=28.0,      # 调优后的值
)
```

## 参数说明

### 主要调优参数

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| Q_photo | 10.0-50.0 | 25.0 | 光合作用权重，越大越重视光合作用 |
| Q_ref | 5.0-50.0 | 25.0 | 参考跟踪权重，越大越紧跟参考值 |
| R_du | 0.001-0.1 | 0.02 | 控制变化惩罚，越大控制越平滑 |
| R_power | 0.001-0.05 | 0.005 | 功率惩罚，越大越节能 |
| u_std | 0.1-0.5 | 0.25 | 控制采样标准差，影响探索性 |
| temperature | 0.5-2.0 | 1.0 | MPPI温度参数，影响随机性 |

### 性能指标

| 指标 | 权重 | 说明 |
|------|------|------|
| 光合作用 | +1.0 | 最大化光合速率 |
| 温度违规 | -0.5 | 最小化温度约束违反 |
| 控制平滑性 | -0.3 | 最小化控制变化 |
| 功率效率 | -0.2 | 最小化功率消耗 |
| 参考跟踪 | -0.4 | 最小化参考跟踪误差 |

## 配置文件

### 自动调参配置 (`auto_tuning_config.json`)

```json
{
  "tuning_method": "bayesian",
  "evaluation_period_hours": 24,
  "parameter_ranges": {
    "Q_photo": [10.0, 50.0],
    "Q_ref": [5.0, 50.0],
    "R_du": [0.001, 0.1],
    "R_power": [0.001, 0.05],
    "horizon": [4, 10],
    "num_samples": [500, 1000],
    "temperature": [0.5, 2.0],
    "u_std": [0.1, 0.5]
  },
  "fitness_weights": {
    "photosynthesis": 1.0,
    "temperature": -0.5,
    "smoothness": -0.3,
    "efficiency": -0.2,
    "tracking": -0.4
  },
  "max_iterations": 50,
  "convergence_threshold": 0.01
}
```

### 自适应调参配置

```python
# 在 mppi_control_adaptive.py 中修改
ADAPTIVE_TUNING_ENABLED = True     # 是否启用自适应调参
ADAPTATION_PERIOD_HOURS = 6        # 适应周期（小时）
LEARNING_RATE = 0.05               # 学习率
```

## 日志和结果

### 日志文件

- `logs/auto_tuning/auto_tuning.log` - 离线调参日志
- `logs/adaptive_tuning/adaptive_tuning_history.json` - 自适应调参历史
- `logs/mppi_adaptive_control_log.csv` - 自适应控制日志
- `logs/tuning_report.md` - 调参报告

### 结果文件

- `logs/auto_tuning_results.json` - 最优参数结果
- `logs/tuning_report.md` - 详细调参报告

## 最佳实践

### 1. 离线调参流程

1. **数据准备**: 确保有足够的历史数据用于评估
2. **参数范围设置**: 根据经验设置合理的参数范围
3. **多方法对比**: 尝试不同的调参方法，选择最佳结果
4. **验证测试**: 在实际环境中验证调优后的参数

### 2. 在线自适应调参

1. **初始参数**: 使用离线调参的结果作为初始参数
2. **适应周期**: 根据环境变化频率设置合适的适应周期
3. **学习率**: 从较小的学习率开始，逐步调整
4. **监控**: 定期检查自适应调参的效果

### 3. 性能监控

```bash
# 查看自适应状态
python applications/control/mppi_control_adaptive.py adaptive_status

# 查看控制日志
tail -f logs/mppi_adaptive_control_simple.log

# 分析性能趋势
python -c "
import pandas as pd
import matplotlib.pyplot as plt

# 读取控制日志
df = pd.read_csv('logs/mppi_adaptive_control_log.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 绘制性能趋势
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(df['timestamp'], df['performance_score'])
plt.title('Performance Score Over Time')

plt.subplot(2, 2, 2)
plt.plot(df['timestamp'], df['pred_pn'])
plt.title('Photosynthesis Rate')

plt.subplot(2, 2, 3)
plt.plot(df['timestamp'], df['pred_temp'])
plt.title('Temperature')

plt.subplot(2, 2, 4)
plt.plot(df['timestamp'], df['cost'])
plt.title('Control Cost')

plt.tight_layout()
plt.show()
"
```

## 故障排除

### 常见问题

1. **调参收敛慢**
   - 增加迭代次数
   - 调整参数范围
   - 检查评估函数

2. **参数震荡**
   - 降低学习率
   - 增加适应周期
   - 调整参数范围

3. **性能下降**
   - 检查环境数据质量
   - 调整性能权重
   - 重新校准模型

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查参数历史
import json
with open('logs/adaptive_tuning_history.json', 'r') as f:
    history = json.load(f)
    print("参数变化历史:", history['parameters'][-10:])
    print("性能历史:", history['performance'][-10:])
```

## 扩展功能

### 自定义评估函数

```python
class CustomMPPITuner(AdaptiveMPPITuner):
    def compute_performance_score(self, **kwargs):
        # 自定义性能评估逻辑
        return custom_score
```

### 多目标优化

```python
# 在贝叶斯优化中支持多目标
from skopt import gbrt_minimize

def multi_objective(params):
    score1 = self.evaluate_photosynthesis(params)
    score2 = self.evaluate_efficiency(params)
    return [score1, score2]  # 返回多个目标
```

## 总结

自动调参是提升MPPI控制器性能的重要手段。建议：

1. 首先使用离线调参获得基础最优参数
2. 在实际控制中启用在线自适应调参
3. 定期监控和调整调参策略
4. 根据实际效果优化评估函数和参数范围

通过合理的自动调参，可以显著提升LED植物照明的控制效果和系统性能。
