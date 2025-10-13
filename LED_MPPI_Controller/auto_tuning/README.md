# MPPI控制器自动调参模块

## 🎯 功能概述

为MPPI控制器提供完整的自动调参解决方案：

### 1. **离线调参** - 基于历史数据的参数优化
- **贝叶斯优化**: 使用scikit-optimize进行高效参数搜索
- **差分进化**: 全局优化算法，无需梯度信息
- **网格搜索**: 简单可靠的参数空间遍历

### 2. **在线调参** - 实时自适应参数调整
- **自适应调参器**: 基于性能反馈的在线参数优化
- **集成控制**: 直接集成到MPPI控制脚本中
- **性能监控**: 实时跟踪和记录控制性能

## 🚀 快速开始

### 方法1: 使用快速启动脚本（推荐）
```bash
cd auto_tuning

# 安装依赖
python run_auto_tuning.py install

# 运行测试
python run_auto_tuning.py test

# 离线调参
python run_auto_tuning.py offline --method evolutionary --iterations 20

# 在线调参
python run_auto_tuning.py online --duration 24

# 查看状态
python run_auto_tuning.py status
```

### 方法2: 直接使用脚本
```bash
cd auto_tuning

# 安装依赖
bash install_tuning_deps.sh

# 离线调参
python auto_tune_mppi.py --method bayesian --iterations 20

# 在线调参
python mppi_control_adaptive.py continuous

# 查看状态
python mppi_control_adaptive.py adaptive_status
```

## 📁 文件结构

```
LED_MPPI_Controller/
├── src/
│   ├── auto_tuning.py              # 完整的离线调参框架
│   └── adaptive_tuning.py          # 简化的在线调参模块
├── applications/
│   ├── control/
│   │   └── mppi_control_adaptive.py  # 集成自适应调参的控制脚本
│   └── tuning/
│       └── auto_tune_mppi.py          # 离线调参执行脚本
├── scripts/
│   └── install_tuning_deps.sh         # 依赖安装脚本
├── tests/
│   └── test_auto_tuning.py            # 自动调参功能测试
└── docs/
    └── auto_tuning_guide.md           # 详细使用指南
```

## 🚀 快速开始

### 1. 安装依赖
```bash
bash scripts/install_tuning_deps.sh
```

### 2. 离线调参（推荐先执行）
```bash
# 贝叶斯优化（需要skopt库）
python applications/tuning/auto_tune_mppi.py --method bayesian --iterations 20

# 差分进化（无需额外依赖）
python applications/tuning/auto_tune_mppi.py --method evolutionary --iterations 30

# 网格搜索
python applications/tuning/auto_tune_mppi.py --method grid --grid-size 3
```

### 3. 在线自适应调参
```bash
# 单次运行
python applications/control/mppi_control_adaptive.py once

# 连续运行
python applications/control/mppi_control_adaptive.py continuous

# 后台运行
python applications/control/mppi_control_adaptive.py background
```

### 4. 查看状态
```bash
# 查看自适应调参状态
python applications/control/mppi_control_adaptive.py adaptive_status
```

## 🔧 主要调优参数

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| **Q_photo** | 10.0-50.0 | 25.0 | 光合作用权重，越大越重视光合作用 |
| **Q_ref** | 5.0-50.0 | 25.0 | 参考跟踪权重，越大越紧跟参考值 |
| **R_du** | 0.001-0.1 | 0.02 | 控制变化惩罚，越大控制越平滑 |
| **R_power** | 0.001-0.05 | 0.005 | 功率惩罚，越大越节能 |
| **u_std** | 0.1-0.5 | 0.25 | 控制采样标准差，影响探索性 |
| **temperature** | 0.5-2.0 | 1.0 | MPPI温度参数，影响随机性 |

## 📊 性能指标

自动调参系统基于以下指标评估性能：

| 指标 | 权重 | 说明 |
|------|------|------|
| 光合作用 | +1.0 | 最大化光合速率 |
| 温度违规 | -0.5 | 最小化温度约束违反 |
| 控制平滑性 | -0.3 | 最小化控制变化 |
| 功率效率 | -0.2 | 最小化功率消耗 |
| 参考跟踪 | -0.4 | 最小化参考跟踪误差 |

## 📈 使用效果

### 测试结果
- ✅ 所有核心功能测试通过
- ✅ 参数范围验证正常
- ✅ 性能评分计算正确
- ✅ 历史数据持久化工作正常

### 预期改进
1. **控制精度提升**: 通过优化权重参数，提高跟踪精度
2. **能耗优化**: 平衡光合作用和功率消耗
3. **稳定性增强**: 减少控制震荡，提高系统稳定性
4. **适应性提升**: 在线调参能够适应环境变化

## 🔍 监控和调试

### 日志文件
- `logs/auto_tuning_results.json` - 离线调参结果
- `logs/adaptive_tuning_history.json` - 自适应调参历史
- `logs/mppi_adaptive_control_log.csv` - 控制日志
- `logs/tuning_report.md` - 调参报告

### 关键监控指标
```python
# 查看自适应状态
summary = tuner.get_performance_summary()
print(f"平均性能分数: {summary['avg_score']:.4f}")
print(f"平均光合速率: {summary['avg_photosynthesis']:.3f}")
print(f"记录总数: {summary['num_records']}")
```

## ⚙️ 配置选项

### 离线调参配置
```json
{
  "tuning_method": "bayesian",
  "evaluation_period_hours": 24,
  "max_iterations": 50,
  "convergence_threshold": 0.01
}
```

### 在线调参配置
```python
ADAPTIVE_TUNING_ENABLED = True     # 启用自适应调参
ADAPTATION_PERIOD_HOURS = 6        # 适应周期
LEARNING_RATE = 0.05               # 学习率
```

## 🎯 最佳实践

### 1. 调参流程
1. **数据准备**: 收集足够的历史控制数据
2. **离线调参**: 使用贝叶斯优化或差分进化获得基础最优参数
3. **在线调参**: 启用自适应调参进行实时优化
4. **监控验证**: 定期检查调参效果和系统性能

### 2. 参数调整建议
- **Q_photo**: 如果光合作用不足，增大此值
- **Q_ref**: 如果跟踪误差大，增大此值
- **R_du**: 如果控制震荡，增大此值
- **R_power**: 如果功耗过高，增大此值

### 3. 性能优化
- 定期分析性能日志
- 根据季节变化调整参数范围
- 监控环境变化对控制性能的影响

## 🔮 扩展功能

### 已实现
- ✅ 多种离线调参算法
- ✅ 在线自适应调参
- ✅ 性能指标计算
- ✅ 历史数据持久化
- ✅ 集成控制脚本

### 可扩展
- 🔄 多目标优化
- 🔄 强化学习调参
- 🔄 环境感知调参
- 🔄 分布式调参

## 📞 技术支持

如有问题，请检查：
1. 依赖是否正确安装
2. 配置文件是否正确
3. 日志文件中的错误信息
4. 运行测试脚本验证功能

## 🎉 总结

自动调参功能为MPPI控制器提供了强大的参数优化能力：

- **离线调参** 提供全局最优参数搜索
- **在线调参** 实现实时参数适应
- **完整集成** 与现有控制系统无缝结合
- **易于使用** 提供简单的命令行接口
- **高度可配置** 支持多种调参策略

通过这些功能，可以显著提升LED植物照明控制系统的性能和适应性。
