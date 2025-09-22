# LED MPPI Controller

基于模型预测路径积分（MPPI）的LED植物生长控制系统，支持多种机器学习模型进行光合作用预测。

## 项目结构

```
LED_MPPI_Controller/
├── src/                    # 源代码
│   ├── mppi.py            # MPPI控制器核心模块
│   └── led.py             # LED模型和热力学模块
├── applications/           # 实际应用脚本
│   ├── control/           # 控制脚本
│   │   ├── mppi_control_real.py      # 实际控制执行
│   │   └── mppi_control_simulate.py  # 模拟控制
│   ├── utils/             # 工具脚本
│   │   └── temperature_reader.py     # 温度读取工具
│   └── scripts/           # 系统脚本
│       ├── start_control.sh          # 统一启动脚本
│       └── test_system.py            # 系统测试
├── tests/                 # 测试文件
│   ├── test_all_models.py # 所有模型测试
│   └── test_solar_vol.py  # 单个模型测试
├── examples/              # 示例代码
│   ├── example_model_usage.py # 模型使用示例
│   ├── mppi_demo.py       # MPPI演示
│   ├── pwm_demo.py        # PWM演示
│   └── thermal_demo.py    # 热力学演示
├── docs/                  # 文档
│   ├── README_solar_vol.md # 模型集成详细说明
│   └── APPLICATIONS.md     # 应用指南
├── config/                # 配置文件
│   ├── default_config.py  # 默认配置
│   └── app_config.py      # 应用配置
├── data/                  # 数据文件
│   ├── calib_data.csv     # 校准数据
│   └── *.png             # 图像文件
├── models/                # 训练好的模型
│   ├── solar_vol/         # 太阳能电压模型
│   ├── ppfd/             # 光合光子通量密度模型
│   └── sp/               # 光谱模型
├── logs/                  # 日志文件
└── README.md             # 项目说明
```

## 功能特性

- **多模型支持**: 集成三种机器学习模型（solar_vol, ppfd, sp）
- **MPPI控制**: 基于模型预测路径积分的优化控制
- **热力学建模**: LED热效应和温度控制
- **实时预测**: 光合作用速率实时预测
- **灵活配置**: 可调节的控制参数和约束条件

## 支持的模型

| 模型 | 输入特征 | 性能 (R²) | 特点 |
|------|----------|-----------|------|
| **Solar_Vol** | Solar_Vol, CO2, T, R:B | 0.9866 | 最稳定，推荐默认 |
| **PPFD** | PPFD, CO2, T, R:B | 0.9807 | 高光合作用速率 |
| **SP** | 8个光谱波段 + CO2, T, R:B | 0.9788 | 光谱信息丰富 |

## 快速开始

### 1. 环境设置

```bash
# 激活conda环境
conda activate ppfd_env

# 进入项目目录
cd LED_MPPI_Controller
```

### 2. 基本使用

```python
from src.mppi import LEDPlant, LEDMPPIController

# 创建LED植物模型（默认使用solar_vol模型）
plant = LEDPlant()

# 单步预测
ppfd, temp, power, photo = plant.step(r_pwm=50.0, b_pwm=10.0, dt=0.1)
print(f"PPFD={ppfd:.2f}, Temp={temp:.2f}°C, Photo={photo:.2f}")

# MPPI控制器
controller = LEDMPPIController(plant, horizon=10, num_samples=500)
action, sequence, success, cost, weights = controller.solve(current_temp=25.0)
print(f"控制动作: R_PWM={action[0]:.2f}, B_PWM={action[1]:.2f}")
```

### 3. 模型选择

```python
# 使用不同模型
plant_solar = LEDPlant(model_name='solar_vol')
plant_ppfd = LEDPlant(model_name='ppfd')
plant_sp = LEDPlant(model_name='sp')
```

## 运行示例

```bash
# 验证项目设置
python verify_setup.py

# 测试所有模型
python tests/test_all_models.py

# 运行使用示例
python examples/example_model_usage.py

# 运行MPPI演示
python examples/mppi_demo.py

# 运行实际控制应用
./applications/scripts/start_control.sh test

# 运行系统集成测试
python tests/test_system.py
```

## 技术细节

- **控制算法**: 模型预测路径积分（MPPI）
- **机器学习**: 支持向量回归（SVR）with RBF核
- **热力学**: 一阶热传递模型
- **优化**: 多目标优化（光合作用、功耗、温度）

## 依赖项

- Python 3.10+
- numpy
- scikit-learn
- matplotlib (用于演示)
- joblib (模型加载)

## 许可证

本项目仅供学术研究使用。

## 更新日志

- **v1.0.0**: 初始版本，支持单模型
- **v2.0.0**: 多模型支持，规范项目结构
