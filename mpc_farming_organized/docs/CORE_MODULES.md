# 核心模块详细说明

## 📁 core/ 目录

### 1. led.py - LED光照模型
**功能**: LED物理建模和环境仿真

**主要类**:
- `LedParams`: LED参数配置类
- `led_step()`: 单步LED仿真函数
- `led_steady_state()`: 稳态计算函数

**关键参数**:
```python
base_ambient_temp = 23.0    # 环境基准温度 (°C)
max_ppfd = 600.0           # 最大PPFD (μmol/m²/s)
max_power = 86.4           # 最大功率 (W)
thermal_resistance = 0.05  # 热阻 (K/W)
time_constant_s = 7.5      # 时间常数 (s)
thermal_mass = 150.0       # 热容 (J/°C)
```

**使用示例**:
```python
from core.led import led_step

ppfd, temp, power, _ = led_step(
    pwm_percent=50.0,        # PWM百分比
    ambient_temp=25.0,       # 环境温度
    base_ambient_temp=22.0,  # 基准温度
    dt=1.0                   # 时间步长
)
```

### 2. mppi_api.py - MPPI控制API
**功能**: 简化的MPPI控制接口

**主要函数**:
- `mppi_next_ppfd()`: 计算下一个PPFD设定值

**输入参数**:
- `current_ppfd`: 当前PPFD测量值 (μmol/m²/s)
- `temperature`: 当前环境温度 (°C)
- `co2`: CO2浓度 (ppm)
- `humidity`: 相对湿度 (%)

**输出**:
- 预测的PPFD设定值 (μmol/m²/s)

**使用示例**:
```python
from core.mppi_api import mppi_next_ppfd

ppfd = mppi_next_ppfd(
    current_ppfd=300.0,
    temperature=25.0,
    co2=400.0,
    humidity=60.0
)
```

### 3. mppi.py - 完整MPPI实现
**功能**: 完整的MPPI控制算法实现

**主要类**:
- `LEDPlant`: LED植物模型类
- `MPPIController`: MPPI控制器类

**关键特性**:
- 随机采样优化
- 约束处理
- 实时控制
- 参数可调

**使用示例**:
```python
from core.mppi import MPPIController, LEDPlant

# 创建植物模型
plant = LEDPlant(
    base_ambient_temp=22.0,
    max_ppfd=700.0,
    max_power=100.0
)

# 创建MPPI控制器
controller = MPPIController(
    plant=plant,
    horizon=10,
    num_samples=800
)

# 控制循环
for step in range(100):
    pwm = controller.compute_control(
        current_ppfd=plant.current_ppfd,
        temperature=plant.ambient_temp,
        co2=400.0,
        humidity=60.0
    )
    plant.step(pwm)
```

### 4. mpc.py - 传统MPC控制器
**功能**: 基于优化的模型预测控制

**主要类**:
- `LEDPlant`: LED植物模型类
- `MPCController`: MPC控制器类

**关键特性**:
- 确定性优化
- 约束处理
- 稳定性保证
- 计算效率

**使用示例**:
```python
from core.mpc import MPCController, LEDPlant

# 创建植物模型
plant = LEDPlant(
    base_ambient_temp=22.0,
    max_ppfd=700.0,
    max_power=100.0
)

# 创建MPC控制器
controller = MPCController(
    plant=plant,
    horizon=10
)

# 控制循环
for step in range(100):
    pwm = controller.compute_control(
        current_ppfd=plant.current_ppfd,
        temperature=plant.ambient_temp,
        co2=400.0,
        humidity=60.0
    )
    plant.step(pwm)
```

## 🔧 模块间关系

```
led.py (物理模型)
    ↓
mppi_api.py (简化接口)
    ↓
mppi.py (完整实现)
    ↓
mpc.py (替代方案)
```

## 📊 性能对比

| 控制器 | 计算速度 | 控制精度 | 稳定性 | 适用场景 |
|--------|----------|----------|--------|----------|
| MPPI API | 最快 | 高 | 好 | 实时控制 |
| MPPI | 快 | 很高 | 很好 | 复杂控制 |
| MPC | 中等 | 高 | 最好 | 精确控制 |

## 🎯 选择建议

### 使用 MPPI API 当:
- 需要快速集成
- 实时控制要求高
- 简单应用场景

### 使用 完整MPPI 当:
- 需要自定义参数
- 复杂控制需求
- 研究开发

### 使用 MPC 当:
- 需要理论保证
- 稳定性要求高
- 确定性控制

## ⚠️ 注意事项

1. **参数调优**: 不同应用场景需要调整控制参数
2. **模型依赖**: 需要准确的环境和植物模型
3. **计算资源**: MPPI需要较多计算资源
4. **实时性**: 确保控制循环的实时性要求
