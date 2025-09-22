# LED MPPI Controller 应用指南

## 概述

本目录包含了LED MPPI控制系统的实际应用脚本，用于控制真实的LED设备进行植物生长优化。

## 📁 应用结构

```
applications/
├── control/                    # 控制脚本
│   ├── mppi_control_real.py    # 实际控制执行（发送命令到设备）
│   └── mppi_control_simulate.py # 模拟控制（仅打印命令）
├── utils/                      # 工具脚本
│   └── temperature_reader.py   # 温度读取工具
└── scripts/                    # 系统脚本
    ├── start_control.sh        # 统一启动脚本
    └── test_system.py          # 系统集成测试
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活conda环境
conda activate ppfd_env

# 进入项目目录
cd LED_MPPI_Controller
```

### 2. 配置参数

编辑 `config/app_config.py` 文件来配置系统参数：

```python
# 温度传感器设备ID
TEMPERATURE_DEVICE_ID = None  # 自动选择设备

# 控制间隔
CONTROL_INTERVAL_MINUTES = 1

# 目标温度
TARGET_TEMPERATURE = 25.0

# 设备IP地址
RED_LED_IP = "192.168.50.94"
BLUE_LED_IP = "192.168.50.69"
```

### 3. 运行控制

```bash
# 使用启动脚本（推荐）
./applications/scripts/start_control.sh once          # 运行一次（模拟模式）
./applications/scripts/start_control.sh continuous    # 连续运行（模拟模式）
./applications/scripts/start_control.sh execute       # 运行一次（实际执行）
./applications/scripts/start_control.sh execute-cont  # 连续运行（实际执行）
./applications/scripts/start_control.sh test          # 系统测试
./applications/scripts/start_control.sh list-devices  # 列出设备
```

## 📊 控制模式

### 模拟模式（推荐用于测试）

- **文件**: `applications/control/mppi_control_simulate.py`
- **功能**: 读取温度，运行MPPI控制，打印命令但不发送到设备
- **日志**: `logs/control_simulate_log.csv`
- **用途**: 算法验证、参数调试、系统测试

### 实际执行模式

- **文件**: `applications/control/mppi_control_real.py`
- **功能**: 读取温度，运行MPPI控制，实际发送命令到LED设备
- **日志**: `logs/control_real_log.csv`
- **用途**: 实际设备控制、生产环境

## 🔧 工具脚本

### 温度读取工具

```bash
python applications/utils/temperature_reader.py
```

- 快速查看当前温度
- 显示数据新鲜度状态
- 支持指定设备ID

### 系统测试

```bash
python tests/test_system.py
```

- 测试温度读取功能
- 测试MPPI控制算法
- 测试设备连接
- 生成测试报告

## 📝 日志系统

### 日志格式

所有日志都保存为CSV格式，包含以下字段：

| 字段 | 说明 |
|------|------|
| timestamp | 时间戳 (YYYY-MM-DD HH:MM:SS) |
| input_temp | 输入温度 (°C) |
| red_pwm | 红光PWM值 |
| blue_pwm | 蓝光PWM值 |
| success | 执行状态 (True/False) |
| cost | MPPI成本值 |
| device_status | 设备状态（仅实际执行模式） |
| notes | 备注信息 |

### 日志位置

- 模拟模式: `logs/control_simulate_log.csv`
- 实际执行: `logs/control_real_log.csv`

## ⚙️ 系统架构

```
温度传感器 → MPPI控制器 → PWM命令 → LED设备
    ↓           ↓           ↓         ↓
 Riotee     优化算法     Shelly   实际LED
 传感器      (src/)     控制器    设备
```

## 🎯 控制流程

1. **温度读取**: 从Riotee传感器读取当前环境温度
2. **MPPI优化**: 基于当前温度运行MPPI算法，计算最优PWM值
3. **命令生成**: 将PWM值转换为设备控制命令
4. **设备控制**: 发送命令到Shelly LED设备
5. **状态检查**: 验证命令执行结果
6. **日志记录**: 记录控制过程和结果

## 🔒 安全特性

### 温度保护

- 如果预测温度超过29°C，自动降低PWM值
- 温度保护缩放因子：0.7

### PWM限制

- 最大PWM值限制：80
- 防止设备过载和损坏

### 数据验证

- 只使用5分钟内的温度数据
- 自动检测数据新鲜度

## 📱 设备管理

### 温度传感器

- **类型**: Riotee传感器
- **数据格式**: CSV文件
- **更新频率**: 实时
- **设备选择**: 支持自动选择或指定设备ID

### LED设备

- **类型**: Shelly智能开关
- **通信**: HTTP RPC
- **控制方式**: 亮度调节 (0-100)
- **网络**: 本地网络 (192.168.50.x)

## 🐛 故障排除

### 常见问题

1. **温度读取失败**
   - 检查Riotee传感器数据文件
   - 确认设备ID配置正确
   - 检查数据时效性

2. **MPPI控制失败**
   - 检查模型文件是否正确加载
   - 确认温度值在合理范围内
   - 检查约束参数设置

3. **设备连接失败**
   - 检查网络连接
   - 确认设备IP地址正确
   - 检查设备在线状态

### 调试模式

```bash
# 运行系统测试查看详细信息
./applications/scripts/start_control.sh test

# 查看温度读取状态
python applications/utils/temperature_reader.py
```

## 📈 性能优化

### 控制参数调优

- **预测时域**: 增加horizon提高预测精度
- **采样数量**: 增加num_samples提高优化质量
- **控制间隔**: 根据系统响应调整间隔时间

### 模型选择

- **solar_vol**: 最稳定，推荐默认使用
- **ppfd**: 高光合作用速率预测
- **sp**: 光谱信息丰富，适合特殊应用

## 🔄 扩展开发

### 添加新的控制策略

1. 在 `applications/control/` 中创建新的控制脚本
2. 继承基础控制类
3. 实现自定义控制逻辑
4. 更新启动脚本支持新策略

### 集成新的传感器

1. 在 `applications/utils/` 中创建传感器读取脚本
2. 实现统一的数据接口
3. 更新控制脚本使用新传感器
4. 添加相应的测试代码

## 📞 技术支持

如遇到问题，请：

1. 查看日志文件获取错误信息
2. 运行系统测试诊断问题
3. 检查配置文件参数设置
4. 确认外部依赖路径正确
