# LED植物生长控制系统 - Main目录

## 📁 文件结构

```
Main/
├── 控制脚本/
│   ├── mppi_control_simulate.py    # 模拟控制（仅打印命令）
│   ├── mppi_control_real.py        # 实际控制（发送命令）
│   └── temperature_reader.py       # 温度读取工具
├── 测试脚本/
│   └── test_system.py              # 系统集成测试
├── 启动脚本/
│   └── start_control.sh            # 统一启动脚本
├── 日志文件/
│   ├── control_simulate_log.csv    # 模拟控制日志
│   └── control_real_log.csv        # 实际控制日志
└── 文档/
    └── README_MPPI_Control.md      # 详细技术文档
```

## 🚀 快速开始

### 1. 模拟模式（推荐用于测试）
```bash
# 运行一次模拟控制
./start_control.sh once

# 连续运行模拟控制
./start_control.sh continuous
```

### 2. 实际执行模式（用于真实控制）
```bash
# 运行一次实际控制
./start_control.sh execute

# 连续运行实际控制
./start_control.sh execute-cont
```

### 3. 系统测试
```bash
# 运行系统集成测试
./start_control.sh test

# 列出可用温度设备
./start_control.sh list-devices
```

## 📊 日志记录

### 模拟模式日志 (`control_simulate_log.csv`)
- 记录控制决策和命令
- 不实际发送到设备
- 用于算法验证和调试

### 实际执行日志 (`control_real_log.csv`)
- 记录实际发送的命令
- 包含设备状态检查结果
- 用于监控实际控制效果

## ⚙️ 配置

所有配置参数都在各Python脚本顶部的宏定义中：

```python
# 温度传感器设备ID
TEMPERATURE_DEVICE_ID = None  # None=自动选择

# 控制循环间隔（分钟）
CONTROL_INTERVAL_MINUTES = 1

# 红蓝比例键
RB_RATIO_KEY = "5:1"
```

## 🔧 工具脚本

- **`temperature_reader.py`**: 快速查看当前温度
- **`test_system.py`**: 系统集成测试
- **`start_control.sh`**: 统一启动脚本

## 📝 日志格式

### CSV字段说明
- **时间戳**: YYYY-MM-DD HH:MM:SS
- **输入温度**: 当前环境温度(°C)
- **红光PWM**: 红光LED的PWM值
- **蓝光PWM**: 蓝光LED的PWM值
- **成功状态**: True/False
- **成本**: MPPI算法计算的成本值
- **设备状态**: 实际设备状态（仅实际执行模式）
- **备注**: 额外信息

## 🎯 使用建议

1. **开发测试**: 使用模拟模式验证算法
2. **实际控制**: 使用实际执行模式控制设备
3. **监控分析**: 查看CSV日志文件分析控制效果
4. **故障排除**: 使用系统测试检查各组件状态
