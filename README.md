# MPPI LED控制循环系统

一个基于模型预测路径积分控制(MPPI)的智能LED植物光照控制系统，集成温度传感器、MPPI控制算法和LED设备控制。

## 🌟 主要功能

- **智能控制**: 使用MPPI算法优化LED设置，最大化光合作用效率
- **温度调节**: 基于实时温度数据自动调整LED功率
- **设备集成**: 支持Shelly智能开关控制LED设备
- **自动化运行**: 支持每分钟自动运行控制循环
- **实时监控**: 提供温度读取和系统状态监控

## 📁 项目结构

```
Project1/
├── Main/                          # 主要控制脚本
│   ├── mppi_control_loop.py      # MPPI控制循环主脚本
│   ├── test_mppi_integration.py  # 集成测试脚本
│   ├── start_mppi_control.sh     # 启动脚本
│   ├── quick_temp.py             # 快速温度显示
│   └── README_MPPI_Control.md    # 详细使用说明
├── AA_Test_9_16/                 # MPPI算法和LED模型
│   ├── mppi.py                   # MPPI控制器核心
│   ├── led.py                    # LED热力学模型
│   └── models/                   # 机器学习模型
├── Test/riotee_sensor/           # 温度传感器模块
└── aioshelly/my_src/             # Shelly设备控制
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- numpy
- 其他依赖见各模块requirements.txt

### 2. 运行测试

```bash
cd Main
./start_mppi_control.sh test
```

### 3. 运行单次控制循环

```bash
cd Main
./start_mppi_control.sh once
```

### 4. 连续运行控制循环

```bash
cd Main
./start_mppi_control.sh continuous
```

## 🎯 系统架构

```
温度传感器数据 → MPPI控制器 → PWM命令 → LED设备
     ↓              ↓           ↓
  Riotee传感器   优化算法     Shelly设备
```

## ⚙️ 控制参数

- **预测时域**: 10步
- **采样数量**: 1000个
- **红蓝比例**: 5:1
- **PWM范围**: 0-80%
- **温度范围**: 20-29°C
- **目标温度**: 25°C

## 📊 控制目标

1. **最大化光合作用**: 主要优化目标
2. **最小化功率消耗**: 节能优化
3. **维持温度约束**: 确保植物生长环境
4. **平滑控制**: 避免剧烈变化

## 🔧 配置说明

### 设备配置

在 `aioshelly/my_src/controller.py` 中配置设备IP：

```python
DEVICES = {
    "Red": "192.168.50.94",   # 红光设备IP
    "Blue": "192.168.50.69",  # 蓝光设备IP
}
```

### MPPI参数调整

在 `Main/mppi_control_loop.py` 中调整控制参数：

```python
# 调整目标温度
self.target_temp = 25.0

# 调整MPPI参数
self.controller = LEDMPPIController(
    horizon=10,           # 预测时域
    num_samples=1000,     # 采样数量
    temperature=1.0,      # 温度参数
    # ...
)
```

## 📈 输出示例

```
🔄 控制循环开始 - 2024-01-15 14:30:00
🌡️  🟢 温度读取: 24.50°C (设备: riotee_001, 45秒前)
🎯 运行MPPI控制 (当前温度: 24.50°C, 目标: 25.00°C)
📊 MPPI结果:
   红光PWM: 45.20
   蓝光PWM: 9.04
   总PWM: 54.24
   成本: 1234.56
📡 发送PWM命令到设备...
🔴 红光设备: brightness=57
🔵 蓝光设备: brightness=11
✅ 控制循环完成
```

## 🛠️ 故障排除

### 常见问题

1. **温度读取失败**: 检查Riotee传感器数据文件
2. **MPPI控制失败**: 检查模型文件是否正确加载
3. **设备连接失败**: 检查网络连接和设备IP地址

### 调试模式

```bash
python test_mppi_integration.py
```

## 📝 更新日志

### v1.0.0 (2024-09-18)
- 初始版本发布
- 集成MPPI控制循环系统
- 支持温度传感器和LED设备控制
- 移除退化模型，强化错误处理

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用MIT许可证。

## 🔗 相关链接

- [GitHub仓库](https://github.com/HaydenYu916/Project1)
- [详细使用说明](Main/README_MPPI_Control.md)
