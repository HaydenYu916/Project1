# 智能植物光照控制系统

一个基于模型预测路径积分控制(MPPI)的智能LED植物光照控制系统，集成温度传感器、MPPI控制算法和LED设备控制。

## 🌟 主要功能

- **智能控制**: 使用MPPI算法优化LED设置，最大化光合作用效率
- **温度调节**: 基于实时温度数据自动调整LED功率
- **设备集成**: 支持Shelly智能开关控制LED设备
- **自动化运行**: 支持每分钟自动运行控制循环
- **实时监控**: 提供温度读取和系统状态监控
- **数据收集**: 自动收集传感器数据并按PPFD时间段分组

## 📁 项目结构

```
Growpro/
├── LED_MPPI_Controller/          # MPPI控制核心模块
│   ├── src/                     # 源代码
│   │   ├── mppi.py             # MPPI控制器核心
│   │   └── led.py              # LED热力学模型
│   ├── applications/            # 实际应用脚本
│   │   ├── control/            # 控制脚本
│   │   ├── utils/              # 工具脚本
│   │   └── scripts/            # 系统脚本
│   ├── models/                 # 机器学习模型
│   ├── tests/                  # 测试文件
│   ├── examples/               # 示例代码
│   └── docs/                   # 文档
├── Tool/                        # 工作区根目录共享工具目录
│   ├── LED_Shelly/             # Shelly设备控制模块
│   └── Sensor_riotee_server/   # Riotee传感器采集模块
└── Sensor/                      # 实验数据目录
    ├── EXP3/                   # 历史实验数据
    └── EXP4/                   # 历史实验数据
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- numpy, scikit-learn, matplotlib
- 其他依赖见各模块requirements.txt

### 2. MPPI控制模块测试

```bash
cd LED_MPPI_Controller
python tests/test_all_models.py
```

### 3. Shelly设备控制

```bash
cd Tool/LED_Shelly
python src/shelly_controller.py Red on
python src/shelly_controller.py Blue brightness 50
```

### 4. PWM调度器运行

```bash
cd Tool/LED_Shelly
python tests/pwm_service.py start
```

### 5. 传感器数据收集

```bash
cd Tool/Sensor_riotee_server
python riotee_system_manager.py start
```

## 🎯 系统架构

```
Sensor模块 → LED_MPPI_Controller → Shelly模块 → LED设备
    ↓              ↓                    ↓
Riotee传感器    MPPI优化算法        Shelly设备
温度/光谱数据   光合作用预测        PWM控制
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

### Shelly设备配置

在工作区根目录的 `Tool/LED_Shelly/config/device_config.py` 中配置设备IP：

```python
DEVICES = {
    "Red": "192.168.0.46",   # 红光设备IP
    "Blue": "192.168.0.63",  # 蓝光设备IP
}
```

### MPPI参数调整

在 `LED_MPPI_Controller/applications/control/mppi_control_real.py` 中调整控制参数：

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

### PWM调度器配置

在工作区根目录的 `Tool/LED_Shelly/tests/` 下配置时间表：

```csv
time,ppfd,r_pwm,b_pwm,phase,phase_name
2025-09-19 07:00:00,100,12,9,heating1,Heating 1
2025-09-19 07:30:00,200,30,10,heating1,Heating 1
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

1. **温度读取失败**: 检查工作区根目录 `Tool/Sensor_riotee_server/logs/` 中的数据文件
2. **MPPI控制失败**: 检查 `LED_MPPI_Controller/models/` 中的模型文件
3. **设备连接失败**: 检查工作区根目录 `Tool/LED_Shelly/config/device_config.py` 中的IP地址
4. **PWM调度器不工作**: 检查工作区根目录 `Tool/LED_Shelly/tests/` 中的时间表文件

### 调试模式

```bash
# MPPI模块调试
cd LED_MPPI_Controller
python tests/test_all_models.py

# Shelly模块调试
cd Tool/LED_Shelly
python tests/pwm_scheduler.py --status

# 传感器模块调试
cd Tool/Sensor_riotee_server
python riotee_system_manager.py status
```

## 📝 更新日志

### v2.0.0 (2025-09-19)
- **项目重组**: 控制核心保留在 `LED_MPPI_Controller`，设备与传感器工具统一到工作区根目录 `Tool/`
- **PWM调度器**: 新增基于时间表的自动PWM控制功能
- **数据收集**: 自动收集传感器数据并按PPFD时间段分组保存
- **实时更新**: 支持实时数据更新和后台运行模式
- **多模型支持**: 集成三种机器学习模型（solar_vol, ppfd, sp）

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
- [LED MPPI控制器详细说明](LED_MPPI_Controller/README.md)
- [Shelly设备控制说明](Tool/LED_Shelly/README.md)
- [PWM调度器使用说明](Tool/LED_Shelly/tests/README_PWM_Scheduler.md)
