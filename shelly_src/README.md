# Shelly Controller

用于控制Shelly设备的Python库，支持RPC命令、实时监听和系统管理。

## 项目结构

```
shelly_src/
├── src/                    # 源代码
│   ├── __init__.py
│   ├── shelly_controller.py    # 核心控制器
│   ├── shelly_listener.py      # 实时监听器
│   ├── shelly_live_api.py      # Live API接口
│   └── shelly_system_manager.py # 系统管理器
├── tests/                  # 测试文件
│   ├── pwm_scheduler.py       # PWM调度器
│   ├── pwm_service.py         # PWM服务
│   ├── test_correct_ppfd.py   # PPFD测试
│   ├── README_PWM_Scheduler.md # PWM调度器文档
│   └── src/                   # 测试数据
│       └── data/              # 图像文件
├── examples/               # 示例代码
│   ├── demo_fill_table.py     # 表格填充演示
│   ├── repair_sweep.py        # 修复扫描
│   └── sweep_pwm.py           # PWM扫描
├── config/                 # 配置文件
│   └── device_config.py       # 设备配置
├── docs/                   # 文档
├── data/                   # 数据文件
├── logs/                   # 日志文件
├── requirements.txt        # 依赖列表
└── README.md              # 项目说明
```

## 功能特性

- **设备控制**: 支持红蓝LED设备的开关、亮度控制
- **RPC通信**: 基于HTTP的RPC命令接口
- **实时监听**: WebSocket实时状态监听
- **PWM调度**: 基于时间表的自动PWM控制
- **数据收集**: 自动收集传感器数据

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基本使用

```python
from src.shelly_controller import rpc, DEVICES

# 控制红光设备
rpc(DEVICES["Red"], "Light.Set", {"id": 0, "on": True, "brightness": 80})

# 控制蓝光设备  
rpc(DEVICES["Blue"], "Light.Set", {"id": 0, "on": True, "brightness": 20})
```

### 3. 命令行使用

```bash
# 打开红光设备
python src/shelly_controller.py Red on

# 设置亮度
python src/shelly_controller.py Red brightness 80

# 获取状态
python src/shelly_controller.py Red get_status
```

## 设备配置

在`config/device_config.py`中配置设备IP地址：

```python
DEVICES = {
    "Red": "192.168.50.94",
    "Blue": "192.168.50.69",
}
```

## PWM调度器

使用PWM调度器进行自动控制：

```bash
# 运行PWM调度器
python tests/pwm_scheduler.py

# 后台运行
python tests/pwm_service.py start
```

## 许可证

本项目仅供学术研究使用。
