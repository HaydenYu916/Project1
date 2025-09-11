# 传感器数据采集系统

统一的多传感器数据采集、处理和分析系统，支持CO2传感器和Riotee传感器的数据收集、实时监控和MQTT发布。

## 📁 项目结构

```
Test/
├── co2_sensor/              # CO2传感器模块
│   ├── __init__.py          # CO2数据读取接口
│   ├── co2_data_collector.py # CO2数据采集器
│   ├── co2_example.py       # CO2使用示例
│   └── co2_system_manager.py # CO2系统管理器
├── riotee_sensor/           # Riotee传感器模块  
│   ├── __init__.py          # Riotee数据读取接口
│   ├── riotee_controller.py # 设备控制器（aioshelly风格）
│   ├── riotee_data_collector.py # 数据采集器
│   ├── riotee_example.py    # 使用示例
│   ├── riotee_live_api.py   # 实时API
│   ├── riotee_system_manager.py # 系统管理器
│   └── README.md            # Riotee模块详细说明
├── sensor_hub/              # 统一数据读取模块
│   └── __init__.py          # 统一接口，同时读取两种传感器
├── examples/                # 使用示例
│   └── sensor_hub_demo.py   # 统一数据读取演示
├── logs/                    # 统一日志目录
│   ├── co2_data.csv         # CO2数据记录
│   ├── riotee_data_all.csv  # Riotee全量数据
│   ├── riotee_data_summary.csv # Riotee摘要数据
│   └── [其他实验数据...]    # 历史实验数据
└── README.md                # 项目总说明（本文件）
```

## 🚀 快速开始

### 1. 启动CO2传感器系统
```bash
cd co2_sensor
python3 co2_system_manager.py start
```

### 2. 启动Riotee传感器系统
```bash
cd riotee_sensor
python3 riotee_system_manager.py start
```

### 3. 使用统一接口读取数据
```bash
cd examples
python3 sensor_hub_demo.py
```

## 📊 核心功能

### CO2传感器 (`co2_sensor/`)
- **数据采集**: 实时读取CO2浓度数据
- **CSV记录**: 自动记录到统一logs目录
- **系统管理**: 启动/停止/状态检查
- **数据接口**: 提供Python API获取当前数据

### Riotee传感器 (`riotee_sensor/`)
- **数据采集**: 温度、湿度、电压、光谱数据
- **设备控制**: aioshelly风格的命令行控制器
- **双CSV系统**: 全量数据 + 智能摘要
- **MQTT发布**: 实时数据发布到MQTT broker
- **Home Assistant**: 自动发现和集成

### 统一数据接口 (`sensor_hub/`)
- **多传感器读取**: 同时获取CO2和Riotee数据
- **数据融合**: 统一的数据格式和时间戳
- **状态检查**: 检查各系统运行状态
- **简化接口**: 便于应用程序集成

## 🎯 使用场景

### 1. 单独使用CO2传感器
```python
from co2_sensor import get_current_co2
data = get_current_co2()
print(f"当前CO2: {data['value']} ppm")
```

### 2. 单独使用Riotee传感器
```python
from riotee_sensor import get_current_riotee
data = get_current_riotee()
print(f"温度: {data['temperature']}°C")
```

### 3. 使用统一接口
```python
from sensor_hub import get_all_current_data
data = get_all_current_data()
print(f"CO2: {data['co2_data']['value']} ppm")
print(f"温度: {data['riotee_data']['temperature']}°C")
```

### 4. 控制Riotee设备
```bash
cd riotee_sensor
python3 riotee_controller.py all sleep 30  # 设置所有设备休眠30秒
python3 riotee_controller.py all status    # 查看设备状态
```

## 📈 数据记录

所有数据统一记录到 `logs/` 目录：

- **CO2数据**: `co2_data.csv`
- **Riotee全量数据**: `riotee_data_all.csv` 
- **Riotee摘要数据**: `riotee_data_summary.csv`
- **历史实验数据**: 按实验名称和时间戳命名

### 数据格式特点
- **会话管理**: 类似aioshelly的Start/Stop标记
- **实验备注**: 支持为每次实验添加备注
- **智能摘要**: 自动记录设备发现和重要状态变化
- **时间戳**: 统一的时间格式和数据时效性检查

## 🔧 系统管理

### 检查系统状态
```bash
# CO2系统
cd co2_sensor && python3 co2_system_manager.py status

# Riotee系统  
cd riotee_sensor && python3 riotee_system_manager.py status

# 统一状态检查
cd examples && python3 sensor_hub_demo.py
```

### 停止系统
```bash
# 停止CO2系统
cd co2_sensor && python3 co2_system_manager.py stop

# 停止Riotee系统
cd riotee_sensor && python3 riotee_system_manager.py stop
```

## 📝 注意事项

1. **环境要求**: 确保已安装相关Python依赖
2. **设备连接**: 检查传感器硬件连接
3. **权限设置**: 可能需要适当的串口访问权限
4. **数据备份**: logs目录包含重要实验数据，注意备份
5. **MQTT配置**: Riotee系统需要配置MQTT broker地址

## 🔗 相关文档

- [Riotee传感器详细说明](riotee_sensor/README.md)
- [使用示例](examples/)
- [数据格式说明](logs/)

---

**项目特点**: 
- ✅ 模块化设计，各传感器独立运行
- ✅ 统一的数据接口和日志系统  
- ✅ aioshelly风格的设备控制
- ✅ 完整的系统管理工具
- ✅ 符合Python包命名规范