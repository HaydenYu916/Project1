# CO2传感器模块

Chamber2 CO2传感器数据采集和Home Assistant集成模块。

## 🎯 功能特性

- **实时数据采集**: 从串口读取CO2浓度数据
- **CSV数据记录**: 自动记录到统一logs目录
- **MQTT发布**: 实时数据发布到MQTT broker
- **Home Assistant集成**: 自动发现和配置

## 📁 文件说明

- **`co2_data_collector.py`** - 主数据采集器
- **`co2_system_manager.py`** - 系统管理器
- **`co2_example.py`** - 使用示例
- **`__init__.py`** - 模块接口

## 🚀 使用方法

### 1. 直接运行数据采集器
```bash
python3 co2_data_collector.py
```

### 2. 使用系统管理器
```bash
# 启动CO2系统
python3 co2_system_manager.py start

# 查看状态
python3 co2_system_manager.py status

# 停止系统
python3 co2_system_manager.py stop
```

## 📊 数据输出

### CSV文件
- **位置**: `../logs/co2_data.csv`
- **格式**: `timestamp,co2`
- **示例**:
```csv
timestamp,co2
2025-09-11 14:30:15,412.5
2025-09-11 14:30:16,413.2
```

### MQTT主题
- **状态**: `co2/chamber2_co2/status` (online/offline)
- **数据**: `co2/chamber2_co2/value` (CO2浓度值)

### Home Assistant
- **设备名称**: "Chamber2 CO2"
- **实体名称**: "Chamber2 CO2 Sensor"
- **设备类**: carbon_dioxide
- **单位**: ppm
- **图标**: mdi:molecule-co2

## ⚙️ 配置参数

### 硬件配置
```python
SERIAL_PORT = '/dev/Chamber2_Co2'  # CO2传感器串口
BAUDRATE = 115200                  # 波特率
```

### MQTT配置
```python
MQTT_CONFIG = {
    "broker": "azure.nocolor.pw",
    "port": 1883,
    "username": "feiyue",
    "password": "123456789",
    "device_name": "chamber2_co2",
}
```

## 📈 输出示例

### 控制台输出
```
2025-09-11 14:30:15, CO2: 412.5 ppm [已发布]
2025-09-11 14:30:16, CO2: 413.2 ppm [已发布]
```

### 日志输出
```
2025-09-11 14:30:10,123 - INFO - MQTT连接成功
2025-09-11 14:30:10,145 - INFO - HA自动发现配置已发布: chamber2_co2_co2
```

## 🔗 Home Assistant集成

启动数据采集器后，会自动在Home Assistant中创建：

1. **设备**: "Chamber2 CO2"
2. **传感器实体**: "Chamber2 CO2 Sensor"
3. **属性**:
   - 设备类: CO2浓度
   - 单位: ppm
   - 状态类: 测量值
   - 图标: CO2分子图标

## ⚠️ 注意事项

1. **串口权限**: 确保有串口访问权限
2. **MQTT配置**: 确保MQTT broker可访问
3. **网络连接**: Home Assistant集成需要网络连接
4. **文件权限**: 确保logs目录可写

## 🛠️ 故障排除

### 串口连接失败
```bash
# 检查设备是否存在
ls -la /dev/Chamber2_Co2

# 检查权限
sudo chmod 666 /dev/Chamber2_Co2
```

### MQTT连接失败
- 检查网络连接
- 验证broker地址和端口
- 确认用户名密码正确

### Home Assistant未显示
- 确认MQTT集成已启用
- 检查自动发现功能
- 查看HA日志了解详情

## 📋 依赖要求

```bash
pip install pyserial paho-mqtt
```
