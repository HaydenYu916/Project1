# CO2传感器数据采集器

## 文件说明

- `co2_simple.py` - **简化数据采集程序（推荐）**
- `co2_collector.py` - 完整数据采集程序
- `co2_system_manager.py` - 系统管理器
- `co2_watchdog.py` - 看门狗服务（自动重启）
- `co2_start.sh` - 简单启动脚本
- `co2_watchdog.service` - systemd服务文件
- `install_watchdog.sh` - 看门狗安装脚本
- `__init__.py` - 数据读取API

## 使用方法

### 简化版本（最推荐）
```bash
# 直接运行，只显示运行状态和串口数据
python3 co2_simple.py
```

### 系统管理器
```bash
# 启动采集器
python3 co2_system_manager.py start

# 查看状态
python3 co2_system_manager.py status

# 停止采集器
python3 co2_system_manager.py stop

# 重启采集器
python3 co2_system_manager.py restart

# 实时查看数据
python3 co2_system_manager.py live

# 启动看门狗服务（自动重启）
python3 co2_system_manager.py start-watchdog

# 停止看门狗服务
python3 co2_system_manager.py stop-watchdog

# 清理中间文件
python3 co2_system_manager.py clean

# 交互模式
python3 co2_system_manager.py
```

### 简单启动
```bash
./co2_start.sh
```

### 直接运行
```bash
python3 co2_collector.py
```

## 功能特点

- ✅ 读取串口数据：`/dev/Chamber2_Co2`
- ✅ 数据保存到：`../logs/co2_data.csv`
- ✅ MQTT上传到Home Assistant
- ✅ 设备名称：`Chamber2_Room`
- ✅ 只显示运行状态和串口数据
- ✅ 无中间文件生成
- ✅ 系统管理：启动、停止、状态监控
- ✅ 自动清理：停止时自动清理临时文件

## 输出文件

- `../logs/co2_data.csv` - CSV数据文件（主要输出）
- 控制台输出 - 运行状态和串口数据

## 简化版本特点

- ✅ 只显示运行状态和串口数据
- ✅ 无中间文件生成（无*.pid, *.log, *.txt, *.json）
- ✅ 直接运行，无需管理
- ✅ 数据保存到CSV文件
- ✅ MQTT上传到Home Assistant

## 停止程序

使用系统管理器：
```bash
python3 co2_system_manager.py stop
```

或按 `Ctrl+C` 停止程序

## 文件说明

- `co2_simple.py` - **简化数据采集程序（推荐）**
- `co2_collector.py` - 完整数据采集程序
- `co2_system_manager.py` - 系统管理器
- `co2_watchdog.py` - 看门狗服务（自动重启）
- `co2_start.sh` - 简单启动脚本
- `co2_watchdog.service` - systemd服务文件
- `install_watchdog.sh` - 看门狗安装脚本
- `__init__.py` - 数据读取API

## 使用方法

### 简化版本（最推荐）
```bash
# 直接运行，只显示运行状态和串口数据
python3 co2_simple.py
```

### 系统管理器
```bash
# 启动采集器
python3 co2_system_manager.py start

# 查看状态
python3 co2_system_manager.py status

# 停止采集器
python3 co2_system_manager.py stop

# 重启采集器
python3 co2_system_manager.py restart

# 实时查看数据
python3 co2_system_manager.py live

# 启动看门狗服务（自动重启）
python3 co2_system_manager.py start-watchdog

# 停止看门狗服务
python3 co2_system_manager.py stop-watchdog

# 清理中间文件
python3 co2_system_manager.py clean

# 交互模式
python3 co2_system_manager.py
```

### 简单启动
```bash
./co2_start.sh
```

### 直接运行
```bash
python3 co2_collector.py
```

## 功能特点

- ✅ 读取串口数据：`/dev/Chamber2_Co2`
- ✅ 数据保存到：`../logs/co2_data.csv`
- ✅ MQTT上传到Home Assistant
- ✅ 设备名称：`Chamber2_Room`
- ✅ 只显示运行状态和串口数据
- ✅ 无中间文件生成
- ✅ 系统管理：启动、停止、状态监控
- ✅ 自动清理：停止时自动清理临时文件

## 输出文件

- `../logs/co2_data.csv` - CSV数据文件（主要输出）
- 控制台输出 - 运行状态和串口数据

## 简化版本特点

- ✅ 只显示运行状态和串口数据
- ✅ 无中间文件生成（无*.pid, *.log, *.txt, *.json）
- ✅ 直接运行，无需管理
- ✅ 数据保存到CSV文件
- ✅ MQTT上传到Home Assistant

## 停止程序

使用系统管理器：
```bash
python3 co2_system_manager.py stop
```

或按 `Ctrl+C` 停止程序