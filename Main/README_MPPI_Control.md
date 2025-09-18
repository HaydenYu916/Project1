# MPPI LED控制循环系统

## 概述

这个系统集成了温度读取、MPPI控制和LED设备控制，实现自动化的植物光照控制。

## 文件说明

- `mppi_control_loop.py` - 主要的控制循环脚本
- `test_mppi_integration.py` - 集成测试脚本
- `quick_temp.py` - 快速温度显示脚本（原有）

## 系统架构

```
温度传感器数据 → MPPI控制器 → PWM命令 → LED设备
     ↓              ↓           ↓
  Riotee传感器   优化算法     Shelly设备
```

## 使用方法

### 1. 运行单次控制循环

```bash
cd /Users/z5540822/Desktop/untitled\ folder/Project1/Main
python mppi_control_loop.py once
```

### 2. 连续运行控制循环（每分钟）

```bash
python mppi_control_loop.py continuous
```

### 3. 自定义间隔时间

```bash
python mppi_control_loop.py continuous 2  # 每2分钟运行一次
```

### 4. 配置设备ID（宏定义方式）

所有脚本都支持通过修改代码顶部的宏定义来配置设备ID：

```python
# 在以下文件顶部都有相同的宏定义：
# - mppi_control_loop.py
# - test_mppi_integration.py  
# - quick_temp.py

TEMPERATURE_DEVICE_ID = "T6ncwg=="  # 指定设备ID
# TEMPERATURE_DEVICE_ID = None      # 自动选择设备
```

### 5. 快速温度查看

```bash
python quick_temp.py
```

### 6. 运行集成测试

```bash
python test_mppi_integration.py
```

## 系统配置

### MPPI控制器参数

- **预测时域**: 10步
- **采样数量**: 1000个
- **时间步长**: 0.1秒
- **红蓝比例**: 5:1
- **PWM范围**: 0-80
- **温度范围**: 20-29°C

### 设备配置

- **红光设备**: 192.168.50.94
- **蓝光设备**: 192.168.50.69

### 控制目标

- **目标温度**: 25°C
- **优化目标**: 最大化光合作用，最小化功率消耗

## 输出说明

### 控制循环输出

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
🔴 红光设备 (192.168.50.94): brightness=57
   命令: {
     "id": 0,
     "on": true,
     "brightness": 57,
     "transition": 1000
   }
🔵 蓝光设备 (192.168.50.69): brightness=11
   命令: {
     "id": 0,
     "on": true,
     "brightness": 11,
     "transition": 1000
   }
✅ 控制循环完成
```

### 数据新鲜度指示

- 🟢 绿色: 数据在2分钟内（新鲜）
- 🟡 黄色: 数据在2-5分钟内（较旧）
- 🔴 红色: 数据超过5分钟（过期）

## 故障排除

### 常见问题

1. **温度读取失败**
   - 检查Riotee传感器数据文件是否存在
   - 确认数据文件路径正确
   - 检查数据是否在有效时间范围内

2. **MPPI控制失败**
   - 检查模型文件是否正确加载
   - 确认温度值在合理范围内
   - 检查约束参数设置

3. **设备连接失败**
   - 检查网络连接
   - 确认设备IP地址正确
   - 检查设备是否在线

### 调试模式

运行测试脚本查看详细错误信息：

```bash
python test_mppi_integration.py
```

## 注意事项

1. **数据时效性**: 系统只使用5分钟内的温度数据
2. **安全限制**: PWM值被限制在0-80范围内
3. **温度保护**: 如果预测温度超过29°C，PWM值会自动降低
4. **命令发送**: 当前版本只打印命令，不实际发送到设备

## 扩展功能

### 启用实际设备控制

要启用实际的设备控制，需要修改 `mppi_control_loop.py` 中的 `send_pwm_commands` 方法，取消注释以下行：

```python
# response = rpc(red_ip, "Light.Set", red_cmd)
# response = rpc(blue_ip, "Light.Set", blue_cmd)
```

### 调整控制参数

修改代码顶部的宏定义来调整各种参数：

```python
# ==================== 配置宏定义 ====================
# 温度传感器设备ID配置
TEMPERATURE_DEVICE_ID = "T6ncwg=="  # None=自动选择

# 控制循环间隔（分钟）
CONTROL_INTERVAL_MINUTES = 2

# 目标温度（°C）
TARGET_TEMPERATURE = 26.0

# 红蓝比例键
RB_RATIO_KEY = "3:1"
# =====================================================
```

### 高级参数调整

如需调整MPPI控制器内部参数，可在 `MPPIControlLoop.__init__` 方法中修改：

```python
# 调整MPPI参数
self.controller = LEDMPCController(
    horizon=15,           # 增加预测时域
    num_samples=2000,     # 增加采样数量
    temperature=0.5,      # 降低温度参数
    # ...
)
```
