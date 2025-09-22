# PWM调度器使用说明

## 功能描述

PWM调度器根据CSV文件中的时间表，自动控制Shelly设备的红蓝PWM值。当真实时间到达预设时间点时，调度器会：

1. 设置对应的红蓝PWM值
2. 等待5秒后检查设备状态
3. 收集同一PPFD时间段内的riotee传感器数据
4. 将数据保存到logs目录的CSV文件中
5. 继续等待下一个时间点

**支持前台和后台运行模式，并自动收集传感器数据！**

## 文件说明

- `pwm_scheduler.py` - 主调度器脚本（支持后台运行和数据收集）
- `pwm_service.py` - 服务管理脚本（推荐使用）
- `test_pwm_scheduler.py` - 测试脚本
- `test_data_collection.py` - 数据收集功能测试脚本
- `extended_schedule_20250919_071157_20250919_071157.csv` - 时间表数据
- `pwm_scheduler.log` - 运行日志文件
- `pwm_scheduler.pid` - 进程ID文件
- `logs/` - 数据收集输出目录
  - `ppfd_XXX_PhaseName_YYYYMMDD_HHMMSS.csv` - PPFD时间段数据文件

## CSV文件格式

CSV文件包含以下列：
- `time` - 执行时间 (YYYY-MM-DD HH:MM:SS)
- `time_str` - 时间字符串显示
- `ppfd` - 光量子通量密度
- `r_pwm` - 红色PWM值 (0-100)
- `b_pwm` - 蓝色PWM值 (0-100)
- `interval` - 间隔编号
- `cycle` - 周期编号
- `phase` - 阶段标识
- `phase_name` - 阶段名称

## 设备配置

在 `shelly_controller.py` 中配置的设备：
- Red设备: 192.168.50.94
- Blue设备: 192.168.50.69

## 使用方法

### 方法一：使用服务管理脚本（推荐）

```bash
cd /home/pi/Desktop/aioshelly/shelly_src/Test

# 1. 测试连接和功能
python3 test_pwm_scheduler.py

# 2. 启动调度器（后台运行）
python3 pwm_service.py start

# 3. 检查运行状态
python3 pwm_service.py status

# 4. 查看日志
python3 pwm_service.py logs

# 5. 停止调度器
python3 pwm_service.py stop

# 6. 重启调度器
python3 pwm_service.py restart

# 7. 前台运行（调试用）
python3 pwm_service.py start --foreground
```

### 方法二：直接使用调度器脚本

```bash
cd /home/pi/Desktop/aioshelly/shelly_src/Test

# 后台运行
python3 pwm_scheduler.py src/extended_schedule_20250919_071157_20250919_071157.csv -d

# 前台运行
python3 pwm_scheduler.py src/extended_schedule_20250919_071157_20250919_071157.csv

# 停止调度器
python3 pwm_scheduler.py --stop

# 检查状态
python3 pwm_scheduler.py --status
```

## 运行流程

1. 调度器启动后加载CSV时间表
2. 找到下一个要执行的时间点
3. 等待到执行时间
4. 结束上一个PPFD时间段的数据收集（如果有）
5. 开始新的PPFD时间段数据收集
6. 执行PWM设置命令：
   - 设置红色设备PWM值
   - 设置蓝色设备PWM值
7. 等待5秒后检查设备状态
8. 继续下一个时间点

## 数据收集功能

调度器会自动收集每个PPFD时间段内的riotee传感器数据：

- **数据来源**: `/home/pi/Desktop/Test/riotee_sensor/logs/` 目录下的所有riotee数据文件
- **收集时机**: 每个PPFD时间段开始和结束时
- **输出位置**: `logs/` 目录
- **文件命名**: `ppfd_{PPFD值}_{阶段名}_{开始时间}.csv`
- **数据内容**: 包含温度、湿度、电压等传感器数据

## 输出示例

### 前台运行输出
```
已加载 33 个时间点
PWM调度器开始运行...

下一个执行时间: 2025-09-19 07:00:00 (Heating 1)
等待 1800.0 秒到 07:00:00

执行时间: 2025-09-19 07:00:00
阶段: Heating 1 (周期 1)
设置PWM值: Red=12, Blue=9
PWM设置成功
等待5秒后检查状态...
检查设备状态...
设备状态检查完成
红色设备: {'on': True, 'brightness': 12, ...}
蓝色设备: {'on': True, 'brightness': 9, ...}
```

### 后台运行状态检查
```bash
$ python3 pwm_service.py status
检查PWM调度器状态...
✓ 调度器状态:
调度器正在运行 (PID: 12345)

最近日志:
2025-09-19 07:00:00 - INFO - 执行时间: 2025-09-19 07:00:00
2025-09-19 07:00:00 - INFO - 阶段: Heating 1 (周期 1)
2025-09-19 07:00:00 - INFO - PPFD值: 100
2025-09-19 07:00:00 - INFO - 开始收集PPFD 100 (Heating 1) 的数据
2025-09-19 07:00:00 - INFO - 设置PWM值: Red=12, Blue=9
2025-09-19 07:00:00 - INFO - PWM设置成功
2025-09-19 07:00:05 - INFO - 等待5秒后检查状态...
2025-09-19 07:30:00 - INFO - 加载了 11 条riotee数据记录
2025-09-19 07:30:00 - INFO - PPFD 100 数据已保存到: src/logs/ppfd_100_Heating 1_20250919_070000.csv
```

### 生成的PPFD数据文件示例
```
# PPFD时间段数据收集
# PPFD值: 200
# 阶段: Heating 1
# 开始时间: 2025-09-19 07:00:00
# 结束时间: 2025-09-19 07:30:00
# 数据条数: 12

id,timestamp,device_id,update_type,temperature,humidity,a1_raw,vcap_raw,sp_415,sp_445,sp_480,sp_515,sp_555,sp_590,sp_630,sp_680,sp_clear,sp_nir,spectral_gain,sleep_time
1,2025-09-19 07:22:14.998,T6ncwg==,PACKET_DATA,21.42,61.87,1.102,3.656,59.0,602.0,280.0,48.0,1485.0,95.0,56.0,61.0,1003.0,1292.0,1,90
2,2025-09-19 07:22:23.406,L_6vSQ==,PACKET_DATA,20.84,60.72,1.014,3.769,57.0,541.0,270.0,44.0,1329.0,85.0,52.0,57.0,926.0,1341.0,1,90
...
```

**数据列说明：**
- **基础数据**: id, timestamp, device_id, update_type, temperature, humidity, a1_raw, vcap_raw
- **光谱数据**: sp_415, sp_445, sp_480, sp_515, sp_555, sp_590, sp_630, sp_680 (不同波长的光谱值)
- **特殊光谱**: sp_clear (全光谱), sp_nir (近红外)
- **系统数据**: spectral_gain, sleep_time

## 注意事项

1. 确保Shelly设备网络连接正常
2. 确保CSV文件中的时间格式正确
3. 调度器会根据当前时间自动找到下一个执行点
4. 如果所有时间点都已过，调度器会自动结束
5. 后台运行时可以通过服务管理脚本控制
6. 前台运行时可以通过Ctrl+C停止调度器

## 后台运行特性

- **守护进程**: 自动转换为后台守护进程
- **PID管理**: 自动管理进程ID文件
- **日志记录**: 所有操作记录到日志文件
- **信号处理**: 支持优雅停止
- **重复运行检测**: 防止多个实例同时运行

## 故障排除

1. **连接失败**: 检查设备IP地址和网络连接
2. **CSV加载失败**: 检查文件路径和格式
3. **PWM设置失败**: 检查设备状态和权限
4. **时间不匹配**: 检查系统时间和CSV时间格式
5. **后台启动失败**: 检查权限和PID文件
6. **日志文件过大**: 定期清理或轮转日志文件

## 常用命令速查

```bash
# 快速启动（后台）
python3 pwm_service.py start

# 检查状态
python3 pwm_service.py status

# 查看日志
python3 pwm_service.py logs

# 停止服务
python3 pwm_service.py stop

# 重启服务
python3 pwm_service.py restart
```
