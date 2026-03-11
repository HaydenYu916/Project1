# Riotee Sensor - My Src 风格

这是基于 aioshelly/my_src 风格重构的 Riotee 传感器控制和监听工具。

## 文件结构

### 核心文件

- **`riotee_controller.py`** - 设备控制器（类似 aioshelly controller.py 风格）
- **`riotee_data_collector.py`** - 数据采集器（主要数据收集功能）
- **`riotee_system_manager.py`** - 系统管理器（启动/停止/状态管理）
- **`riotee_live_api.py`** - 实时API接口
- **`riotee_example.py`** - 使用示例

### 数据文件

- **`../logs/riotee_data_all.csv`** - 全量数据记录（所有数据包）
- **`../logs/riotee_data_summary.csv`** - 摘要数据记录（重要事件和状态变化）

## 使用方法

### 1. 设备控制器 (riotee_controller.py)

用于发送命令控制设备：

```bash
# 设置所有设备休眠时间为10秒
python3 riotee_controller.py all sleep 10

# 设置指定设备休眠时间为5秒
python3 riotee_controller.py ABC123 sleep 5

# 列出所有设备
python3 riotee_controller.py all list

# 显示系统状态
python3 riotee_controller.py all status
```

### 2. 数据采集器 (riotee_data_collector.py)

用于实时采集和记录设备数据：

```bash
# 开始数据采集（默认追加模式）
python3 riotee_data_collector.py

# 指定实验名称和备注
python3 riotee_data_collector.py experiment_name "实验备注信息"

# 新文件模式（创建带时间戳的新文件）
python3 riotee_data_collector.py --new-file experiment_name

# 停止采集
按 Ctrl+C
```

### 3. 系统管理器 (riotee_system_manager.py)

用于管理整个Riotee系统：

```bash
# 启动系统
python3 riotee_system_manager.py start

# 停止系统
python3 riotee_system_manager.py stop

# 查看状态
python3 riotee_system_manager.py status
```

## 特性

### 控制器特性
- ✅ 简单命令行接口
- ✅ 支持单设备和所有设备操作
- ✅ 自动启动 Gateway 服务器
- ✅ 类似 aioshelly controller 的使用方式

### 数据采集器特性
- ✅ 实时数据采集和记录
- ✅ 双 CSV 文件系统（全量 + 智能摘要）
- ✅ 会话管理（Start/Stop 标记）
- ✅ 实验备注支持
- ✅ MQTT数据发布
- ✅ Home Assistant自动发现
- ✅ 追加模式（默认）和新文件模式

## 文件说明

### riotee_controller.py
```python
# 基本用法
python3 riotee_controller.py <device> <command> [args...]

# 支持的命令
sleep <seconds>  # 设置休眠时间
list            # 列出设备
status          # 显示状态
```

### riotee_listener.py
```python
# 基本用法
python3 riotee_listener.py [备注信息]

# 数据记录到
../logs/riotee_log_all.csv    # 所有数据
../logs/riotee_log_event.csv  # 重要事件
```

## 与原 command_sender_binary.py 的对比

| 特性 | 原文件 | 新结构 |
|------|--------|--------|
| 文件数量 | 1个大文件 | 2个专门化文件 |
| 使用方式 | 交互式 | 命令行 + 后台监听 |
| 代码风格 | 单体应用 | aioshelly 风格 |
| 数据记录 | 无 | 双 CSV 系统 |
| 设备管理 | 内置 | 分离的控制器 |

## 兼容性

- ✅ 与现有 riotee_data_collector.py 兼容
- ✅ 与统一 logs/ 目录结构兼容
- ✅ 与现有 Gateway 系统兼容
- ✅ 支持所有原有功能

## 环境要求

1. **riotee_gateway 模块**: 已自动配置路径到 `/home/pi/Desktop/riotee-env/`
2. **riotee-gateway 命令**: 需要在 PATH 中或手动启动 Gateway 服务器
3. **Python 模块**: numpy (已包含在 riotee 环境中)

## 注意事项

1. 确保 riotee-gateway 服务可用
2. logs/ 目录会自动创建  
3. 监听器会自动处理 Gateway 启动
4. 使用 Ctrl+C 优雅停止监听器
5. 如果 riotee-gateway 命令不可用，需要手动启动 Gateway 服务器

## 故障排除

### 模块导入错误
```bash
# 错误: 无法导入riotee_gateway模块
# 解决: 文件会自动添加路径，如果仍有问题，请检查虚拟环境
```

### Gateway 服务器启动失败
```bash
# 错误: [Errno 2] No such file or directory: 'riotee-gateway'
# 解决: 手动启动 Gateway 或添加 riotee-gateway 到 PATH
```

### 设备未找到
```bash
# 错误: 未找到设备
# 解决: 确保设备已连接且 Gateway 正在运行
```
