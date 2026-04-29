# PWM发送确认与重试机制说明

## 概述

为 `mppi_control_real_v2.py` 添加了自动确认和重试机制，确保PWM命令成功发送到Shelly设备。

## 主要改进

### 1. 配置参数（第31-33行）

```python
STATUS_CHECK_DELAY = 5.0  # 从3.0秒增加到5.0秒，等待Shelly稳定
PWM_RETRY_MAX = 3  # PWM发送失败时的最大重试次数
PWM_TOLERANCE = 3  # PWM占空比验证容差（0-100范围内，允许±3的偏差）
```

- **STATUS_CHECK_DELAY**: 发送PWM后等待5秒再验证，确保设备完全稳定
- **PWM_RETRY_MAX**: 如果验证失败，最多重试3次
- **PWM_TOLERANCE**: 允许PWM占空比值与期望值有±3的偏差（0-100范围内）

### 2. 新增方法

#### `_verify_pwm(device_name, expected_brightness)` (第255-307行)

验证设备的PWM是否成功设置：

1. 调用 `Shelly.GetStatus` RPC方法获取设备状态
2. 提取 `light:0` 的 `brightness` 和 `output` 字段
3. 检查设备是否开启 (`output == True`)
4. 检查亮度是否在容差范围内 (`|actual - expected| <= 3`)

**返回**: `True` 验证成功，`False` 验证失败

#### `_send_pwm_single_device(device_name, brightness, retry_count)` (第309-367行)

发送单个设备的PWM命令并自动重试：

1. 发送 `Light.Set` RPC命令
2. 等待5秒让设备稳定
3. 调用 `_verify_pwm()` 验证
4. 如果验证失败且未达最大重试次数，递归重试
5. 返回详细的结果字典，包含：
   - `success`: 是否成功
   - `verified`: 是否通过验证
   - `retries`: 重试次数
   - `error`: 错误信息（如果有）

#### `_send_pwm(r_pwm, b_pwm)` (第369-415行)

主PWM发送方法，已重构为：

1. 分别调用 `_send_pwm_single_device()` 处理红光和蓝光设备
2. 记录详细的成功/失败日志
3. 在非后台模式下打印验证结果
4. 返回两个设备的完整状态

### 3. 日志增强（第497-520行）

在 `run_once()` 方法中更新了日志记录：

- 成功且有重试：记录 `red_retry:N` 或 `blue_retry:N`
- 失败：记录 `red_error:错误信息` 或 `blue_error:错误信息`
- 多个状态用 `|` 分隔

## 工作流程

```
发送PWM命令
    ↓
等待5秒 (STATUS_CHECK_DELAY)
    ↓
读取设备状态 (Shelly.GetStatus)
    ↓
验证PWM占空比是否匹配 (±3的容差，范围0-100)
    ↓
    ├─ 成功 → 返回
    └─ 失败 → 
         ↓
    重试次数 < 3?
         ↓
    ├─ 是 → 重新发送PWM (递归)
    └─ 否 → 返回失败
```

## 使用示例

### 前台运行（单次）
```bash
cd /home/pi/Desktop/LED_MPPI_Controller/applications/control
python3 mppi_control_real_v2.py once
```

输出示例：
```
📡 正在发送PWM命令: 红 65% / 蓝 35%
✅ Red 验证成功: 亮度 65
📝 [2025-10-22 14:30:15] ✅ 红光设备设置成功: 65%
✅ Blue 验证成功: 亮度 35
📝 [2025-10-22 14:30:21] ✅ 蓝光设备设置成功: 35%
```

如果需要重试：
```
📡 正在发送PWM命令: 红 65% / 蓝 35%
❌ Red 亮度不匹配: 期望 65, 实际 60 (差值 5)
📝 [2025-10-22 14:30:16] 🔄 Red 验证失败，正在重试 (1/3)...
✅ Red 验证成功: 亮度 65
📝 [2025-10-22 14:30:22] ✅ 红光设备设置成功: 65% (重试 1 次)
```

### 后台运行（连续控制）
```bash
python3 mppi_control_real_v2.py start
```

日志文件 `logs/mppi_v2_control_simple.log` 会记录所有重试和验证信息。

### 查看状态
```bash
python3 mppi_control_real_v2.py status
```

### 停止后台进程
```bash
python3 mppi_control_real_v2.py stop
```

## CSV日志格式

在 `logs/mppi_v2_control_log.csv` 的 `note` 列中会记录：

| note值 | 含义 |
|--------|------|
| `ok` | 两个设备都成功，无重试 |
| `red_retry:1` | 红光设备重试1次后成功 |
| `blue_retry:2` | 蓝光设备重试2次后成功 |
| `red_retry:1\|blue_retry:1` | 两个设备都重试1次后成功 |
| `red_error:达到最大重试次数 (3)` | 红光设备失败 |
| `blue_error:RPC错误: timeout` | 蓝光设备RPC超时 |

## 配置调整

如果需要调整重试行为，可以修改顶部常量：

```python
# 增加等待时间（如果网络延迟较高）
STATUS_CHECK_DELAY = 8.0  # 等待8秒

# 增加重试次数
PWM_RETRY_MAX = 5  # 最多重试5次

# 放宽验证容差
PWM_TOLERANCE = 5  # 允许±5的偏差（0-100范围内）
```

## 注意事项

1. **延迟增加**: 每次发送PWM现在需要至少5秒验证时间，如果有重试可能需要更长时间
2. **网络稳定性**: 确保树莓派与Shelly设备之间网络连接稳定
3. **日志监控**: 建议定期检查日志文件，关注频繁重试的情况
4. **容差设置**: PWM_TOLERANCE = 3（允许±3的偏差，0-100范围内）适用于大多数场景，如果设备响应不稳定可适当增加

## 故障排除

### 问题：设备总是验证失败
- 检查网络连接
- 增加 `STATUS_CHECK_DELAY` 到 8-10 秒
- 检查Shelly设备是否正常响应 `Shelly.GetStatus` 命令

### 问题：经常需要重试
- 可能是网络不稳定，考虑优化网络环境
- 可能是设备固件问题，考虑升级Shelly固件
- 检查设备负载是否过高

### 问题：重试次数不够
- 增加 `PWM_RETRY_MAX` 的值（但不建议超过5次）
- 如果经常达到最大重试仍失败，应该检查根本原因而非增加重试

## 兼容性

- 完全向后兼容之前的代码
- 日志格式保持不变，只是 `note` 字段增加了更多信息
- 不影响MPPI控制器的核心逻辑

