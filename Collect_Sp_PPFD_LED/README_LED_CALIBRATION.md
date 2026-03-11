# LED Calibration Data Collection

本 README 只覆盖数据采集流程（暂不包含建模）。

## 1. 目标

在每个 LED 条件段（segment）中，同时采集：

1. Riotee 光谱传感器时序数据（BLE 上报，保留原始记录，不按 BLE 接收时间对齐）
2. 光谱仪 PPFD 标签（每段触发 3 次）
3. 主控段信息表（`segment_table.csv`）

关键对齐策略：
- 使用光学 marker（`OFF -> BLUE ONLY`）做段边界检测。
- 后处理按传感器光学通道（如 `sp_clear/sp_445`）分段，不按 BLE 接收时间分段。

## 2. 文件位置

新增主控脚本：
- `/home/hao/Desktop/Growpro/Collect_Sp_PPFD_LED/run_led_calibration.py`

## 3. 依赖与前置

1. Shelly 红蓝灯 IP 已在工作区根目录的 `Tool/LED_Shelly/config/device_config.py` 配置。
2. 光谱仪串口可用（默认 `/dev/ttyACM0`）。
3. Python 环境可导入：
   - `requests`
   - `pyserial`
   - Riotee 相关依赖（用于 `riotee_data_collector.py`）
4. Riotee 设备已上线。

## 4. 默认实验逻辑

默认 `Ts = 10s` 时：

1. Marker:
   - `T_off = 2*Ts = 20s`（LED OFF）
   - `T_blue = 2*Ts = 20s`（BLUE ONLY，`pwm_b=100,pwm_r=0`）
2. 目标条件：
   - 切到目标 PWM 后 `T_settle = max(10,2*Ts) = 20s`
3. 测量窗口：
   - `T_meas = 10*Ts = 100s`
   - 在窗口内均匀触发 `N_spec=3` 次光谱仪
4. 每段总时长：
   - `T_seg = T_off + T_blue + T_settle + T_meas`

默认网格：
- 比例 `ratios = 1:0,4:1,2:1,1:1,1:2,1:4,0:1`
- 总强度 `totals = 20,40,60,80`
- `pwm_r = round(T * r/(r+b))`
- `pwm_b = round(T * b/(r+b))`

## 5. 快速开始

在根目录执行：

```bash
cd /home/hao/Desktop/Growpro/Collect_Sp_PPFD_LED
python3 run_led_calibration.py
```

## 6. 常用命令

1. 指定 Ts 与串口：

```bash
python3 run_led_calibration.py --ts 10 --spec-port /dev/ttyACM0
```

2. 自定义网格：

```bash
python3 run_led_calibration.py \
  --ratios "1:0,2:1,1:1,1:2,0:1" \
  --totals "10,20,40,60,80,100"
```

3. 每 8 段插入漂移参考段（60,60）：

```bash
python3 run_led_calibration.py --drift-every 8 --drift-pwm-r 60 --drift-pwm-b 60
```

4. 不自动启动 Riotee 采集器（外部已启动）：

```bash
python3 run_led_calibration.py --sensor-mode external
```

5. 完全关闭传感器采集（仅调灯+光谱仪）：

```bash
python3 run_led_calibration.py --sensor-mode off
```

6. 只为“跑通流程”的快速模式（不改正式默认）：

```bash
python3 run_led_calibration.py \
  --quick-run \
  --ratios "1:1" \
  --totals "20" \
  --n-spec 1
```

7. 如果你明确需要由主控下发 sleep_time（默认不下发）：

```bash
python3 run_led_calibration.py \
  --set-sensor-sleep \
  --sensor-sleep-retries 12 \
  --sensor-sleep-retry-sec 5
```

8. 手动缩放时序（例如所有阶段乘 0.2）：

```bash
python3 run_led_calibration.py --time-scale 0.2
```

9. 光谱仪串口不稳定时增加重试（掉线自动重连）：

```bash
python3 run_led_calibration.py \
  --spec-open-retries 20 \
  --spec-open-retry-sec 2 \
  --spec-meas-retries 5 \
  --spec-meas-retry-sec 1
```

10. LED 设备偶发不可达时，默认会持续等待重试（不退出）：

```bash
python3 run_led_calibration.py \
  --led-failure-policy wait \
  --led-retry-sec 5 \
  --led-status-timeout-sec 8 \
  --led-status-poll-sec 0.2
```

若你希望 LED 失败立即退出：

```bash
python3 run_led_calibration.py --led-failure-policy abort
```

## 7. 输出目录结构

每次运行会在 `outputs/<run_id>/` 生成：

1. `run_config.json`
2. `segment_table.csv`
3. `run_log.txt`
4. `sensor_collector.log`（Riotee 采集器 stdout/stderr）
5. `sensor_timeseries.csv`（当 `sensor-mode=auto` 且 Riotee 采集正常时）
6. `logs/sensor_timeseries_all.csv`、`logs/sensor_timeseries_summary.csv`（Riotee 原始输出）
7. `archive/<YYYY-MM-DD>/standard_csv/raw_bytes_*.csv`（光谱仪标准CSV，按本次run目录保存）

示例：

```text
Collect_Sp_PPFD_LED/
  outputs/
    20260304_213000/
      run_config.json
      run_log.txt
      sensor_collector.log
      segment_table.csv
      sensor_timeseries.csv
      logs/
        sensor_timeseries_all.csv
        sensor_timeseries_summary.csv
      archive/
        2026-03-06/
          standard_csv/
            raw_bytes_20260306_150554.csv
```

## 8. `segment_table.csv` 关键字段

1. 段信息：
   - `segment_id, segment_type, ratio_r, ratio_b, pwm_r, pwm_b, total_pwm`
2. 主控时间戳：
   - `segment_start_master_time`
   - `marker_off_request_master_time`
   - `marker_off_start_master_time`
   - `marker_blue_request_master_time`
   - `marker_blue_start_master_time`
   - `marker_end_master_time`
   - `target_pwm_request_master_time`
   - `target_pwm_applied_master_time`
   - `steady_start_master_time`
   - `steady_end_master_time`
   - `segment_end_master_time`
3. 时序参数：
   - `Ts, T_off, T_blue, T_settle, T_meas, N_spec`
4. 光谱仪标签：
   - `ppfd_1..ppfd_N`
   - `ppfd_1_time..ppfd_N_time`
   - `spec_file_1..spec_file_N`
   - `PPFD_spec_mean`

## 9. 注意事项

1. `sensor_timeseries.csv` 不做 BLE 收包时间对齐，仅保留原始记录。
2. 后续分段应基于 marker 的光学形态（`sp_clear/sp_445`）。
3. 运行中断（Ctrl+C）后脚本会尝试：
   - 关闭 LED（设为 0/0）
   - 停止 Riotee 采集器
4. 默认不下发 sensor sleep_time（假设你已在设备端固定配置）。
5. `sensor-mode=auto` 时，如果暂时没有 sensor 数据，脚本会一直等待到第一条数据出现后再开始实验段。
6. 若启动后长时间没数据（默认 1 小时），脚本会自动把红蓝灯设为 `50/50` 给 sensor 充电，再持续等待数据；通常对应首次运行或间隔很久后的启动。可用 `--sensor-charge-after-sec` 调整阈值。
7. 光谱仪触发若发生串口 I/O 错误，脚本会自动重连并重试；若仍失败，该次 `ppfd` 记空值，实验继续执行后续段。
8. 默认 `--led-failure-policy wait`：若 Shelly RPC 超时，脚本会持续等待并重试，不会直接中断整次实验。
9. 每次运行都会对光谱仪执行会话初始化（`8C 00` + 自动积分模式）。
10. 从 2026-03-06 起，主控会在每次 `Light.Set` 后轮询 `Shelly.GetStatus`，确认 `brightness` 到达目标值后再进入该阶段计时；`segment_table.csv` 同时保留 request/applied 两套时间，便于分析 LED 或网络延迟。

## 10. 下一步（建模前）

当前阶段建议先做两件事：

1. 验证 `segment_table.csv` 的段时序是否符合预期（特别是 marker 与测量窗）。
2. 用 `sensor_timeseries.csv` 的 `sp_clear/sp_445` 做 marker 检测，确认 segment_id 重建稳定。

## 11. 生成 `condition_summary.csv`

默认在每次运行结束后，脚本会自动调用 `build_condition_summary.py` 生成 `condition_summary.csv`。

若你需要手动重建，可执行：

```bash
python3 build_condition_summary.py \
  --run-dir /home/hao/Desktop/Growpro/Collect_Sp_PPFD_LED/outputs/<run_id>
```

输出文件：
- `outputs/<run_id>/condition_summary.csv`

注意：
- 如果你用了 `--quick-run` 或很小的 `--time-scale`，而 sensor `sleep_time` 比 marker 段更长，marker 检测会明显变差（段数匹配不上）。这属于“跑通模式”的预期现象，不建议用于正式建模数据。
- 当 marker 检测不到足够段时，脚本会自动按 `segment_table.csv` 的主控时间窗兜底，保证每个 `segment_id` 仍有一行 summary（字段 `window_source=master_time_fallback`）。

如需关闭自动生成：

```bash
python3 run_led_calibration.py --skip-condition-summary
```
