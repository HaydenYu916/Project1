# 两高度对比扫描计划

## 目标

对当前 `3红1蓝、水平布灯` 的 LED，在两个候选高度上做空间扫描，并在每个空间点位上继续扫描 `30` 个控制点，建立：

1. `PPFD` 与 `PWM_r / PWM_b` 的关系
2. 不同高度下的空间均匀性
3. 红蓝光谱比例与控制输入的对应关系

这里要特别区分：

1. `3红1蓝` 是灯具结构
2. `--ratios` 是控制分配比例（`PWM_r : PWM_b`）
3. 真实输出比例要以后续测得的 `PPFD-R / PPFD-B` 或 sensor 光谱比例为准

## 高度定义

- `高度1`
- `高度2`

实际数值先不写，现场确定后再填。

## 点位定义

每个高度固定测 `5` 个点：

| 点位 | 位置 | 说明 |
|---|---|---|
| `C` | 中心 | 灯板正下方中心 |
| `L` | 左 | 与中心等距 |
| `R` | 右 | 与中心等距 |
| `U` | 上 | 与中心等距 |
| `D` | 下 | 与中心等距 |

建议中心到四个方向点的距离：

- `50 mm` 或 `75 mm`

## 每个点位的扫描策略

每个空间点位都不只测一个条件，而是扫描 `30` 个控制点。

采用现有脚本最容易执行的方式：

1. `5` 个控制比例
2. `6` 个总 PWM
3. 共 `5 x 6 = 30` 个 segment

## 硬件限制

由于当前 LED 硬件限制：

- 当某一路 `0 < PWM < 4` 时，该路虽然有控制占空比，但 LED 实际不会亮

因此需要区分：

1. `PWM = 0`：该路本来就是关闭，属于合法状态
2. `PWM = 1,2,3`：该路理论非零，但硬件实际上不亮，属于无效出光点

因此本计划需要保证每个被执行的组合里，非零通道的 `PWM >= 4`。

### 控制比例列表

以下比例都表示 `PWM_r : PWM_b`，不是灯珠数量比例。

这组比例按你的实际使用区间调整为“偏红分布更开”：

| 序号 | 控制比例 | 含义 |
|---|---|---|
| 1 | `1:1` | 红蓝各半 |
| 2 | `3:1` | 偏红 |
| 3 | `5:1` | 更偏红 |
| 4 | `7:1` | 强偏红 |
| 5 | `1:0` | 纯红边界 |

### 总 PWM 列表

```text
10,30,45,60,80,100
```

## 单点位测试条件

每个空间点位固定执行：

- `ratios = 1:1,3:1,5:1,7:1,1:0`
- `totals = 10,30,45,60,80,100`
- `N_spec = 1`
- `time-scale = 0.5`

理论上这样每个点位得到 `30` 个测试点。

但加入 `total=10` 之后，会新增 `3` 个硬件无效组合：

| 控制比例 | total_pwm | 计算后的 PWM | 无效原因 |
|---|---|---|---|
| `3:1` | `10` | `pwm_r=8, pwm_b=2` | 蓝灯 `PWM=2`，实际不亮 |
| `5:1` | `10` | `pwm_r=8, pwm_b=2` | 蓝灯 `PWM=2`，实际不亮 |
| `7:1` | `10` | `pwm_r=9, pwm_b=1` | 蓝灯 `PWM=1`，实际不亮 |

因此更准确地说：

- `30` 个理论测试点
- `27` 个有效测试点
- `3` 个无效测试点

## 执行顺序

### 高度1

| 顺序 | 点位 | run-name |
|---|---|---|
| 1 | `C` | `h1_pC` |
| 2 | `L` | `h1_pL` |
| 3 | `R` | `h1_pR` |
| 4 | `U` | `h1_pU` |
| 5 | `D` | `h1_pD` |

### 高度2

| 顺序 | 点位 | run-name |
|---|---|---|
| 6 | `C` | `h2_pC` |
| 7 | `L` | `h2_pL` |
| 8 | `R` | `h2_pR` |
| 9 | `U` | `h2_pU` |
| 10 | `D` | `h2_pD` |

## 启动命令

### 第一个点位启动命令

第一个点位保留 `1` 分钟预热，且采用缩短版时序：

```bash
cd /home/hao/Desktop/Growpro/Collect_Sp_PPFD_LED && python3 run_led_calibration.py \
  --spec-port /dev/ttyACM1 \
  --sensor-mode auto \
  --time-scale 0.5 \
  --ratios "1:1,3:1,5:1,7:1,1:0" \
  --totals "10,30,45,60,80,100" \
  --n-spec 3 \
  --prewarm-min 1 \
  --run-name h1_pL-0310
```

### 后续点位启动命令

后续点位关闭预热：

```bash
cd /home/hao/Desktop/Growpro/Collect_Sp_PPFD_LED && python3 run_led_calibration.py \
  --spec-port /dev/ttyACM1 \
  --sensor-mode auto \
  --time-scale 0.5 \
  --ratios "1:1,3:1,5:1,7:1,1:0" \
  --totals "10,30,45,60,80,100" \
  --n-spec 3 \
  --prewarm-min 0 \
  --run-name h1_pL+y-0311
```

例如：

```bash
cd /home/hao/Desktop/Growpro/Collect_Sp_PPFD_LED && python3 run_led_calibration.py \
  --spec-port /dev/ttyACM1 \
  --sensor-mode auto \
  --time-scale 0.5 \
  --ratios "1:1,3:1,5:1,7:1,1:0" \
  --totals "10,30,45,60,80,100" \
  --n-spec 1 \
  --prewarm-min 0 \
  --run-name h2_pL
```

```bash
cd /home/hao/Desktop/Growpro/Collect_Sp_PPFD_LED && python3 run_led_calibration.py \
  --spec-port /dev/ttyACM1 \
  --sensor-mode auto \
  --time-scale 0.5 \
  --ratios "1:1,3:1,5:1,7:1,1:0" \
  --totals "10,30,45,60,80,100" \
  --n-spec 1 \
  --prewarm-min 0 \
  --run-name h3_pR
```

## 工作量

每个点位：

- `30` 个理论 segment
- `27` 个有效 segment
- `3` 个无效 segment

每个高度：

- `5` 个点位
- 共 `150` 个理论 segment
- 共 `135` 个有效 segment

两个高度合计：

- `10` 个点位
- 共 `300` 个理论 segment
- 共 `270` 个有效 segment

## 每个点位建议记录的输出

每个点位最终建议整理出一张表，至少保留以下字段：

| 字段 | 含义 |
|---|---|
| `height` | `高度1` / `高度2` |
| `point_id` | `C/L/R/U/D` |
| `run_name` | 运行名称 |
| `pwm_r` | 红灯 PWM |
| `pwm_b` | 蓝灯 PWM |
| `total_pwm` | 总 PWM |
| `PPFD_spec_mean` | 总 PPFD |
| `PPFD-B` | 蓝光 PPFD |
| `PPFD-G` | 绿光 PPFD |
| `PPFD-R` | 红光 PPFD |
| `PPFD-FR` | 远红 PPFD |
| `PPFD-IR` | 红外 PPFD |
| `sp_415_mean` | 415nm |
| `sp_445_mean` | 445nm |
| `sp_480_mean` | 480nm |
| `sp_515_mean` | 515nm |
| `sp_555_mean` | 555nm |
| `sp_590_mean` | 590nm |
| `sp_630_mean` | 630nm |
| `sp_680_mean` | 680nm |
| `sp_clear_mean` | 全光谱通道 |
| `sp_nir_mean` | 近红外通道 |
| `is_valid_hw_point` | 是否满足硬件最小点亮限制 |
| `invalid_reason` | 若无效，记录具体原因 |

## 建议计算的比例指标

| 指标 | 公式 |
|---|---|
| `Blue_share` | `PPFD-B / PPFD_spec_mean` |
| `Green_share` | `PPFD-G / PPFD_spec_mean` |
| `Red_share` | `PPFD-R / PPFD_spec_mean` |
| `FR_share` | `PPFD-FR / PPFD_spec_mean` |
| `RB_ppfd_ratio` | `PPFD-R / PPFD-B` |
| `RB_sensor_ratio` | `(sp_630 + sp_680) / (sp_445 + sp_480)` |
| `Blue_sensor_share` | `(sp_445 + sp_480) / sp_clear` |
| `Red_sensor_share` | `(sp_630 + sp_680) / sp_clear` |
| `Green_sensor_share` | `(sp_515 + sp_555 + sp_590) / sp_clear` |

## 最终判断方式

最终比较 `高度1` 和 `高度2` 时，建议看两层：

### 第一层：空间表现

对同一个 PWM 条件，例如：

- `1:1 @ total=50`
- `2:1 @ total=80`
- `0:1 @ total=100`

比较 5 个点之间的：

1. `PPFD` 平均值
2. 最低点 / 平均值
3. 红蓝比例是否稳定

### 第二层：控制映射能力

比较两个高度下：

1. `PWM -> PPFD` 是否更平滑
2. `PWM -> 红蓝输出比例` 是否更单调、更容易选工作点
3. 哪个高度更容易找到“目标 PPFD + 目标谱比”的组合

当前这版计划默认就是为了先看“接近线性映射”和“可控性”，所以优先保留：

1. `1:1`
2. `3:1`
3. `5:1`
4. `7:1`
5. `1:0`

后续如果你确认蓝侧也需要补，再单独加 `1:3` 或 `0:1`。

优先选择那个：

1. 空间均匀性更好
2. 红蓝输出比例更稳定
3. 控制到目标工作点更容易的高度
