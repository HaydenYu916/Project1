# Riotee Sensor To PPFD Dataset

## 目标

这份文档只回答一件事：

如何把 `Riotee sensor` 的数据整理成一个可用于建模的数据表，使模型学习：

`Riotee 光谱特征 + 空间位置 -> PPFD`

这里的 `PPFD` 标签来自光谱仪，不是 Riotee 自己算出来的值。

## 一句话结论

如果你的目标是做 `Riotee sensor -> PPFD`，主输入表应优先使用每个 run 下的 `condition_summary.csv`，而不是直接从 `sensor_timeseries.csv` 生造训练样本。

原因：

1. `condition_summary.csv` 已经按 segment 对齐好了 sensor 窗口和光谱仪 `PPFD_spec_mean`
2. `sensor_timeseries.csv` 只是原始时间序列，前处理成本高，而且 BLE 时间本身不适合作为直接分段依据
3. 位置是你自己补充的外部字段，应在 `condition_summary.csv` 这一层合并

## 每个 run 下哪些文件有用

每个 run 一般在：

`outputs/<run_id>/`

常见文件作用如下：

| 文件 | 用途 | 是否进入训练主表 |
|---|---|---|
| `run_config.json` | 判断 run 参数是否属于同一实验计划 | 间接使用 |
| `segment_table.csv` | 校验每个 segment 的主控时序和光谱仪标签 | 间接使用 |
| `condition_summary.csv` | 每个 segment 一行，已汇总 sensor 特征和 `PPFD_spec_mean` | 是，主表来源 |
| `sensor_timeseries.csv` | 原始 Riotee 时序数据 | 否，主要做追溯 |
| `logs/sensor_timeseries_all.csv` | 更原始的 Riotee 输出 | 否，主要做追溯 |
| `archive/.../raw_bytes_*.csv` | 光谱仪原始文件 | 否，主要做追溯 |

## 推荐的数据粒度

推荐用 `segment` 作为一条样本。

也就是：

- 一种空间位置
- 一种 LED 条件
- 一组 Riotee 汇总光谱特征
- 一个光谱仪 PPFD 标签

对应关系为：

`1 row = 1 segment = 1 training sample`

## 位置数据怎么补

位置不是当前脚本自动生成的，需要你自己准备一个位置映射表，再并到训练表。

建议至少准备这些字段：

| 字段 | 含义 |
|---|---|
| `run_name` | 如 `h1_pC` |
| `height_id` | 如 `h1` |
| `point_id` | 如 `C` / `L` / `R` / `U` / `D` |
| `x_mm` | 横向坐标 |
| `y_mm` | 纵向坐标 |
| `z_mm` | 高度坐标 |

示例：

```csv
run_name,height_id,point_id,x_mm,y_mm,z_mm
h1_pC,h1,C,0,0,300
h1_pL,h1,L,-75,0,300
h1_pR,h1,R,75,0,300
h2_pC,h2,C,0,0,400
```

如果现场暂时没有绝对坐标，也至少要保留：

- `height_id`
- `point_id`

## 建模时推荐保留哪些列

推荐把 `condition_summary.csv` 中这些列作为候选输入：

### 目标列

- `PPFD_spec_mean`

### Riotee 光谱特征

- `sp_415_mean`
- `sp_445_mean`
- `sp_480_mean`
- `sp_515_mean`
- `sp_555_mean`
- `sp_590_mean`
- `sp_630_mean`
- `sp_680_mean`
- `sp_clear_mean`
- `sp_nir_mean`

### 可选派生特征

- `sum_vis`
- `norm_445`
- `norm_480`
- `norm_630`
- `norm_680`
- `rb_sensor`
- `temperature_mean`
- `gain_mode`

### 控制条件

- `pwm_r`
- `pwm_b`
- `total_pwm`
- `rb_ratio_pwm`

### 位置特征

- `height_id`
- `point_id`
- `x_mm`
- `y_mm`
- `z_mm`

### 追溯字段

- `run_id`
- `run_name`
- `segment_id`

## 不建议直接拿来当主训练样本的文件

### `sensor_timeseries.csv`

这个文件不建议直接拿来做训练主表，原因是：

1. 一条 BLE 上报不等于一个稳定光照条件
2. 文件前面可能有 `# Start ...` 注释行
3. 它没有直接绑定到光谱仪的 `PPFD_spec_mean`
4. 脚本文档已经说明，这个文件不按 BLE 接收时间做段对齐

所以：

- `sensor_timeseries.csv` 适合追查异常
- `condition_summary.csv` 才适合做建模主输入

## 清洗流程

### 1. 先筛 run

只保留“完整且属于同一实验计划”的 run。

一个 run 至少应满足：

1. 有 `run_config.json`
2. 有 `segment_table.csv`
3. 有 `condition_summary.csv`
4. `segment_table.csv` 行数等于 `run_config.json` 里的 `segment_count`
5. `condition_summary.csv` 行数等于 `segment_count`

如果任一条件不满足，这个 run 应先排除。

### 2. 再筛实验计划

如果你要做“当前高度比较计划”的统一模型，建议只保留同一配置：

- `segment_count = 30`
- `n_spec = 1`
- `time_scale = 0.5`
- `ratios = 1:1,3:1,5:1,7:1,1:0`
- `totals = 10,30,45,60,80,100`

不要把旧的 `60` 段 run 和现在的 `30` 段 run 混在一起。

### 3. 处理重复点位

如果同一个逻辑点位有多个成功 run，例如同一个 `run_name` 出现多次：

优先规则建议为：

1. 保留配置匹配当前计划的 run
2. 保留完整 run
3. 保留时间最新的 run

旧 run、半截 run、配置不一致 run 只保留作追溯，不进入训练集。

### 4. 按行筛样本

从每个合格 run 的 `condition_summary.csv` 中筛掉明显不可靠的 segment。

建议最小过滤规则如下：

1. `PPFD_spec_mean` 为空或 `<= 0` 的行剔除
2. `sensor_window_count <= 0` 的行剔除
3. `window_source != marker` 的行先剔除
4. 关键光谱列为空的行剔除

更严格的版本可加：

1. `sensor_window_count < 5` 的行剔除
2. `gain_mode` 明显异常的行剔除
3. 温度异常漂移的行单独标记

说明：

- `window_source = marker` 说明 sensor 窗口是通过光学 marker 匹配出来的
- `window_source = master_time_fallback` 说明 marker 没匹配稳，只是用主控时间窗兜底
- 第一版模型建议只用 `marker`

### 5. 合并位置

把你自己维护的位置表按 `run_name` 合并进来。

合并后，每一行除了光谱特征和 PPFD，还应有：

- `height_id`
- `point_id`
- `x_mm`
- `y_mm`
- `z_mm`

### 6. 保留追溯信息

即使做模型，也建议保留下面这些字段，方便回查：

- `run_id`
- `run_name`
- `segment_id`
- `window_source`
- `sensor_window_count`
- `pwm_r`
- `pwm_b`
- `total_pwm`

## 当前项目里最适合做模型的表

建议最终整理出一张长表，命名可以类似：

`riotee_ppfd_training_table.csv`

推荐结构如下：

| 列名 | 来源 | 说明 |
|---|---|---|
| `run_id` | 目录名 | 如 `20260308_174608_h3_pR` |
| `run_name` | `run_config.json` 或目录后缀 | 如 `h3_pR` |
| `height_id` | 位置表 | 如 `h3` |
| `point_id` | 位置表 | 如 `R` |
| `x_mm` | 位置表 | 位置坐标 |
| `y_mm` | 位置表 | 位置坐标 |
| `z_mm` | 位置表 | 位置坐标 |
| `segment_id` | `condition_summary.csv` | 每个条件一行 |
| `pwm_r` | `condition_summary.csv` | 控制输入 |
| `pwm_b` | `condition_summary.csv` | 控制输入 |
| `total_pwm` | `condition_summary.csv` | 控制输入 |
| `rb_ratio_pwm` | `condition_summary.csv` | 控制输入 |
| `sp_415_mean` 到 `sp_nir_mean` | `condition_summary.csv` | Riotee 光谱特征 |
| `sum_vis` | `condition_summary.csv` | 可选特征 |
| `rb_sensor` | `condition_summary.csv` | 可选特征 |
| `temperature_mean` | `condition_summary.csv` | 可选特征 |
| `gain_mode` | `condition_summary.csv` | 可选特征 |
| `sensor_window_count` | `condition_summary.csv` | 质量控制 |
| `window_source` | `condition_summary.csv` | 质量控制 |
| `PPFD_spec_mean` | `condition_summary.csv` | 训练目标 |

## 建议的排除规则

以下情况建议不要进第一版模型：

1. run 不完整
2. run 配置不属于当前目标计划
3. `window_source != marker`
4. `PPFD_spec_mean` 缺失
5. `sensor_window_count` 太低
6. 位置信息缺失

## 关于硬件无效点

当前高度比较计划里，存在一些理论控制输入合法、但硬件实际上不出光的组合：

- 若某一路满足 `0 < PWM < 4`，该路 LED 实际不会亮

这类点怎么处理，取决于你的建模目标：

### 如果模型目标是“真实系统行为”

可以保留。

因为模型最终学到的是：

`Riotee 观测到的真实光谱 + 位置 -> 光谱仪测得的真实 PPFD`

### 如果模型目标是“理想控制规律”

建议剔除这类点。

因为它们会把“控制命令”和“真实出光”之间的硬件非线性混进来。

## 第一版建模建议

如果你想先做一个稳的 baseline，建议这样做：

1. 只用完整 run
2. 只用当前主计划
3. 只用 `window_source = marker`
4. 只用有位置信息的样本
5. 目标列只用 `PPFD_spec_mean`
6. 特征先用：
   - `sp_415_mean` 到 `sp_nir_mean`
   - `x_mm, y_mm, z_mm`
   - 可选加 `temperature_mean`

先不要在第一版里混入：

- 原始 `sensor_timeseries.csv` 的逐点时间序列
- `master_time_fallback` 样本
- 不同实验计划的 run

## 最终结论

如果你的目标是：

`Riotee sensor 数据 -> 输出 PPFD`

推荐的清洗路径是：

1. 以 `condition_summary.csv` 为主
2. 先筛完整 run 和同计划 run
3. 再筛可靠 segment
4. 合并你自己提供的位置字段
5. 最终得到一张 `segment` 级别的训练表

不要直接拿 `sensor_timeseries.csv` 逐行建模，除非你准备自己重做 marker 分段和窗口汇总。
