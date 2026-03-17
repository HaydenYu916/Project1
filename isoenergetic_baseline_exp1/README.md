# Iso-Energetic Baseline Exp1

这个文件夹用于归档 `mppi_v2_control_log_exp1.csv` 的等能常光基线分析，包含：

- `isoenergetic_baseline_analysis.py`
  复现实验分析的脚本
- `mppi_v2_control_log_exp1.csv`
  输入控制日志
- `mppi_v2_control_log_exp1_isoenergetic_timeseries.csv`
  逐时刻重算后的结果
- `mppi_v2_control_log_exp1_isoenergetic_summary.json`
  摘要结果
- `缺点与局限.md`
  该比较方法的局限说明

## 当前口径

默认采用以下比较定义：

- `dt` 使用名义控制周期 `900 s`
- 常光基线按整个实验周期的平均 `PPFD` 构造，而不是按天单独重算
- `PPFD` 由当前仓库的 `Solar_Vol_clean.csv` 插值得到
- 基线 `solar_vol_const` 由同一条 `solar_vol <-> PPFD` 查找关系反求
- 默认主结果使用“segmented thermal replay”：
  每个连续开灯段都从该段起始 `input_temp` 出发，动态与常光各自连续回放到该段结束
- `CO2` 仍固定使用日志中的 `co2_ppm`
- 逐时刻 CSV 输出动态/常光的分段连续热回放温度、功率和 `Pn` 轨迹
- 连续 carry-over 的 `closed-loop replay` 仅保留为可选诊断
- 这是一种 `model-based predicted growth proxy` 比较，不是实测生物量

如果需要旧的“严格按日志 `pred_power` 对账”的做法，仍可使用：

```bash
python /home/hao/Desktop/Growpro/isoenergetic_baseline_exp1/isoenergetic_baseline_analysis.py \
  --log-path /home/hao/Desktop/Growpro/isoenergetic_baseline_exp1/mppi_v2_control_log_exp1.csv \
  --baseline-mode empirical_power
```

## 当前结果

来自 `mppi_v2_control_log_exp1_isoenergetic_summary.json`（当前默认 `baseline_mode=mean_ppfd`，主结果为 `segmented_thermal_replay`）：

- 动态平均 PPFD：`238.073369 umol m^-2 s^-1`
- 常光固定 PPFD：`238.073369 umol m^-2 s^-1`
- 常光固定设定：`solar_vol_const = 1.333437`
- 常光固定设定：`R_PWM = 44.591633`
- 常光固定设定：`B_PWM = 12.277490`
- 动态总能量：`16,748,695.054 J`
- 动态总能量：`4,652.415 Wh`
- 常光总能量：`16,542,259.186 J`
- 常光总能量：`4,595.072 Wh`
- 常光相对动态能量偏差：`-1.232549 %`
- 动态 segmented 累计预测光合代理：`4,722,583.331 umol m^-2`
- 常光 segmented 累计预测光合代理：`5,084,313.790 umol m^-2`
- 动态 segmented 单位能量预测光合：`1015.082067 umol m^-2 Wh^-1`
- 常光 segmented 单位能量预测光合：`1106.470975 umol m^-2 Wh^-1`
- 动态相对常光主结果变化：`-7.114637 %`
- 回放分段数：`9`
- 动态 segmented 平均预测温度：`32.649097 °C`
- 常光 segmented 平均预测温度：`30.137884 °C`
- 动态 segmented 最高预测温度：`38.764221 °C`
- 常光 segmented 最高预测温度：`35.313193 °C`

注意：修复 phase 重置累积并重标定 heating 快分量后，这组 segmented thermal replay 已明显稳定；但当前热模型单步仍有约 `+1.66 °C` 的正偏，因此温度绝对值仍不应直接当成物理真实值。

补充结果（固定日志温度轨迹的 exogenous replay）：

- 动态累计预测光合代理：`4,875,537.843 umol m^-2`
- 常光累计预测光合代理：`5,244,988.391 umol m^-2`
- 动态相对常光变化：`-7.043877 %`

如果需要连续 carry-over 的诊断性热回放：

- `--closed-loop`

## 使用方法

先进入 `Growpro` 环境：

```bash
source /home/hao/miniconda3/etc/profile.d/conda.sh
conda activate Growpro
```

在仓库根目录运行：

```bash
python /home/hao/Desktop/Growpro/isoenergetic_baseline_exp1/isoenergetic_baseline_analysis.py \
  --log-path /home/hao/Desktop/Growpro/isoenergetic_baseline_exp1/mppi_v2_control_log_exp1.csv
```

如果要重新输出文件：

```bash
python /home/hao/Desktop/Growpro/isoenergetic_baseline_exp1/isoenergetic_baseline_analysis.py \
  --log-path /home/hao/Desktop/Growpro/isoenergetic_baseline_exp1/mppi_v2_control_log_exp1.csv \
  --output-json /home/hao/Desktop/Growpro/isoenergetic_baseline_exp1/mppi_v2_control_log_exp1_isoenergetic_summary.json \
  --output-csv /home/hao/Desktop/Growpro/isoenergetic_baseline_exp1/mppi_v2_control_log_exp1_isoenergetic_timeseries.csv
```

如果只想看连续 carry-over 的诊断性热回放：

```bash
python /home/hao/Desktop/Growpro/isoenergetic_baseline_exp1/isoenergetic_baseline_analysis.py \
  --log-path /home/hao/Desktop/Growpro/isoenergetic_baseline_exp1/mppi_v2_control_log_exp1.csv \
  --closed-loop
```

## 主要公式

- 动态平均 PPFD：`PPFD_mean = sum(PPFD_k * dt_k) / sum(dt_k)`
- 常光基线：`PPFD_const = PPFD_mean`
- 反求常光设定：`u_const = g^{-1}(PPFD_const)`
- 动态分段热回放：`T_dyn,k+1 = h(T_dyn,k, u_k, CO2_k)`
- 基线分段热回放：`T_base,k+1 = h(T_base,k, u_const, CO2_k)`
- 每个开灯段初值：`T_dyn,0 = T_base,0 = T_log,start`
- 动态重算：`Pn_dyn,k = f(u_k, T_dyn,k, CO2_k)`
- 基线重算：`Pn_base,k = f(u_const, T_base,k, CO2_k)`
- 累计代理：`C = sum(Pn_k * dt_k)`
- 单位能量效率：`eta = C / E`

这里 `g` 是当前仓库中的 `solar_vol <-> PPFD` 查找关系，`h` 是当前仓库中的热模型，`f` 是当前仓库中的 `solar_vol` 光合作用预测模型。
