# Basil光合速率预测器

简化版Basil光合速率预测工具

## 文件说明

- `predict.py` - 主预测脚本
- `lssvr_ga_trained_model.pkl` - 训练好的LSSVR模型
- `standard_scaler_lssvr_ppfd_t_co2_400_rb_083.pkl` - 数据标准化器

## 使用方法

1. 打开 `predict.py` 文件
2. 修改以下变量：
   ```python
   PPFD = 500          # 光量子密度 (umol/m2/s)
   TEMPERATURE = 25    # 温度 (°C)
   ```
3. 运行脚本：
   ```bash
   python predict.py
   ```

## 模型信息

- 模型条件: CO2=400ppm, R:B=0.83 (固定)
- 输入特征: PPFD + 温度
- 输出: 光合速率 (umol/m2/s)
- 算法: LSSVR with RBF kernel

## 建议输入范围

- PPFD: 0-1000 umol/m2/s
- 温度: 18-30 °C 