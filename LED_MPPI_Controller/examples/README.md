# LED控制系统 - 示例代码

本目录包含LED控制系统的示例代码和演示脚本，帮助用户理解和使用系统功能。

## 目录结构

```
examples/
├── README.md                    # 本文件
├── ppfd_demo.py                 # PWM-PPFD转换系统演示
└── result/                      # 输出结果目录
    ├── ppfd_model_fitting.png   # 模型拟合质量图
    ├── ppfd_comparison.png      # 不同标签对比图
    ├── ppfd_pwm_table_1_1.csv  # 1:1标签的PPFD-PWM对应表
    └── ppfd_pwm_table_5_1.csv   # 5:1标签的PPFD-PWM对应表
```

## 示例脚本

### 1. ppfd_demo.py - PWM-PPFD转换系统演示

**功能**: 演示LED控制系统中PWM-PPFD转换系统的完整功能

**主要特性**:
- 模型加载和拟合
- 前向预测功能（根据PPFD预测PWM）
- 反向求解功能（使用solve_pwm_for_target_ppfd函数）
- 模型拟合质量可视化
- 不同比例对比分析
- PPFD-PWM对应表导出

**使用方法**:
```bash
cd examples
python ppfd_demo.py
```

**输出文件**:
- `result/ppfd_model_fitting.png`: 模型拟合质量图
- `result/ppfd_comparison.png`: 不同标签对比图
- `result/ppfd_pwm_table_*.csv`: PPFD-PWM对应表

**演示内容**:

1. **模型加载和拟合**
   - 从标定数据CSV文件加载数据
   - 对每个标签分别拟合线性模型：
     - R_PWM = α × PPFD + β
     - B_PWM = γ × PPFD + δ
   - 显示拟合系数和R²值

2. **前向预测演示**
   - 测试不同PPFD值下的PWM预测
   - 显示R_PWM、B_PWM和Total_PWM值

3. **反向求解演示**
   - 使用solve_pwm_for_target_ppfd函数
   - 测试不同目标PPFD下的PWM求解

4. **可视化分析**
   - 绘制每个标签的拟合质量图
   - 对比不同标签的模型参数
   - 显示PPFD vs Total PWM的预测曲线

5. **数据导出**
   - 生成PPFD-PWM对应表
   - 支持0-600 PPFD范围的完整查找表

## 输出结果说明

### 图表文件

- **ppfd_model_fitting.png**: 显示每个标签的拟合质量
  - 左图：PPFD vs R_PWM，显示红光通道的拟合效果
  - 右图：PPFD vs B_PWM，显示蓝光通道的拟合效果
  - 包含数据点、拟合直线和R²值

- **ppfd_comparison.png**: 不同标签的对比分析
  - 左上：R_PWM斜率对比
  - 右上：B_PWM斜率对比
  - 左下：PPFD vs Total PWM预测对比
  - 右下：R²值对比

### 数据文件

- **ppfd_pwm_table_*.csv**: PPFD-PWM对应表
  - 包含PPFD、R_PWM、B_PWM、Total_PWM四列
  - PPFD范围：0-600 μmol/m²/s，步长10
  - 可用于快速查找和验证

## 技术细节

### 拟合方法
- 使用最小二乘法进行线性回归
- 分别拟合R_PWM和B_PWM对PPFD的线性关系
- 计算R²决定系数评估拟合质量

### 数据格式
- 输入：标定数据CSV文件（calib_data.csv）
- 输出：PNG图表和CSV数据表
- 编码：UTF-8

### 依赖库
- numpy: 数值计算
- matplotlib: 图表绘制
- csv: CSV文件处理
- pathlib: 路径处理

## 使用建议

1. **首次使用**: 先运行ppfd_demo.py了解系统功能
2. **数据验证**: 检查生成的图表和数据表是否符合预期
3. **参数调整**: 根据需要修改脚本中的测试参数
4. **结果分析**: 重点关注R²值和拟合曲线的合理性

## 注意事项

- 确保标定数据文件存在且格式正确
- 生成的图表使用英文标签避免字体问题
- 结果文件保存在result/目录中
- 脚本会自动创建result目录（如果不存在）

## 扩展功能

如需添加新的示例脚本，建议：
1. 遵循现有的代码结构和命名规范
2. 添加详细的文档说明
3. 将输出文件放在result/目录中
4. 更新本README文件

---

**最后更新**: 2025年9月25日  
**版本**: 1.0  
**作者**: LED控制系统开发团队
