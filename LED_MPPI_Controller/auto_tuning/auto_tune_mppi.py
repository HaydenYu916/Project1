#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPPI控制器自动调参脚本

使用方法:
python auto_tune_mppi.py --method bayesian --iterations 20
python auto_tune_mppi.py --method online --duration 24
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 添加项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
RIOTEE_SENSOR_DIR = os.path.join(PROJECT_ROOT, "..", "Sensor", "riotee_sensor")
SHELLY_DIR = os.path.join(PROJECT_ROOT, "..", "Shelly", "src")

for path in (SRC_DIR, RIOTEE_SENSOR_DIR, SHELLY_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from auto_tuning import MPPIAutoTuner, MPPIParameters
from mppi_v2 import LEDPlant
from led import PWMtoPowerModel


def create_test_plant():
    """创建测试用的LEDPlant实例"""
    try:
        # 加载功率模型
        calib_csv = os.path.join(PROJECT_ROOT, "data", "calib_data.csv")
        if not os.path.exists(calib_csv):
            print(f"警告: 功率标定文件不存在: {calib_csv}")
            power_model = None
        else:
            power_model = PWMtoPowerModel(include_intercept=True)
            power_model = power_model.fit(calib_csv)
        
        # 创建LEDPlant
        plant = LEDPlant(
            base_ambient_temp=25.0,
            max_solar_vol=2.0,
            thermal_model_type='thermal',
            model_dir='Thermal/exported_models',
            power_model=power_model,
            r_b_ratio=0.83,
            use_solar_vol_model=True,
        )
        
        return plant
    except Exception as e:
        print(f"创建LEDPlant失败: {e}")
        return None


def run_bayesian_tuning(tuner: MPPIAutoTuner, iterations: int):
    """运行贝叶斯优化"""
    print(f"开始贝叶斯优化，迭代次数: {iterations}")
    start_time = time.time()
    
    try:
        optimal_params = tuner.bayesian_optimization(iterations)
        
        elapsed_time = time.time() - start_time
        print(f"贝叶斯优化完成，耗时: {elapsed_time:.2f}秒")
        print(f"最优参数: {optimal_params.to_dict()}")
        
        return optimal_params
    except Exception as e:
        print(f"贝叶斯优化失败: {e}")
        return None


def run_evolutionary_tuning(tuner: MPPIAutoTuner, iterations: int):
    """运行差分进化优化"""
    print(f"开始差分进化优化，迭代次数: {iterations}")
    start_time = time.time()
    
    try:
        optimal_params = tuner.differential_evolution_tuning(iterations)
        
        elapsed_time = time.time() - start_time
        print(f"差分进化优化完成，耗时: {elapsed_time:.2f}秒")
        print(f"最优参数: {optimal_params.to_dict()}")
        
        return optimal_params
    except Exception as e:
        print(f"差分进化优化失败: {e}")
        return None


def run_grid_search(tuner: MPPIAutoTuner, grid_size: int):
    """运行网格搜索"""
    print(f"开始网格搜索，网格大小: {grid_size}")
    start_time = time.time()
    
    try:
        optimal_params = tuner.grid_search_tuning(grid_size)
        
        elapsed_time = time.time() - start_time
        print(f"网格搜索完成，耗时: {elapsed_time:.2f}秒")
        print(f"最优参数: {optimal_params.to_dict()}")
        
        return optimal_params
    except Exception as e:
        print(f"网格搜索失败: {e}")
        return None


def run_online_tuning(tuner: MPPIAutoTuner, duration_hours: float):
    """运行在线自适应调参"""
    print(f"开始在线自适应调参，持续时间: {duration_hours}小时")
    print("按 Ctrl+C 停止")
    
    try:
        tuner.online_adaptive_tuning(
            adaptation_period_hours=1.0,  # 每小时评估一次
            learning_rate=0.1
        )
    except KeyboardInterrupt:
        print("在线调参被用户中断")
    except Exception as e:
        print(f"在线调参失败: {e}")


def apply_optimal_parameters(optimal_params: MPPIParameters, 
                           config_file_path: str):
    """将最优参数应用到实际控制脚本"""
    print(f"将最优参数应用到: {config_file_path}")
    
    # 读取现有配置
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        print(f"配置文件不存在: {config_file_path}")
        return
    
    # 替换参数值
    param_updates = {
        'DEFAULT_REFERENCE_WEIGHT': optimal_params.Q_ref,
        'CONTROL_INTERVAL_MINUTES': 15.0,  # 保持不变
    }
    
    for param_name, param_value in param_updates.items():
        # 查找并替换参数定义
        import re
        pattern = rf"^{param_name}\s*=\s*[\d.]+"
        replacement = f"{param_name} = {param_value}"
        
        if re.search(pattern, content, re.MULTILINE):
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            print(f"更新参数: {param_name} = {param_value}")
        else:
            print(f"警告: 未找到参数 {param_name}")
    
    # 写入更新后的配置
    with open(config_file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("参数更新完成")


def generate_parameter_report(optimal_params: MPPIParameters, 
                            tuner: MPPIAutoTuner):
    """生成参数调优报告"""
    report_file = os.path.join(tuner.log_dir, "tuning_report.md")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# MPPI控制器自动调参报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 最优参数配置\n\n")
        f.write("```python\n")
        f.write("self.controller.set_weights(\n")
        f.write(f"    Q_photo={optimal_params.Q_photo:.2f},\n")
        f.write(f"    R_du={optimal_params.R_du:.4f},\n")
        f.write(f"    R_power={optimal_params.R_power:.4f},\n")
        f.write(f"    Q_ref={optimal_params.Q_ref:.2f},\n")
        f.write(")\n")
        f.write("\n")
        f.write("self.controller.set_constraints(\n")
        f.write(f"    u_min={optimal_params.u_min:.2f},\n")
        f.write(f"    u_max={optimal_params.u_max:.2f},\n")
        f.write(f"    temp_min={optimal_params.temp_min:.1f},\n")
        f.write(f"    temp_max={optimal_params.temp_max:.1f},\n")
        f.write(")\n")
        f.write("\n")
        f.write("self.controller.set_mppi_params(\n")
        f.write(f"    u_std={optimal_params.u_std:.3f},\n")
        f.write(f"    dt=900.0,\n")
        f.write(")\n")
        f.write("```\n\n")
        
        f.write("## 参数说明\n\n")
        f.write(f"- **Q_photo**: {optimal_params.Q_photo:.2f} - 光合作用权重\n")
        f.write(f"- **Q_ref**: {optimal_params.Q_ref:.2f} - 参考跟踪权重\n")
        f.write(f"- **R_du**: {optimal_params.R_du:.4f} - 控制变化惩罚\n")
        f.write(f"- **R_power**: {optimal_params.R_power:.4f} - 功率惩罚\n")
        f.write(f"- **horizon**: {optimal_params.horizon} - 预测时域\n")
        f.write(f"- **num_samples**: {optimal_params.num_samples} - 采样数量\n")
        f.write(f"- **temperature**: {optimal_params.temperature:.2f} - MPPI温度参数\n")
        f.write(f"- **u_std**: {optimal_params.u_std:.3f} - 控制标准差\n")
        
        # 如果有历史记录，添加性能趋势
        if tuner.history:
            f.write("\n## 性能历史\n\n")
            f.write("| 时间 | 适应度 |\n")
            f.write("|------|--------|\n")
            for record in tuner.history[-10:]:  # 显示最近10条记录
                timestamp = record['timestamp'][:19]  # 去掉微秒
                fitness = record['fitness']
                f.write(f"| {timestamp} | {fitness:.4f} |\n")
    
    print(f"调参报告已生成: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="MPPI控制器自动调参")
    parser.add_argument("--method", 
                       choices=["bayesian", "evolutionary", "grid", "online"],
                       default="bayesian",
                       help="调参方法")
    parser.add_argument("--iterations", type=int, default=20,
                       help="优化迭代次数")
    parser.add_argument("--grid-size", type=int, default=3,
                       help="网格搜索的网格大小")
    parser.add_argument("--duration", type=float, default=24.0,
                       help="在线调参持续时间(小时)")
    parser.add_argument("--apply", action="store_true",
                       help="自动应用最优参数到控制脚本")
    parser.add_argument("--log-dir", default="logs/auto_tuning",
                       help="日志目录")
    
    args = parser.parse_args()
    
    # 创建测试植物模型
    print("初始化LEDPlant...")
    plant = create_test_plant()
    if plant is None:
        print("错误: 无法创建LEDPlant，退出")
        return 1
    
    # 创建自动调参器
    print("创建自动调参器...")
    tuner = MPPIAutoTuner(plant, log_dir=args.log_dir)
    
    # 运行调参
    optimal_params = None
    
    if args.method == "bayesian":
        optimal_params = run_bayesian_tuning(tuner, args.iterations)
    elif args.method == "evolutionary":
        optimal_params = run_evolutionary_tuning(tuner, args.iterations)
    elif args.method == "grid":
        optimal_params = run_grid_search(tuner, args.grid_size)
    elif args.method == "online":
        run_online_tuning(tuner, args.duration)
        optimal_params = tuner.current_params
    
    if optimal_params is not None:
        print("\n" + "="*50)
        print("调参结果:")
        print("="*50)
        
        # 生成报告
        generate_parameter_report(optimal_params, tuner)
        
        # 应用参数
        if args.apply:
            config_file = os.path.join(PROJECT_ROOT, "applications", "control", "mppi_control_real_v2.py")
            apply_optimal_parameters(optimal_params, config_file)
        
        print(f"\n最优参数已保存到: {tuner.log_dir}")
        print("可以将这些参数手动应用到控制脚本中")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
