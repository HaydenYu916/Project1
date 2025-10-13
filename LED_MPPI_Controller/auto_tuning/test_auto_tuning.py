#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动调参功能测试脚本
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

try:
    from adaptive_tuning import AdaptiveMPPITuner
    print("✅ 成功导入 AdaptiveMPPITuner")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def test_adaptive_tuner_basic():
    """测试自适应调参器基本功能"""
    print("\n🧪 测试自适应调参器基本功能...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        # 创建调参器
        tuner = AdaptiveMPPITuner(
            log_dir=temp_dir,
            adaptation_period=0.1,  # 很短的周期用于测试
            learning_rate=0.1
        )
        
        # 测试参数获取
        params = tuner.get_current_parameters()
        assert isinstance(params, dict)
        assert 'Q_photo' in params
        assert 'Q_ref' in params
        print("✅ 参数获取正常")
        
        # 测试性能记录
        score = tuner.record_performance(
            photosynthesis_rate=5.0,
            temp_violation=0.0,
            control_change=0.1,
            power=50.0,
            ref_error=0.05
        )
        assert isinstance(score, (int, float))
        print(f"✅ 性能记录正常，分数: {score:.4f}")
        
        # 测试参数适应
        improved = tuner.adapt_parameters()
        print(f"✅ 参数适应测试完成，是否改进: {improved}")
        
        # 测试性能摘要
        summary = tuner.get_performance_summary()
        assert isinstance(summary, dict)
        print(f"✅ 性能摘要正常，记录数: {summary.get('num_records', 0)}")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("✅ 自适应调参器基本功能测试通过")


def test_parameter_ranges():
    """测试参数范围"""
    print("\n🧪 测试参数范围...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        tuner = AdaptiveMPPITuner(log_dir=temp_dir)
        
        # 检查参数范围
        ranges = tuner.param_ranges
        required_params = ['Q_photo', 'Q_ref', 'R_du', 'R_power', 'u_std', 'temperature']
        
        for param in required_params:
            assert param in ranges, f"缺少参数范围: {param}"
            min_val, max_val = ranges[param]
            assert min_val < max_val, f"参数范围无效: {param}"
            print(f"✅ {param}: [{min_val}, {max_val}]")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("✅ 参数范围测试通过")


def test_performance_scoring():
    """测试性能评分"""
    print("\n🧪 测试性能评分...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        tuner = AdaptiveMPPITuner(log_dir=temp_dir)
        
        # 测试不同场景的性能评分
        scenarios = [
            {
                'name': '理想场景',
                'photosynthesis': 8.0,
                'temp_violation': 0.0,
                'control_change': 0.05,
                'power': 40.0,
                'ref_error': 0.02
            },
            {
                'name': '温度违规',
                'photosynthesis': 6.0,
                'temp_violation': 2.0,
                'control_change': 0.1,
                'power': 60.0,
                'ref_error': 0.1
            },
            {
                'name': '控制抖动',
                'photosynthesis': 7.0,
                'temp_violation': 0.5,
                'control_change': 1.0,
                'power': 50.0,
                'ref_error': 0.05
            }
        ]
        
        for scenario in scenarios:
            score = tuner.compute_performance_score(
                photosynthesis_rate=scenario['photosynthesis'],
                temp_violation=scenario['temp_violation'],
                control_smoothness=scenario['control_change'],
                power_efficiency=scenario['power'],
                reference_tracking_error=scenario['ref_error']
            )
            print(f"✅ {scenario['name']}: {score:.4f}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("✅ 性能评分测试通过")


def test_history_persistence():
    """测试历史数据持久化"""
    print("\n🧪 测试历史数据持久化...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # 创建调参器并添加一些数据
        tuner = AdaptiveMPPITuner(log_dir=temp_dir)
        
        # 添加一些性能记录
        for i in range(5):
            tuner.record_performance(
                photosynthesis_rate=5.0 + i * 0.5,
                temp_violation=0.0,
                control_change=0.1,
                power=50.0,
                ref_error=0.05
            )
        
        # 保存历史
        tuner.save_history()
        
        # 创建新的调参器实例
        tuner2 = AdaptiveMPPITuner(log_dir=temp_dir)
        
        # 检查历史数据是否加载
        assert len(tuner2.performance_history) == 5
        print(f"✅ 历史数据加载正常，记录数: {len(tuner2.performance_history)}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("✅ 历史数据持久化测试通过")


def main():
    """运行所有测试"""
    print("🚀 开始自动调参功能测试")
    print("=" * 50)
    
    try:
        test_adaptive_tuner_basic()
        test_parameter_ranges()
        test_performance_scoring()
        test_history_persistence()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！自动调参功能正常")
        print("\n下一步:")
        print("1. 安装依赖: bash scripts/install_tuning_deps.sh")
        print("2. 运行离线调参: python applications/tuning/auto_tune_mppi.py --method evolutionary --iterations 5")
        print("3. 运行在线调参: python applications/control/mppi_control_adaptive.py once")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
