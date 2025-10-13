#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPPI控制器自动调参模块

提供完整的自动调参功能，包括离线调参和在线自适应调参。
"""

import os
import sys

# 添加项目路径到sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
RIOTEE_SENSOR_DIR = os.path.join(PROJECT_ROOT, "..", "Sensor", "riotee_sensor")
SHELLY_DIR = os.path.join(PROJECT_ROOT, "..", "Shelly", "src")

for path in (SRC_DIR, RIOTEE_SENSOR_DIR, SHELLY_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# 导入核心模块
try:
    from .auto_tuning import MPPIAutoTuner, MPPIParameters, PerformanceMetrics
    from .adaptive_tuning import AdaptiveMPPITuner, integrate_with_controller
except ImportError as e:
    print(f"警告: 自动调参模块导入失败: {e}")
    print("请确保已安装所需依赖: bash install_tuning_deps.sh")

__version__ = "1.0.0"
__author__ = "MPPI Controller Team"

__all__ = [
    "MPPIAutoTuner",
    "MPPIParameters", 
    "PerformanceMetrics",
    "AdaptiveMPPITuner",
    "integrate_with_controller"
]
