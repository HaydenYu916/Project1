"""
LED MPPI Controller Package

基于模型预测路径积分（MPPI）的LED植物生长控制系统
"""

from .mppi_v2 import LEDPlant, LEDMPPIController  # 统一导出合并后的实现
from .led import LedThermalParams, create_model, PWMtoPowerModel, PWMtoPPFDModel

__version__ = "2.0.0"
__author__ = "LED Control Research Team"
__email__ = "research@ledcontrol.com"

__all__ = [
    'LEDPlant',            # 来自 mppi_v2.py（Solar Vol 控制）
    'LEDMPPIController',   # 来自 mppi_v2.py（Solar Vol 控制）
    'LedThermalParams',
    'create_model',
    'PWMtoPowerModel',
    'PWMtoPPFDModel',
]
