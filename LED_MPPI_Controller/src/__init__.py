"""
LED MPPI Controller Package

基于模型预测路径积分（MPPI）的LED植物生长控制系统
"""

from .mppi import LEDPlant, LEDMPPIController
from .led import LedThermalParams, create_model, PWMtoPowerModel, PWMtoPPFDModel

__version__ = "2.0.0"
__author__ = "LED Control Research Team"
__email__ = "research@ledcontrol.com"

__all__ = [
    'LEDPlant',
    'LEDMPPIController', 
    'LedThermalParams',
    'create_model',
    'PWMtoPowerModel',
    'PWMtoPPFDModel'
]

