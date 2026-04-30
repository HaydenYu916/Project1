"""
LED MPPI Controller 默认配置文件
"""

# 模型配置
MODEL_CONFIG = {
    'default_model': 'solar_vol',  # 默认使用的模型
    'available_models': ['solar_vol', 'ppfd', 'sp'],
    'model_paths': {
        'solar_vol': 'models/solar_vol/',
        'ppfd': 'models/ppfd/',
        'sp': 'models/sp/'
    }
}

# MPPI控制器参数
MPPI_CONFIG = {
    'horizon': 10,              # 预测时域
    'num_samples': 500,         # 采样数量
    'dt': 0.1,                  # 时间步长
    'temperature': 1.0,         # 温度参数
    'maintain_rb_ratio': True,  # 是否维持红蓝比例
    'rb_ratio_key': "5:1",      # 默认红蓝比例
}

# 权重配置
WEIGHTS = {
    'Q_photo': 10.0,      # 光合作用权重
    'R_pwm': 0.001,       # PWM权重
    'R_dpwm': 0.05,       # PWM变化权重
    'R_power': 0.01,      # 功率权重
}

# 约束条件
CONSTRAINTS = {
    'pwm_min': 5.0,       # PWM最小值
    'pwm_max': 95.0,      # PWM最大值
    'temp_min': 20.0,     # 温度最小值
    'temp_max': 30.0,     # 温度最大值
}

# 惩罚参数
PENALTIES = {
    'temp_penalty': 100000.0,  # 温度惩罚
    'pwm_penalty': 1000.0,     # PWM惩罚
}

# PWM标准差
PWM_STD = [15.0, 15.0]  # [红色, 蓝色]

# LED参数
LED_CONFIG = {
    'model_type': 'first_order',  # 热力学模型类型
    'use_efficiency': False,      # 是否使用效率模型
    'heat_scale': 1.0,           # 热缩放因子
}

# 数据文件路径
DATA_PATHS = {
    'calib_csv': 'data/calib_data.csv',
    'default_calib': 'data/calib_data.csv',
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/led_mppi.log'
}
