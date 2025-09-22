"""
LED MPPI Controller 应用配置文件
统一管理所有控制应用的配置参数
"""

# ==================== 温度传感器配置 ====================
# 温度传感器设备ID配置
TEMPERATURE_DEVICE_ID = None  # None=自动选择, "T6ncwg=="=指定设备1, "L_6vSQ=="=指定设备2

# 温度数据最大有效期（秒）
MAX_TEMPERATURE_AGE_SECONDS = 300  # 5分钟

# ==================== 控制循环配置 ====================
# 控制循环间隔（分钟）
CONTROL_INTERVAL_MINUTES = 1

# 红蓝比例键
RB_RATIO_KEY = "5:1"

# 目标温度（°C）
TARGET_TEMPERATURE = 25.0

# ==================== 设备配置 ====================
# LED设备IP地址
RED_LED_IP = "192.168.50.94"
BLUE_LED_IP = "192.168.50.69"

# 状态检查延迟（秒）
STATUS_CHECK_DELAY = 3

# ==================== 日志配置 ====================
# 日志文件路径（相对于项目根目录）
LOG_DIR = "logs"
CONTROL_SIMULATE_LOG = "control_simulate_log.csv"
CONTROL_REAL_LOG = "control_real_log.csv"

# ==================== MPPI控制器配置 ====================
# MPPI控制器参数
MPPI_HORIZON = 10
MPPI_NUM_SAMPLES = 1000
MPPI_DT = 0.1
MPPI_TEMPERATURE = 1.0

# PWM约束
PWM_MIN = 0
PWM_MAX = 80

# 温度约束
TEMP_MIN = 20.0
TEMP_MAX = 29.0

# ==================== 模型配置 ====================
# 默认使用的模型
DEFAULT_MODEL_NAME = "solar_vol"  # solar_vol, ppfd, sp

# ==================== 路径配置 ====================
# 外部依赖路径（相对于项目根目录）
RIOTEE_SENSOR_DIR = "../Test/riotee_sensor"
CONTROLLER_DIR = "../shelly_src/src"

# ==================== 调试配置 ====================
# 是否启用详细日志
VERBOSE_LOGGING = True

# 是否启用模拟模式（不实际发送命令到设备）
SIMULATION_MODE = True

# ==================== 安全配置 ====================
# 最大允许的PWM值
MAX_SAFE_PWM = 80

# 温度保护阈值
TEMP_PROTECTION_THRESHOLD = 29.0

# 温度保护时的PWM缩放因子
TEMP_PROTECTION_SCALE = 0.7
