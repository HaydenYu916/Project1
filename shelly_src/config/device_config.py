"""
Shelly设备配置
"""

# Shelly设备IP地址配置
DEVICES = {
    "Red": "192.168.50.94",
    "Blue": "192.168.50.69",
}

# 设备默认参数
DEFAULT_BRIGHTNESS = 50
DEFAULT_TRANSITION_MS = 1000
DEFAULT_TIMEOUT = 5

# 日志配置
LOG_DIR = "logs"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
