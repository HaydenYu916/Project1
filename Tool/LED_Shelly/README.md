# Shelly Controller

A Python library for controlling Shelly devices, supporting RPC commands, real-time monitoring, and system management.

## Project Structure

```
shelly_src/
├── src/                    # Source code
│   ├── __init__.py
│   ├── shelly_controller.py    # Core controller
│   ├── shelly_listener.py      # Real-time listener
│   ├── shelly_live_api.py      # Live API interface
│   └── shelly_system_manager.py # System manager
├── tests/                  # Test files
│   ├── pwm_scheduler.py       # PWM scheduler
│   ├── pwm_service.py         # PWM service
│   ├── test_correct_ppfd.py   # PPFD test
│   ├── README_PWM_Scheduler.md # PWM scheduler documentation
│   └── src/                   # Test data
│       └── data/              # Image files
├── examples/               # Example code
│   ├── demo_fill_table.py     # Table filling demo
│   ├── repair_sweep.py        # Repair sweep
│   └── sweep_pwm.py           # PWM sweep
├── config/                 # Configuration files
│   └── device_config.py       # Device configuration
├── docs/                   # Documentation
├── data/                   # Data files
├── logs/                   # Log files
├── requirements.txt        # Dependencies
└── README.md              # Project description
```

## Features

- **Device Control**: Support for on/off and brightness control of red and blue LED devices
- **RPC Communication**: HTTP-based RPC command interface
- **Real-time Monitoring**: WebSocket real-time status monitoring
- **PWM Scheduling**: Automatic PWM control based on schedule
- **Data Collection**: Automatic sensor data collection

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from src.shelly_controller import rpc, DEVICES

# Control red light device
rpc(DEVICES["Red"], "Light.Set", {"id": 0, "on": True, "brightness": 80})

# Control blue light device  
rpc(DEVICES["Blue"], "Light.Set", {"id": 0, "on": True, "brightness": 20})
```

### 3. Command Line Usage

```bash
# Turn on red light device
python src/shelly_controller.py Red on

# Set brightness
python src/shelly_controller.py Red brightness 80

# Get status
python src/shelly_controller.py Red get_status
```

## Device Configuration

Configure device IP addresses in `config/device_config.py`:

```python
DEVICES = {
    "Red": "192.168.0.46",
    "Blue": "192.168.0.63",
}
```

## PWM Scheduler

Use the PWM scheduler for automatic control:

```bash
# Run PWM scheduler
python tests/pwm_scheduler.py

# Run in background
python tests/pwm_service.py start
```

## License

This project is for academic research use only.
