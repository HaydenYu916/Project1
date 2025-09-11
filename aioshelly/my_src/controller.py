import sys
import requests

DEVICES = {
    "Red": "192.168.50.94",
    "Blue": "192.168.50.69",
}

def rpc(ip, method, params=None):
    url = f"http://{ip}/rpc/{method}"
    r = requests.get(url, params=params)
    return r.json()

def print_status(status):
    """只打印关键字段"""
    light = status.get("light:0", {})
    print({
        "on": light.get("output"),
        "brightness": light.get("brightness"),
        "apower": light.get("apower"),
        "voltage": light.get("voltage"),
        "current": light.get("current"),
        "temperature_C": light.get("temperature", {}).get("tC"),
    })

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python3 controller.py <device> <command> [args...]")
        print("支持命令: on | off | brightness <val> | transition <val> <ms> | get_status | sys")
        sys.exit(1)

    device = sys.argv[1]
    action = sys.argv[2]

    if device not in DEVICES:
        print(f"未知设备 {device}，可用: {list(DEVICES.keys())}")
        sys.exit(1)

    ip = DEVICES[device]

    if action == "on":
        print(rpc(ip, "Light.Set", {"id": 0, "on": True}))

    elif action == "off":
        print(rpc(ip, "Light.Set", {"id": 0, "on": False}))

    elif action == "brightness" and len(sys.argv) == 4:
        value = int(sys.argv[3])
        print(rpc(ip, "Light.Set", {"id": 0, "on": True, "brightness": value}))

    elif action == "transition" and len(sys.argv) == 5:
        value = int(sys.argv[3])
        ms = int(sys.argv[4])
        print(rpc(ip, "Light.Set", {"id": 0, "on": True, "brightness": value, "transition": ms}))

    elif action == "get_status":
        status = rpc(ip, "Shelly.GetStatus")
        print_status(status)   # 只输出关键数据

    elif action == "sys":
        print(rpc(ip, "Shelly.GetDeviceInfo"))

    else:
        print("参数错误")
