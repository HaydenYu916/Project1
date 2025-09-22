#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Riotee设备控制器 - 类似aioshelly controller.py风格
==================================================

功能:
- 向Riotee设备发送二进制休眠时间配置命令
- 简单命令行控制接口
- 类似aioshelly controller的使用方式

使用方法:
python3 riotee_controller.py <device_id> <command> [args...]
python3 riotee_controller.py all sleep <seconds>
python3 riotee_controller.py ABC123 sleep 10
"""

import sys
import time
import subprocess
import struct
import socket
import os

# 添加riotee_gateway路径
riotee_env_path = "/home/pi/Desktop/riotee-env/lib/python3.11/site-packages"
if os.path.exists(riotee_env_path) and riotee_env_path not in sys.path:
    sys.path.insert(0, riotee_env_path)

try:
    from riotee_gateway import GatewayClient
    from riotee_gateway.packet_model import PacketApiSend
    import numpy as np
except ImportError as e:
    print(f"错误: 无法导入riotee_gateway模块: {e}")
    print("请确保已安装riotee_gateway或激活相应的虚拟环境")
    sys.exit(1)

def check_gateway_running():
    """检查Gateway服务器是否运行"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        return result == 0
    except:
        return False

def start_gateway_server():
    """启动Gateway服务器（如果未运行）"""
    if check_gateway_running():
        return None
    
    try:
        print("正在启动Gateway服务器...")
        gateway_process = subprocess.Popen(
            ["riotee-gateway", "server", "-p", "8000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(3)
        print("Gateway服务器已启动")
        return gateway_process
    except Exception as e:
        print(f"启动Gateway服务器失败: {e}")
        return None

def build_sleep_command(seconds):
    """构建休眠时间二进制命令
    格式: 'S' (0x53) + uint16 little-endian 秒数
    """
    if seconds < 1 or seconds > 3600:
        raise ValueError("休眠时间必须在1-3600秒之间")
    
    return struct.pack('<BH', ord('S'), seconds)

def send_command(client, device_id, command):
    """向设备发送命令"""
    try:
        pkt_id = int(np.random.randint(0, 2**16))
        pkt = PacketApiSend.from_binary(command, pkt_id=pkt_id)
        client.send_packet(device_id, pkt)
        return True
    except Exception as e:
        print(f"发送失败: {e}")
        return False

def get_devices(client):
    """获取设备列表"""
    try:
        return list(client.get_devices())
    except:
        return []

def print_usage():
    """打印使用说明"""
    print("用法: python3 riotee_controller.py <device> <command> [args...]")
    print("支持命令:")
    print("  sleep <seconds>  - 设置休眠时间(1-3600秒)")
    print("  list            - 列出所有设备")
    print("  status          - 显示设备状态")
    print("")
    print("设备:")
    print("  all       - 所有设备")
    print("  <ID>      - 指定设备ID")
    print("")
    print("示例:")
    print("  python3 riotee_controller.py all sleep 10")
    print("  python3 riotee_controller.py ABC123 sleep 5")
    print("  python3 riotee_controller.py all list")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)

    device = sys.argv[1]
    action = sys.argv[2]

    # 启动Gateway
    gateway_process = start_gateway_server()
    
    try:
        # 连接Gateway
        client = GatewayClient(host="localhost", port=8000)
        devices = get_devices(client)
        
        if not devices:
            print("未找到设备")
            sys.exit(1)

        # 验证设备
        if device != "all" and device not in devices:
            print(f"未知设备 {device}，可用: {devices}")
            sys.exit(1)

        # 执行命令
        if action == "sleep" and len(sys.argv) == 4:
            sleep_time = int(sys.argv[3])
            command = build_sleep_command(sleep_time)
            
            if device == "all":
                success = 0
                for dev_id in devices:
                    if send_command(client, dev_id, command):
                        print(f"✓ {dev_id}: 休眠时间设置为 {sleep_time}秒")
                        success += 1
                    else:
                        print(f"✗ {dev_id}: 命令发送失败")
                print(f"完成: {success}/{len(devices)} 设备成功")
            else:
                if send_command(client, device, command):
                    print(f"✓ {device}: 休眠时间设置为 {sleep_time}秒")
                else:
                    print(f"✗ {device}: 命令发送失败")

        elif action == "list":
            print(f"找到 {len(devices)} 个设备:")
            for i, dev_id in enumerate(devices):
                print(f"  {i+1}. {dev_id}")

        elif action == "status":
            print(f"Gateway: {'运行中' if check_gateway_running() else '未运行'}")
            print(f"设备数量: {len(devices)}")
            print(f"设备列表: {devices}")

        else:
            print("无效命令")
            print_usage()

    except Exception as e:
        print(f"错误: {e}")
    finally:
        if gateway_process:
            gateway_process.terminate()
