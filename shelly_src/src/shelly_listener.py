import asyncio
import aiohttp
import csv
import os
import sys
import signal
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, Optional
from aioshelly.rpc_device.device import RpcDevice, ConnectionOptions, WsServer

# 默认设备配置（可通过命令行覆盖）
DEVICES_DEFAULT = {
    "Red": "192.168.50.94",
    "Blue": "192.168.50.69",
}

# 日志目录与文件（运行时初始化）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CSV_ALL: Optional[str] = None
CSV_EVENT: Optional[str] = None

# 启动时间（用于计算 Duration）
start_time = None


def init_csv(note: str):
    """启动时写入 Start（附带备注）"""
    global start_time
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if note.strip():
        start_line = f"# Start @ {timestamp} {note}\n"
    else:
        start_line = f"# Start @ {timestamp}\n"

    os.makedirs(LOGS_DIR, exist_ok=True)

    for file, header in [
        (CSV_ALL, ["time", "update_type", "device", "on", "brightness", "apower", "voltage", "current"]),
        (CSV_EVENT, ["time", "update_type", "device", "on", "brightness", "apower", "voltage", "current"]),
    ]:
        if not os.path.exists(file):
            with open(file, "w", newline="") as f:
                f.write(start_line)
                writer = csv.writer(f)
                writer.writerow(header)
        else:
            with open(file, "a") as f:
                f.write(start_line)


def stop_csv():
    """退出时写 Stop @ 时间戳 (Duration)"""
    global start_time
    end_time = time.time()
    duration_sec = int(end_time - start_time) if start_time else 0
    h, m, s = duration_sec // 3600, (duration_sec % 3600) // 60, duration_sec % 60
    duration_str = f"{h}h{m}m{s}s" if h else f"{m}m{s}s"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stop_line = f"# Stop @ {timestamp} (Duration: {duration_str})\n"

    for file in [CSV_ALL, CSV_EVENT]:
        with open(file, "a") as f:
            f.write(stop_line)
            f.write("# ----\n")  # 分隔线，方便区分不同实验


async def listen_device(name: str, ip: str, ws_context: WsServer, last_brightness: Dict[str, Optional[int]], stop_event: asyncio.Event, reconnect_initial_delay: float = 2.0, reconnect_max_delay: float = 30.0):
    backoff = reconnect_initial_delay
    while not stop_event.is_set():
        try:
            options = ConnectionOptions(ip)
            async with aiohttp.ClientSession() as session:
                device = await RpcDevice.create(session, ws_context, options)
                await device.initialize()

                logging.info(f"{name}@{ip} 已连接，初始状态记录")

                def log_status(status, update_type="Init"):
                    try:
                        light = status.get("light:0", {})
                        row = [
                            datetime.now().isoformat(),
                            update_type,
                            name,
                            light.get("output"),
                            light.get("brightness"),
                            light.get("apower"),
                            light.get("voltage"),
                            light.get("current"),
                        ]
                        with open(CSV_ALL, "a", newline="") as f:
                            csv.writer(f).writerow(row)

                        brightness = light.get("brightness")
                        if last_brightness.get(name) != brightness:
                            last_brightness[name] = brightness
                            with open(CSV_EVENT, "a", newline="") as f:
                                csv.writer(f).writerow(row)

                        if update_type != "Init":
                            logging.debug(f"{name} 事件: b={brightness}")
                    except Exception as e:
                        logging.error(f"写入CSV失败或回调异常: {e}")

                # 初始状态
                log_status(device.status, "Init")

                # 订阅更新
                device.subscribe_updates(lambda dev, ut: log_status(dev.status, ut))

                # 保持运行直到收到停止信号
                while not stop_event.is_set():
                    await asyncio.sleep(1)

                return  # 收到停止信号，正常返回

        except asyncio.CancelledError:
            logging.info(f"{name} 任务被取消")
            raise
        except Exception as e:
            logging.warning(f"{name}@{ip} 连接/运行异常: {e}，将在 {backoff:.1f}s 后重试")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, reconnect_max_delay)


async def main():
    parser = argparse.ArgumentParser(description="Shelly 灯设备监听器")
    parser.add_argument("note", nargs="*", help="实验备注，可选")
    parser.add_argument("--devices", "-d", type=str, default=None, help="设备列表，如 Red=192.168.1.2,Blue=192.168.1.3")
    parser.add_argument("--port", type=int, default=8123, help="WebSocket 服务端口，默认8123")
    parser.add_argument("--debug", action="store_true", help="开启调试日志")
    args = parser.parse_args()

    # 日志初始化
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")

    # 设备解析
    if args.devices:
        devices: Dict[str, str] = {}
        for pair in args.devices.split(','):
            if not pair.strip():
                continue
            if '=' not in pair:
                logging.warning(f"忽略无效设备配置: {pair}")
                continue
            name, ip = pair.split('=', 1)
            devices[name.strip()] = ip.strip()
    else:
        devices = DEVICES_DEFAULT

    # CSV 路径
    global CSV_ALL, CSV_EVENT
    CSV_ALL = os.path.join(LOGS_DIR, "shelly_log_all.csv")
    CSV_EVENT = os.path.join(LOGS_DIR, "shelly_log_event.csv")

    # 备注
    note = " ".join(args.note) if args.note else ""
    init_csv(note)

    # 启动 WS 上下文
    ws_context = WsServer()
    bind_port = args.port
    max_tries = 10
    for attempt in range(max_tries):
        try:
            await ws_context.initialize(bind_port)
            logging.info(f"WsServer 启动在端口 {bind_port}")
            break
        except OSError as e:
            logging.warning(f"端口 {bind_port} 占用或不可用: {e}. 尝试下一个端口")
            bind_port += 1
    else:
        logging.error("无法绑定任何可用端口，退出")
        return

    # 任务集合
    last_brightness: Dict[str, Optional[int]] = {}
    stop_event = asyncio.Event()

    tasks = [
        asyncio.create_task(
            listen_device(name, ip, ws_context, last_brightness, stop_event)
        )
        for name, ip in devices.items()
    ]

    # 捕捉 Ctrl+C/SIGTERM，触发优雅退出
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    try:
        await stop_event.wait()
    finally:
        # 取消任务并等待结束
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # 关闭 ws_context（如果支持）
        try:
            if hasattr(ws_context, "close"):
                await ws_context.close()
        except Exception:
            pass
        stop_csv()


if __name__ == "__main__":
    asyncio.run(main())

