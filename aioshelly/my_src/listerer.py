import asyncio
import aiohttp
import csv
import os
import sys
import signal
import time
from datetime import datetime
from aioshelly.rpc_device.device import RpcDevice, ConnectionOptions, WsServer

# 设备配置
DEVICES = {
    "Red": "192.168.50.94",
    "Blue": "192.168.50.69",
}

# 日志文件
CSV_ALL = "shelly_log_all.csv"
CSV_EVENT = "shelly_log_event.csv"

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


async def listen_device(name, ip, ws_context, last_brightness):
    options = ConnectionOptions(ip)
    async with aiohttp.ClientSession() as session:
        device = await RpcDevice.create(session, ws_context, options)
        await device.initialize()

        def log_status(status, update_type="Init"):
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
            # 写全量日志
            with open(CSV_ALL, "a", newline="") as f:
                csv.writer(f).writerow(row)

            # 只在亮度变化时写事件日志
            brightness = light.get("brightness")
            if last_brightness.get(name) != brightness:
                last_brightness[name] = brightness
                with open(CSV_EVENT, "a", newline="") as f:
                    csv.writer(f).writerow(row)

        # 初始状态
        log_status(device.status, "Init")

        # 订阅更新
        device.subscribe_updates(lambda dev, ut: log_status(dev.status, ut))

        while True:
            await asyncio.sleep(2)


async def main():
    # 备注
    note = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    init_csv(note)

    ws_context = WsServer()
    await ws_context.initialize(8123)

    last_brightness = {}
    tasks = [listen_device(name, ip, ws_context, last_brightness) for name, ip in DEVICES.items()]

    # 捕捉 Ctrl+C，写 Stop
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: stop_csv() or loop.stop())

    try:
        await asyncio.gather(*tasks)
    finally:
        stop_csv()


if __name__ == "__main__":
    asyncio.run(main())

