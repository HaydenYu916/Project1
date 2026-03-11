"""Simple command‑line entry point that uses :mod:`psm_lib`.

This script provides a command‑line entry point for the
PSM‑60s library.  It opens a serial port, performs periodic measurements
and appends results to a CSV file.

Usage:
    python cli.py [--port PORT]

"""
import argparse
import datetime
import glob
import os
import time

import serial

from lib import (
    get_csv_filename,
    complete_spectrum_measurement,
    append_parsed_measurement_to_csv,
    process_pending_raw_files,
    run_single_measurement,
)


def _list_port_candidates() -> list[str]:
    """List likely serial ports for user guidance."""
    candidates = set()
    preferred_prefixes = ("/dev/ttyACM", "/dev/ttyUSB", "/dev/cu.usb", "/dev/tty.usb")
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*", "/dev/cu.usb*", "/dev/tty.usb*"):
        candidates.update(glob.glob(pattern))

    # Prefer pyserial's enumerator when available.
    try:
        from serial.tools import list_ports

        for info in list_ports.comports():
            dev = getattr(info, "device", "")
            if dev and dev.startswith(preferred_prefixes):
                candidates.add(dev)
    except Exception:
        pass

    return sorted(candidates)


def _print_port_error(port: str, error: Exception | None = None) -> None:
    """Print a concise serial-port failure message with suggestions."""
    if error is None:
        print(f"❌ 串口不存在: {port}")
    else:
        print(f"❌ 串口打开失败: {port} ({error})")

    ports = _list_port_candidates()
    if ports:
        print("🔎 当前可见串口：")
        for p in ports:
            print(f"  - {p}")
        print("请使用 `--port <设备路径>` 指定正确串口。")
    else:
        print("🔎 当前未发现常见 USB 串口设备（/dev/ttyACM* 或 /dev/ttyUSB*）。")
        print("请检查设备连接、线缆和权限后重试。")


def _print_pending_summary(stats: dict[str, int], source_dir: str) -> None:
    """Print a compact startup summary for pending raw files."""
    print(
        f"📂 启动扫描 {source_dir}: "
        f"发现 {stats['found']}，入库 {stats['processed']}，"
        f"重复 {stats['duplicate']}，缺失 {stats['missing_file']}，"
        f"写入失败 {stats['write_error']}，归档失败 {stats['archive_failed']}"
    )


def measurement_loop(
    port: str,
    archive_dir: str | None = None,
):
    """Run the per‑minute measurement cycle (no test mode).

    测量完成后会输出标准格式 CSV（类似 ``580.csv``）并写入每日汇总。
    ``archive_dir`` 仅影响启动时处理旧 ``raw_bytes_*.txt`` 文件的归档。
    """
    csv_filename = get_csv_filename()
    ser = None

    try:
        ser = serial.Serial(port, baudrate=115200, timeout=1)
        print(f"🔌 已打开串口 {port}")

        while True:
            current_time = datetime.datetime.now()
            if current_time.second == 0:
                print(f"\n⏰ {current_time.strftime('%Y-%m-%d %H:%M:%S')} - 开始光谱测量")
                # perform real measurement; append results when available
                result = complete_spectrum_measurement(ser)
                # result may be (raw_bytes, standard_csv, parsed), total_time, integration
                if isinstance(result, tuple):
                    spectrum_result = result[0]
                    if isinstance(spectrum_result, tuple) and len(spectrum_result) >= 3:
                        standard_csv_path = spectrum_result[1]
                        parsed = spectrum_result[2]
                        if standard_csv_path and parsed:
                            append_parsed_measurement_to_csv(
                                parsed,
                                csv_filename,
                                source_file=standard_csv_path,
                            )
                print(f"✅ 测量完成，等待下一分钟...")
                while True:
                    now = datetime.datetime.now()
                    if now.second == 0:
                        break
                    seconds_to_wait = 60 - now.second
                    print(f"\r⏳ 距离下次采样还有 {seconds_to_wait} 秒... [{now.strftime('%H:%M:%S')}]",
                          end="", flush=True)
                    time.sleep(1)
                print()
            else:
                seconds_to_wait = 60 - current_time.second
                print(f"\r⏳ 等待下次采样，还有 {seconds_to_wait} 秒...", end="", flush=True)
                time.sleep(1)

    except Exception as e:
        if hasattr(serial, 'SerialException') and isinstance(e, serial.SerialException):
            print(f"❌ 串口连接失败: {e}")
        else:
            # re-raise keyboard interrupts so they propagate cleanly
            if isinstance(e, KeyboardInterrupt):
                raise
            print(f"❌ 遇到错误: {e}")
    except KeyboardInterrupt:
        print("\n🛑 程序被用户中断")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("🔌 串口已关闭")


def main():
    parser = argparse.ArgumentParser(description="PSM-60s 光谱仪采集")
    parser.add_argument("--port", default="/dev/ttyACM0",
                        help="串口设备路径")
    parser.add_argument("--source-dir", default=".",
                        help="启动时扫描待入库 raw 文件的目录")
    parser.add_argument("--archive", metavar="DIR", default=None,
                        help="将处理过的 raw 文件移动到指定目录作为历史")
    parser.add_argument("--once", action="store_true",
                        help="只执行一次测量然后退出（不进入循环）")
    args = parser.parse_args()

    csv_filename = get_csv_filename()
    try:
        stats = process_pending_raw_files(
            csv_filename,
            source_dir=args.source_dir,
            archive_dir=args.archive,
        )
    except (FileNotFoundError, NotADirectoryError) as e:
        parser.error(str(e))
    _print_pending_summary(stats, args.source_dir)

    if args.port.startswith("/dev/") and not os.path.exists(args.port):
        _print_port_error(args.port)
        return 2

    if args.once:
        try:
            run_single_measurement(
                args.port,
                csv_filename=csv_filename,
                archive_dir=args.archive,
            )
        except serial.SerialException as e:
            _print_port_error(args.port, e)
            return 2
    else:
        try:
            measurement_loop(
                args.port,
                archive_dir=args.archive,
            )
        except serial.SerialException as e:
            _print_port_error(args.port, e)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
