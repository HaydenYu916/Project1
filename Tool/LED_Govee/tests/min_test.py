#!/usr/bin/env python3
"""Minimal Govee H6056 BLE smoke test.

Usage:
    python3 min_test.py D0:C9:07:7D:17:AC

Sequence: connect -> power on -> set red -> brightness 0..100 ramp -> power off.
Prints per-command latency to validate protocol bytes and connection stability
before integrating into the calibration runner.
"""

import argparse
import asyncio
import time

from bleak import BleakClient

WRITE_CHAR = "00010203-0405-0607-0809-0a0b0c0d2b11"


def _frame(payload: bytes) -> bytes:
    if len(payload) > 19:
        raise ValueError("payload too long")
    buf = bytearray(20)
    buf[: len(payload)] = payload
    chk = 0
    for b in buf[:19]:
        chk ^= b
    buf[19] = chk
    return bytes(buf)


def cmd_power(on: bool) -> bytes:
    return _frame(bytes([0x33, 0x01, 0x01 if on else 0x00]))


def cmd_brightness(level: int) -> bytes:
    level = max(0, min(100, int(level)))
    return _frame(bytes([0x33, 0x04, level]))


def cmd_color(r: int, g: int, b: int) -> bytes:
    return _frame(bytes([0x33, 0x05, 0x02, r & 0xFF, g & 0xFF, b & 0xFF]))


async def write(client: BleakClient, label: str, payload: bytes) -> None:
    t0 = time.perf_counter()
    await client.write_gatt_char(WRITE_CHAR, payload, response=False)
    dt = (time.perf_counter() - t0) * 1000
    print(f"  {label:<20s} {payload.hex()}  ({dt:6.1f} ms)")


async def main(mac: str) -> None:
    print(f"Connecting to {mac} ...")
    t0 = time.perf_counter()
    async with BleakClient(mac, timeout=20.0) as client:
        print(f"Connected in {(time.perf_counter() - t0)*1000:.0f} ms")

        await write(client, "power on", cmd_power(True))
        await asyncio.sleep(0.3)

        await write(client, "color red", cmd_color(255, 0, 0))
        await asyncio.sleep(0.3)

        for level in (0, 20, 40, 60, 80, 100):
            await write(client, f"bright {level:3d}", cmd_brightness(level))
            await asyncio.sleep(0.4)

        await write(client, "power off", cmd_power(False))
        await asyncio.sleep(0.3)

    print("Done. Disconnected cleanly.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mac", help="H6056 MAC address, e.g. D0:C9:07:7D:17:AC")
    args = ap.parse_args()
    asyncio.run(main(args.mac.upper()))
