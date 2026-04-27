#!/usr/bin/env python3
"""H6056 segment-control probe.

Goal: confirm the H6053-family segment protocol works on H6056, and discover
which segment bitmask corresponds to the left bar vs the right bar.

Sequence:
  1. Power on, set global brightness 100.
  2. Light segments 0..11 one at a time in white -> user reports which bar
     each segment is on (so we know the left/right split).
  3. With a guessed split (low 6 bits = left, high 6 bits = right), set
     left=red, right=blue, brightness=50.
  4. Ramp left brightness 0..100 (right stays at 50).
  5. Ramp right brightness 0..100 (left stays at 50).
  6. Power off.
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


def cmd_global_brightness(level: int) -> bytes:
    return _frame(bytes([0x33, 0x04, max(0, min(100, level))]))


def cmd_seg_color(seg_mask: int, r: int, g: int, b: int) -> bytes:
    seg_lo = seg_mask & 0xFF
    seg_hi = (seg_mask >> 8) & 0xFF
    return _frame(
        bytes(
            [0x33, 0x05, 0x15, 0x01, r & 0xFF, g & 0xFF, b & 0xFF,
             0, 0, 0, 0, 0, seg_lo, seg_hi]
        )
    )


def cmd_seg_brightness(seg_mask: int, pct: int) -> bytes:
    pct = max(0, min(100, int(pct)))
    seg_lo = seg_mask & 0xFF
    seg_hi = (seg_mask >> 8) & 0xFF
    return _frame(bytes([0x33, 0x05, 0x15, 0x02, pct, seg_lo, seg_hi]))


async def write(client: BleakClient, label: str, payload: bytes) -> None:
    t0 = time.perf_counter()
    await client.write_gatt_char(WRITE_CHAR, payload, response=False)
    dt = (time.perf_counter() - t0) * 1000
    print(f"  {label:<32s} {payload.hex()}  ({dt:5.1f} ms)")


async def main(mac: str, mode: str) -> None:
    print(f"Connecting to {mac} ...")
    async with BleakClient(mac, timeout=20.0) as client:
        print("Connected.")

        await write(client, "power on", cmd_power(True))
        await asyncio.sleep(0.3)
        await write(client, "global brightness 100", cmd_global_brightness(100))
        await asyncio.sleep(0.3)

        if mode == "probe":
            print("\n>>> PROBE MODE: lighting segments 0..11 one at a time.")
            print(">>> Watch the bars. For each segment, note which bar (LEFT/RIGHT) lights up.")
            print(">>> Each segment shows white for 2s, then turns black before next.\n")
            for seg_idx in range(12):
                mask = 1 << seg_idx
                # First clear all to dim red so we can see context
                await write(client, f"all-off (black)", cmd_seg_color(0x0FFF, 0, 0, 0))
                await asyncio.sleep(0.1)
                await write(client, f"seg {seg_idx:2d} -> WHITE", cmd_seg_color(mask, 255, 255, 255))
                await asyncio.sleep(2.0)

            await write(client, "all black", cmd_seg_color(0x0FFF, 0, 0, 0))

        elif mode == "split":
            # Confirmed split for H6056: segments 0-5 = front bar, 6-11 = back bar
            FRONT = 0x003F
            BACK = 0x0FC0
            print("\n>>> SPLIT MODE: front=RED, back=BLUE, then independent brightness ramps.\n")

            await write(client, "front -> RED",  cmd_seg_color(FRONT, 255, 0, 0))
            await asyncio.sleep(0.2)
            await write(client, "back  -> BLUE", cmd_seg_color(BACK, 0, 0, 255))
            await asyncio.sleep(0.2)
            await write(client, "front brightness 50", cmd_seg_brightness(FRONT, 50))
            await asyncio.sleep(0.2)
            await write(client, "back  brightness 50", cmd_seg_brightness(BACK, 50))
            await asyncio.sleep(2.0)

            print("\n--- ramping FRONT (red) 0..100, back held at 50 ---")
            for pct in (0, 20, 40, 60, 80, 100):
                await write(client, f"front bright {pct:3d}", cmd_seg_brightness(FRONT, pct))
                await asyncio.sleep(1.2)

            print("\n--- ramping BACK (blue) 0..100, front held at 100 ---")
            for pct in (0, 20, 40, 60, 80, 100):
                await write(client, f"back  bright {pct:3d}", cmd_seg_brightness(BACK, pct))
                await asyncio.sleep(1.2)

        await write(client, "power off", cmd_power(False))
        await asyncio.sleep(0.3)

    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mac")
    ap.add_argument("--mode", choices=("probe", "split"), default="probe",
                    help="probe = light each segment one-by-one; split = left=red right=blue ramp")
    args = ap.parse_args()
    asyncio.run(main(args.mac.upper(), args.mode))
