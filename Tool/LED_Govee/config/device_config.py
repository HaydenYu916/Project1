"""Govee H6056 device configuration.

Hardware: ONE H6056 set = 1 BLE controller + 2 light bars (12 segments total).
Segments 0-5 = front bar, segments 6-11 = back bar.

To preserve the experiment semantics from the Shelly version (independent
red/blue PWM channels), we map:
    "Red"  -> front bar, color forced to pure red,  pwm controls seg-brightness
    "Blue" -> back  bar, color forced to pure blue, pwm controls seg-brightness
"""

DEVICE_MAC = "CE:F8:00:86:19:40"

SEG_FRONT = 0x003F  # bits 0..5
SEG_BACK = 0x0FC0   # bits 6..11

DEVICES = {
    "Red":  {"mac": DEVICE_MAC, "segment_mask": SEG_FRONT, "color": (255, 0, 0)},
    "Blue": {"mac": DEVICE_MAC, "segment_mask": SEG_BACK,  "color": (0, 0, 255)},
}

DEFAULT_BLE_ADAPTER = "hci0"
DEFAULT_SETTLE_MS = 300
DEFAULT_CONNECT_TIMEOUT = 20.0

WRITE_CHAR_UUID = "00010203-0405-0607-0809-0a0b0c0d2b11"
