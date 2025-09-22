import csv
import sys
import time
from datetime import datetime
from pathlib import Path

from shelly_controller import rpc, DEVICES


def _retry(callable_fn, max_attempts=5, base_delay=0.5):
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return callable_fn()
        except Exception as e:
            last_err = str(e)
        time.sleep(base_delay * (2 ** (attempt - 1)))
    return {"error": last_err or "unknown"}


def set_brightness(device_name: str, pwm_value: int, transition_ms: int = 0):
    ip = DEVICES[device_name]
    value = max(0, min(100, int(pwm_value)))
    if value <= 0:
        return _retry(lambda: rpc(ip, "Light.Set", {"id": 0, "on": False, "brightness": 0, "transition": transition_ms}))
    return _retry(lambda: rpc(ip, "Light.Set", {"id": 0, "on": True, "brightness": value, "transition": transition_ms}))


def get_status(device_name: str, source: str = "em"):
    ip = DEVICES[device_name]
    status = _retry(lambda: rpc(ip, "Shelly.GetStatus"))
    if not isinstance(status, dict):
        return {"error": status}
    light = status.get("light:0", {}) if isinstance(status.get("light:0", {}), dict) else {}
    if source.lower() in ("em", "emeter", "meter"):
        em = status.get("em:0", {}) if isinstance(status.get("em:0", {}), dict) else {}
        apower = em.get("apower") if em.get("apower") is not None else light.get("apower")
        voltage = em.get("voltage") if em.get("voltage") is not None else light.get("voltage")
        current = em.get("current") if em.get("current") is not None else light.get("current")
    else:
        apower = light.get("apower")
        voltage = light.get("voltage")
        current = light.get("current")

    return {
        "on": light.get("output"),
        "brightness": light.get("brightness"),
        "apower": apower,
        "voltage": voltage,
        "current": current,
        "temperature_C": (light.get("temperature") or {}).get("tC"),
        "raw": status,
    }


def ensure_other_off(current_device: str):
    for name in DEVICES.keys():
        if name != current_device:
            set_brightness(name, 0)


def sweep(device_order, start=0, stop=100, step=1, settle_sec=3.0, out_dir: str = ".", *,
          source: str = "em", samples: int = 3, sample_interval: float = 0.5, transition_ms: int = 0):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir) / f"sweep_pwm_{timestamp}.csv"

    header = [
        "time",
        "device",
        "pwm",
        "source",
        "samples",
        "on",
        "brightness",
        "apower",
        "voltage",
        "current",
        "temperature_C",
        "error",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"# Sweep start @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        writer.writerow(header)

        for device in device_order:
            ensure_other_off(device)
            for pwm in range(int(start), int(stop) + 1, int(step)):
                set_res = set_brightness(device, pwm, transition_ms)
                time.sleep(settle_sec)

                ok_samples = []
                last_err = None
                for _ in range(max(1, int(samples))):
                    stat = get_status(device, source)
                    if isinstance(stat, dict) and not stat.get("error"):
                        ok_samples.append(stat)
                    else:
                        last_err = stat.get("error") if isinstance(stat, dict) else str(stat)
                    time.sleep(max(0.0, float(sample_interval)))

                if ok_samples:
                    # 平均功率、电压、电流，brightness 取最后一次
                    apower_avg = sum(s.get("apower") or 0.0 for s in ok_samples) / len(ok_samples)
                    voltage_avg = sum(s.get("voltage") or 0.0 for s in ok_samples) / len(ok_samples)
                    current_avg = sum(s.get("current") or 0.0 for s in ok_samples) / len(ok_samples)
                    last_stat = ok_samples[-1]
                    writer.writerow([
                        datetime.now().isoformat(),
                        device,
                        pwm,
                        source,
                        len(ok_samples),
                        last_stat.get("on"),
                        last_stat.get("brightness"),
                        round(apower_avg, 3),
                        round(voltage_avg, 3),
                        round(current_avg, 3),
                        last_stat.get("temperature_C"),
                        (set_res.get("error") if isinstance(set_res, dict) else None) or None,
                    ])
                else:
                    # 全部失败，仍写一行但包含错误，避免空洞
                    writer.writerow([
                        datetime.now().isoformat(),
                        device,
                        pwm,
                        source,
                        0,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        last_err or (set_res.get("error") if isinstance(set_res, dict) else None),
                    ])
                f.flush()

        writer.writerow([f"# Sweep stop @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])

    return str(out_path)


def parse_args(argv):
    # 用法: python3 sweep_pwm.py [Red|Blue|Both] [start] [stop] [step] [settle_sec] [out_dir] [source] [samples] [sample_interval] [transition_ms]
    device_arg = argv[1] if len(argv) >= 2 else "Both"
    start = int(argv[2]) if len(argv) >= 3 else 0
    stop = int(argv[3]) if len(argv) >= 4 else 100
    step = int(argv[4]) if len(argv) >= 5 else 1
    try:
        settle = float(argv[5]) if len(argv) >= 6 else 3.0
    except ValueError:
        settle = 3.0
    out_dir = argv[6] if len(argv) >= 7 else "."
    source = argv[7] if len(argv) >= 8 else "em"
    try:
        samples = int(argv[8]) if len(argv) >= 9 else 3
    except ValueError:
        samples = 3
    try:
        sample_interval = float(argv[9]) if len(argv) >= 10 else 0.5
    except ValueError:
        sample_interval = 0.5
    try:
        transition_ms = int(argv[10]) if len(argv) >= 11 else 0
    except ValueError:
        transition_ms = 0

    if device_arg.lower() == "red":
        order = ["Red"]
    elif device_arg.lower() == "blue":
        order = ["Blue"]
    else:
        # 先红后蓝
        order = ["Red", "Blue"]

    return order, start, stop, step, settle, out_dir, source, samples, sample_interval, transition_ms


def main():
    order, start, stop, step, settle, out_dir, source, samples, sample_interval, transition_ms = parse_args(sys.argv)
    out = sweep(order, start, stop, step, settle, out_dir, source=source, samples=samples, sample_interval=sample_interval, transition_ms=transition_ms)
    print(f"结果已写出: {out}")


if __name__ == "__main__":
    main()


