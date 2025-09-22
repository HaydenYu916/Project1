import csv
import sys
import time
from pathlib import Path

from shelly_controller import rpc, DEVICES


DEFAULT_INPUT = "Parsed_log_table.csv"
DEFAULT_OUTPUT = "Parsed_log_table_filled.csv"


def normalize_split(line: str) -> list[str]:
    """将一行使用英文逗号和中文逗号同时切分，并去除空白。"""
    # 统一分隔符
    unified = line.replace("，", ",")
    parts = [p.strip() for p in unified.split(",")]
    # 去空元素（结尾逗号导致的空串保留一位用于占位）
    return parts


def parse_row(parts: list[str]):
    """解析一行，返回 (key, ppfd, r_pwm, b_pwm)。不够字段时返回 None。"""
    if len(parts) < 4:
        return None
    key = parts[0]
    try:
        ppfd = float(parts[1]) if parts[1] != "" else None
    except ValueError:
        ppfd = None
    try:
        r_pwm = float(parts[2]) if parts[2] != "" else None
    except ValueError:
        r_pwm = None
    try:
        b_pwm = float(parts[3]) if parts[3] != "" else None
    except ValueError:
        b_pwm = None
    return key, ppfd, r_pwm, b_pwm


def _retry(callable_fn, max_attempts=3, base_delay=0.3):
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return callable_fn()
        except Exception as e:
            last_err = str(e)
        time.sleep(base_delay * (2 ** (attempt - 1)))
    return {"error": last_err or "unknown"}


def set_device_brightness(device_name: str, pwm_value: float | None):
    ip = DEVICES[device_name]
    if pwm_value is None:
        return {"skipped": True}
    value = max(0, min(100, int(round(pwm_value))))
    if value <= 0:
        # 亮度为 0 时，尝试直接关断
        return _retry(lambda: rpc(ip, "Light.Set", {"id": 0, "on": False, "brightness": 0}))
    # 设为开并设置亮度
    return _retry(lambda: rpc(ip, "Light.Set", {"id": 0, "on": True, "brightness": value}))


def get_device_status(device_name: str):
    ip = DEVICES[device_name]
    status = _retry(lambda: rpc(ip, "Shelly.GetStatus"))
    if not isinstance(status, dict):
        return {"error": status}
    light = status.get("light:0", {}) if isinstance(status.get("light:0", {}), dict) else {}
    return {
        "on": light.get("output"),
        "brightness": light.get("brightness"),
        "apower": light.get("apower"),
        "voltage": light.get("voltage"),
        "current": light.get("current"),
        "temperature_C": (light.get("temperature") or {}).get("tC"),
        "raw": status,
    }


def process_file(input_path: str, output_path: str, settle_sec: float = 1.0):
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    rows_out: list[list] = []
    # 输出表头
    header = [
        "KEY",
        "PPFD",
        "R_PWM",
        "B_PWM",
        "R_POWER",
        "B_POWER",
        "R_VOLTAGE",
        "B_VOLTAGE",
        "R_CURRENT",
        "B_CURRENT",
        "R_ON",
        "B_ON",
        "R_TEMP_C",
        "B_TEMP_C",
        "R_ERROR",
        "B_ERROR",
    ]
    rows_out.append(header)

    with open(input_path, "r", encoding="utf-8") as f:
        # 逐行读取，手动解析，兼容中英文逗号
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 跳过可能的表头行（包含字段名）
            if any(x in line for x in ["PPFD", "PWM", "VOLTAGE", "CURRENT", "POWER"]):
                continue
            parts = normalize_split(line)
            parsed = parse_row(parts)
            if not parsed:
                continue
            key, ppfd, r_pwm, b_pwm = parsed

            # 设置亮度
            r_set = set_device_brightness("Red", r_pwm)
            b_set = set_device_brightness("Blue", b_pwm)

            # 等待稳定
            time.sleep(settle_sec)

            # 读取状态
            r_stat = get_device_status("Red")
            b_stat = get_device_status("Blue")

            r_power = r_stat.get("apower") if isinstance(r_stat, dict) else None
            b_power = b_stat.get("apower") if isinstance(b_stat, dict) else None
            r_voltage = r_stat.get("voltage") if isinstance(r_stat, dict) else None
            b_voltage = b_stat.get("voltage") if isinstance(b_stat, dict) else None
            r_current = r_stat.get("current") if isinstance(r_stat, dict) else None
            b_current = b_stat.get("current") if isinstance(b_stat, dict) else None

            r_on = r_stat.get("on") if isinstance(r_stat, dict) else None
            b_on = b_stat.get("on") if isinstance(b_stat, dict) else None
            r_temp = r_stat.get("temperature_C") if isinstance(r_stat, dict) else None
            b_temp = b_stat.get("temperature_C") if isinstance(b_stat, dict) else None

            rows_out.append([
                key,
                ppfd,
                r_pwm,
                b_pwm,
                r_power,
                b_power,
                r_voltage,
                b_voltage,
                r_current,
                b_current,
                r_on,
                b_on,
                r_temp,
                b_temp,
                (r_set.get("error") if isinstance(r_set, dict) else None) or (r_stat.get("error") if isinstance(r_stat, dict) else None),
                (b_set.get("error") if isinstance(b_set, dict) else None) or (b_stat.get("error") if isinstance(b_stat, dict) else None),
            ])

    # 写出 CSV（使用英文逗号），UTF-8 编码
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows_out)

    return output_path


def main():
    input_path = DEFAULT_INPUT
    output_path = DEFAULT_OUTPUT
    settle_sec = 1.0

    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    if len(sys.argv) >= 4:
        try:
            settle_sec = float(sys.argv[3])
        except ValueError:
            pass

    out = process_file(input_path, output_path, settle_sec)
    print(f"写出完成: {out}")


if __name__ == "__main__":
    main()


