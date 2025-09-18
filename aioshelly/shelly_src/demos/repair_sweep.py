import csv
import os
import sys
import tempfile
from pathlib import Path
from collections import defaultdict
import subprocess


def read_csv_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows


def write_csv_rows(path: str, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def extract_header_idx(rows):
    start_idx = 0
    if rows and rows[0] and rows[0][0].startswith("# "):
        start_idx = 1
    headers = rows[start_idx]
    data_rows = rows[start_idx + 1 :]
    return start_idx, headers, data_rows


def contiguous_ranges(sorted_values):
    if not sorted_values:
        return []
    ranges = []
    start = prev = sorted_values[0]
    for v in sorted_values[1:]:
        if v == prev + 1:
            prev = v
        else:
            ranges.append((start, prev))
            start = prev = v
    ranges.append((start, prev))
    return ranges


def build_key(device: str, pwm: int):
    return f"{device}__{pwm}"


def run_sweep(device: str, start: int, stop: int, *, settle_sec: float, out_dir: str, source: str, samples: int, sample_interval: float, transition_ms: int) -> Path:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("sweep_pwm.py")),
        device,
        str(start),
        str(stop),
        "1",
        str(settle_sec),
        out_dir,
        source,
        str(samples),
        str(sample_interval),
        str(transition_ms),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    # 输出形如: 结果已写出: sweep_pwm_YYYYMMDD_HHMMSS.csv
    out_line = (proc.stdout or "").strip().splitlines()[-1]
    out_file = out_line.split(":", 1)[-1].strip()
    p = Path(out_file)
    return p if p.is_absolute() else Path(os.getcwd()) / p


def load_sweep_map(path: Path):
    rows = read_csv_rows(str(path))
    _, headers, data_rows = extract_header_idx(rows)
    h = {k: i for i, k in enumerate(headers)}
    result = {}
    max_needed_idx = max(h.values()) if h else 0

    def safe_get(row, col):
        idx = h.get(col)
        if idx is None:
            return ""
        if idx >= len(row):
            return ""
        return row[idx]
    for r in data_rows:
        if not r:
            continue
        if r[0].startswith("# "):
            continue
        if len(r) <= max_needed_idx:
            continue
        device = r[h["device"]]
        pwm_str = r[h["pwm"]]
        if pwm_str == "":
            continue
        try:
            pwm = int(float(pwm_str))
        except Exception:
            continue
        result[build_key(device, pwm)] = r
    return headers, result


def repair(input_csv: str, *, settle_sec: float = 5.0, source: str = "em", samples: int = 7, sample_interval: float = 0.5, transition_ms: int = 0) -> str:
    rows = read_csv_rows(input_csv)
    start_idx, headers, data_rows = extract_header_idx(rows)
    h = {k: i for i, k in enumerate(headers)}
    if "device" not in h or "pwm" not in h:
        raise ValueError("输入 CSV 缺少必要列: device/pwm")

    # 支持已标注文件：如果存在 anomaly 列，则只挑选 anomaly 非空行；否则选择 error 非空或明显异常的行
    anomaly_idx = h.get("anomaly")
    error_idx = h.get("error")

    # 本函数内也定义一次工具以避免作用域问题
    max_needed_idx = max(h.values()) if h else 0

    def safe_get(row, col):
        idx = h.get(col)
        if idx is None:
            return ""
        if idx >= len(row):
            return ""
        return row[idx]

    device_to_pwms = defaultdict(set)
    for r in data_rows:
        if not r:
            continue
        device = safe_get(r, "device")
        pwm_str = safe_get(r, "pwm")
        try:
            pwm = int(float(pwm_str))
        except Exception:
            continue
        take = False
        if anomaly_idx is not None and anomaly_idx < len(r):
            take = bool(r[anomaly_idx])
        else:
            # fallback: error 非空或 voltage 不在合理范围
            err = (r[error_idx] if error_idx is not None and error_idx < len(r) else "")
            if err:
                take = True
            else:
                v_str = safe_get(r, "voltage")
                try:
                    v = float(v_str)
                    if v < 200 or v > 260:
                        take = True
                except Exception:
                    pass
        if take:
            device_to_pwms[device].add(pwm)

    if not device_to_pwms:
        # 无需修复，直接返回
        return input_csv

    # 为每个设备生成连续区间并重扫
    merged_map = {}
    sweep_headers = None
    with tempfile.TemporaryDirectory() as tmpdir:
        for device, pwm_set in device_to_pwms.items():
            ranges = contiguous_ranges(sorted(pwm_set))
            for s, e in ranges:
                out_csv = run_sweep(
                    device,
                    s,
                    e,
                    settle_sec=settle_sec,
                    out_dir=tmpdir,
                    source=source,
                    samples=samples,
                    sample_interval=sample_interval,
                    transition_ms=transition_ms,
                )
                sweep_headers, sweep_map = load_sweep_map(out_csv)
                merged_map.update(sweep_map)

    # 用新结果替换原行
    out_rows = rows[: start_idx + 1]  # 保留注释与表头
    for r in data_rows:
        if not r:
            continue
        if r[0].startswith("# "):
            continue
        if len(r) <= max_needed_idx:
            out_rows.append(r)
            continue
        device = safe_get(r, "device")
        try:
            pwm = int(float(safe_get(r, "pwm")))
        except Exception:
            out_rows.append(r)
            continue
        key = build_key(device, pwm)
        if key in merged_map:
            out_rows.append(merged_map[key])
        else:
            out_rows.append(r)

    # 写出临时修复文件
    repaired_path = str(Path(input_csv).with_name(Path(input_csv).stem + "_repaired.csv"))
    write_csv_rows(repaired_path, out_rows)

    # 调用标注脚本生成 _repaired_annotated.csv
    ann_script = Path(__file__).with_name("annotate_sweep_anomalies.py")
    ann_out = str(Path(repaired_path).with_name(Path(repaired_path).stem + "_annotated.csv"))
    subprocess.run([sys.executable, str(ann_script), repaired_path, ann_out], check=False)

    return repaired_path


def main():
    if len(sys.argv) < 2:
        print("用法: python3 repair_sweep.py <input_csv> [source em|light] [settle_sec] [samples] [sample_interval] [transition_ms]")
        sys.exit(1)
    input_csv = sys.argv[1]
    source = sys.argv[2] if len(sys.argv) >= 3 else "em"
    try:
        settle = float(sys.argv[3]) if len(sys.argv) >= 4 else 5.0
    except Exception:
        settle = 5.0
    try:
        samples = int(sys.argv[4]) if len(sys.argv) >= 5 else 7
    except Exception:
        samples = 7
    try:
        sample_interval = float(sys.argv[5]) if len(sys.argv) >= 6 else 0.5
    except Exception:
        sample_interval = 0.5
    try:
        transition_ms = int(sys.argv[6]) if len(sys.argv) >= 7 else 0
    except Exception:
        transition_ms = 0

    out = repair(input_csv, settle_sec=settle, source=source, samples=samples, sample_interval=sample_interval, transition_ms=transition_ms)
    print(f"已修复写出: {out}")


if __name__ == "__main__":
    main()


