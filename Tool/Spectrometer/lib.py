"""PSM-60s 光谱仪控制与数据处理库。

该模块包含原来 ``PSM_60s.py`` 中的核心功能，可被其它脚本或
单元测试导入使用。

示例用法：

    import serial
    from Collect_Sp_PPFD_LED import get_csv_filename, complete_spectrum_measurement

    ser = serial.Serial(port, baudrate=115200, timeout=1)
    csv_file = get_csv_filename()
    result, total_time, integration = complete_spectrum_measurement(ser)

"""

try:
    import serial
except ImportError:  # serial module is optional for offline use
    serial = None

import time
import datetime
import os
import struct
import csv
import glob
import re
import base64


STANDARD_METRIC_LAYOUT: list[tuple[str, int]] = [
    ("PAR(mW/cm²)", 1),
    ("PPFD(umol/㎡/s)", 2),
    ("PPFD-UV(umol/㎡/s)", 3),
    ("PPFD-B(umol/㎡/s)", 4),
    ("PPFD-G(umol/㎡/s)", 5),
    ("PPFD-R(umol/㎡/s)", 6),
    ("PPFD-FR(umol/㎡/s)", 7),
    ("PPFD-IR(umol/㎡/s)", 8),
    ("Kppfv(umol/s/klm)", 9),
    ("Erb Ratio", 10),
    ("YPFD(umol/m²/s)", 11),
    ("Ech-A(mW/cm²)", 12),
    ("Ech-B(mW/cm²)", 13),
    ("DLI(mol/m²d)", 14),
    ("CLI(mol/m²)", 15),
    ("E(lx)", 16),
    ("Candle E(fc)", 17),
    ("CCT(K)", 18),
    ("Duv", 19),
    ("x", 20),
    ("y", 21),
    ("u", 22),
    ("v", 23),
    ("u’", 24),
    ("v’", 25),
    ("SDCM", 26),
    ("Ra", 27),
    ("R1", 28),
    ("R2", 29),
    ("R3", 30),
    ("R4", 31),
    ("R5", 32),
    ("R6", 33),
    ("R7", 34),
    ("R8", 35),
    ("R9", 36),
    ("R10", 37),
    ("R11", 38),
    ("R12", 39),
    ("R13", 40),
    ("R14", 41),
    ("R15", 42),
    ("Ee(mW/cm²)", 43),
    ("S/P", 44),
    ("Dominant(nm)", 45),
    ("Purity(%)", 46),
    ("Half width(nm)", 47),
    ("Peak(nm)", 48),
    ("Center(nm)", 49),
    ("Centroid(nm)", 50),
    ("R ratio(%)", 51),
    ("G ratio(%)", 52),
    ("B ratio(%)", 53),
    ("Integrat.Time(ms)", 59),
    ("Peak Signal", 60),
    ("Dark Signal", 61),
    ("Compensate level", 62),
]

SUMMARY_METRIC_REMAP: list[tuple[str, str]] = [
    ("PAR(mW/cm2)", "PAR(mW/cm²)"),
    ("PPFD(umol/m2/s)", "PPFD(umol/㎡/s)"),
    ("PPFD-UV(umol/m2/s)", "PPFD-UV(umol/㎡/s)"),
    ("PPFD-B(umol/m2/s)", "PPFD-B(umol/㎡/s)"),
    ("PPFD-G(umol/m2/s)", "PPFD-G(umol/㎡/s)"),
    ("PPFD-R(umol/m2/s)", "PPFD-R(umol/㎡/s)"),
    ("PPFD-FR(umol/m2/s)", "PPFD-FR(umol/㎡/s)"),
    ("PPFD-IR(umol/m2/s)", "PPFD-IR(umol/㎡/s)"),
    ("Integrat.Time(ms)", "Integrat.Time(ms)"),
]

SUMMARY_SAMPLE_WAVELENGTHS: list[int] = [415, 445, 480, 515, 555, 590, 630, 680]

SUMMARY_PARAM_NAMES: list[str] = [
    "PAR(mW/cm2)",
    "PPFD(umol/m2/s)",
    "PPFD-UV(umol/m2/s)",
    "PPFD-B(umol/m2/s)",
    "PPFD-G(umol/m2/s)",
    "PPFD-R(umol/m2/s)",
    "PPFD-FR(umol/m2/s)",
    "PPFD-IR(umol/m2/s)",
    "Integrat.Time(ms)",
    "Test Date",
    "415nm",
    "445nm",
    "480nm",
    "515nm",
    "555nm",
    "590nm",
    "630nm",
    "680nm",
    "StartTestWave",
    "EndTestWave",
]

SUMMARY_UNITS: list[str] = [
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "intensity",
    "intensity",
    "intensity",
    "intensity",
    "intensity",
    "intensity",
    "intensity",
    "intensity",
    "nm",
    "nm",
]

_HEX_DUMP_LINE_RE = re.compile(
    r"^(?:[0-9A-Fa-f]{2})(?:\s+[0-9A-Fa-f]{2})*(?:\s+#.*)?$"
)

# Track serial sessions that already completed one-time initialization.
_SESSION_INITIALIZED: set[int] = set()


def _is_session_initialized(ser) -> bool:
    """Return whether serial session has completed one-time init."""
    if bool(getattr(ser, "_growpro_session_initialized", False)):
        return True
    return id(ser) in _SESSION_INITIALIZED


def _mark_session_initialized(ser) -> None:
    """Mark serial session as initialized (attribute-first, set fallback)."""
    try:
        setattr(ser, "_growpro_session_initialized", True)
    except Exception:
        _SESSION_INITIALIZED.add(id(ser))


def _parse_version_string(version_bytes: bytes) -> str:
    """Parse device version bytes into ``major.minor.patch`` string."""
    if len(version_bytes) < 2:
        return ""
    version_code = int.from_bytes(version_bytes[:2], byteorder="little")
    major = version_code // 1000
    minor = (version_code % 1000) // 10
    patch = version_code % 10
    return f"{major}.{minor:02d}.{patch:02d}"


def _infer_full_spectrum_points(model: str) -> int | None:
    """Infer full-spectrum point count from model family when known.

    根据协议文档：
    - 300/310/320/330 系列返回 671 点（380-1050nm）
    - 250/350 系列返回 901 点
    """
    model_upper = model.upper()
    if any(tag in model_upper for tag in ("-300", "-310", "-320", "-330")):
        return 671
    if any(tag in model_upper for tag in ("-250", "-350")):
        return 901
    return None


def _extract_raw_bytes_from_text_dump(filename: str) -> bytes:
    """Extract real packet bytes from a human-readable hex dump text file."""
    hex_tokens: list[str] = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or not _HEX_DUMP_LINE_RE.match(stripped):
                continue
            line_without_comment = stripped.split("#", 1)[0].strip()
            hex_tokens.extend(line_without_comment.split())

    if not hex_tokens:
        raise ValueError(f"未从文件提取到十六进制字节: {filename}")
    return bytes(int(token, 16) for token in hex_tokens)


def _parse_test_datetime(payload: bytes) -> str:
    """Parse timestamp bytes from payload into display text."""
    ts_start = 62 * 4
    ts_end = ts_start + 20
    if len(payload) < ts_end:
        return ""

    ts_raw = payload[ts_start:ts_end].decode("ascii", errors="ignore")
    parts = [part for part in ts_raw.split("\x00") if part]
    if not parts:
        return ""

    if len(parts) >= 2:
        date_part, time_part = parts[0], parts[1]
        dt_text = f"{date_part} {time_part}"
        for in_fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                dt = datetime.datetime.strptime(dt_text, in_fmt)
                return dt.strftime("%Y/%m/%d %H:%M")
            except ValueError:
                continue
        return dt_text

    return parts[0]


def _decode_measurement_from_raw_bytes(raw_data: bytes) -> dict:
    """Decode one spectrum packet into standard metrics and spectrum."""
    if len(raw_data) < 40:
        raise ValueError(f"数据长度不足，无法解析: {len(raw_data)}")

    if raw_data[0] != 0x8C:
        raise ValueError(f"帧头错误: 期望 0x8C，实际 0x{raw_data[0]:02X}")
    if raw_data[1] != 0x13:
        raise ValueError(f"功能码错误: 期望 0x13，实际 0x{raw_data[1]:02X}")

    data_length = int.from_bytes(raw_data[2:4], byteorder="big")
    expected_total = data_length + 4
    if len(raw_data) != expected_total:
        raise ValueError(f"数据包长度不匹配: 预期 {expected_total} 字节，实际 {len(raw_data)} 字节")

    payload = raw_data[40:expected_total]
    if len(payload) < (62 * 4 + 20 + 8):
        raise ValueError("负载长度不足，无法解析参数和波长范围")

    model = raw_data[4:14].decode("ascii", errors="ignore").rstrip("\x00")
    version = _parse_version_string(raw_data[14:16])
    test_date = _parse_test_datetime(payload)

    float_values: list[float] = []
    for offset in range(0, 62 * 4, 4):
        float_values.append(struct.unpack("<f", payload[offset:offset + 4])[0])

    metrics: dict[str, float] = {}
    for field_name, field_idx in STANDARD_METRIC_LAYOUT:
        if 1 <= field_idx <= len(float_values):
            metrics[field_name] = float_values[field_idx - 1]
        else:
            metrics[field_name] = 0.0

    start_wave = int(round(struct.unpack("<f", payload[-8:-4])[0]))
    end_wave = int(round(struct.unpack("<f", payload[-4:])[0]))
    if start_wave <= 0 or end_wave < start_wave:
        raise ValueError(f"异常波长范围: start={start_wave}, end={end_wave}")

    spectrum_start = 62 * 4 + 20
    expected_points = end_wave - start_wave + 1
    spectrum_bytes = len(payload) - 8 - spectrum_start
    if spectrum_bytes < 0:
        raise ValueError("负载长度不足，无法解析光谱数据段")
    if spectrum_bytes % 4 != 0:
        raise ValueError(f"光谱数据段字节数异常: {spectrum_bytes}（非4字节对齐）")
    available_points = spectrum_bytes // 4
    if available_points < expected_points:
        raise ValueError(
            f"光谱点数不足: 测试范围需要 {expected_points} 点，实际仅 {available_points} 点"
        )

    start_index = 0
    if available_points > expected_points:
        full_points = _infer_full_spectrum_points(model)
        if full_points is None:
            raise ValueError(
                "光谱点数大于测试范围，且机型未知，无法安全确定光谱对齐规则"
            )
        if available_points != full_points:
            raise ValueError(
                f"光谱点数与机型不匹配: 机型 {model} 期望满谱 {full_points} 点，"
                f"实际 {available_points} 点"
            )
        # 已知 300/310/320/330 系列为 380-1050nm 满谱返回。
        if full_points == 671:
            full_start_wave = 380
            start_index = start_wave - full_start_wave
            if start_index < 0 or (start_index + expected_points) > available_points:
                raise ValueError(
                    f"测试范围无法在满谱中定位: start={start_wave}, end={end_wave}, "
                    f"满谱起点={full_start_wave}, 点数={available_points}"
                )
        else:
            raise ValueError(
                f"机型 {model} 的满谱映射规则未实现，无法安全截取测试范围"
            )

    spectrum: list[tuple[int, float]] = []
    for i in range(expected_points):
        base = spectrum_start + (start_index + i) * 4
        value = struct.unpack("<f", payload[base:base + 4])[0]
        spectrum.append((start_wave + i, value))

    return {
        "model": model,
        "version": version,
        "test_date": test_date,
        "metrics": metrics,
        "start_wave": start_wave,
        "end_wave": end_wave,
        "spectrum": spectrum,
    }


def _write_standard_csv(parsed: dict, csv_path: str) -> None:
    """Write decoded measurement in the same layout as example/580.csv."""
    rows: list[list[object]] = [
        ["Model", parsed["model"]],
        ["Version", parsed["version"]],
        ["Mark"],
        ["Test Date", parsed["test_date"]],
    ]

    metrics: dict[str, float] = parsed["metrics"]
    for field_name, _ in STANDARD_METRIC_LAYOUT:
        rows.append([field_name, metrics.get(field_name, 0.0)])

    rows.append(["StartTestWave", parsed["start_wave"]])
    rows.append(["EndTestWave", parsed["end_wave"]])
    for wavelength, intensity in parsed["spectrum"]:
        rows.append([wavelength, intensity])

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)


def export_standard_csv_from_raw_file(
    txt_filename: str,
    output_dir: str | None = None,
) -> str:
    """Convert ``raw_bytes_*.txt`` into a standard full CSV format file."""
    parsed = _decode_measurement_from_raw_bytes(
        _extract_raw_bytes_from_text_dump(txt_filename)
    )

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(txt_filename)),
            "standard_csv",
        )
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(txt_filename))[0] + ".csv"
    csv_path = os.path.join(output_dir, base)
    _write_standard_csv(parsed, csv_path)
    return csv_path


def _default_standard_csv_dir(now: datetime.datetime | None = None) -> str:
    """Return default output directory for standard CSV exports."""
    if now is None:
        now = datetime.datetime.now()
    return os.path.join("archive", now.strftime("%F"), "standard_csv")


def _build_unique_csv_path(output_dir: str, basename: str) -> str:
    """Build a non-conflicting CSV path under *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    base_stem, base_ext = os.path.splitext(basename)
    if not base_ext:
        base_ext = ".csv"

    candidate = os.path.join(output_dir, f"{base_stem}{base_ext}")
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(output_dir, f"{base_stem}_{counter:02d}{base_ext}")
        counter += 1
    return candidate


def export_standard_csv_from_raw_bytes(
    raw_data: bytes,
    output_dir: str | None = None,
    basename: str | None = None,
) -> tuple[str, dict]:
    """Convert one packet of raw bytes into standard CSV, return path+parsed."""
    parsed = _decode_measurement_from_raw_bytes(raw_data)
    now = datetime.datetime.now()
    if output_dir is None:
        output_dir = _default_standard_csv_dir(now)
    if basename is None:
        basename = f"raw_bytes_{now.strftime('%Y%m%d_%H%M%S')}.csv"

    csv_path = _build_unique_csv_path(output_dir, basename)
    _write_standard_csv(parsed, csv_path)
    return csv_path, parsed


def _extract_summary_data_from_parsed(parsed: dict) -> dict[str, float | str]:
    """Extract summary fields (used by daily spectrum_data CSV) from parsed data."""
    metrics: dict[str, float] = parsed["metrics"]
    spectrum_map = dict(parsed["spectrum"])

    extracted_data: dict[str, float | str] = {
        "Test Date": parsed.get("test_date", ""),
        "StartTestWave": float(parsed.get("start_wave", 0.0)),
        "EndTestWave": float(parsed.get("end_wave", 0.0)),
    }
    for summary_name, standard_name in SUMMARY_METRIC_REMAP:
        extracted_data[summary_name] = metrics.get(standard_name, 0.0)
    for wavelength in SUMMARY_SAMPLE_WAVELENGTHS:
        extracted_data[f"{wavelength}nm"] = spectrum_map.get(wavelength, 0.0)
    return extracted_data


def append_parsed_measurement_to_csv(
    parsed: dict,
    csv_filename: str,
    source_file: str | None = None,
) -> bool:
    """Append parsed measurement directly to daily summary CSV."""
    if not os.path.exists(csv_filename):
        print(f"❌ CSV文件不存在: {csv_filename}")
        return False

    existing_keys = _load_existing_record_keys(csv_filename)
    if source_file:
        record_key = _normalize_record_key(source_file)
    else:
        now = datetime.datetime.now()
        record_key = f"live_{now.strftime('%Y%m%d_%H%M%S_%f')}"

    if record_key in existing_keys:
        print(f"🔁 文件已记录，跳过汇总CSV: {record_key}")
        return False

    extracted_data = _extract_summary_data_from_parsed(parsed)
    row = [record_key]
    for param_name in SUMMARY_PARAM_NAMES:
        row.append(extracted_data.get(param_name, 0.0))

    with open(csv_filename, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)
    print(f"✅ 实时测量已写入汇总CSV: {csv_filename}")
    return True


def send_command(ser, cmd_hex: str):
    """发送十六进制命令到设备。

    ``ser`` 应为类似串口的对象，支持 ``write`` 和 ``in_waiting``
    属性。
    """
    cmd = bytes.fromhex(cmd_hex)
    ser.write(cmd)
    print(f"📤 命令已发送：{cmd_hex.upper()}")


def clear_serial_buffers(ser) -> None:
    """Best-effort clear of serial RX/TX buffers before a protocol step."""
    cleared_rx = 0
    try:
        reset_in = getattr(ser, "reset_input_buffer", None)
        if callable(reset_in):
            waiting = getattr(ser, "in_waiting", 0) or 0
            reset_in()
            cleared_rx = int(waiting)
        else:
            waiting = getattr(ser, "in_waiting", 0) or 0
            if waiting:
                cleared_rx = len(ser.read(waiting))
    except Exception as exc:
        print(f"⚠️ 清空输入缓冲失败: {exc}")

    try:
        reset_out = getattr(ser, "reset_output_buffer", None)
        if callable(reset_out):
            reset_out()
    except Exception as exc:
        print(f"⚠️ 清空输出缓冲失败: {exc}")

    if cleared_rx:
        print(f"🧹 已清空串口输入缓冲: {cleared_rx} 字节")


def read_response(ser, timeout=1):
    """从串口（或类串口对象）读取所有可用数据。"""
    response = ser.read(ser.in_waiting or 1024)
    if response:
        print(f"📥 接收响应: {len(response)} 字节 - {response.hex(' ').upper()}")
    return response


def _decode_b64_token(token: bytes) -> bytes | None:
    """Decode urlsafe base64 token used by gateway-wrapped responses."""
    try:
        text = token.decode("ascii", errors="ignore").strip()
        if not text:
            return None
        text += "=" * (-len(text) % 4)
        return base64.urlsafe_b64decode(text)
    except Exception:
        return None


def _parse_status_response(response: bytes) -> tuple[int | None, int | None, int | None]:
    """Parse status tuple from direct serial or gateway-wrapped 8C 03 response.

    Returns ``(data_status, test_status, test_mode)``.
    """
    if not response:
        return None, None, None

    # Direct device response.
    if len(response) >= 4 and response[0] == 0x8C:
        data_status = response[2]
        test_status = response[3]
        test_mode = response[5] if len(response) >= 6 else None
        return data_status, test_status, test_mode

    # Gateway wrapped format:
    # [<dev_b64>\0<cmd_b64>\0<status_b64>\0<payload_b64>\0]
    if response.startswith(b"[") and response.endswith(b"]"):
        body = response[1:-1]
        tokens = [tok for tok in body.split(b"\x00") if tok]
        decoded = [_decode_b64_token(tok) for tok in tokens]
        if len(decoded) >= 3 and decoded[2] and len(decoded[2]) >= 2:
            data_status = decoded[2][0]
            test_status = decoded[2][1]
            test_mode = None
            if len(decoded) >= 4 and decoded[3] and len(decoded[3]) >= 2:
                # payload[1] is often mode-like marker on current firmware
                test_mode = decoded[3][1]
            return data_status, test_status, test_mode

    return None, None, None


def save_raw_bytes_to_text(raw_data: bytes, filename=None):
    """将原始字节数据保存为可阅读的十六进制文本文件。

    格式与原始脚本相同：前几行分块、时间戳合并、带标签的
    波长数据等。
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"raw_bytes_{timestamp}.txt"

    with open(filename, 'w') as f:
        # header
        f.write(f"原始字节数据 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据长度: {len(raw_data)} 字节\n")
        f.write("=" * 50 + "\n\n")

        # first three lines are fixed lengths
        if len(raw_data) >= 14:
            first_line = raw_data[0:14].hex(' ').upper()
            f.write(f"{first_line}\n")
        if len(raw_data) >= 16:
            second_line = raw_data[14:16].hex(' ').upper()
            f.write(f"{second_line}\n")
        if len(raw_data) >= 40:
            third_line = raw_data[16:40].hex(' ').upper()
            f.write(f"{third_line}\n")

        f.write("\n")

        remaining_data = raw_data[40:]
        line_count = 0
        i = 0

        start_wavelength = None
        end_wavelength = None
        if len(remaining_data) >= 8:
            start_bytes = remaining_data[-8:-4][::-1]
            end_bytes = remaining_data[-4:][::-1]
            start_wavelength = struct.unpack('>f', start_bytes)[0]
            end_wavelength = struct.unpack('>f', end_bytes)[0]

        spectrum_data_count = len(remaining_data) - 8 if len(remaining_data) >= 8 else len(remaining_data)
        spectrum_lines = spectrum_data_count // 4

        while i < len(remaining_data) - 8:
            chunk = remaining_data[i:i+4]
            hex_line = chunk.hex(' ').upper()
            if len(chunk) < 4:
                hex_line += '   ' * (4 - len(chunk))

            line_count += 1

            if 63 <= line_count <= 67:
                if line_count == 63:
                    timestamp_line = hex_line
                elif line_count == 67:
                    timestamp_line += " " + hex_line
                    f.write(f"{timestamp_line}\n")
                    f.write("\n")
                else:
                    timestamp_line += " " + hex_line
            else:
                if line_count > 67 and start_wavelength is not None and end_wavelength is not None:
                    data_point_index = line_count - 68
                    if data_point_index < spectrum_lines:
                        wavelength = int(start_wavelength) + data_point_index
                        if wavelength <= int(end_wavelength):
                            f.write(f"{hex_line}  # {wavelength}nm\n")
                        else:
                            f.write(f"{hex_line}\n")
                    else:
                        f.write(f"{hex_line}\n")
                else:
                    f.write(f"{hex_line}\n")

            i += 4

        # wavelength footer
        if len(remaining_data) >= 8:
            start_line = remaining_data[-8:-4].hex(' ').upper()
            f.write(f"{start_line}  # 起始扫描波长: {start_wavelength:.1f}nm\n")
        if len(remaining_data) >= 4:
            end_line = remaining_data[-4:].hex(' ').upper()
            f.write(f"{end_line}  # 终止扫描波长: {end_wavelength:.1f}nm\n")

    print(f"💾 原始字节数据已保存到: {filename}")
    return filename


def hex_to_float_little_endian(hex_str: str):
    """将十六进制字符串按小端格式转换为浮点数。"""
    hex_str = hex_str.replace(' ', '')
    bytes_data = bytes.fromhex(hex_str)
    return struct.unpack('<f', bytes_data)[0]


def hex_to_ascii(hex_str: str):
    """将十六进制字符串解码为 ASCII 文本，忽略无法解码的字节。"""
    hex_str = hex_str.replace(' ', '')
    bytes_data = bytes.fromhex(hex_str)
    try:
        return bytes_data.decode('ascii', errors='ignore').strip('\x00')
    except Exception:
        return ""


def process_single_raw_file(filename: str) -> dict:
    """从原始文本转储中提取汇总 CSV 所需的参数。"""
    parsed = _decode_measurement_from_raw_bytes(
        _extract_raw_bytes_from_text_dump(filename)
    )
    return _extract_summary_data_from_parsed(parsed)


def initialize_spectrometer_session(ser) -> bool:
    """Initialize spectrometer session once: connect + auto integration."""
    if _is_session_initialized(ser):
        print("\n1️⃣ 会话初始化：已完成，跳过联机/积分模式设置")
        return False

    print("\n1️⃣ 会话初始化（仅首次）...")
    clear_serial_buffers(ser)
    print("   - 联机")
    send_command(ser, "8C 00")
    time.sleep(0.1)
    read_response(ser)

    print("   - 设置积分方式为自动")
    send_command(ser, "8C 02 01")
    time.sleep(0.3)
    read_response(ser)
    _mark_session_initialized(ser)
    print("   ✅ 会话初始化完成")
    return True


def complete_spectrum_measurement(ser):
    """Perform the full measurement sequence and return the raw data.

    Returns a tuple ``(raw_bytes, filename)`` plus timing information in the
    caller's return value.
    """
    print("🚀 开始完整的光谱测量流程")
    measurement_start_time = time.time()

    # 1. 会话初始化：联机 + 自动积分（每个串口会话仅执行一次）
    initialize_spectrometer_session(ser)

    # 2. 单次采样
    print("\n2️⃣ 触发单次采样...")
    send_command(ser, "8C 0E 01")
    time.sleep(0.1)
    read_response(ser)

    # 3. 等待采样完成
    print("\n3️⃣ 轮询采样状态...")
    max_wait_time = max(5.0, float(os.getenv("GROWPRO_SPEC_STATUS_WAIT_SEC", "120")))
    poll_interval = max(0.1, float(os.getenv("GROWPRO_SPEC_STATUS_POLL_SEC", "1.0")))
    start_time = time.time()
    check_count = 0
    measurement_finished = False
    while time.time() - start_time < max_wait_time:
        send_command(ser, "8C 03")
        time.sleep(poll_interval)
        response = read_response(ser)
        check_count += 1
        if response:
            # 协议定义:
            # data_status -> 测试数据状态: 00已读/01未读(可取)
            # test_status -> 测试状态: 00测试中/01测试结束
            data_status, test_status, test_mode = _parse_status_response(response)
            if data_status is None or test_status is None:
                print(f"   第{check_count}次检查 - 状态包解析失败，继续等待...")
                continue

            mode_text = ""
            if test_mode is not None:
                mode_text = f", 测试模式: 0x{test_mode:02X}"

            print(
                f"   第{check_count}次检查 - 数据状态: 0x{data_status:02X}, "
                f"测试状态: 0x{test_status:02X}{mode_text}"
            )

            # 某些固件会在测试结束后一直返回 data_status=0x00，
            # 因此这里以“测试结束”为主判据，数据状态仅作提示。
            if test_status == 0x01:
                measurement_finished = True
                if data_status == 0x01:
                    print("   ✅ 采样完成且数据状态=未读，开始读取结果")
                else:
                    print("   ✅ 采样完成（数据状态=已读/未知），按兼容模式继续读取结果")
                break
            else:
                print("   ⏳ 采样进行中...")
            if check_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"   ⏳ 已等待 {elapsed:.1f} 秒，继续等待...")
        else:
            print("   ⏳ 等待采样完成...")
    if not measurement_finished:
        print("⚠️ 采样状态轮询超时，尝试继续读取数据...")
        print("   ↩️ 发送停止命令尝试复位设备状态...")
        send_command(ser, "8C 25")
        time.sleep(0.1)
        read_response(ser)

    # 4. 读取测试结果
    print("\n4️⃣ 读取测试结果...")
    total_measurement_time = time.time() - measurement_start_time
    print(f"⏱️ 总测量时间: {total_measurement_time:.1f} 秒")
    spectrum_result = read_spectrum_data(ser)
    actual_integration_time = None
    if isinstance(spectrum_result, tuple) and len(spectrum_result) >= 3:
        parsed = spectrum_result[2]
        if isinstance(parsed, dict):
            metrics = parsed.get("metrics", {})
            integration_ms = metrics.get("Integrat.Time(ms)")
            if isinstance(integration_ms, (int, float)):
                actual_integration_time = float(integration_ms) / 1000.0
                print(
                    f"   📊 结果包积分时间: {float(integration_ms):.0f}ms "
                    f"({actual_integration_time:.3f}秒)"
                )
    if isinstance(spectrum_result, tuple):
        return spectrum_result, total_measurement_time, actual_integration_time
    else:
        return spectrum_result, total_measurement_time, actual_integration_time


def read_spectrum_data(ser, save_raw_text: bool = False, max_retries: int = 2):
    """Pull spectrum bytes, validate strictly, and export standard CSV.

    严格校验项：
    1) 帧头/功能码必须为 ``8C 13``；
    2) 包长度必须与长度字段完全一致；
    3) 光谱点数必须与 ``StartTestWave/EndTestWave`` 完整匹配。

    校验失败会丢弃本次结果并自动重试读取（最多 ``max_retries`` 次重试）。
    """
    print("📊 开始读取光谱数据...")

    def _read_packet_once() -> bytes:
        clear_serial_buffers(ser)

        print("\n🔁 读取光谱数据")
        send_command(ser, "8C 13")

        raw_data = bytearray()
        expected_total = None
        read_start = time.time()
        last_rx = read_start
        total_timeout = 8.0
        idle_timeout = 0.5

        # 使用长度字段收包，避免固定延迟导致截断。
        while time.time() - read_start < total_timeout:
            waiting = ser.in_waiting
            if waiting > 0:
                chunk = ser.read(waiting)
                raw_data.extend(chunk)
                last_rx = time.time()

                if expected_total is None and len(raw_data) >= 4:
                    data_length = int.from_bytes(raw_data[2:4], byteorder='big')
                    expected_total = data_length + 4
                if expected_total is not None and len(raw_data) >= expected_total:
                    break
            else:
                if raw_data and (time.time() - last_rx) >= idle_timeout:
                    break
                time.sleep(0.05)

        print(f"📥 接收数据: {len(raw_data)} 字节")
        if len(raw_data) >= 4:
            data_length = int.from_bytes(raw_data[2:4], byteorder='big')
            expected_total = data_length + 4
            print(f"📏 数据长度字段: {data_length} 字节")
            print(f"📏 预期总包长度: {expected_total} 字节")
            print(f"📏 实际接收长度: {len(raw_data)} 字节")

        return bytes(raw_data)

    total_attempts = max(1, max_retries + 1)
    for attempt in range(1, total_attempts + 1):
        if total_attempts > 1:
            print(f"📡 光谱读取尝试 {attempt}/{total_attempts}")

        raw_bytes = _read_packet_once()
        if not raw_bytes:
            print("⚠️ 未接收到数据")
            if attempt < total_attempts:
                print("↩️ 本次读取失败，重试...")
                time.sleep(0.2)
                continue
            return bytearray(), None, None

        try:
            # 先做严格协议校验，校验通过才允许导出标准CSV。
            parsed = _decode_measurement_from_raw_bytes(raw_bytes)
        except Exception as e:
            print(f"❌ 光谱包校验失败: {e}")
            if len(raw_bytes) >= 20:
                print(f"   前20个字节: {' '.join(f'{b:02X}' for b in raw_bytes[:20])}")
            if attempt < total_attempts:
                print("↩️ 丢弃本次结果并重试读取...")
                time.sleep(0.2)
                continue
            print("❌ 达到重试上限，放弃本次测量结果")
            return bytearray(), None, None

        standard_csv_path = None
        try:
            standard_csv_path, parsed = export_standard_csv_from_raw_bytes(raw_bytes)
            print(f"\n📊 数据处理完成:")
            print(f"  标准CSV文件: {standard_csv_path}")
        except Exception as e:
            print(f"⚠️ 标准CSV导出失败: {e}")

        if save_raw_text:
            raw_txt = save_raw_bytes_to_text(raw_bytes)
            print(f"  原始字节文件: {raw_txt}")
        print(f"  数据长度: {len(raw_bytes)} 字节")
        if len(raw_bytes) >= 20:
            print(f"\n前20个字节: {' '.join(f'{b:02X}' for b in raw_bytes[:20])}")
        return raw_bytes, standard_csv_path, parsed

    return bytearray(), None, None


def create_test_data():
    """Generate fake sensor data for offline testing."""
    print("🧪 使用测试数据演示文件保存功能")
    start_wave = 380
    end_wave = 780
    point_count = end_wave - start_wave + 1

    # 62 standard metric floats
    metrics = [0.0] * 62
    metrics[0] = 1.23   # PAR
    metrics[1] = 45.6   # PPFD
    metrics[58] = 150.0  # Integrat.Time(ms), index=59 in protocol (1-based)
    metric_bytes = b"".join(struct.pack("<f", v) for v in metrics)

    # 20-byte timestamp area
    timestamp = b"2026-03-04\x0012:34:56\x00"[:20].ljust(20, b"\x00")

    spectrum_bytes = b"".join(
        struct.pack("<f", 0.1 + i * 0.01) for i in range(point_count)
    )
    wave_range = struct.pack("<f", float(start_wave)) + struct.pack("<f", float(end_wave))
    payload = metric_bytes + timestamp + spectrum_bytes + wave_range

    # Header(4) + model(10) + version(2) + reserved(24) + payload
    model = b"HPCS-310P\x00"  # 10 bytes
    version_code = (20009).to_bytes(2, byteorder="little", signed=False)
    reserved = bytes(24)

    data_length = len(model) + len(version_code) + len(reserved) + len(payload)
    packet = bytearray()
    packet += bytes([0x8C, 0x13])
    packet += data_length.to_bytes(2, byteorder="big")
    packet += model
    packet += version_code
    packet += reserved
    packet += payload
    return bytes(packet)


def get_csv_filename(today: datetime.date | datetime.datetime | None = None):
    """Return the daily CSV path, creating it with a header if missing."""
    if today is None:
        today = datetime.date.today()
    if isinstance(today, datetime.datetime):
        today = today.date()

    day_str = today.strftime("%Y-%m-%d")
    csv_dir = os.path.join("archive", day_str)
    csv_filename = os.path.join(csv_dir, f"spectrum_data_{day_str}.csv")
    legacy_filename = f"spectrum_data_{day_str}.csv"

    os.makedirs(csv_dir, exist_ok=True)

    # Backward compatibility: move previously generated daily summary
    # from working directory into archive/<day>/ when first seen.
    if os.path.exists(legacy_filename) and not os.path.exists(csv_filename):
        os.replace(legacy_filename, csv_filename)

    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            first_row = ["File"]
            for i, param_name in enumerate(SUMMARY_PARAM_NAMES):
                if SUMMARY_UNITS[i]:
                    first_row.append(f"{param_name}({SUMMARY_UNITS[i]})")
                else:
                    first_row.append(param_name)
            writer.writerow(first_row)
        print(f"📄 创建新的CSV文件: {csv_filename}")
    return csv_filename


def archive_raw_file(txt_filename: str, base_dir: str = "archive") -> None:
    """Move *txt_filename* into ``base_dir/YYYY-MM-DD``.

    如果目标目录不存在会创建它；失败时抛出异常。此函数用于
    将原始字节文件按日期分类归档。
    """
    today = datetime.datetime.now().strftime("%F")
    dest_dir = os.path.join(base_dir, today)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(txt_filename))
    os.replace(txt_filename, dest)


def list_raw_files(source_dir: str = ".") -> list[str]:
    """返回 *source_dir* 下所有待处理的原始数据文件。"""
    source_dir_abs = os.path.abspath(source_dir)
    if not os.path.isdir(source_dir_abs):
        raise NotADirectoryError(f"source_dir 不存在或不是目录: {source_dir}")
    pattern = os.path.join(source_dir_abs, "raw_bytes_*.txt")
    return sorted(glob.glob(pattern))


def _normalize_record_key(txt_filename: str, source_dir: str | None = None) -> str:
    """Build a stable key for CSV de-duplication."""
    abs_path = os.path.abspath(txt_filename)
    base_dir = os.path.abspath(source_dir) if source_dir is not None else os.getcwd()
    try:
        key = os.path.relpath(abs_path, start=base_dir)
    except ValueError:
        key = abs_path
    return os.path.normpath(key).replace("\\", "/")


def _load_existing_record_keys(csv_filename: str) -> set[str]:
    """Load first-column keys from existing CSV rows."""
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"CSV文件不存在: {csv_filename}")

    keys: set[str] = set()
    with open(csv_filename, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            if key and key != "File":
                keys.add(key)
    return keys


def process_pending_raw_files(
    csv_filename: str,
    source_dir: str = ".",
    archive_dir: str | None = None,
) -> dict[str, int]:
    """处理目录中所有 ``raw_bytes_*.txt`` 文件并写入 CSV。"""
    existing_keys = _load_existing_record_keys(csv_filename)
    raw_files = list_raw_files(source_dir)
    stats = {
        "found": len(raw_files),
        "processed": 0,
        "skipped": 0,
        "duplicate": 0,
        "missing_file": 0,
        "write_error": 0,
        "archived": 0,
        "archive_failed": 0,
    }

    for txt_filename in raw_files:
        record_key = _normalize_record_key(txt_filename, source_dir=source_dir)
        status = _append_to_csv_with_status(
            txt_filename,
            csv_filename,
            archive_dir=archive_dir,
            existing_keys=existing_keys,
            record_key=record_key,
        )
        if status == "processed_archived":
            stats["processed"] += 1
            stats["archived"] += 1
        elif status == "processed_archive_failed":
            stats["processed"] += 1
            stats["archive_failed"] += 1
        elif status == "duplicate":
            stats["duplicate"] += 1
            stats["skipped"] += 1
        elif status == "missing_file":
            stats["missing_file"] += 1
            stats["skipped"] += 1
        else:
            stats["write_error"] += 1
            stats["skipped"] += 1
    return stats


def _append_to_csv_with_status(
    txt_filename: str,
    csv_filename: str,
    archive_dir: str | None = None,
    existing_keys: set[str] | None = None,
    record_key: str | None = None,
) -> str:
    """Append a parsed file into CSV and return a status string."""
    if record_key is None:
        record_key = _normalize_record_key(txt_filename)

    if not os.path.exists(csv_filename):
        print(f"❌ CSV文件不存在: {csv_filename}")
        return "write_error"

    if existing_keys is None:
        try:
            existing_keys = _load_existing_record_keys(csv_filename)
        except Exception:
            print(f"❌ 加载CSV索引失败: {csv_filename}")
            return "write_error"

    if not os.path.exists(txt_filename):
        print(f"❌ 文件不存在: {txt_filename}")
        return "missing_file"

    # Always try to export the full standard CSV (580.csv layout).
    try:
        standard_csv_path = export_standard_csv_from_raw_file(txt_filename)
        print(f"🧾 已生成标准CSV: {standard_csv_path}")
    except Exception as e:
        print(f"⚠️ 标准CSV导出失败（将继续写汇总CSV）: {e}")

    if record_key in existing_keys:
        print(f"🔁 文件已记录，跳过汇总CSV: {txt_filename}")
        return "duplicate"

    try:
        print(f"🔍 开始处理文件: {txt_filename}")
        extracted_data = process_single_raw_file(txt_filename)
        print(f"📊 提取到 {len(extracted_data)} 个参数")

        row = [record_key]
        for param_name in SUMMARY_PARAM_NAMES:
            value = extracted_data.get(param_name, 0.0)
            row.append(value)
            print(f"   {param_name}: {value}")

        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
        existing_keys.add(record_key)
        print(f"✅ 数据已添加到CSV文件: {csv_filename}")
        print(f"📊 写入 {len(row)} 个字段")
    except Exception as e:
        print(f"❌ 添加数据到CSV时出错: {e}")
        import traceback
        traceback.print_exc()
        return "write_error"

    # archive source file, but never delete it on failure
    if archive_dir is None:
        try:
            archive_raw_file(txt_filename)
            print(f"📁 已按日期归档 {txt_filename}")
            return "processed_archived"
        except Exception:
            print(f"⚠️ 日期归档失败，已保留原始文件: {txt_filename}")
            return "processed_archive_failed"
    try:
        os.makedirs(archive_dir, exist_ok=True)
        dest = os.path.join(archive_dir, os.path.basename(txt_filename))
        os.replace(txt_filename, dest)
        print(f"📁 已归档 {txt_filename} 到 {archive_dir}")
        return "processed_archived"
    except Exception:
        print(f"⚠️ 无法归档 {txt_filename}，已保留原始文件")
        return "processed_archive_failed"


def append_to_csv(
    txt_filename: str,
    csv_filename: str,
    archive_dir: str | None = None,
) -> bool:
    """根据 *txt_filename* 的内容追加一行到 *csv_filename*。

    在写入前会扫描 CSV，防止重复记录同一个文件名。

    追加成功后会将原始文件归档（默认按日期归档到 ``archive``，
    或归档到 *archive_dir* 指定目录）。归档失败时会保留原文件，
    不会删除。返回值为是否成功追加到 CSV。
    """
    status = _append_to_csv_with_status(
        txt_filename,
        csv_filename,
        archive_dir=archive_dir,
    )
    return status in {"processed_archived", "processed_archive_failed"}


def run_single_measurement(
    port: str,
    csv_filename: str | None = None,
    archive_dir: str | None = None,
):
    """打开指定串口，执行一次测量，并将结果写入 CSV。

    返回与 :func:`complete_spectrum_measurement` 相同的元组
    ``(result, total_time, integration_time)``。测量完成后会将解析结果
    直接写入 *csv_filename*，并输出标准格式 CSV 文件。

    ``csv_filename`` 默认为 :func:`get_csv_filename` 得到的路径。
    ``archive_dir`` 仅用于兼容旧流程，函数结束前会关闭串口。
    """
    if csv_filename is None:
        csv_filename = get_csv_filename()

    if serial is None:
        raise RuntimeError("serial module not installed; cannot access device")

    ser = None
    try:
        ser = serial.Serial(port, baudrate=115200, timeout=1)
        result = complete_spectrum_measurement(ser)
        # attach to daily summary CSV when parsed data is available
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
        return result
    finally:
        if ser and ser.is_open:
            ser.close()


def run_periodic_measurements(
    port: str,
    period: float,
    csv_filename: str | None = None,
    stop_event=None,
    archive_dir: str | None = None,
):
    """每隔 *period* 秒重复执行测量。

    *port* 为串口设备名称，*period* 为时间间隔（秒，可为浮点）。
    默认使用 :func:`get_csv_filename` 返回的 CSV 文件，或者传入
    *csv_filename*。此循环会一直运行，直到传入的 *stop_event*
    被设置或者用户按下 Ctrl+C。

    此函数供其它脚本调用，类似 ``measurement_loop``，但时间间
    隔可配置。
    """
    import threading
    if csv_filename is None:
        csv_filename = get_csv_filename()

    if stop_event is None:
        stop_event = threading.Event()

    ser = None
    try:
        if serial is None:
            raise RuntimeError("serial module not available")
        ser = serial.Serial(port, baudrate=115200, timeout=1)
        print(f"🔌 打开串口 {port} (周期 {period}s)")
        while not stop_event.is_set():
            start = time.time()
            result = complete_spectrum_measurement(ser)
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
            # wait remaining time
            elapsed = time.time() - start
            to_sleep = period - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
    except KeyboardInterrupt:
        print("\n🛑 周期采样已中断")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("🔌 串口已关闭")


def run_on_minute_slot(port: str, csv_filename: str | None = None, stop_event=None):
    """在每分钟的整点执行采样。

    此函数打开串口并保持连接，等待秒数变为 0 时进行一次测量，
    并将结果写入 CSV。之后睡至下一分钟，如此循环，直到访问
    指定的 *stop_event* 或按 Ctrl+C。
    """
    import threading
    if csv_filename is None:
        csv_filename = get_csv_filename()
    if stop_event is None:
        stop_event = threading.Event()

    if serial is None:
        raise RuntimeError("serial module not available")

    ser = serial.Serial(port, baudrate=115200, timeout=1)
    print(f"🔌 串口 {port} 已打开，将每分钟整点采样")
    try:
        while not stop_event.is_set():
            now = datetime.datetime.now()
            # wait until the next minute boundary
            seconds_to_wait = 60 - now.second
            time.sleep(seconds_to_wait)
            if stop_event.is_set():
                break
            start = time.time()
            result = complete_spectrum_measurement(ser)
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
            # small pause to avoid re-triggering within same second
            elapsed = time.time() - start
            if elapsed < 1:
                time.sleep(1 - elapsed)
    except KeyboardInterrupt:
        print("\n🛑 整点采样已中断")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("🔌 串口已关闭")
