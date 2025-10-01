#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MPPI v2 实际控制脚本

使用基于 Solar Vol 的 MPPI 控制器计算指令，并通过 Shelly 设备下发 PWM。
支持单次运行、持续循环以及后台守护模式，日志输出与 v2 仿真脚本兼容。
"""

from __future__ import annotations

import argparse
import atexit
import csv
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# 固定默认参数，可按需修改
CONTROL_INTERVAL_MINUTES = 15.0
DEFAULT_TARGET_SOLAR_VOL = 1.6  # 固定的Solar Vol目标值
DEFAULT_REFERENCE_WEIGHT = 25.0  # Solar Vol参考跟踪权重
RB_RATIO = 0.83
STATUS_CHECK_DELAY = 3.0
NIGHT_START_HOUR = 23
NIGHT_END_HOUR = 7

# 路径设置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "mppi_v2_control_log.csv")
SIMPLE_LOG_FILE = os.path.join(LOG_DIR, "mppi_v2_control_simple.log")
PID_FILE = os.path.join(LOG_DIR, "mppi_v2_control.pid")
BACKGROUND_LOG_FILE = os.path.join(LOG_DIR, "mppi_v2_control_background.log")

# 附加依赖路径
RIOTEE_SENSOR_DIR = os.path.join(PROJECT_ROOT, "..", "Sensor", "riotee_sensor")
SHELLY_DIR = os.path.join(PROJECT_ROOT, "..", "Shelly", "src")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

for path in (SRC_DIR, RIOTEE_SENSOR_DIR, SHELLY_DIR, CONFIG_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# 兼容 sensor_reader
import sensor_reader as _sensor_reader  # type: ignore

if not hasattr(_sensor_reader, "SensorReading"):
    _sensor_reader.SensorReading = _sensor_reader.DemoSensorReader  # type: ignore[attr-defined]
if not hasattr(_sensor_reader, "RIOTEE_DATA_PATH"):
    _sensor_reader.RIOTEE_DATA_PATH = _sensor_reader.DEFAULT_RIOTEE_DATA_PATH  # type: ignore[attr-defined]
if not hasattr(_sensor_reader, "CO2_DATA_PATH"):
    _sensor_reader.CO2_DATA_PATH = None  # type: ignore[attr-defined]
sys.modules.setdefault("sensor_reading", _sensor_reader)

from led import PWMtoPowerModel
from mppi_v2 import LEDMPPIController, LEDPlant
from sensor_reader import DEFAULT_CO2_PPM
from shelly_controller import DEVICES, rpc  # type: ignore


def ensure_log_dir() -> None:
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)


class MPPIControlV2:
    """实际控制执行器"""

    def __init__(self, *, background: bool = False) -> None:
        ensure_log_dir()
        self.background = background
        self.control_interval_seconds = CONTROL_INTERVAL_MINUTES * 60.0
        self.target_solar_vol = DEFAULT_TARGET_SOLAR_VOL
        self.reference_weight = DEFAULT_REFERENCE_WEIGHT
        self.current_temp: Optional[float] = None
        self.devices = DEVICES
        self.co2_fallback = DEFAULT_CO2_PPM
        self._init_logging()
        self._init_models()
        self._init_log_file()

        if self.background:
            signal.signal(signal.SIGTERM, self._handle_background_signal)
            signal.signal(signal.SIGINT, self._handle_background_signal)
            atexit.register(self._cleanup_pid_file)

    # ---------- 初始化 ----------
    def _init_logging(self) -> None:
        if self.background:
            logging.basicConfig(
                filename=BACKGROUND_LOG_FILE,
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
            )

    def _init_models(self) -> None:
        power_model = self._load_power_model()
        self.plant = LEDPlant(
            base_ambient_temp=25.0,
            max_solar_vol=2.0,
            thermal_resistance=0.05,
            time_constant_s=900.0,
            thermal_mass=150.0,
            power_model=power_model,
            r_b_ratio=RB_RATIO,
            use_solar_vol_model=True,
        )

        self.controller = LEDMPPIController(
            plant=self.plant,
            horizon=6,
            num_samples=700,
            dt=self.control_interval_seconds,
            temperature=1.0,
        )
        self.controller.set_constraints(
            u_min=0.05,
            u_max=float(self.plant.max_solar_vol),
            temp_min=20.0,
            temp_max=29.8,
        )
        self.controller.set_weights(
            Q_photo=25.0,
            R_du=0.02,
            R_power=0.005,
            Q_ref=self.reference_weight,
        )
        self.controller.set_mppi_params(u_std=0.25, dt=self.control_interval_seconds)

    def _load_power_model(self) -> PWMtoPowerModel:
        calib_csv = os.path.join(PROJECT_ROOT, "data", "calib_data.csv")
        if not os.path.exists(calib_csv):
            raise FileNotFoundError(f"功率标定文件缺失: {calib_csv}")
        model = PWMtoPowerModel(include_intercept=True)
        return model.fit(calib_csv)

    def _init_log_file(self) -> None:
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "sensor_timestamp",
                        "input_temp",
                        "co2_ppm",
                        "solar_vol_cmd",
                        "r_pwm",
                        "b_pwm",
                        "pred_temp",
                        "pred_power",
                        "pred_pn",
                        "target_solar_vol",
                        "cost",
                        "success",
                        "note",
                    ]
                )

    # ---------- 基础工具 ----------
    def _handle_background_signal(self, _signum, _frame) -> None:  # type: ignore[override]
        self._log_simple("收到终止信号，准备退出")
        self._cleanup_pid_file()
        sys.exit(0)

    def _cleanup_pid_file(self) -> None:
        try:
            if os.path.exists(PID_FILE):
                os.unlink(PID_FILE)
        except OSError:
            pass

    def _log_simple(self, message: str) -> None:
        entry = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
        if self.background:
            logging.info(message)
        else:
            print(f"📝 {entry}")
        with open(SIMPLE_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(entry + "\n")

    def _is_night_time(self) -> bool:
        hour = datetime.now().hour
        if NIGHT_START_HOUR > NIGHT_END_HOUR:
            return hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR
        return NIGHT_START_HOUR <= hour < NIGHT_END_HOUR

    def _read_environment(self) -> Dict[str, Any]:
        temp, solar_vol, pn, ts = self.plant.sensor_reader.read_latest_riotee_data()
        co2 = self.plant.sensor_reader.read_latest_co2_data()
        return {
            "temp": None if temp is None else float(temp),
            "solar_vol": None if solar_vol is None else float(solar_vol),
            "pn": None if pn is None else float(pn),
            "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts) if ts else None,
            "co2": self.co2_fallback if co2 is None else float(co2),
        }

    def _make_solar_vol_reference(self) -> Optional[np.ndarray]:
        # 生成固定的Solar Vol参考序列
        if self.target_solar_vol is None or self.reference_weight <= 0:
            return None
        return np.full(self.controller.horizon, self.target_solar_vol, dtype=float)

    def _send_pwm(self, r_pwm: float, b_pwm: float) -> Dict[str, Any]:
        result: Dict[str, Any] = {"red": None, "blue": None}
        brightness_r = int(np.clip(np.round(r_pwm), 0, 100))
        brightness_b = int(np.clip(np.round(b_pwm), 0, 100))

        if "Red" in self.devices:
            payload = {"id": 0, "on": True, "brightness": brightness_r, "transition": 1000}
            try:
                resp = rpc(self.devices["Red"], "Light.Set", payload)
                result["red"] = resp
            except Exception as exc:  # noqa: BLE001
                result["red"] = {"error": str(exc)}

        if "Blue" in self.devices:
            payload = {"id": 0, "on": True, "brightness": brightness_b, "transition": 1000}
            try:
                resp = rpc(self.devices["Blue"], "Light.Set", payload)
                result["blue"] = resp
            except Exception as exc:  # noqa: BLE001
                result["blue"] = {"error": str(exc)}

        if not self.background:
            print(
                f"📡 已发送命令: 红 {brightness_r} / 蓝 {brightness_b}; "
                f"响应: {result}"
            )
        time.sleep(STATUS_CHECK_DELAY)
        return result

    def _log_cycle(self, payload: Dict[str, Any]) -> None:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    payload.get("timestamp"),
                    payload.get("sensor_timestamp"),
                    payload.get("input_temp"),
                    payload.get("co2_ppm"),
                    payload.get("solar_vol_cmd"),
                    payload.get("r_pwm"),
                    payload.get("b_pwm"),
                    payload.get("pred_temp"),
                    payload.get("pred_power"),
                    payload.get("pred_pn"),
                    payload.get("target_pn"),
                    payload.get("cost"),
                    payload.get("success"),
                    payload.get("note"),
                ]
            )

    # ---------- 核心控制 ----------
    def run_once(self) -> None:
        if self._is_night_time():
            self._log_simple("夜间模式，跳过本次控制")
            return

        env = self._read_environment()
        measured_temp = env.get("temp")
        if measured_temp is not None:
            self.current_temp = measured_temp
        elif self.current_temp is None:
            self.current_temp = 25.0

        current_temp = float(self.current_temp)
        # 使用固定的Solar Vol参考值进行跟踪控制
        solar_vol_ref = self._make_solar_vol_reference()

        try:
            optimal_sv, optimal_seq, success, cost, _weights = self.controller.solve(
                current_temp=current_temp,
                solar_vol_ref_seq=solar_vol_ref,
            )
        except Exception as exc:  # noqa: BLE001
            self._log_simple(f"MPPI 求解失败: {exc}")
            return

        cycle_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not success:
            self._log_simple("MPPI 求解未成功，跳过下发")
            self._log_cycle(
                {
                    "timestamp": cycle_ts,
                    "sensor_timestamp": env.get("timestamp"),
                    "input_temp": measured_temp,
                    "co2_ppm": env.get("co2"),
                    "solar_vol_cmd": None,
                    "r_pwm": None,
                    "b_pwm": None,
                    "pred_temp": None,
                    "pred_power": None,
                    "pred_pn": None,
                    "target_solar_vol": self.target_solar_vol,
                    "cost": None,
                    "success": False,
                    "note": "solve_failed",
                }
            )
            return

        r_pwm, b_pwm = self.plant._solar_vol_to_pwm(optimal_sv)  # noqa: SLF001
        preds = self.plant.predict(optimal_seq, current_temp, dt=self.control_interval_seconds)
        (_sv_series, temp_pred, power_pred, pn_pred, _r_series, _b_series) = preds

        next_temp = float(temp_pred[0]) if len(temp_pred) else current_temp
        next_power = float(power_pred[0]) if len(power_pred) else 0.0
        next_pn = float(pn_pred[0]) if len(pn_pred) else 0.0

        status = self._send_pwm(r_pwm, b_pwm)
        note = ""
        if isinstance(status.get("red"), dict) and "error" in status["red"]:
            note += f"red_error:{status['red']['error']}"
        if isinstance(status.get("blue"), dict) and "error" in status["blue"]:
            if note:
                note += "|"
            note += f"blue_error:{status['blue']['error']}"

        self.current_temp = next_temp
        if not self.background:
            print("=" * 70)
            print(f"🔄 控制循环 @ {cycle_ts}")
            print(f"🌡️ 输入温度: {measured_temp if measured_temp is not None else 'N/A'} °C")
            print(f"🌬️ CO₂: {env.get('co2')} ppm")
            print(f"🎯 Solar Vol 指令: {float(optimal_sv):.3f}")
            print(f"🔴 红光PWM: {r_pwm:.2f} | 🔵 蓝光PWM: {b_pwm:.2f}")
            print(f"📈 预测温度: {next_temp:.2f} °C")
            print(f"⚡ 预测功率: {next_power:.2f} W")
            print(f"🌱 预测光合速率: {next_pn:.3f} (目标Solar Vol: {self.target_solar_vol:.3f})")
            print(f"💰 代价: {float(cost):.2f}")

        self._log_cycle(
            {
                "timestamp": cycle_ts,
                "sensor_timestamp": env.get("timestamp"),
                "input_temp": measured_temp,
                "co2_ppm": env.get("co2"),
                "solar_vol_cmd": float(optimal_sv),
                "r_pwm": float(r_pwm),
                "b_pwm": float(b_pwm),
                "pred_temp": next_temp,
                "pred_power": next_power,
                "pred_pn": next_pn,
                "target_solar_vol": self.target_solar_vol,
                "cost": float(cost),
                "success": True,
                "note": note or "ok",
            }
        )

    def run_continuous(self) -> None:
        self._log_simple("进入连续控制模式")
        while True:
            start = time.time()
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001
                self._log_simple(f"控制循环异常: {exc}")
            elapsed = time.time() - start
            sleep_time = max(0.0, self.control_interval_seconds - elapsed)
            time.sleep(sleep_time)


# ---------- 后台管理 ----------
def _is_running() -> bool:
    if not os.path.exists(PID_FILE):
        return False
    try:
        with open(PID_FILE, "r", encoding="utf-8") as fh:
            pid = int(fh.read().strip())
        os.kill(pid, 0)
        return True
    except Exception:  # noqa: BLE001
        try:
            os.unlink(PID_FILE)
        except OSError:
            pass
        return False


def _start_background() -> None:
    if _is_running():
        print("✅ MPPI v2 控制后台已在运行")
        return

    ensure_log_dir()
    cmd = [sys.executable, os.path.abspath(__file__), "background"]
    with open(BACKGROUND_LOG_FILE, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=CURRENT_DIR,
            text=True,
        )
    with open(PID_FILE, "w", encoding="utf-8") as fh:
        fh.write(str(process.pid))
    print(f"🚀 已启动后台进程 (PID: {process.pid})")


def _stop_background() -> None:
    if not _is_running():
        print("ℹ️ 后台进程未运行")
        return
    with open(PID_FILE, "r", encoding="utf-8") as fh:
        pid = int(fh.read().strip())
    print(f"⏹️ 正在停止后台进程 (PID: {pid}) ...")
    os.kill(pid, signal.SIGTERM)
    for _ in range(10):
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.5)
    if os.path.exists(PID_FILE):
        os.unlink(PID_FILE)
    print("✅ 后台进程已停止")


def _show_status() -> None:
    print("📊 MPPI v2 控制状态")
    print("=" * 40)
    if _is_running():
        with open(PID_FILE, "r", encoding="utf-8") as fh:
            pid = fh.read().strip()
        print(f"🟢 后台进程: 运行中 (PID: {pid})")
    else:
        print("🔴 后台进程: 未运行")
    if os.path.exists(BACKGROUND_LOG_FILE):
        size = os.path.getsize(BACKGROUND_LOG_FILE)
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(BACKGROUND_LOG_FILE)))
        print(f"📄 后台日志: {BACKGROUND_LOG_FILE} ({size} bytes, 更新 {mtime})")
    else:
        print("📄 后台日志: 未生成")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPPI v2 实际控制运行器")
    parser.add_argument(
        "command",
        nargs="?",
        default="once",
        choices=(
            "once",
            "continuous",
            "background",
            "start",
            "stop",
            "restart",
            "status",
        ),
        help="控制模式",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "start":
        _start_background()
        return
    if args.command == "stop":
        _stop_background()
        return
    if args.command == "restart":
        _stop_background()
        time.sleep(1.0)
        _start_background()
        return
    if args.command == "status":
        _show_status()
        return

    background_mode = args.command == "background"
    controller = MPPIControlV2(background=background_mode)

    if args.command == "continuous":
        controller._log_simple("启动连续控制 (前台)")
        controller.run_continuous()
    elif args.command == "background":
        with open(PID_FILE, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))
        controller._log_simple("后台守护进程启动")
        controller.run_continuous()
    else:  # once
        controller._log_simple("执行单次控制循环")
        controller.run_once()


if __name__ == "__main__":
    main()
