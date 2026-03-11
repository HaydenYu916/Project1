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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# 固定默认参数，可按需修改
CONTROL_INTERVAL_MINUTES = 15.0
DEFAULT_TARGET_SOLAR_VOL = 1.6  # 固定的Solar Vol目标值
DEFAULT_REFERENCE_WEIGHT = 50.0  # Solar Vol参考跟踪权重 (从25.0->30.0->50.0，提高跟踪效果)
RB_RATIO = 0.83
STATUS_CHECK_DELAY = 5.0  # 从3.0秒增加到5.0秒，等待Shelly稳定
PWM_RETRY_MAX = 3  # PWM发送失败时的最大重试次数
PWM_TOLERANCE = 3  # PWM占空比验证容差（0-100范围内，允许±3的偏差）
NIGHT_START_HOUR = 1
NIGHT_END_HOUR = 9

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
TOOL_ROOT = os.path.join(PROJECT_ROOT, "..", "Tool")
RIOTEE_SENSOR_DIR = os.path.join(TOOL_ROOT, "Sensor_riotee_server")
SHELLY_DIR = os.path.join(TOOL_ROOT, "LED_Shelly", "src")
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
            thermal_model_type='thermal',  # 使用热力学模型
            model_dir='Thermal/exported_models',  # 指定模型目录
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
            temp_max=35.0,  # 从29.8°C放宽到32°C，增加2.2°C允许范围，减少温度约束压制
        )
        self.controller.set_weights(
            Q_photo=25.0,  # 从25.0降低到15.0，减少光合作用权重，让参考跟踪更突出
            R_du=0.01,     # 从0.02降低到0.01，减少控制变化惩罚，允许更积极的控制
            R_power=0.002, # 从0.005降低到0.002，减少功率惩罚，允许更高功率输出
            Q_ref=self.reference_weight,  # 50.0的参考跟踪权重，最高优先级
        )
        self.controller.set_mppi_params(u_std=0.25, dt=self.control_interval_seconds)
        # 大幅降低温度惩罚，让控制器更关注 Solar Vol 跟踪
        self.controller.set_penalties(temp_penalty=1e3)  # 从1e5(100000)降低到1e3(1000)，减少100倍温度约束压制

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

    def _seconds_until_next_boundary(self) -> float:
        """计算距离下一个对齐边界的秒数。

        边界定义:
        - 当 CONTROL_INTERVAL_MINUTES 为 15 -> 分钟对齐到 0/15/30/45
        - 当 为 30 -> 分钟对齐到 0/30
        - 其它值 -> 按分钟粒度，向上取整到下一个间隔倍数（跨小时自动进位）
        """
        now = datetime.now()
        interval_min = int(round(CONTROL_INTERVAL_MINUTES))
        if interval_min <= 0:
            return 0.0

        # 计算下一对齐分钟
        current_minute = now.minute
        block_index = (current_minute // interval_min) + 1
        next_minute = block_index * interval_min

        target = now.replace(second=0, microsecond=0)
        if next_minute >= 60:
            next_minute -= 60
            target = (target.replace(minute=0) + timedelta(hours=1)).replace(minute=next_minute)
        else:
            target = target.replace(minute=next_minute)

        delta = (target - now).total_seconds()
        # 如果已经非常接近边界（< 0.5s），视为到达，返回 0
        if delta < 0.5:
            return 0.0
        return max(0.0, delta)

    def _sleep_until_schedule_boundary(self) -> None:
        wait_s = self._seconds_until_next_boundary()
        if wait_s > 0:
            self._log_simple(f"等待至对齐时间点 {CONTROL_INTERVAL_MINUTES}min 间隔，{wait_s:.1f}s 后开始")
            time.sleep(wait_s)

    def _make_solar_vol_reference(self) -> Optional[np.ndarray]:
        # 生成固定的Solar Vol参考序列
        if self.target_solar_vol is None or self.reference_weight <= 0:
            return None
        return np.full(self.controller.horizon, self.target_solar_vol, dtype=float)

    def _verify_pwm(self, device_name: str, expected_brightness: int) -> bool:
        """验证设备PWM是否成功设置
        
        Args:
            device_name: 设备名称 ("Red" 或 "Blue")
            expected_brightness: 期望的亮度值 (0-100)
        
        Returns:
            bool: 验证成功返回True，失败返回False
        """
        if device_name not in self.devices:
            return False
        
        try:
            # 获取设备状态
            status = rpc(self.devices[device_name], "Shelly.GetStatus")
            
            # 检查是否有错误
            if isinstance(status, dict) and "error" in status:
                self._log_simple(f"❌ {device_name} 状态查询失败: {status['error']}")
                return False
            
            # 提取light:0的状态
            light_status = status.get("light:0", {})
            actual_brightness = light_status.get("brightness")
            is_on = light_status.get("output")
            
            # 验证设备是否开启且亮度正确（允许一定容差）
            if not is_on:
                self._log_simple(f"❌ {device_name} 设备未开启")
                return False
            
            if actual_brightness is None:
                self._log_simple(f"❌ {device_name} 无法读取亮度值")
                return False
            
            # 检查PWM值是否在容差范围内
            diff = abs(actual_brightness - expected_brightness)
            if diff > PWM_TOLERANCE:
                self._log_simple(
                    f"❌ {device_name} 亮度不匹配: 期望 {expected_brightness}, "
                    f"实际 {actual_brightness} (差值 {diff})"
                )
                return False
            
            # 验证成功
            if not self.background:
                print(f"✅ {device_name} 验证成功: 亮度 {actual_brightness}")
            return True
            
        except Exception as exc:  # noqa: BLE001
            self._log_simple(f"❌ {device_name} 验证异常: {exc}")
            return False

    def _send_pwm_single_device(
        self, device_name: str, brightness: int, retry_count: int = 0
    ) -> Dict[str, Any]:
        """发送单个设备的PWM命令并验证
        
        Args:
            device_name: 设备名称 ("Red" 或 "Blue")
            brightness: 亮度值 (0-100)
            retry_count: 当前重试次数
        
        Returns:
            Dict: 包含发送结果和验证状态的字典
        """
        result = {
            "success": False,
            "response": None,
            "error": None,
            "retries": retry_count,
            "verified": False,
        }
        
        if device_name not in self.devices:
            result["error"] = f"设备 {device_name} 不存在"
            return result
        
        # 发送PWM命令
        payload = {"id": 0, "on": True, "brightness": brightness, "transition": 1000}
        try:
            resp = rpc(self.devices[device_name], "Light.Set", payload)
            result["response"] = resp
            
            # 检查RPC响应是否有错误
            if isinstance(resp, dict) and "error" in resp:
                result["error"] = f"RPC错误: {resp['error']}"
                return result
            
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"发送异常: {exc}"
            return result
        
        # 等待设备稳定
        time.sleep(STATUS_CHECK_DELAY)
        
        # 验证PWM是否成功设置
        if self._verify_pwm(device_name, brightness):
            result["success"] = True
            result["verified"] = True
            return result
        
        # 验证失败，如果还有重试机会，则重试
        if retry_count < PWM_RETRY_MAX:
            self._log_simple(
                f"🔄 {device_name} 验证失败，正在重试 ({retry_count + 1}/{PWM_RETRY_MAX})..."
            )
            return self._send_pwm_single_device(device_name, brightness, retry_count + 1)
        
        # 已达最大重试次数
        result["error"] = f"达到最大重试次数 ({PWM_RETRY_MAX})"
        return result

    def _send_pwm(self, r_pwm: float, b_pwm: float) -> Dict[str, Any]:
        """发送PWM命令到红蓝设备，带重试和验证机制
        
        Args:
            r_pwm: 红光PWM值 (0-100)
            b_pwm: 蓝光PWM值 (0-100)
        
        Returns:
            Dict: 包含两个设备发送结果的字典
        """
        result: Dict[str, Any] = {"red": None, "blue": None}
        brightness_r = int(np.clip(np.round(r_pwm), 0, 100))
        brightness_b = int(np.clip(np.round(b_pwm), 0, 100))

        if not self.background:
            print(f"📡 正在发送PWM命令: 红 {brightness_r}% / 蓝 {brightness_b}%")

        # 发送红光设备PWM
        if "Red" in self.devices:
            result["red"] = self._send_pwm_single_device("Red", brightness_r)
            if result["red"]["success"]:
                msg = f"✅ 红光设备设置成功: {brightness_r}%"
                if result["red"]["retries"] > 0:
                    msg += f" (重试 {result['red']['retries']} 次)"
                self._log_simple(msg)
            else:
                self._log_simple(
                    f"❌ 红光设备设置失败: {result['red'].get('error', '未知错误')}"
                )

        # 发送蓝光设备PWM
        if "Blue" in self.devices:
            result["blue"] = self._send_pwm_single_device("Blue", brightness_b)
            if result["blue"]["success"]:
                msg = f"✅ 蓝光设备设置成功: {brightness_b}%"
                if result["blue"]["retries"] > 0:
                    msg += f" (重试 {result['blue']['retries']} 次)"
                self._log_simple(msg)
            else:
                self._log_simple(
                    f"❌ 蓝光设备设置失败: {result['blue'].get('error', '未知错误')}"
                )

        if not self.background:
            print(f"📊 发送完成 - 红光: {result['red']}, 蓝光: {result['blue']}")

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
                        payload.get("target_solar_vol"),
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
                    "sensor_timestamp": env.get("timestamp") or "",
                    "input_temp": measured_temp if measured_temp is not None else "",
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
        
        # 构建日志注释信息
        note = ""
        red_status = status.get("red")
        blue_status = status.get("blue")
        
        if red_status:
            if red_status.get("success"):
                if red_status.get("retries", 0) > 0:
                    note += f"red_retry:{red_status['retries']}"
            elif red_status.get("error"):
                note += f"red_error:{red_status['error']}"
        
        if blue_status:
            if blue_status.get("success"):
                if blue_status.get("retries", 0) > 0:
                    if note:
                        note += "|"
                    note += f"blue_retry:{blue_status['retries']}"
            elif blue_status.get("error"):
                if note:
                    note += "|"
                note += f"blue_error:{blue_status['error']}"

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
                "sensor_timestamp": env.get("timestamp") or "",
                "input_temp": measured_temp if measured_temp is not None else "",
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
        self._log_simple("进入连续控制模式（按墙钟对齐到固定时间点）")
        while True:
            # 在每次循环开始前先对齐到下一个边界（0/15/30/45 或 0/30 等）
            try:
                self._sleep_until_schedule_boundary()
            except Exception as exc:  # noqa: BLE001
                self._log_simple(f"对齐时间点异常: {exc}")
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001
                self._log_simple(f"控制循环异常: {exc}")


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
            cwd=PROJECT_ROOT,
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
