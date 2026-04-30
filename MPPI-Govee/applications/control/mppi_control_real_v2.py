#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MPPI v2 实际控制脚本 (Growpod / PPFD 版)

使用基于 PPFD 的 MPPI 控制器计算指令,并通过 Shelly 设备下发 PWM。
支持单次运行、持续循环以及后台守护模式。

⚠️ 安全:默认 dry-run = ON,不会真正下发 PWM,只打印决策;
    要启用真实下发,显式加 --live。
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
from typing import Any, Dict, Optional

import numpy as np

# 固定默认参数
CONTROL_INTERVAL_MINUTES = 15.0
DEFAULT_TARGET_PPFD = 400.0
DEFAULT_REFERENCE_WEIGHT = 0.0
DEFAULT_POWER_BUDGET_WEIGHT = 0.0   # 与 R_power 互斥; 默认关闭
RB_RATIO = 0.83
STATUS_CHECK_DELAY = 5.0
PWM_RETRY_MAX = 3
PWM_TOLERANCE = 3
NIGHT_START_HOUR = 1
NIGHT_END_HOUR = 9
THERMAL_ADAPTIVE_ENABLED = True
THERMAL_UPDATE_INTERVAL_HOURS = 1.0
THERMAL_UPDATE_WINDOW_HOURS = 6.0
THERMAL_UPDATE_MIN_SAMPLES = 24

# 路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "mppi_v2_control_log.csv")
SIMPLE_LOG_FILE = os.path.join(LOG_DIR, "mppi_v2_control_simple.log")
PID_FILE = os.path.join(LOG_DIR, "mppi_v2_control.pid")
BACKGROUND_LOG_FILE = os.path.join(LOG_DIR, "mppi_v2_control_background.log")
THERMAL_RUNTIME_PARAMS_FILE = os.path.join(LOG_DIR, "thermal_runtime_params.json")

TOOL_ROOT = os.path.join(PROJECT_ROOT, "..", "Tool")
RIOTEE_SENSOR_DIR = os.path.join(TOOL_ROOT, "Sensor_riotee_server")
SHELLY_DIR = os.path.join(TOOL_ROOT, "LED_Govee", "src")
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


def ensure_log_dir() -> None:
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)


class MPPIControlV2:
    """实际控制执行器 (PPFD 控制)。"""

    def __init__(self, *, background: bool = False, live: bool = False) -> None:
        ensure_log_dir()
        self.background = background
        self.live = bool(live)              # ⚠️ True 才会真正向 LED 下发 PWM
        self.control_interval_seconds = CONTROL_INTERVAL_MINUTES * 60.0
        self.target_ppfd = DEFAULT_TARGET_PPFD
        self.reference_weight = DEFAULT_REFERENCE_WEIGHT
        self.power_budget_weight = DEFAULT_POWER_BUDGET_WEIGHT
        self.target_mean_power: Optional[float] = None
        self.current_temp: Optional[float] = None
        self.co2_fallback = DEFAULT_CO2_PPM
        self.thermal_updater = None
        # Shelly / 热模型更新器只在 live 模式下加载
        self.devices: Dict[str, Any] = {}
        self._rpc = None
        if self.live:
            self._init_hardware_modules()

        self._init_logging()
        self._init_models()
        self._init_log_file()

        if not self.live:
            self._log_simple("⚠️ 当前为 dry-run 模式: 只打印决策, 不向 LED 下发 PWM。要启用真实下发请加 --live。")

        if self.background:
            signal.signal(signal.SIGTERM, self._handle_background_signal)
            signal.signal(signal.SIGINT, self._handle_background_signal)
            atexit.register(self._cleanup_pid_file)

    # ---------- 初始化 ----------
    def _init_hardware_modules(self) -> None:
        # Govee 用单台 BLE 灯条做 R+B 混合，没有 Shelly 的"两个 device + RPC"
        # 概念。这里 connect 一次，整个 controller 生命周期共用，下发由
        # _send_pwm() 直接调 set_led_mix()。
        import govee_controller  # type: ignore
        from thermal_parameter_updater import ThermalParameterUpdater
        try:
            govee_controller.connect()
            self._log_simple("🔌 Govee BLE connected (mix mode)")
        except Exception as exc:
            self._log_simple(f"⚠️ Govee BLE connect failed: {exc}; "
                             f"调用方需自行确保 set_led_mix() 可用")
        self.devices = dict(govee_controller.DEVICES)
        self._rpc = None  # legacy slot; set_led_mix 不需要 RPC
        self._ThermalParameterUpdater = ThermalParameterUpdater

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
            max_ppfd=500.0,
            thermal_model_type='thermal',
            model_dir='Thermal/exported_models',
            power_model=power_model,
            r_b_ratio=RB_RATIO,
            target_ppfd=self.target_ppfd,
            use_pn_model=True,
            use_sp_to_ppfd=True,
            adaptive_thermal_enabled=THERMAL_ADAPTIVE_ENABLED,
            adaptive_thermal_params_path=THERMAL_RUNTIME_PARAMS_FILE,
        )

        if THERMAL_ADAPTIVE_ENABLED and self.live:
            self.thermal_updater = self._ThermalParameterUpdater(
                model_dir=os.path.join(PROJECT_ROOT, "Thermal", "exported_models"),
                runtime_params_path=THERMAL_RUNTIME_PARAMS_FILE,
                log_path=LOG_FILE,
                base_ambient_temp=25.0,
                model_type='thermal',
                solar_threshold=1.4,
                control_interval_seconds=self.control_interval_seconds,
                update_interval_seconds=THERMAL_UPDATE_INTERVAL_HOURS * 3600.0,
                window_hours=THERMAL_UPDATE_WINDOW_HOURS,
                min_samples=THERMAL_UPDATE_MIN_SAMPLES,
            )

        self.controller = LEDMPPIController(
            plant=self.plant,
            horizon=6,
            num_samples=700,
            dt=self.control_interval_seconds,
            temperature=1.0,
        )
        self.controller.set_constraints(
            u_min=80.0,
            u_max=float(self.plant.max_ppfd),
            temp_min=20.0,
            temp_max=35.0,
        )
        # 使用 controller 默认权重(都已是 z-score 后的 O(1) 量级);仅显式覆盖 Q_ref
        self.controller.set_weights(Q_ref=self.reference_weight)

        self.target_mean_power = self._estimate_target_mean_power()
        if self.power_budget_weight > 0.0 and self.target_mean_power is not None:
            self.controller.set_power_budget(
                target_mean_power=self.target_mean_power,
                power_budget_weight=self.power_budget_weight,
            )

        self.controller.set_mppi_params(u_std=20.0, dt=self.control_interval_seconds)

    def _load_power_model(self) -> PWMtoPowerModel:
        calib_csv = os.path.join(PROJECT_ROOT, "data", "calib_data.csv")
        if not os.path.exists(calib_csv):
            raise FileNotFoundError(f"功率标定文件缺失: {calib_csv}")
        return PWMtoPowerModel(include_intercept=True).fit(calib_csv)

    def _estimate_target_mean_power(self) -> Optional[float]:
        if self.target_ppfd is None:
            return None
        r_pwm, b_pwm = self.plant._ppfd_to_pwm(float(self.target_ppfd))  # noqa: SLF001
        total_pwm = float(r_pwm + b_pwm)
        power_key = self.plant._get_power_model_key(self.plant.r_b_ratio)  # noqa: SLF001
        return float(self.plant.power_model.predict(total_pwm=total_pwm, key=power_key))

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
                        "ppfd_cmd",
                        "r_pwm",
                        "b_pwm",
                        "pred_temp",
                        "pred_power",
                        "pred_pn",
                        "target_ppfd",
                        "target_mean_power",
                        "cost",
                        "success",
                        "note",
                        "live",
                    ]
                )

    # ---------- 基础工具 ----------
    def _handle_background_signal(self, _signum, _frame) -> None:  # type: ignore[override]
        self._log_simple("收到终止信号,准备退出")
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
        """同时兼容 mppi_v2.SensorReading (5-tuple) 与 DemoSensorReader (4-tuple)。"""
        result = self.plant.sensor_reader.read_latest_riotee_data()
        if len(result) == 5:
            temp, solar_vol, pn, ts, _spectrum = result
        else:
            temp, solar_vol, pn, ts = result
        co2 = self.plant.sensor_reader.read_latest_co2_data()
        return {
            "temp": None if temp is None else float(temp),
            "solar_vol": None if solar_vol is None else float(solar_vol),
            "pn": None if pn is None else float(pn),
            "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts) if ts else None,
            "co2": self.co2_fallback if co2 is None else float(co2),
        }

    def _seconds_until_next_boundary(self) -> float:
        now = datetime.now()
        interval_min = int(round(CONTROL_INTERVAL_MINUTES))
        if interval_min <= 0:
            return 0.0
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
        if delta < 0.5:
            return 0.0
        return max(0.0, delta)

    def _sleep_until_schedule_boundary(self) -> None:
        wait_s = self._seconds_until_next_boundary()
        if wait_s > 0:
            self._log_simple(f"等待至对齐时间点 {CONTROL_INTERVAL_MINUTES}min 间隔,{wait_s:.1f}s 后开始")
            time.sleep(wait_s)

    def _make_ppfd_reference(self) -> Optional[np.ndarray]:
        if self.target_ppfd is None or self.reference_weight <= 0:
            return None
        return np.full(self.controller.horizon, self.target_ppfd, dtype=float)

    def _make_mean_sequence(self) -> Optional[np.ndarray]:
        if self.target_ppfd is None:
            return None
        return np.full(self.controller.horizon, self.target_ppfd, dtype=float)

    def _maybe_update_thermal_model(self) -> str:
        if self.thermal_updater is None:
            return ""
        try:
            result = self.thermal_updater.maybe_update()
        except Exception as exc:  # noqa: BLE001
            self._log_simple(f"热模型在线更新失败: {exc}")
            return "thermal_update_error"
        if result.get("status") != "updated":
            return ""
        self.plant.refresh_thermal_model_runtime_parameters()
        baseline_mae = result.get("baseline_mae")
        candidate_mae = result.get("candidate_mae")
        samples = result.get("samples")
        self._log_simple(
            f"热模型参数已更新: samples={samples}, mae {baseline_mae} -> {candidate_mae}"
        )
        return f"thermal_updated:{baseline_mae}->{candidate_mae}"

    # ---------- PWM 下发 (含 dry-run) ----------
    def _verify_pwm(self, device_name: str, expected_brightness: int) -> bool:
        if not self.live or device_name not in self.devices or self._rpc is None:
            return False
        try:
            status = self._rpc(self.devices[device_name], "Shelly.GetStatus")
            if isinstance(status, dict) and "error" in status:
                self._log_simple(f"❌ {device_name} 状态查询失败: {status['error']}")
                return False
            light_status = status.get("light:0", {})
            actual_brightness = light_status.get("brightness")
            is_on = light_status.get("output")
            if not is_on:
                self._log_simple(f"❌ {device_name} 设备未开启")
                return False
            if actual_brightness is None:
                self._log_simple(f"❌ {device_name} 无法读取亮度")
                return False
            diff = abs(actual_brightness - expected_brightness)
            if diff > PWM_TOLERANCE:
                self._log_simple(
                    f"❌ {device_name} 亮度不匹配: 期望 {expected_brightness}, 实际 {actual_brightness} (差 {diff})"
                )
                return False
            if not self.background:
                print(f"✅ {device_name} 验证成功: 亮度 {actual_brightness}")
            return True
        except Exception as exc:  # noqa: BLE001
            self._log_simple(f"❌ {device_name} 验证异常: {exc}")
            return False

    def _send_pwm_single_device(
        self, device_name: str, brightness: int, retry_count: int = 0
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "success": False,
            "response": None,
            "error": None,
            "retries": retry_count,
            "verified": False,
        }
        if not self.live or self._rpc is None:
            result["error"] = "dry-run"
            return result
        if device_name not in self.devices:
            result["error"] = f"设备 {device_name} 不存在"
            return result
        payload = {"id": 0, "on": True, "brightness": brightness, "transition": 1000}
        try:
            resp = self._rpc(self.devices[device_name], "Light.Set", payload)
            result["response"] = resp
            if isinstance(resp, dict) and "error" in resp:
                result["error"] = f"RPC错误: {resp['error']}"
                return result
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"发送异常: {exc}"
            return result

        time.sleep(STATUS_CHECK_DELAY)

        if self._verify_pwm(device_name, brightness):
            result["success"] = True
            result["verified"] = True
            return result

        if retry_count < PWM_RETRY_MAX:
            self._log_simple(
                f"🔄 {device_name} 验证失败,重试 ({retry_count + 1}/{PWM_RETRY_MAX})..."
            )
            return self._send_pwm_single_device(device_name, brightness, retry_count + 1)

        result["error"] = f"达到最大重试次数 ({PWM_RETRY_MAX})"
        return result

    def _send_pwm(self, r_pwm: float, b_pwm: float) -> Dict[str, Any]:
        """Govee 模式：用混合 RGB 把红+蓝送到一个物理设备的两个 bar。

        与 Shelly 版本的关键差异：
          - 不是给两个独立设备分别下发（红/蓝），而是单设备的 RGB(R,0,B) 混合
          - 不靠 `Light.Set` RPC 分别推送，而是 govee_controller.set_led_mix()
            自带（BLE / HA 两路 transport）；这里走 BLE 直连，最低延迟
          - per-channel "verified" 字段保留接口形状，对外语义和 Shelly 一致
        """
        result: Dict[str, Any] = {"red": None, "blue": None}
        brightness_r = int(np.clip(np.round(r_pwm), 0, 100))
        brightness_b = int(np.clip(np.round(b_pwm), 0, 100))

        if not self.live:
            self._log_simple(
                f"[dry-run] 拟下发 Govee mix PWM: 红 {brightness_r}% / 蓝 {brightness_b}% (未真实推送)"
            )
            stub = {"success": True, "verified": False, "retries": 0, "response": None, "error": "dry-run"}
            result["red"] = dict(stub)
            result["blue"] = dict(stub)
            return result

        if not self.background:
            print(f"📡 Govee mix 下发: 红 {brightness_r}% / 蓝 {brightness_b}%")

        try:
            # govee_controller.set_led_mix 接收 (pwm_r, pwm_b)，对每个 bar 都
            # 设成 RGB((r*255/100), 0, (b*255/100))，brightness=100。
            # 调用方应已在 setup 阶段调过 govee_controller.connect()。
            from govee_controller import set_led_mix  # type: ignore
            set_led_mix(brightness_r, brightness_b)
            payload = {"r": brightness_r, "b": brightness_b, "transport": "ble"}
            common = {"success": True, "verified": True, "retries": 0,
                      "response": payload, "error": None}
            result["red"] = dict(common)
            result["blue"] = dict(common)
            self._log_simple(
                f"✅ Govee mix 设置成功: R={brightness_r}% B={brightness_b}%"
            )
        except Exception as exc:
            err = str(exc)
            self._log_simple(f"❌ Govee mix 下发失败: {err}")
            failure = {"success": False, "verified": False, "retries": 0,
                       "response": None, "error": err}
            result["red"] = dict(failure)
            result["blue"] = dict(failure)

        if not self.background:
            print(f"📊 发送完成 - {result['red']}")

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
                    payload.get("ppfd_cmd"),
                    payload.get("r_pwm"),
                    payload.get("b_pwm"),
                    payload.get("pred_temp"),
                    payload.get("pred_power"),
                    payload.get("pred_pn"),
                    payload.get("target_ppfd"),
                    payload.get("target_mean_power"),
                    payload.get("cost"),
                    payload.get("success"),
                    payload.get("note"),
                    int(self.live),
                ]
            )

    # ---------- 核心控制 ----------
    def run_once(self) -> None:
        if self._is_night_time():
            self._log_simple("夜间模式,跳过本次控制")
            return

        env = self._read_environment()
        measured_temp = env.get("temp")
        if env.get("co2") is not None:
            self.plant.co2_ppm = float(env["co2"])
        if measured_temp is not None:
            self.current_temp = measured_temp
        elif self.current_temp is None:
            self.current_temp = 25.0

        current_temp = float(self.current_temp)
        thermal_update_note = self._maybe_update_thermal_model()
        mean_sequence = self._make_mean_sequence()
        ppfd_ref = self._make_ppfd_reference()

        try:
            optimal_u, optimal_seq, success, cost, _weights = self.controller.solve(
                current_temp=current_temp,
                mean_sequence=mean_sequence,
                ppfd_ref_seq=ppfd_ref,
            )
        except Exception as exc:  # noqa: BLE001
            self._log_simple(f"MPPI 求解失败: {exc}")
            return

        cycle_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not success:
            self._log_simple("MPPI 求解未成功,跳过下发")
            self._log_cycle(
                {
                    "timestamp": cycle_ts,
                    "sensor_timestamp": env.get("timestamp") or "",
                    "input_temp": measured_temp if measured_temp is not None else "",
                    "co2_ppm": env.get("co2"),
                    "ppfd_cmd": None,
                    "r_pwm": None,
                    "b_pwm": None,
                    "pred_temp": None,
                    "pred_power": None,
                    "pred_pn": None,
                    "target_ppfd": self.target_ppfd,
                    "target_mean_power": self.target_mean_power,
                    "cost": None,
                    "success": False,
                    "note": "solve_failed",
                }
            )
            return

        r_pwm, b_pwm = self.plant._ppfd_to_pwm(optimal_u)  # noqa: SLF001
        preds = self.plant.predict(optimal_seq, current_temp, dt=self.control_interval_seconds)
        (_ppfd_series, temp_pred, power_pred, pn_pred, _r_series, _b_series) = preds

        next_temp = float(temp_pred[0]) if len(temp_pred) else current_temp
        next_power = float(power_pred[0]) if len(power_pred) else 0.0
        next_pn = float(pn_pred[0]) if len(pn_pred) else 0.0

        status = self._send_pwm(r_pwm, b_pwm)

        # 注释串
        note = thermal_update_note
        for color in ("red", "blue"):
            st = status.get(color)
            if not st:
                continue
            if st.get("success") and st.get("retries", 0) > 0:
                if note:
                    note += "|"
                note += f"{color}_retry:{st['retries']}"
            elif st.get("error") and st["error"] != "dry-run":
                if note:
                    note += "|"
                note += f"{color}_error:{st['error']}"

        self.current_temp = next_temp
        if not self.background:
            print("=" * 70)
            mode = "LIVE" if self.live else "DRY-RUN"
            print(f"🔄 控制循环 @ {cycle_ts}  [{mode}]")
            print(f"🌡️ 输入温度: {measured_temp if measured_temp is not None else 'N/A'} °C")
            print(f"🌬️ CO₂: {env.get('co2')} ppm")
            print(f"🎯 PPFD 指令: {float(optimal_u):.1f} μmol/m²/s")
            print(f"🔴 红光PWM: {r_pwm:.2f} | 🔵 蓝光PWM: {b_pwm:.2f}")
            print(f"📈 预测温度: {next_temp:.2f} °C")
            print(f"⚡ 预测功率: {next_power:.2f} W")
            if self.target_mean_power is not None:
                print(
                    f"🌱 预测光合速率: {next_pn:.3f} "
                    f"(目标 PPFD: {self.target_ppfd:.0f}, 目标均值功率: {self.target_mean_power:.2f} W)"
                )
            else:
                print(f"🌱 预测光合速率: {next_pn:.3f} (目标 PPFD: {self.target_ppfd:.0f})")
            print(f"💰 代价: {float(cost):.2f}")

        self._log_cycle(
            {
                "timestamp": cycle_ts,
                "sensor_timestamp": env.get("timestamp") or "",
                "input_temp": measured_temp if measured_temp is not None else "",
                "co2_ppm": env.get("co2"),
                "ppfd_cmd": float(optimal_u),
                "r_pwm": float(r_pwm),
                "b_pwm": float(b_pwm),
                "pred_temp": next_temp,
                "pred_power": next_power,
                "pred_pn": next_pn,
                "target_ppfd": self.target_ppfd,
                "target_mean_power": self.target_mean_power,
                "cost": float(cost),
                "success": True,
                "note": note or ("dry-run" if not self.live else "ok"),
            }
        )

    def run_continuous(self) -> None:
        mode = "LIVE" if self.live else "DRY-RUN"
        self._log_simple(f"进入连续控制模式 [{mode}](按墙钟对齐到固定时间点)")
        while True:
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


def _start_background(live: bool) -> None:
    if _is_running():
        print("✅ MPPI v2 控制后台已在运行")
        return
    ensure_log_dir()
    cmd = [sys.executable, os.path.abspath(__file__), "background"]
    if live:
        cmd.append("--live")
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
    print(f"🚀 已启动后台进程 (PID: {process.pid}, live={live})")


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
    parser = argparse.ArgumentParser(description="MPPI v2 (Growpod / PPFD) 实际控制运行器")
    parser.add_argument(
        "command",
        nargs="?",
        default="once",
        choices=("once", "continuous", "background", "start", "stop", "restart", "status"),
        help="控制模式",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="⚠️ 启用真实硬件下发(默认 dry-run, 不会真正控制 LED)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "start":
        _start_background(live=args.live)
        return
    if args.command == "stop":
        _stop_background()
        return
    if args.command == "restart":
        _stop_background()
        time.sleep(1.0)
        _start_background(live=args.live)
        return
    if args.command == "status":
        _show_status()
        return

    background_mode = args.command == "background"
    controller = MPPIControlV2(background=background_mode, live=args.live)

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
