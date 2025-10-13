#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于 mppi_v2 的最新仿真脚本。

通过 Demo 传感器数据驱动 LEDPlant 与 LEDMPPIController，输出 Solar Vol 控制序列。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

CONTROL_INTERVAL_MINUTES = 15.0
DEFAULT_TARGET_SOLAR_VOL = 1.6
DEFAULT_REFERENCE_WEIGHT = 25.0
RB_RATIO = 0.83

# 项目路径设置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "mppi_v2_simulation_log.csv")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 兼容 mppi_v2 期望的 sensor_reading 接口
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

np.random.seed(42)


class MPPISimulationV2:
    """MPPI v2 仿真器，使用与实际控制脚本一致的 Solar Vol 参考配置。"""

    def __init__(
        self,
        *,
        control_interval_minutes: float = CONTROL_INTERVAL_MINUTES,
        horizon: int = 6,
        num_samples: int = 700,
        temperature: float = 1.0,
        u_std: float = 0.25,
        reference_weight: float = DEFAULT_REFERENCE_WEIGHT,
        target_solar_vol: Optional[float] = DEFAULT_TARGET_SOLAR_VOL,
    ) -> None:
        ensure_log_dir()
        self.control_interval_minutes = float(control_interval_minutes)
        self.dt_seconds = self.control_interval_minutes * 60.0
        self.target_solar_vol = float(target_solar_vol) if target_solar_vol is not None else None
        self.reference_weight = float(reference_weight)
        self.co2_fallback = float(DEFAULT_CO2_PPM)
        self.r_b_ratio = RB_RATIO

        power_model = self._load_power_model()
        self.plant = LEDPlant(
            base_ambient_temp=25.0,
            max_solar_vol=2.0,
            thermal_model_type='thermal',  # 使用热力学模型
            model_dir='Thermal/exported_models',  # 指定模型目录
            power_model=power_model,
            r_b_ratio=self.r_b_ratio,
            use_solar_vol_model=True,
        )

        self.controller = LEDMPPIController(
            plant=self.plant,
            horizon=int(horizon),
            num_samples=int(num_samples),
            dt=self.dt_seconds,
            temperature=float(temperature),
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
        self.controller.set_mppi_params(u_std=u_std, dt=self.dt_seconds)

        self.sensor_reader = self.plant.sensor_reader
        self.current_sim_temp: Optional[float] = None
        self._init_log()

    def _load_power_model(self) -> PWMtoPowerModel:
        calib_csv = os.path.join(PROJECT_ROOT, "data", "calib_data.csv")
        if not os.path.exists(calib_csv):
            raise FileNotFoundError(f"Power calibration file missing: {calib_csv}")
        model = PWMtoPowerModel(include_intercept=True)
        return model.fit(calib_csv)

    def _init_log(self) -> None:
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
                        "sequence_preview",
                    ]
                )

    def _read_sensors(self) -> Dict[str, Any]:
        temp, solar_vol, pn, ts = self.sensor_reader.read_latest_riotee_data()
        co2_val = self.sensor_reader.read_latest_co2_data()

        timestamp = ts.isoformat() if hasattr(ts, "isoformat") else (str(ts) if ts else None)
        fallback = co2_val is None

        return {
            "temp": float(temp) if temp is not None else None,
            "solar_vol": float(solar_vol) if solar_vol is not None else None,
            "pn": float(pn) if pn is not None else None,
            "timestamp": timestamp,
            "co2": self.co2_fallback if fallback else float(co2_val),
            "co2_fallback": fallback,
        }

    def _log_cycle(self, row: Dict[str, Any]) -> None:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    row.get("timestamp"),
                    row.get("sensor_timestamp"),
                    row.get("input_temp"),
                    row.get("co2_ppm"),
                    row.get("solar_vol_cmd"),
                    row.get("r_pwm"),
                    row.get("b_pwm"),
                    row.get("pred_temp"),
                    row.get("pred_power"),
                    row.get("pred_pn"),
                    row.get("target_solar_vol"),
                    row.get("cost"),
                    row.get("success"),
                    row.get("note"),
                    row.get("sequence_preview"),
                ]
            )

    def _make_solar_vol_reference(self) -> Optional[np.ndarray]:
        if self.target_solar_vol is None or self.reference_weight <= 0:
            return None
        return np.full(self.controller.horizon, self.target_solar_vol, dtype=float)

    def run_cycle(self, cycle_index: int) -> Dict[str, Any]:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        env = self._read_sensors()
        measured_temp = env.get("temp")

        if measured_temp is not None:
            self.current_sim_temp = measured_temp
        elif self.current_sim_temp is None:
            self.current_sim_temp = 25.0

        current_temp = float(self.current_sim_temp)
        solar_vol_ref = self._make_solar_vol_reference()
        solar_ref_list = solar_vol_ref.tolist() if solar_vol_ref is not None else None
        notes: list[str] = []

        if env.get("co2_fallback"):
            notes.append("co2_fallback")

        try:
            optimal_sv, optimal_seq, success, cost, _weights = self.controller.solve(
                current_temp=current_temp,
                solar_vol_ref_seq=solar_vol_ref,
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"solve_error:{exc}")
            row = {
                "timestamp": timestamp,
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
                "note": "|".join(notes) if notes else "solve_exception",
                "sequence_preview": "",
                "solar_vol_ref_seq": solar_ref_list,
            }
            self._log_cycle(row)
            raise

        if not success:
            notes.append("solve_failed")
            row = {
                "timestamp": timestamp,
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
                "cost": float(cost),
                "success": False,
                "note": "|".join(notes),
                "sequence_preview": "",
                "solar_vol_ref_seq": solar_ref_list,
            }
            self._log_cycle(row)
            raise RuntimeError("MPPI solve failed")

        r_pwm, b_pwm = self.plant._solar_vol_to_pwm(optimal_sv)
        _sv_series, temp_pred, power_pred, pn_pred, r_series, b_series = self.plant.predict(
            optimal_seq,
            current_temp,
            dt=self.dt_seconds,
        )

        next_temp = float(temp_pred[0]) if len(temp_pred) > 0 else current_temp
        next_power = float(power_pred[0]) if len(power_pred) > 0 else 0.0
        next_pn = float(pn_pred[0]) if len(pn_pred) > 0 else 0.0

        self.current_sim_temp = next_temp

        preview = f"{optimal_seq[0]:.3f}" if len(optimal_seq) else ""
        if len(optimal_seq) > 1:
            preview += "|" + "|".join(f"{u:.3f}" for u in optimal_seq[1:3])

        row = {
            "timestamp": timestamp,
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
            "note": "|".join(notes) if notes else "ok",
            "sequence_preview": preview,
            "solar_vol_ref_seq": solar_ref_list,
        }

        self._log_cycle(row)
        self._print_cycle(
            cycle_index,
            row,
            optimal_seq,
            temp_pred,
            power_pred,
            pn_pred,
            r_series,
            b_series,
        )
        return row

    def _print_cycle(
        self,
        cycle_index: int,
        row: Dict[str, Any],
        optimal_seq: np.ndarray,
        temp_pred: np.ndarray,
        power_pred: np.ndarray,
        pn_pred: np.ndarray,
        r_series: np.ndarray,
        b_series: np.ndarray,
    ) -> None:
        print("=" * 70)
        print(f"🔄 仿真循环 {cycle_index + 1}")
        print(f"🕒 时间: {row['timestamp']}")
        print(f"🌡️ 输入温度: {row['input_temp'] if row['input_temp'] is not None else 'N/A'} °C")
        print(f"🌬️ CO₂: {row['co2_ppm'] if row['co2_ppm'] is not None else 'N/A'} ppm")
        print(f"🎯 Solar Vol 指令: {row['solar_vol_cmd']:.3f}")
        print(f"🔴 红光PWM: {row['r_pwm']:.2f} | 🔵 蓝光PWM: {row['b_pwm']:.2f}")
        print(f"📈 预测温度: {row['pred_temp']:.2f} °C")
        print(f"⚡ 预测功率: {row['pred_power']:.2f} W")
        target_sv = row.get("target_solar_vol")
        if target_sv is not None:
            print(f"🌱 预测光合速率: {row['pred_pn']:.3f} (目标Solar Vol: {target_sv:.3f})")
        else:
            print(f"🌱 预测光合速率: {row['pred_pn']:.3f}")
        print(f"💰 代价: {row['cost']:.2f}")
        print(f"🗒️ 备注: {row['note']}")
        print(f"🔍 序列前瞻: {row['sequence_preview']}")
        if len(optimal_seq) > 0:
            print(f"   控制序列: {[round(x, 4) for x in optimal_seq.tolist()]}")
        if len(temp_pred) > 0:
            print(f"   温度预测序列: {[round(x, 3) for x in temp_pred.tolist()]}")
        if len(power_pred) > 0:
            print(f"   功率预测序列: {[round(x, 3) for x in power_pred.tolist()]}")
        if len(pn_pred) > 0:
            print(f"   光合预测序列: {[round(x, 3) for x in pn_pred.tolist()]}")
        ref_seq = row.get("solar_vol_ref_seq")
        if ref_seq:
            print(f"   光合参考序列: {[round(x, 3) for x in ref_seq]}")
        if len(r_series) > 0 and len(b_series) > 0:
            rb_pairs = list(zip(r_series.tolist(), b_series.tolist()))
            print(f"   PWM预测序列: {[('%.1f' % rp, '%.1f' % bp) for rp, bp in rb_pairs]}")

    def run(self, steps: int) -> None:
        """执行仿真主循环。

        参数：
        - steps: 要运行的循环次数。
        """
        for i in range(steps):
            try:
                self.run_cycle(i)
            except Exception as exc:  # noqa: BLE001
                print(f"❌ 仿真循环失败: {exc}")
                break


def parse_args() -> argparse.Namespace:
    """解析命令行参数（中文帮助）。"""
    parser = argparse.ArgumentParser(description="MPPI v2 仿真运行器")
    # 基础运行参数
    parser.add_argument("--steps", type=int, default=8, help="仿真循环次数")  # 运行步数
    parser.add_argument(
        "--interval",
        type=float,
        default=CONTROL_INTERVAL_MINUTES,
        help="控制间隔（分钟），影响 dt",
    )  # 控制间隔
    parser.add_argument("--horizon", type=int, default=6, help="MPPI 预测地平线长度（步）")  # 预测步长

    # MPPI 采样/温度
    parser.add_argument("--samples", type=int, default=700, help="MPPI 采样数量，越大越准但更慢")  # 采样数
    parser.add_argument("--temperature", type=float, default=1.0, help="MPPI 温度参数（熵强度）")  # 温度
    parser.add_argument("--ustd", type=float, default=0.25, help="控制增量采样标准差（探索幅度）")  # 采样扰动尺度
    parser.add_argument(
        "--target-solar",
        "--target-pn",
        dest="target_solar",
        type=float,
        default=DEFAULT_TARGET_SOLAR_VOL,
        help=f"Solar Vol 参考值，用于基准跟踪，默认 {DEFAULT_TARGET_SOLAR_VOL}",
    )
    parser.add_argument(
        "--ref-weight",
        type=float,
        default=DEFAULT_REFERENCE_WEIGHT,
        help=f"Solar Vol 参考误差惩罚权重 (Q_ref)，默认 {DEFAULT_REFERENCE_WEIGHT}",  # 参考跟踪权重
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口：构建仿真器并按参数运行。"""
    args = parse_args()
    sim = MPPISimulationV2(
        control_interval_minutes=args.interval,
        horizon=args.horizon,
        num_samples=args.samples,
        temperature=args.temperature,
        u_std=args.ustd,
        target_solar_vol=args.target_solar,
        reference_weight=args.ref_weight,
    )
    sim.run(args.steps)


if __name__ == "__main__":
    main()
