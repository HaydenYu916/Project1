#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于 mppi_v2 的 Growpod 仿真脚本 (PPFD 控制)。

通过 Demo 传感器数据驱动 LEDPlant 与 LEDMPPIController,输出 PPFD 控制序列。
链路: spectrum -> PPFD (观测), PPFD -> (R_PWM,B_PWM) (执行), PPFD -> Pn (评估)。
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
DEFAULT_TARGET_PPFD = 400.0
DEFAULT_REFERENCE_WEIGHT = 0.0
DEFAULT_POWER_BUDGET_WEIGHT = 0.0   # 与 R_power 互斥; 默认关闭, 让 R_power 主导
RB_RATIO = 0.83

# 项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "mppi_v2_simulation_log.csv")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 兼容 mppi_v2 的 sensor_reading 接口
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
    """MPPI v2 仿真器 (PPFD 控制)。"""

    def __init__(
        self,
        *,
        control_interval_minutes: float = CONTROL_INTERVAL_MINUTES,
        horizon: int = 6,
        num_samples: int = 700,
        temperature: float = 1.0,
        u_std: float = 20.0,
        reference_weight: float = DEFAULT_REFERENCE_WEIGHT,
        power_budget_weight: float = DEFAULT_POWER_BUDGET_WEIGHT,
        target_ppfd: Optional[float] = DEFAULT_TARGET_PPFD,
    ) -> None:
        ensure_log_dir()
        self.control_interval_minutes = float(control_interval_minutes)
        self.dt_seconds = self.control_interval_minutes * 60.0
        self.target_ppfd = float(target_ppfd) if target_ppfd is not None else None
        self.reference_weight = float(reference_weight)
        self.power_budget_weight = float(power_budget_weight)
        self.target_mean_power: Optional[float] = None
        self.co2_fallback = float(DEFAULT_CO2_PPM)
        self.r_b_ratio = RB_RATIO

        power_model = self._load_power_model()
        self.plant = LEDPlant(
            base_ambient_temp=25.0,
            max_ppfd=500.0,
            thermal_model_type='thermal',
            model_dir='Thermal/exported_models',
            power_model=power_model,
            r_b_ratio=self.r_b_ratio,
            target_ppfd=self.target_ppfd if self.target_ppfd is not None else DEFAULT_TARGET_PPFD,
            use_pn_model=True,
            use_sp_to_ppfd=False,   # 仿真不读光谱
        )

        self.controller = LEDMPPIController(
            plant=self.plant,
            horizon=int(horizon),
            num_samples=int(num_samples),
            dt=self.dt_seconds,
            temperature=float(temperature),
        )
        self.controller.set_constraints(
            u_min=80.0,
            u_max=float(self.plant.max_ppfd),
            temp_min=20.0,
            temp_max=29.8,
        )
        # Q_ref 由命令行控制; 其它权重用 controller 默认值
        self.controller.set_weights(Q_ref=self.reference_weight)

        # 功率预算 (与 R_power 互斥; >0 时 controller 会自动把 R_power 清零)
        self.target_mean_power = self._estimate_target_mean_power()
        if self.target_mean_power is not None and self.power_budget_weight > 0.0:
            self.controller.set_power_budget(
                target_mean_power=self.target_mean_power,
                power_budget_weight=self.power_budget_weight,
            )

        self.controller.set_mppi_params(u_std=u_std, dt=self.dt_seconds)

        self.sensor_reader = self.plant.sensor_reader
        self.current_sim_temp: Optional[float] = None
        self._init_log()

    def _load_power_model(self) -> PWMtoPowerModel:
        calib_csv = os.path.join(PROJECT_ROOT, "data", "calib_data.csv")
        if not os.path.exists(calib_csv):
            raise FileNotFoundError(f"Power calibration file missing: {calib_csv}")
        return PWMtoPowerModel(include_intercept=True).fit(calib_csv)

    def _estimate_target_mean_power(self) -> Optional[float]:
        if self.target_ppfd is None:
            return None
        r_pwm, b_pwm = self.plant._ppfd_to_pwm(float(self.target_ppfd))
        total_pwm = float(r_pwm + b_pwm)
        power_key = self.plant._get_power_model_key(self.plant.r_b_ratio)
        return float(self.plant.power_model.predict(total_pwm=total_pwm, key=power_key))

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
                        "sequence_preview",
                    ]
                )

    def _read_sensors(self) -> Dict[str, Any]:
        """适配 mppi_v2.SensorReading (5-tuple) 与 DemoSensorReader (4-tuple) 两种返回。"""
        result = self.sensor_reader.read_latest_riotee_data()
        if len(result) == 5:
            temp, solar_vol, pn, ts, _spectrum = result
        else:
            temp, solar_vol, pn, ts = result
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
                    row.get("ppfd_cmd"),
                    row.get("r_pwm"),
                    row.get("b_pwm"),
                    row.get("pred_temp"),
                    row.get("pred_power"),
                    row.get("pred_pn"),
                    row.get("target_ppfd"),
                    row.get("target_mean_power"),
                    row.get("cost"),
                    row.get("success"),
                    row.get("note"),
                    row.get("sequence_preview"),
                ]
            )

    def _make_ppfd_reference(self) -> Optional[np.ndarray]:
        if self.target_ppfd is None or self.reference_weight <= 0:
            return None
        return np.full(self.controller.horizon, self.target_ppfd, dtype=float)

    def _make_mean_sequence(self) -> Optional[np.ndarray]:
        if self.target_ppfd is None:
            return None
        return np.full(self.controller.horizon, self.target_ppfd, dtype=float)

    def run_cycle(self, cycle_index: int) -> Dict[str, Any]:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        env = self._read_sensors()
        measured_temp = env.get("temp")
        if env.get("co2") is not None:
            self.plant.co2_ppm = float(env["co2"])

        if measured_temp is not None:
            self.current_sim_temp = measured_temp
        elif self.current_sim_temp is None:
            self.current_sim_temp = 25.0

        current_temp = float(self.current_sim_temp)
        mean_sequence = self._make_mean_sequence()
        ppfd_ref = self._make_ppfd_reference()
        ref_list = ppfd_ref.tolist() if ppfd_ref is not None else None
        notes: list[str] = []

        if env.get("co2_fallback"):
            notes.append("co2_fallback")

        try:
            optimal_u, optimal_seq, success, cost, _weights = self.controller.solve(
                current_temp=current_temp,
                mean_sequence=mean_sequence,
                ppfd_ref_seq=ppfd_ref,
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"solve_error:{exc}")
            row = {
                "timestamp": timestamp,
                "sensor_timestamp": env.get("timestamp"),
                "input_temp": measured_temp,
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
                "note": "|".join(notes) if notes else "solve_exception",
                "sequence_preview": "",
                "ppfd_ref_seq": ref_list,
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
                "ppfd_cmd": None,
                "r_pwm": None,
                "b_pwm": None,
                "pred_temp": None,
                "pred_power": None,
                "pred_pn": None,
                "target_ppfd": self.target_ppfd,
                "target_mean_power": self.target_mean_power,
                "cost": float(cost),
                "success": False,
                "note": "|".join(notes),
                "sequence_preview": "",
                "ppfd_ref_seq": ref_list,
            }
            self._log_cycle(row)
            raise RuntimeError("MPPI solve failed")

        r_pwm, b_pwm = self.plant._ppfd_to_pwm(optimal_u)
        _ppfd_series, temp_pred, power_pred, pn_pred, r_series, b_series = self.plant.predict(
            optimal_seq,
            current_temp,
            dt=self.dt_seconds,
        )

        next_temp = float(temp_pred[0]) if len(temp_pred) > 0 else current_temp
        next_power = float(power_pred[0]) if len(power_pred) > 0 else 0.0
        next_pn = float(pn_pred[0]) if len(pn_pred) > 0 else 0.0
        self.current_sim_temp = next_temp

        preview = f"{optimal_seq[0]:.1f}" if len(optimal_seq) else ""
        if len(optimal_seq) > 1:
            preview += "|" + "|".join(f"{u:.1f}" for u in optimal_seq[1:3])

        row = {
            "timestamp": timestamp,
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
            "note": "|".join(notes) if notes else "ok",
            "sequence_preview": preview,
            "ppfd_ref_seq": ref_list,
        }

        self._log_cycle(row)
        self._print_cycle(cycle_index, row, optimal_seq, temp_pred, power_pred, pn_pred, r_series, b_series)
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
        print(f"🎯 PPFD 指令: {row['ppfd_cmd']:.1f} μmol/m²/s")
        print(f"🔴 红光PWM: {row['r_pwm']:.2f} | 🔵 蓝光PWM: {row['b_pwm']:.2f}")
        print(f"📈 预测温度: {row['pred_temp']:.2f} °C")
        print(f"⚡ 预测功率: {row['pred_power']:.2f} W")
        target_power = row.get("target_mean_power")
        target_ppfd = row.get("target_ppfd")
        if target_power is not None and target_ppfd is not None:
            print(
                f"🌱 预测光合速率: {row['pred_pn']:.3f} "
                f"(目标 PPFD: {target_ppfd:.0f}, 目标均值功率: {target_power:.2f} W)"
            )
        elif target_ppfd is not None:
            print(f"🌱 预测光合速率: {row['pred_pn']:.3f} (目标 PPFD: {target_ppfd:.0f})")
        else:
            print(f"🌱 预测光合速率: {row['pred_pn']:.3f}")
        print(f"💰 代价: {row['cost']:.2f}")
        print(f"🗒️ 备注: {row['note']}")
        print(f"🔍 序列前瞻: {row['sequence_preview']}")
        if len(optimal_seq) > 0:
            print(f"   控制序列: {[round(x, 1) for x in optimal_seq.tolist()]}")
        if len(temp_pred) > 0:
            print(f"   温度预测序列: {[round(x, 3) for x in temp_pred.tolist()]}")
        if len(power_pred) > 0:
            print(f"   功率预测序列: {[round(x, 3) for x in power_pred.tolist()]}")
        if len(pn_pred) > 0:
            print(f"   光合预测序列: {[round(x, 3) for x in pn_pred.tolist()]}")
        ref_seq = row.get("ppfd_ref_seq")
        if ref_seq:
            print(f"   PPFD 参考序列: {[round(x, 1) for x in ref_seq]}")
        if len(r_series) > 0 and len(b_series) > 0:
            rb_pairs = list(zip(r_series.tolist(), b_series.tolist()))
            print(f"   PWM预测序列: {[('%.1f' % rp, '%.1f' % bp) for rp, bp in rb_pairs]}")

    def run(self, steps: int) -> None:
        for i in range(steps):
            try:
                self.run_cycle(i)
            except Exception as exc:  # noqa: BLE001
                print(f"❌ 仿真循环失败: {exc}")
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPPI v2 (Growpod / PPFD) 仿真运行器")
    parser.add_argument("--steps", type=int, default=8, help="仿真循环次数")
    parser.add_argument("--interval", type=float, default=CONTROL_INTERVAL_MINUTES, help="控制间隔(分钟)")
    parser.add_argument("--horizon", type=int, default=6, help="MPPI 预测地平线长度(步)")
    parser.add_argument("--samples", type=int, default=700, help="MPPI 采样数")
    parser.add_argument("--temperature", type=float, default=1.0, help="MPPI 温度参数(熵强度)")
    parser.add_argument("--ustd", type=float, default=20.0, help="PPFD 控制噪声标准差")
    parser.add_argument(
        "--target-ppfd",
        dest="target_ppfd",
        type=float,
        default=DEFAULT_TARGET_PPFD,
        help=f"PPFD 参考值,默认 {DEFAULT_TARGET_PPFD}",
    )
    parser.add_argument(
        "--ref-weight",
        type=float,
        default=DEFAULT_REFERENCE_WEIGHT,
        help=f"PPFD 参考误差惩罚权重 (Q_ref),默认 {DEFAULT_REFERENCE_WEIGHT}",
    )
    parser.add_argument(
        "--power-budget-weight",
        type=float,
        default=DEFAULT_POWER_BUDGET_WEIGHT,
        help=f"平均功率预算惩罚权重(>0 时与 R_power 互斥),默认 {DEFAULT_POWER_BUDGET_WEIGHT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim = MPPISimulationV2(
        control_interval_minutes=args.interval,
        horizon=args.horizon,
        num_samples=args.samples,
        temperature=args.temperature,
        u_std=args.ustd,
        target_ppfd=args.target_ppfd,
        reference_weight=args.ref_weight,
        power_budget_weight=args.power_budget_weight,
    )
    sim.run(args.steps)


if __name__ == "__main__":
    main()
