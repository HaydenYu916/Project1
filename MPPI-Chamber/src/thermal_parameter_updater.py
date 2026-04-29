from __future__ import annotations

import copy
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from led import LedThermalParams, ThermalModelManager


@dataclass
class ThermalLogRow:
    timestamp: datetime
    input_temp: float
    solar_vol_cmd: float


class ThermalParameterUpdater:
    """基于控制日志的小步长热模型参数校准器。

    设计目标：
    - 基础热模型始终可用；
    - 在线校准仅在日志足够、且验证误差确实下降时才生效；
    - 校准失败时继续沿用旧参数，不影响控制主循环。
    """

    def __init__(
        self,
        *,
        model_dir: str,
        runtime_params_path: str,
        log_path: str,
        base_ambient_temp: float,
        model_type: str = "thermal",
        solar_threshold: float = 1.4,
        solar_change_tolerance: float = 0.02,
        control_interval_seconds: float = 900.0,
        update_interval_seconds: float = 3600.0,
        window_hours: float = 6.0,
        min_samples: int = 24,
        min_improvement_c: float = 0.12,
        min_improvement_ratio: float = 0.05,
    ) -> None:
        self.model_dir = model_dir
        self.runtime_params_path = runtime_params_path
        self.log_path = log_path
        self.base_ambient_temp = float(base_ambient_temp)
        self.model_type = model_type
        self.solar_threshold = float(solar_threshold)
        self.solar_change_tolerance = float(solar_change_tolerance)
        self.control_interval_seconds = float(control_interval_seconds)
        self.update_interval_seconds = float(update_interval_seconds)
        self.window_hours = float(window_hours)
        self.min_samples = int(min_samples)
        self.min_improvement_c = float(min_improvement_c)
        self.min_improvement_ratio = float(min_improvement_ratio)
        self.max_gap_seconds = max(self.control_interval_seconds * 2.5, self.control_interval_seconds + 60.0)
        self.state_path = f"{self.runtime_params_path}.state.json"
        self._scale_candidates = [0.7, 0.85, 1.0, 1.15, 1.3]

    def maybe_update(self, now: Optional[datetime] = None) -> Dict[str, object]:
        current_time = now or datetime.now()
        if not self._is_due(current_time):
            return {"status": "skipped", "reason": "rate_limited"}

        rows = self._load_recent_rows(current_time)
        if len(rows) < self.min_samples:
            self._write_state(
                {
                    "last_attempt_at": current_time.isoformat(),
                    "last_status": "skipped",
                    "reason": f"insufficient_samples:{len(rows)}",
                }
            )
            return {"status": "skipped", "reason": "insufficient_samples", "samples": len(rows)}

        current_payload = self._load_runtime_payload() or self._empty_runtime_payload()
        baseline_mae = self._score_rows(rows, current_payload)
        candidate_payload = self._fit_payload(rows)
        candidate_mae = self._score_rows(rows, candidate_payload)

        improvement = baseline_mae - candidate_mae
        ratio_improvement = 0.0 if baseline_mae <= 1e-9 else improvement / baseline_mae
        if (
            not math.isfinite(candidate_mae)
            or candidate_mae >= baseline_mae - self.min_improvement_c
            and ratio_improvement < self.min_improvement_ratio
        ):
            self._write_state(
                {
                    "last_attempt_at": current_time.isoformat(),
                    "last_status": "skipped",
                    "reason": "no_material_improvement",
                    "baseline_mae": baseline_mae,
                    "candidate_mae": candidate_mae,
                }
            )
            return {
                "status": "skipped",
                "reason": "no_material_improvement",
                "samples": len(rows),
                "baseline_mae": round(baseline_mae, 4),
                "candidate_mae": round(candidate_mae, 4),
            }

        candidate_payload["meta"] = {
            "updated_at": current_time.isoformat(),
            "window_hours": self.window_hours,
            "samples": len(rows),
            "baseline_mae": baseline_mae,
            "candidate_mae": candidate_mae,
        }
        self._write_runtime_payload(candidate_payload)
        self._write_state(
            {
                "last_attempt_at": current_time.isoformat(),
                "last_status": "updated",
                "baseline_mae": baseline_mae,
                "candidate_mae": candidate_mae,
            }
        )
        return {
            "status": "updated",
            "samples": len(rows),
            "baseline_mae": round(baseline_mae, 4),
            "candidate_mae": round(candidate_mae, 4),
        }

    def _is_due(self, now: datetime) -> bool:
        state = self._read_state()
        last_attempt_at = state.get("last_attempt_at")
        if not isinstance(last_attempt_at, str):
            return True
        try:
            last_attempt = datetime.fromisoformat(last_attempt_at)
        except ValueError:
            return True
        return (now - last_attempt).total_seconds() >= self.update_interval_seconds

    def _read_state(self) -> Dict[str, object]:
        if not os.path.exists(self.state_path):
            return {}
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _write_state(self, payload: Dict[str, object]) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        tmp_path = f"{self.state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        os.replace(tmp_path, self.state_path)

    def _load_recent_rows(self, now: datetime) -> List[ThermalLogRow]:
        if not os.path.exists(self.log_path):
            return []

        parsed: List[ThermalLogRow] = []
        with open(self.log_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_text = (row.get("timestamp") or "").strip()
                temp_text = (row.get("input_temp") or "").strip()
                solar_text = (row.get("solar_vol_cmd") or "").strip()
                success_text = (row.get("success") or "").strip().lower()
                if success_text not in {"true", "1", "yes"}:
                    continue
                if not ts_text or not temp_text or not solar_text:
                    continue
                try:
                    timestamp = datetime.fromisoformat(ts_text)
                    input_temp = float(temp_text)
                    solar_vol_cmd = float(solar_text)
                except ValueError:
                    continue
                parsed.append(
                    ThermalLogRow(
                        timestamp=timestamp,
                        input_temp=input_temp,
                        solar_vol_cmd=solar_vol_cmd,
                    )
                )

        if not parsed:
            return []

        parsed.sort(key=lambda row: row.timestamp)
        anchor = min(now, parsed[-1].timestamp)
        cutoff = anchor - timedelta(hours=self.window_hours)
        return [row for row in parsed if row.timestamp >= cutoff]

    def _load_base_phase_params(self, phase: str) -> Dict[str, object]:
        filename = f"{phase}_thermal_model.json"
        path = os.path.join(self.model_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise RuntimeError(f"{filename} 格式无效")
        return payload

    def _empty_runtime_payload(self) -> Dict[str, object]:
        return {"heating": {}, "cooling": {}, "meta": {}}

    def _load_runtime_payload(self) -> Optional[Dict[str, object]]:
        if not os.path.exists(self.runtime_params_path):
            return None
        try:
            with open(self.runtime_params_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _write_runtime_payload(self, payload: Dict[str, object]) -> None:
        os.makedirs(os.path.dirname(self.runtime_params_path), exist_ok=True)
        tmp_path = f"{self.runtime_params_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        os.replace(tmp_path, self.runtime_params_path)

    def _build_phase_override(
        self,
        base_phase: Dict[str, object],
        *,
        gain_scale: float,
        tau_scale: float,
        alpha_scale: float,
    ) -> Dict[str, object]:
        parameters = copy.deepcopy(base_phase.get("parameters", {}))
        for key in ("K1_base", "K2_base"):
            if isinstance(parameters.get(key), (int, float)):
                parameters[key] = float(parameters[key]) * gain_scale
        for key in ("tau1", "tau2"):
            if isinstance(parameters.get(key), (int, float)):
                parameters[key] = max(1e-6, float(parameters[key]) * tau_scale)
        if isinstance(parameters.get("alpha_solar"), (int, float)):
            parameters["alpha_solar"] = float(parameters["alpha_solar"]) * alpha_scale
        override = {"parameters": parameters}
        if isinstance(base_phase.get("a1_ref"), (int, float)):
            override["a1_ref"] = float(base_phase["a1_ref"])
        return override

    def _phase_for_interval(self, rows: List[ThermalLogRow], idx: int) -> str:
        if idx >= 2:
            control_change = rows[idx - 1].solar_vol_cmd - rows[idx - 2].solar_vol_cmd
            if abs(control_change) > self.solar_change_tolerance:
                return "heating" if control_change > 0 else "cooling"
        return "heating" if rows[idx - 1].solar_vol_cmd > self.solar_threshold else "cooling"

    def _build_manager(self, payload: Dict[str, object]) -> ThermalModelManager:
        params = LedThermalParams(
            base_ambient_temp=self.base_ambient_temp,
            model_type=self.model_type,
            model_dir=self.model_dir,
            solar_threshold=self.solar_threshold,
            solar_change_tolerance=self.solar_change_tolerance,
            adaptive_enabled=False,
            adaptive_params_path="",
        )
        manager = ThermalModelManager(params)
        manager.apply_runtime_parameters_payload(payload)
        return manager

    def _score_rows(
        self,
        rows: List[ThermalLogRow],
        payload: Dict[str, object],
        phase_filter: Optional[str] = None,
    ) -> float:
        if len(rows) < 2:
            return math.inf

        manager = self._build_manager(payload)
        manager.reset(rows[0].input_temp)
        errors: List[float] = []

        for idx in range(1, len(rows)):
            previous_row = rows[idx - 1]
            current_row = rows[idx]
            dt_seconds = (current_row.timestamp - previous_row.timestamp).total_seconds()
            if dt_seconds <= 0:
                continue
            if dt_seconds > self.max_gap_seconds:
                manager.reset(current_row.input_temp)
                continue

            control_change = None
            if idx >= 2:
                control_change = previous_row.solar_vol_cmd - rows[idx - 2].solar_vol_cmd

            predicted_temp = manager.step(
                power=0.0,
                dt=dt_seconds,
                solar_vol=previous_row.solar_vol_cmd,
                control_change=control_change,
            )
            interval_phase = self._phase_for_interval(rows, idx)
            if phase_filter is None or interval_phase == phase_filter:
                errors.append(abs(predicted_temp - current_row.input_temp))
            manager.sync_observation(current_row.input_temp)

        if not errors:
            return math.inf
        return sum(errors) / len(errors)

    def _fit_payload(self, rows: List[ThermalLogRow]) -> Dict[str, object]:
        base_heating = self._load_base_phase_params("heating")
        base_cooling = self._load_base_phase_params("cooling")
        payload = self._empty_runtime_payload()

        phase_to_base = {"heating": base_heating, "cooling": base_cooling}
        min_phase_samples = max(4, self.min_samples // 4)

        for phase, base_phase in phase_to_base.items():
            phase_count = sum(1 for idx in range(1, len(rows)) if self._phase_for_interval(rows, idx) == phase)
            if phase_count < min_phase_samples:
                payload[phase] = {}
                continue

            best_scales = {"gain_scale": 1.0, "tau_scale": 1.0, "alpha_scale": 1.0}
            for _ in range(2):
                for scale_name in ("gain_scale", "tau_scale", "alpha_scale"):
                    best_score = math.inf
                    best_value = best_scales[scale_name]
                    for candidate in self._scale_candidates:
                        trial_scales = dict(best_scales)
                        trial_scales[scale_name] = candidate
                        trial_payload = copy.deepcopy(payload)
                        trial_payload[phase] = self._build_phase_override(base_phase, **trial_scales)
                        score = self._score_rows(rows, trial_payload, phase_filter=phase)
                        if score < best_score:
                            best_score = score
                            best_value = candidate
                    best_scales[scale_name] = best_value

            payload[phase] = self._build_phase_override(base_phase, **best_scales)

        return payload
