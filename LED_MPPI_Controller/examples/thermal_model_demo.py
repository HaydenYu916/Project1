#!/usr/bin/env python3
"""
热力学模型示例（Thermal Model Demo）

包含两部分：
1) 统一模型（UnifiedPPFDThermalModel）：以 PPFD 作为输入的阶跃响应（单图）
2) 统一模型多图：PPFD 从 100 到 600，步长 100，分别输出单独图片

将在 examples/result 目录下输出一张对比图 thermal_model_demo.png。
"""

import os
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 将 src 加入路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from led import (
    LedThermalParams,
    UnifiedPPFDThermalModel,
    UnifiedPPFDParams,
)


def load_fitted_params(json_path: str | None) -> UnifiedPPFDParams | None:
    """从07脚本导出的JSON加载拟合系数。如果找不到则返回None。"""
    if not json_path:
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        p = data.get('parameters', {})
        return UnifiedPPFDParams(
            k1_a=float(p['K1_a']), k1_b=float(p['K1_b']),
            t1_a=float(p['tau1_a']), t1_b=float(p['tau1_b']),
            k2_a=float(p['K2_a']),  k2_p=float(p['K2_b']),
            t2_a=float(p['tau2_a']), t2_p=float(p['tau2_b']),
        )
    except Exception:
        return None


def simulate_unified_ppfd(ppfd: float = 300.0, total_hours: float = 6.0, dt_hours: float = 0.05, *, model_params: UnifiedPPFDParams | None = None):
    # 与 07 脚本对齐：时间单位使用“小时”，输出温差 ΔT（不叠加基线）
    params = LedThermalParams(base_ambient_temp=25.0)
    model = UnifiedPPFDThermalModel(params, initial_temp=25.0, model_params=model_params)
    t_h = np.arange(0.0, total_hours + dt_hours, dt_hours)
    temps = []
    for _ in t_h:
        temps.append(model.step(ppfd=ppfd, dt=dt_hours))  # 这里 dt 的单位与 07 保持一致（小时）
    temps = np.array(temps)
    deltas = temps - params.base_ambient_temp
    return t_h, deltas


def simulate_unified_ppfd_with_cooling(ppfd: float, on_hours: float, off_hours: float, dt_hours: float, *, model_params: UnifiedPPFDParams | None = None):
    """先加热(on_hours)后降温(off_hours)。返回拼接后的时间(小时)与ΔT序列。"""
    params = LedThermalParams(base_ambient_temp=25.0)
    model = UnifiedPPFDThermalModel(params, initial_temp=25.0, model_params=model_params)
    # 加热阶段
    t_on = np.arange(0.0, on_hours + dt_hours, dt_hours)
    temps_on = []
    for _ in t_on:
        temps_on.append(model.step(ppfd=ppfd, dt=dt_hours))
    # 降温阶段（PPFD=0）
    t_off = np.arange(dt_hours, off_hours + dt_hours, dt_hours)
    temps_off = []
    for _ in t_off:
        temps_off.append(model.step(ppfd=0.0, dt=dt_hours))
    # 组合
    t = np.concatenate([t_on, on_hours + t_off])
    temps = np.array(list(temps_on) + list(temps_off))
    deltas = temps - params.base_ambient_temp
    return t, deltas


def main():
    base_dir = ROOT / "examples" / "result"
    out_dir = base_dir / "thermal_model_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 可选：自动读取07脚本导出的JSON参数
    # 你可以按需修改路径
    candidate_jsons = [
        ROOT.parent / "Thermal-Test" / "src5" / "models" / "unified_ppfd_temp_diff_second_order_model.json",
        ROOT / ".." / ".." / "Thermal-Test" / "src5" / "models" / "unified_ppfd_temp_diff_second_order_model.json",
    ]
    fit_params = None
    for jp in candidate_jsons:
        if os.path.exists(jp):
            fit_params = load_fitted_params(str(jp))
            if fit_params:
                break

    # 仿真参数（可根据需要调整）
    # 与 07 对齐：0-6 小时
    total_hours = 6.0
    dt_hours = 0.05

    # part 1: 单图（统一模型，示例 PPFD=300）
    ppfd_step = 300.0
    t3, y3 = simulate_unified_ppfd(ppfd=ppfd_step, total_hours=total_hours, dt_hours=dt_hours, model_params=fit_params)
    plt.figure(figsize=(10, 6))
    plt.plot(t3, y3, label=f"Unified (PPFD={ppfd_step})", linewidth=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature Difference (°C)")
    plt.title("Unified Thermal Model (Single PPFD) - ΔT")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path_single = out_dir / "unified_ppfd_300.png"
    plt.tight_layout()
    plt.savefig(out_path_single, dpi=300)
    print(f"Saved: {out_path_single}")

    # part 2: 多图（PPFD: 100..600, step 100）
    ppfd_values = list(range(100, 601, 100))
    series = []  # 收集时间与温度序列用于叠加图
    for v in ppfd_values:
        t, y = simulate_unified_ppfd(ppfd=float(v), total_hours=total_hours, dt_hours=dt_hours, model_params=fit_params)
        series.append((v, t, y))
        plt.figure(figsize=(10, 6))
        plt.plot(t, y, label=f"Unified (PPFD={v})", linewidth=2)
        plt.xlabel("Time (hours)")
        plt.ylabel("Temperature Difference (°C)")
        plt.title(f"Unified Thermal Model (PPFD={v}) - ΔT")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path = out_dir / f"unified_ppfd_{v}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        print(f"Saved: {out_path}")

    # part 3: 叠加图（100..600 一张图）
    plt.figure(figsize=(12, 7))
    for v, t, y in series:
        plt.plot(t, y, linewidth=2, label=f"PPFD={v}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature Difference (°C)")
    plt.title("Unified Thermal Model (PPFD 100..600) - ΔT")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Curves")
    out_overlay = out_dir / "unified_ppfd_overlay.png"
    plt.tight_layout()
    plt.savefig(out_overlay, dpi=300)
    print(f"Saved: {out_overlay}")

    # part 4: 叠加图（带降温：先加热1.0h，后降温1.0h）
    on_hours = 1.0
    off_hours = 1.0
    plt.figure(figsize=(12, 7))
    for v in ppfd_values:
        t, y = simulate_unified_ppfd_with_cooling(ppfd=float(v), on_hours=on_hours, off_hours=off_hours, dt_hours=dt_hours, model_params=fit_params)
        plt.plot(t, y, linewidth=2, label=f"PPFD={v}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature Difference (°C)")
    plt.title(f"Unified Thermal Model Cooling (Heat {on_hours}h → Off {off_hours}h)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Curves")
    out_overlay_cool = out_dir / "unified_ppfd_overlay_cooling.png"
    plt.tight_layout()
    plt.savefig(out_overlay_cool, dpi=300)
    print(f"Saved: {out_overlay_cool}")


if __name__ == "__main__":
    main()


