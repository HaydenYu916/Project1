"""
PWM 标定与求解演示（不涉及热学）

功能概览：
- 从 CSV 标定数据拟合 PWM→PPFD 的线性模型（按 R:B 比例 key 分组 + 整体）
- 预测不同 PWM / 不同 R:B 比例下的 PPFD
- 反解给定目标 PPFD 下的 R/B PWM（支持按标注比例或经验比例）

运行：
    python AA_Test_9_16/pwm_demo.py
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

# 确保导入本目录下的 led.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from led import (
    PWMtoPPFDModel,
    DEFAULT_CALIB_CSV,
    solve_pwm_for_target_ppfd,
    PWMtoPowerModel,
)


def demo_fit_and_report(model: PWMtoPPFDModel) -> None:
    print("=" * 60)
    print("1) 拟合与系数概览")
    print("=" * 60)
    print(f"使用标定CSV: {model.csv_path}")

    # 显示整体系数
    if model.overall:
        c = model.overall
        print(f"overall: a_r={c.a_r:.3f}, a_b={c.a_b:.3f}, intercept={c.intercept:.3f}")

    # 显示部分 key 的系数
    for key in ["1:1", "3:1", "5:1", "7:1", "r1"]:
        ck = model.by_key.get(key.lower().replace(" ", ""))
        if ck:
            print(f"{key:>3}: a_r={ck.a_r:.3f}, a_b={ck.a_b:.3f}, intercept={ck.intercept:.3f}")
    print()


def _total_to_rb(total_pwm: float, ratio: Tuple[float, float]) -> Tuple[float, float]:
    w_r, w_b = ratio
    return total_pwm * w_r, total_pwm * w_b


def _parse_ratio_key(key: str) -> Tuple[float, float]:
    s = key.strip().lower().replace(" ", "")
    if s == "r1" or s == "r:1":
        return 1.0, 0.0
    if ":" in s:
        a, b = s.split(":", 1)
        try:
            ra = float(a)
            rb = float(b)
        except ValueError:
            return 0.5, 0.5
        ssum = ra + rb if (ra + rb) > 0 else 1.0
        return ra / ssum, rb / ssum
    return 0.5, 0.5


def demo_predict_sweep(model: PWMtoPPFDModel) -> None:
    print("=" * 60)
    print("2) PWM 扫描与 PPFD 预测")
    print("=" * 60)

    pwm_vals = np.linspace(0, 100, 21)
    ratios = {
        "1:1": (0.5, 0.5),
        "3:1": (3/4, 1/4),
        "5:1": (5/6, 1/6),
        "7:1": (7/8, 1/8),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, ratio in ratios.items():
        pred_ppfd: List[float] = []
        for p in pwm_vals:
            r_pwm, b_pwm = _total_to_rb(float(p), ratio)
            y = model.predict(r_pwm=r_pwm, b_pwm=b_pwm, key=name)
            pred_ppfd.append(y)
        ax.plot(pwm_vals, pred_ppfd, label=f"{name}")
        print(f"{name} at 50%: predict={pred_ppfd[len(pred_ppfd)//2]:.1f} μmol/m²/s")

    ax.set_title("Predicted PPFD vs Total PWM")
    ax.set_xlabel("Total PWM (%)")
    ax.set_ylabel("PPFD (μmol/m²/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    print()


class PowerInterpolator:
    """按比例键(如 1:1/3:1/5:1/7:1)对 Total PWM→Total Power(W) 做一维线性插值。"""

    def __init__(self) -> None:
        self.by_key: Dict[str, Tuple[List[float], List[float]]] = {}

    @staticmethod
    def _normalize_key(key: str) -> str:
        return key.strip().lower().replace(" ", "")

    @classmethod
    def from_csv(cls, csv_path: str) -> "PowerInterpolator":
        import csv
        inst = cls()
        by_key_pairs: Dict[str, List[Tuple[float, float]]] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader((line for line in f if line.strip() and not line.lstrip().startswith("#")))
            for row in reader:
                key = row.get("R:B") or row.get("ratio") or row.get("Key") or row.get("KEY")
                r_pwm = row.get("R_PWM") or row.get("r_pwm") or row.get("R PWM")
                b_pwm = row.get("B_PWM") or row.get("b_pwm") or row.get("B PWM")
                r_pow = row.get("R_POWER") or row.get("r_power")
                b_pow = row.get("B_POWER") or row.get("b_power")
                if key is None or r_pwm is None or b_pwm is None or r_pow is None or b_pow is None:
                    continue
                try:
                    r_pwm_f = float(r_pwm); b_pwm_f = float(b_pwm)
                    r_pow_f = float(r_pow); b_pow_f = float(b_pow)
                except ValueError:
                    continue
                total_pwm = r_pwm_f + b_pwm_f
                total_pow = r_pow_f + b_pow_f
                if total_pwm < 0:
                    continue
                k = cls._normalize_key(key)
                by_key_pairs.setdefault(k, []).append((total_pwm, total_pow))

        # 排序并去重 x；构建 x/y 列表
        for k, pairs in by_key_pairs.items():
            pairs.sort(key=lambda t: t[0])
            xs: List[float] = []
            ys: List[float] = []
            last_x: Optional[float] = None
            for x, y in pairs:
                if last_x is not None and abs(x - last_x) < 1e-9:
                    # 如果有重复 x，取最新值
                    ys[-1] = y
                else:
                    xs.append(x); ys.append(y)
                    last_x = x
            if len(xs) >= 2:
                inst.by_key[k] = (xs, ys)
        return inst

    def predict_power(self, *, total_pwm: float, key: str, clamp: bool = True) -> float:
        import bisect
        k = self._normalize_key(key)
        if k not in self.by_key:
            raise KeyError(f"calib中不存在比例键: {key}")
        xs, ys = self.by_key[k]
        x = float(total_pwm)
        if clamp:
            if x <= xs[0]:
                return float(ys[0])
            if x >= xs[-1]:
                return float(ys[-1])
        # 二分查找区间
        i = bisect.bisect_left(xs, x)
        i = max(1, min(i, len(xs) - 1))
        x0, x1 = xs[i - 1], xs[i]
        y0, y1 = ys[i - 1], ys[i]
        t = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
        return float(y0 + t * (y1 - y0))


def demo_power_fit_and_energy(model: PWMtoPPFDModel) -> None:
    print("=" * 60)
    print("4) 功率直线拟合与能耗 (kWh)")
    print("=" * 60)
    pwr_model = PWMtoPowerModel(include_intercept=True).fit(DEFAULT_CALIB_CSV)

    # 绘制不同比例下：Total PWM → Total Power(W)（直线）
    pwm_vals = np.linspace(0, 120, 25)  # 允许到 120 以覆盖 5:1 的高点
    ratios = ["1:1", "3:1", "5:1", "7:1"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in ratios:
        y_list = []
        for p in pwm_vals:
            y = pwr_model.predict(total_pwm=float(p), key=key)
            y_list.append(y)
        ax.plot(pwm_vals, y_list, label=key)
    ax.set_title("Interpolated Power vs Total PWM")
    ax.set_xlabel("Total PWM (%)")
    ax.set_ylabel("Power (W)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 举例：用 PPFD 目标先解出整数 PWM，再用插值得到功率与 1 小时能耗
    examples = [
        ("5:1", 300.0, 1.0),
        ("7:1", 450.0, 1.0),
    ]
    for key, target_ppfd, hours in examples:
        r_pwm, b_pwm, total = solve_pwm_for_target_ppfd(
            model=model,
            target_ppfd=target_ppfd,
            key=key,
            ratio_strategy="label",
            integer_output=True,
        )
        p_w = pwr_model.predict(total_pwm=float(r_pwm + b_pwm), key=key)
        e_kwh = p_w / 1000.0 * float(hours)
        print(
            f"比例 {key}, 目标PPFD={target_ppfd:.1f} => PWM: R={r_pwm}% B={b_pwm}% (总={r_pwm + b_pwm}%) | "
            f"功率≈{p_w:.1f} W, {hours} 小时能耗≈{e_kwh:.3f} kWh"
        )
    print()

def demo_solve_targets(model: PWMtoPPFDModel) -> None:
    print("=" * 60)
    print("3) 目标 PPFD 的 PWM 求解")
    print("=" * 60)
    targets = [150.0, 300.0, 450.0]
    for key in ["1:1", "3:1", "5:1", "7:1"]:
        print(f"比例 {key}:")
        # 计算该比例下的可达上限（按各通道最大100，保持比例）
        w_r, w_b = _parse_ratio_key(key)
        s_max = float("inf")
        if w_r > 0:
            s_max = min(s_max, 100.0 / w_r)
        if w_b > 0:
            s_max = min(s_max, 100.0 / w_b)
        r_max = min(100.0, s_max * w_r)
        b_max = min(100.0, s_max * w_b)
        y_max = model.predict(r_pwm=r_max, b_pwm=b_max, key=key)
        print(f"  可达上限估计：PPFD≈{y_max:.1f} (R≤100, B≤100, 固定比例)")
        for t in targets:
            r_pwm, b_pwm, total = solve_pwm_for_target_ppfd(
                model=model,
                target_ppfd=t,
                key=key,
                ratio_strategy="label",  # 也可以试试 "empirical_global" / "empirical_local"
                ratio_window_ppfd=10.0,
                integer_output=True,
            )
            # r_pwm, b_pwm, total 已为整数
            y = model.predict(r_pwm=float(r_pwm), b_pwm=float(b_pwm), key=key)
            line = f"  target={t:6.1f} => total={total:3d}%, R={r_pwm:3d}%, B={b_pwm:3d}% | check={y:6.1f}"
            # 可达性提示
            if t > y_max + 1e-6:
                line += "  [提示: 目标超出该比例可达上限, 已按上限给出最优近似]"
            # 量化误差提示
            if abs(y - t) > 5.0:
                line += f"  [量化/模型误差≈{y - t:+.1f}]"
            print(line)
        print()


def main() -> None:
    print("PWM 演示：拟合→预测→求解")
    model = PWMtoPPFDModel(include_intercept=True).fit(DEFAULT_CALIB_CSV)

    demo_fit_and_report(model)
    demo_predict_sweep(model)
    demo_solve_targets(model)
    demo_power_fit_and_energy(model)

    print("完成。")


if __name__ == "__main__":
    main()
