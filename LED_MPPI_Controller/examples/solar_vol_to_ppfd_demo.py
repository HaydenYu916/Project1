from __future__ import annotations

"""Solar_Vol → PPFD 线性映射演示

功能:
- 从 data/Solar_Vol_clean.csv 读取数据
- 仅拟合指定比例键（默认 5:1，即 R:B≈0.83）下的 Solar_Vol→PPFD 直线
- 打印系数、做双向预测示例
- 可视化散点与拟合直线并保存到 examples/result/

运行:
    python -m A.Project1.LED_MPPI_Controller.examples.solar_vol_to_ppfd_demo \
        --focus_key 5:1 \
        --csv data/Solar_Vol_clean.csv
"""

import argparse
import os
import math
from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def _abs_path(rel_path: str) -> str:
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, os.pardir, rel_path))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=_abs_path("data/Solar_Vol_clean.csv"), help="CSV 路径")
    parser.add_argument("--focus_key", type=str, default="5:1", help="比例键(例如 '5:1' 或 '0.83')")
    parser.add_argument("--out", type=str, default=_abs_path("examples/result/solar_vol_to_ppfd_demo.png"), help="结果图保存路径")
    args = parser.parse_args(argv)

    try:
        from A.Project1.LED_MPPI_Controller.src.led import SolarVolToPPFDModel
    except ModuleNotFoundError:
        import sys
        # 允许直接运行本脚本：将本项目的 src 目录加入 sys.path
        sys.path.append(_abs_path("src"))
        from led import SolarVolToPPFDModel

    model = SolarVolToPPFDModel.from_csv(args.csv, focus_key=args.focus_key)
    info = model.get_line_info(key=args.focus_key)
    a = info["a"]; c = info["c"]
    print(f"[fit] PPFD = {a:.6f} * Solar_Vol + {c:.6f}  (key={args.focus_key})")

    # 示例预测
    sv_example = 1.56
    ppfd_pred = model.predict_ppfd(solar_vol=sv_example, key=args.focus_key)
    sv_inv = model.predict_solar_vol(ppfd=400.0, key=args.focus_key)
    print(f"[predict] Solar_Vol={sv_example:.3f} → PPFD={ppfd_pred:.2f}")
    print(f"[invert ] PPFD=400 → Solar_Vol={sv_inv:.3f}")

    # 可视化（可选）
    if _HAS_MPL:
        # 重新采样一段线作图
        xs = np.linspace(0.0, 2.0, 200)
        ys = a * xs + c

        # 尝试从 CSV 取同键散点（用于展示）
        import csv
        scatter_x, scatter_y = [], []
        with open(args.csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader((line for line in f if line.strip() and not line.lstrip().startswith("#")))

            def _get(row: dict, *names: str):
                for name in names:
                    if name in row:
                        return row[name]
                for name in names:
                    for k, v in row.items():
                        if k.replace(" ", "").lower() == name.replace(" ", "").lower():
                            return v
                return None

            def _norm_key(s: str) -> str:
                return str(s).strip().lower().replace(" ", "")

            fk = _norm_key(args.focus_key)
            for row in reader:
                sv = _get(row, "Solar_Vol", "solar_vol")
                ppfd = _get(row, "PPFD", "ppfd")
                key_raw = _get(row, "R:B", "ratio", "Label", "KEY", "Key")
                if sv is None or ppfd is None:
                    continue
                if key_raw is not None and _norm_key(key_raw) != fk:
                    # 允许 0.83 等数值近似到 5:1 的情况
                    try:
                        if abs(float(key_raw) - 0.83) < 0.02 and fk in ("5:1", "5:1"):
                            pass
                        else:
                            continue
                    except Exception:
                        continue
                try:
                    scatter_x.append(float(sv)); scatter_y.append(float(ppfd))
                except Exception:
                    pass

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        plt.figure(figsize=(6.5, 4.2))
        if scatter_x:
            plt.scatter(scatter_x, scatter_y, s=24, alpha=0.75, label=f"data ({args.focus_key})")
        plt.plot(xs, ys, "r-", label=f"fit: y={a:.3f}x+{c:.3f}")
        plt.xlabel("Solar_Vol")
        plt.ylabel("PPFD")
        plt.title(f"Solar_Vol → PPFD (key={args.focus_key})")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out, dpi=160)
        print(f"[save] figure -> {args.out}")
    else:
        print("[warn] 未安装 matplotlib，跳过绘图")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


