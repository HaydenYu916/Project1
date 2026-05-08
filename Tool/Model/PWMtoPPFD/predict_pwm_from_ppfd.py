#!/usr/bin/env python3

"""
Predict red and blue PWM values from a target PPFD with a chosen blue share.

Two-channel linear model (no intercept, forced through origin):
    PPFD = a_r * pwm_r + b_b * pwm_b

The caller picks `blue_share = pwm_b / (pwm_r + pwm_b)` in [0, max_blue_share].
The default cap is 0.5, which enforces pwm_r >= pwm_b.

Examples:
    python3 predict_pwm_from_ppfd.py --ppfd 200                # default blue_share=0.0 (pure red)
    python3 predict_pwm_from_ppfd.py --ppfd 400 --blue-share 0.3
"""

import argparse
import json
from pathlib import Path


DEFAULT_MODEL = {
    "model_name": "pwm_to_ppfd_linear_2ch",
    "a_r": 3.5731,
    "b_b": 1.8293,
    "max_blue_share": 0.5,
    "default_blue_share": 0.0,
    "sample_count": 20,
    "train_pwm_min": 0,
    "train_pwm_max": 100,
    "train_ppfd_min": 71.3,
    "train_ppfd_max": 526.3,
    "rmse": 6.25,
    "max_abs_err": 14.0,
    "trained_at": "2026-05-08",
    "trained_run": "Collect_Sp_PPFD_LED/outputs/20260508_131754_single_C_z25_RgeB_chamber2",
}


def load_model(model_json: str | None):
    if not model_json:
        return DEFAULT_MODEL

    path = Path(model_json)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_json}")

    package = json.loads(path.read_text())
    model = DEFAULT_MODEL.copy()
    model.update(package)
    return model


def build_result(target_ppfd: float, blue_share: float, model: dict):
    if target_ppfd < 0:
        raise ValueError("Target PPFD must be >= 0")

    a_r = float(model["a_r"])
    b_b = float(model["b_b"])
    max_share = float(model.get("max_blue_share", 0.5))

    s = max(0.0, min(float(blue_share), max_share))

    # PPFD = a_r * R + b_b * B; with B = s*(R+B), R = (1-s)*(R+B):
    #   PPFD = (R+B) * (a_r*(1-s) + b_b*s)
    denom = a_r * (1.0 - s) + b_b * s
    total = 0.0 if denom <= 0 else target_ppfd / denom
    pwm_r = (1.0 - s) * total
    pwm_b = s * total

    # Saturate per-channel to [0, 100]; if red caps, push remaining demand to blue
    # but never exceed pwm_r (R >= B invariant).
    saturated = False
    if pwm_r > 100.0:
        saturated = True
        pwm_r = 100.0
        remaining = max(0.0, target_ppfd - a_r * pwm_r)
        pwm_b = min(pwm_r, remaining / b_b if b_b > 0 else 0.0)
    if pwm_b > 100.0:
        saturated = True
        pwm_b = 100.0
    if pwm_b > pwm_r:
        pwm_b = pwm_r  # enforce R >= B even after numerical drift

    predicted_ppfd = a_r * pwm_r + b_b * pwm_b

    return {
        "target_ppfd": target_ppfd,
        "predicted_ppfd": predicted_ppfd,
        "blue_share": s,
        "red_pwm_float": pwm_r,
        "blue_pwm_float": pwm_b,
        "red_pwm": int(round(pwm_r)),
        "blue_pwm": int(round(pwm_b)),
        "saturated": saturated,
        "in_training_ppfd_range": (
            model["train_ppfd_min"] <= target_ppfd <= model["train_ppfd_max"]
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Input a target PPFD and blue share, get recommended red/blue PWM."
    )
    parser.add_argument("--ppfd", type=float, required=True, help="Target PPFD")
    parser.add_argument(
        "--blue-share",
        type=float,
        default=DEFAULT_MODEL["default_blue_share"],
        help="Blue share = B/(R+B), in [0, max_blue_share]. Default 0.0 (pure red).",
    )
    parser.add_argument("--model-json", type=str, default=None, help="Optional model JSON path")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model_json)
    result = build_result(args.ppfd, args.blue_share, model)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("Model:", model["model_name"])
    print(f"Equation: PPFD = {model['a_r']:.4f}*R + {model['b_b']:.4f}*B")
    print(f"Target PPFD: {result['target_ppfd']:.2f}")
    print(f"Blue share : {result['blue_share']:.3f}")
    print(f"Red  PWM   : {result['red_pwm_float']:.2f} (rounded {result['red_pwm']})")
    print(f"Blue PWM   : {result['blue_pwm_float']:.2f} (rounded {result['blue_pwm']})")
    print(f"Predicted  : {result['predicted_ppfd']:.2f}")
    if result["saturated"]:
        print("Warning: hardware PWM saturated; actual PPFD may be below target.")
    if not result["in_training_ppfd_range"]:
        print(
            f"Warning: target PPFD outside training range "
            f"{model['train_ppfd_min']:.2f}~{model['train_ppfd_max']:.2f} (extrapolation)."
        )


if __name__ == "__main__":
    main()
