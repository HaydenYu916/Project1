#!/usr/bin/env python3

"""
Predict red and blue PWM values from a target PPFD.

This script uses the linear model trained on samples near an R:B ratio of 5:1:
    PPFD = intercept + slope * pwm_r

By default, it returns a PWM pair using R:B = 5:1.
You can provide a different ratio if needed, but accuracy may decrease
when the output ratio moves far away from 5:1.

Example:
    python3 predict_pwm_from_ppfd.py --ppfd 200
"""

import argparse
import json
from pathlib import Path


DEFAULT_MODEL = {
    "model_name": "pwm_to_ppfd_linear_5to1_nearby",
    "intercept": 1.994258,
    "slope": 3.695779,
    "ratio_window": [4.7, 5.3],
    "recommended_rb_ratio": 5.0,
    "sample_count": 16,
    "train_pwm_r_min": 21.0,
    "train_pwm_r_max": 83.0,
    "train_ppfd_min": 80.17,
    "train_ppfd_max": 311.65,
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


def predict_red_pwm(target_ppfd: float, intercept: float, slope: float) -> float:
    if slope <= 0:
        raise ValueError("Model slope must be greater than 0")
    return max(0.0, (target_ppfd - intercept) / slope)


def build_result(target_ppfd: float, rb_ratio: float, model: dict):
    if target_ppfd <= 0:
        raise ValueError("Target PPFD must be greater than 0")
    if rb_ratio <= 0:
        raise ValueError("R:B ratio must be greater than 0")

    pwm_r = predict_red_pwm(target_ppfd, model["intercept"], model["slope"])
    pwm_b = pwm_r / rb_ratio
    predicted_ppfd = model["intercept"] + model["slope"] * pwm_r

    return {
        "target_ppfd": target_ppfd,
        "predicted_ppfd": predicted_ppfd,
        "rb_ratio": rb_ratio,
        "red_pwm_float": pwm_r,
        "blue_pwm_float": pwm_b,
        "red_pwm": round(pwm_r),
        "blue_pwm": round(pwm_b),
        "in_training_ppfd_range": (
            model["train_ppfd_min"] <= target_ppfd <= model["train_ppfd_max"]
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Input a target PPFD and get recommended red and blue PWM values."
    )
    parser.add_argument(
        "--ppfd",
        type=float,
        required=True,
        help="Target PPFD, for example 200",
    )
    parser.add_argument(
        "--rb-ratio",
        type=float,
        default=DEFAULT_MODEL["recommended_rb_ratio"],
        help="R:B ratio used for output, default is 5.0",
    )
    parser.add_argument(
        "--model-json",
        type=str,
        default=None,
        help="Optional path to a model JSON exported by the notebook",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result in JSON format",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model_json)
    result = build_result(args.ppfd, args.rb_ratio, model)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("Model:", model["model_name"])
    print(
        "Equation: PPFD = %.6f + %.6f * pwm_r"
        % (model["intercept"], model["slope"])
    )
    print("Target PPFD: %.2f" % result["target_ppfd"])
    print("R:B ratio: %.3f:1" % result["rb_ratio"])
    print("Suggested red PWM: %.2f (rounded: %d)" % (
        result["red_pwm_float"],
        result["red_pwm"],
    ))
    print("Suggested blue PWM: %.2f (rounded: %d)" % (
        result["blue_pwm_float"],
        result["blue_pwm"],
    ))
    print("Predicted PPFD: %.2f" % result["predicted_ppfd"])

    if not result["in_training_ppfd_range"]:
        print(
            "Warning: target PPFD is outside the training range %.2f ~ %.2f, so this is an extrapolation."
            % (model["train_ppfd_min"], model["train_ppfd_max"])
        )

    if abs(result["rb_ratio"] - model["recommended_rb_ratio"]) > 0.5:
        print("Warning: the requested output ratio is far from 5:1, so accuracy may decrease.")


if __name__ == "__main__":
    main()
