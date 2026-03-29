#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PACKAGE = BASE_DIR / "model" / "best_model_package.joblib"
DEFAULT_MODEL_METADATA = BASE_DIR / "model" / "best_model_metadata.json"
DEFAULT_INPUT_CSV = BASE_DIR / "input" / "new_samples.csv"
DEFAULT_OUTPUT_CSV = BASE_DIR / "predictions.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict Pn from T, CO2, R:B, and PPFD using the bundled model."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help=f"Input CSV path, default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path, default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument(
        "--model-package",
        type=Path,
        default=DEFAULT_MODEL_PACKAGE,
        help=f"Model package path, default: {DEFAULT_MODEL_PACKAGE}",
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Print bundled model metadata before prediction.",
    )
    parser.add_argument("--T", type=float, default=None, help="Single prediction input: temperature")
    parser.add_argument("--CO2", type=float, default=None, help="Single prediction input: CO2")
    parser.add_argument("--RB", type=float, default=None, help="Single prediction input: R:B ratio")
    parser.add_argument("--PPFD", type=float, default=None, help="Single prediction input: PPFD")
    return parser.parse_args()


def load_package(package_path: Path):
    if not package_path.exists():
        raise FileNotFoundError(f"Model package not found: {package_path}")
    return joblib.load(package_path)


def load_metadata():
    if not DEFAULT_MODEL_METADATA.exists():
        return None
    return json.loads(DEFAULT_MODEL_METADATA.read_text(encoding="utf-8"))


def prepare_features(df: pd.DataFrame, feature_columns: list[str]):
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required feature columns: {missing}")
    return df[feature_columns].copy()


def round_prediction(values):
    return [round(float(value), 2) for value in values]


def build_single_input_dataframe(args) -> pd.DataFrame | None:
    value_map = {
        "T": args.T,
        "CO2": args.CO2,
        "R:B": args.RB,
        "PPFD": args.PPFD,
    }
    if all(value is None for value in value_map.values()):
        return None

    missing = [key for key, value in value_map.items() if value is None]
    if missing:
        raise ValueError(f"Single prediction mode is missing values: {missing}")

    return pd.DataFrame([value_map])


def main():
    args = parse_args()
    package = load_package(args.model_package)
    metadata = load_metadata()

    pipeline = package["pipeline"]
    feature_columns = package["feature_columns"]
    model_name = package["model_name"]

    if args.show_metadata and metadata is not None:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))

    single_df = build_single_input_dataframe(args)
    if single_df is not None:
        X = prepare_features(single_df, feature_columns)
        pred = round_prediction(pipeline.predict(X))
        result = dict(single_df.iloc[0])
        result["Pn_pred"] = pred[0]
        print("Model:", model_name)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    input_csv = args.input_csv or DEFAULT_INPUT_CSV
    df = pd.read_csv(input_csv)
    X = prepare_features(df, feature_columns)
    pred = round_prediction(pipeline.predict(X))

    out = df.copy()
    out["Pn_pred"] = pred
    out.to_csv(args.output_csv, index=False)

    print("Model:", model_name)
    print("Feature columns:", feature_columns)
    print("Input rows:", len(df))
    print("Saved predictions to:", args.output_csv)


if __name__ == "__main__":
    main()
