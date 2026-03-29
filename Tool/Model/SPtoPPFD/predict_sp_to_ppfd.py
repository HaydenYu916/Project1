#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PACKAGE = BASE_DIR / "model" / "best_model_package.joblib"
DEFAULT_MODEL_METADATA = BASE_DIR / "model" / "best_model_metadata.json"
DEFAULT_INPUT_CSV = BASE_DIR / "input" / "new_samples.csv"
DEFAULT_OUTPUT_CSV = BASE_DIR / "predictions.csv"


class PLSRRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X, y):
        n_comp = min(self.n_components, X.shape[1], max(1, X.shape[0] - 1))
        self.model_ = PLSRegression(n_components=max(1, n_comp))
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        pred = self.model_.predict(X)
        return np.asarray(pred).ravel()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Input spectral features and predict PPFD using the bundled model."
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
    parser.add_argument("--sp-415", type=float, default=None, help="Single prediction input: sp_415_mean")
    parser.add_argument("--sp-445", type=float, default=None, help="Single prediction input: sp_445_mean")
    parser.add_argument("--sp-480", type=float, default=None, help="Single prediction input: sp_480_mean")
    parser.add_argument("--sp-515", type=float, default=None, help="Single prediction input: sp_515_mean")
    parser.add_argument("--sp-555", type=float, default=None, help="Single prediction input: sp_555_mean")
    parser.add_argument("--sp-590", type=float, default=None, help="Single prediction input: sp_590_mean")
    parser.add_argument("--sp-630", type=float, default=None, help="Single prediction input: sp_630_mean")
    parser.add_argument("--sp-680", type=float, default=None, help="Single prediction input: sp_680_mean")
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
        "sp_415_mean": args.sp_415,
        "sp_445_mean": args.sp_445,
        "sp_480_mean": args.sp_480,
        "sp_515_mean": args.sp_515,
        "sp_555_mean": args.sp_555,
        "sp_590_mean": args.sp_590,
        "sp_630_mean": args.sp_630,
        "sp_680_mean": args.sp_680,
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
        result["PPFD_pred"] = pred[0]
        print("Model:", model_name)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    input_csv = args.input_csv or DEFAULT_INPUT_CSV
    df = pd.read_csv(input_csv)
    X = prepare_features(df, feature_columns)
    pred = round_prediction(pipeline.predict(X))

    out = df.copy()
    out["PPFD_pred"] = pred
    out.to_csv(args.output_csv, index=False)

    print("Model:", model_name)
    print("Feature columns:", feature_columns)
    print("Input rows:", len(df))
    print("Saved predictions to:", args.output_csv)


if __name__ == "__main__":
    main()
