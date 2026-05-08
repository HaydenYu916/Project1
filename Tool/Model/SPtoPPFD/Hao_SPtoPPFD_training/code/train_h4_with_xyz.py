#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, loguniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "data" / "Train_data-h4_pC.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output" / "h4_no_xyz_output"
SPECTRAL_COLS = [
    "sp_415_mean",
    "sp_445_mean",
    "sp_480_mean",
    "sp_515_mean",
    "sp_555_mean",
    "sp_590_mean",
    "sp_630_mean",
    "sp_680_mean",
    "x_cm",
    "y_cm",
    "z_cm",
]


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
        description="Train PPFD regression models using spectral features only (no x/y/z)."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path, default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for exported results, default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="PPFD_spec_mean",
        help="Target column name, default: PPFD_spec_mean",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="ID",
        help="Optional sample ID column to keep in prediction outputs, default: ID",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio, default: 0.2",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation ratio within train+val pool, default: 0.2",
    )
    parser.add_argument(
        "--tune-iters",
        type=int,
        default=30,
        help="RandomizedSearchCV iterations per tunable model, default: 30",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Inner CV folds for tuning, default: 5",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip RandomizedSearchCV and use only untuned models.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed, default: 42",
    )
    return parser.parse_args()


def regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    mean_y = float(np.mean(y_true))
    y_range = float(np.max(y_true) - np.min(y_true))
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Bias": bias,
        "NRMSE_mean": float(rmse / mean_y) if mean_y != 0 else np.nan,
        "NRMSE_range": float(rmse / y_range) if y_range != 0 else np.nan,
    }


def make_preprocessor(scale: bool = True):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def make_models(random_state: int):
    models = {
        "DummyMean": Pipeline(
            [("preprocess", make_preprocessor(scale=False)), ("model", DummyRegressor(strategy="mean"))]
        ),
        "LinearRegression": Pipeline(
            [("preprocess", make_preprocessor(scale=True)), ("model", LinearRegression())]
        ),
        "Ridge": Pipeline(
            [("preprocess", make_preprocessor(scale=True)), ("model", Ridge(alpha=1.0, random_state=random_state))]
        ),
        "Lasso": Pipeline(
            [
                ("preprocess", make_preprocessor(scale=True)),
                ("model", Lasso(alpha=0.01, random_state=random_state, max_iter=5000)),
            ]
        ),
        "ElasticNet": Pipeline(
            [
                ("preprocess", make_preprocessor(scale=True)),
                (
                    "model",
                    ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state, max_iter=5000),
                ),
            ]
        ),
        "Huber": Pipeline(
            [("preprocess", make_preprocessor(scale=True)), ("model", HuberRegressor())]
        ),
        "PLSR": Pipeline(
            [("preprocess", make_preprocessor(scale=True)), ("model", PLSRRegressor(n_components=2))]
        ),
        "PCR_Ridge": Pipeline(
            [
                ("preprocess", make_preprocessor(scale=True)),
                ("pca", PCA(n_components=min(5, len(SPECTRAL_COLS)), random_state=random_state)),
                ("model", Ridge(alpha=1.0, random_state=random_state)),
            ]
        ),
        "KNN": Pipeline(
            [("preprocess", make_preprocessor(scale=True)), ("model", KNeighborsRegressor(n_neighbors=5))]
        ),
        "SVR_RBF": Pipeline(
            [("preprocess", make_preprocessor(scale=True)), ("model", SVR(C=10.0, epsilon=0.1, gamma="scale"))]
        ),
        "KernelRidge_RBF": Pipeline(
            [("preprocess", make_preprocessor(scale=True)), ("model", KernelRidge(alpha=1.0, kernel="rbf"))]
        ),
        "DecisionTree": Pipeline(
            [
                ("preprocess", make_preprocessor(scale=False)),
                ("model", DecisionTreeRegressor(max_depth=4, random_state=random_state)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("preprocess", make_preprocessor(scale=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "ExtraTrees": Pipeline(
            [
                ("preprocess", make_preprocessor(scale=False)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [("preprocess", make_preprocessor(scale=False)), ("model", GradientBoostingRegressor(random_state=random_state))]
        ),
        "HistGBR": Pipeline(
            [
                ("preprocess", make_preprocessor(scale=False)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_depth=4,
                        max_iter=300,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "GPR": Pipeline(
            [
                ("preprocess", make_preprocessor(scale=True)),
                (
                    "model",
                    GaussianProcessRegressor(
                        kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
                        + WhiteKernel(noise_level=1.0),
                        normalize_y=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            [
                ("preprocess", make_preprocessor(scale=True)),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        max_iter=2000,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = Pipeline(
            [
                ("preprocess", make_preprocessor(scale=False)),
                (
                    "model",
                    CatBoostRegressor(
                        depth=5,
                        learning_rate=0.05,
                        iterations=500,
                        loss_function="RMSE",
                        verbose=False,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    except Exception:
        pass

    return models


def build_tuning_search(model_name: str, random_state: int):
    if model_name == "Ridge":
        pipe = Pipeline(
            [("preprocess", make_preprocessor(scale=True)), ("model", Ridge(random_state=random_state))]
        )
        param_dist = {"model__alpha": loguniform(1e-3, 1e3)}

    elif model_name == "PLSR":
        pipe = Pipeline(
            [("preprocess", make_preprocessor(scale=True)), ("model", PLSRRegressor())]
        )
        param_dist = {"model__n_components": randint(1, min(len(SPECTRAL_COLS), 6) + 1)}

    elif model_name == "RandomForest":
        pipe = Pipeline(
            [
                ("preprocess", make_preprocessor(scale=False)),
                ("model", RandomForestRegressor(random_state=random_state, n_jobs=-1)),
            ]
        )
        param_dist = {
            "model__n_estimators": randint(200, 800),
            "model__max_depth": randint(2, 12),
            "model__min_samples_leaf": randint(1, 6),
            "model__max_features": ["sqrt", "log2", None],
        }

    elif model_name == "HistGBR":
        pipe = Pipeline(
            [
                ("preprocess", make_preprocessor(scale=False)),
                ("model", HistGradientBoostingRegressor(random_state=random_state)),
            ]
        )
        param_dist = {
            "model__learning_rate": loguniform(1e-2, 2e-1),
            "model__max_depth": randint(2, 10),
            "model__max_iter": randint(100, 600),
            "model__min_samples_leaf": randint(5, 30),
        }

    elif model_name == "SVR_RBF":
        pipe = Pipeline(
            [("preprocess", make_preprocessor(scale=True)), ("model", SVR(kernel="rbf"))]
        )
        param_dist = {
            "model__C": loguniform(1e-1, 1e3),
            "model__epsilon": loguniform(1e-3, 1.0),
            "model__gamma": loguniform(1e-3, 1.0),
        }

    elif model_name == "GPR":
        pipe = Pipeline(
            [
                ("preprocess", make_preprocessor(scale=True)),
                (
                    "model",
                    GaussianProcessRegressor(
                        kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
                        + WhiteKernel(noise_level=1.0),
                        normalize_y=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        param_dist = {"model__alpha": loguniform(1e-10, 1e-1)}

    elif model_name == "CatBoost":
        from catboost import CatBoostRegressor

        pipe = Pipeline(
            [
                ("preprocess", make_preprocessor(scale=False)),
                (
                    "model",
                    CatBoostRegressor(
                        loss_function="RMSE",
                        verbose=False,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        param_dist = {
            "model__depth": randint(3, 9),
            "model__learning_rate": loguniform(1e-2, 2e-1),
            "model__iterations": randint(200, 800),
            "model__l2_leaf_reg": loguniform(1e-2, 10.0),
        }

    else:
        raise ValueError(f"Unknown model for tuning: {model_name}")

    return pipe, param_dist


def load_dataset(input_csv: Path, target_col: str):
    df = pd.read_csv(input_csv)
    missing = [c for c in SPECTRAL_COLS + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep_cols = [c for c in ["ID"] if c in df.columns] + SPECTRAL_COLS + [target_col]
    df_model = df[keep_cols].copy()
    df_model = df_model.dropna(subset=[target_col]).reset_index(drop=True)
    return df, df_model


def evaluate_model(estimator, model_name: str, version: str, X_train, y_train, X_val, y_val, extra=None):
    pred_train = estimator.predict(X_train)
    pred_val = estimator.predict(X_val)
    train_metrics = regression_metrics(y_train, pred_train)
    val_metrics = regression_metrics(y_val, pred_val)
    row = {
        "model": model_name,
        "version": version,
        "train_RMSE": train_metrics["RMSE"],
        "train_MAE": train_metrics["MAE"],
        "train_R2": train_metrics["R2"],
        "val_RMSE": val_metrics["RMSE"],
        "val_MAE": val_metrics["MAE"],
        "val_R2": val_metrics["R2"],
        "val_Bias": val_metrics["Bias"],
        "val_NRMSE_mean": val_metrics["NRMSE_mean"],
        "val_NRMSE_range": val_metrics["NRMSE_range"],
    }
    if extra:
        row.update(extra)
    return row


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    raw_df, df_model = load_dataset(args.input_csv, args.target_col)
    X_full = df_model[SPECTRAL_COLS].copy()
    y_full = df_model[args.target_col].copy()

    if len(df_model) < 10:
        raise ValueError("Not enough rows for train/val/test split. Need at least 10 rows.")

    test_ratio = args.test_size
    val_ratio_within_trainval = args.val_size
    if not (0 < test_ratio < 0.5):
        raise ValueError("--test-size must be between 0 and 0.5")
    if not (0 < val_ratio_within_trainval < 0.5):
        raise ValueError("--val-size must be between 0 and 0.5")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_full,
        y_full,
        test_size=test_ratio,
        random_state=args.random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio_within_trainval,
        random_state=args.random_state,
    )

    split_summary = pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "n_rows": [len(X_train), len(X_val), len(X_test)],
            "ppfd_mean": [y_train.mean(), y_val.mean(), y_test.mean()],
            "ppfd_std": [y_train.std(), y_val.std(), y_test.std()],
        }
    )
    split_summary.to_csv(args.output_dir / "split_summary.csv", index=False)

    models_untuned = make_models(args.random_state)
    untuned_rows = []
    for model_name, model in models_untuned.items():
        est = clone(model)
        est.fit(X_train, y_train)
        untuned_rows.append(
            evaluate_model(est, model_name, "untuned", X_train, y_train, X_val, y_val)
        )

    untuned_val_df = pd.DataFrame(untuned_rows).sort_values("val_RMSE").reset_index(drop=True)
    untuned_val_df.to_csv(args.output_dir / "untuned_validation_results.csv", index=False)

    tuned_val_df = pd.DataFrame()
    best_estimators: dict[str, Pipeline] = {}

    if not args.skip_tuning:
        tunable_candidates = ["Ridge", "PLSR", "RandomForest", "HistGBR", "SVR_RBF", "GPR", "CatBoost"]
        tunable_models = [m for m in tunable_candidates if m in models_untuned]
        tuned_rows = []
        inner_cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

        for model_name in tunable_models:
            pipe, param_dist = build_tuning_search(model_name, args.random_state)
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_dist,
                n_iter=args.tune_iters,
                scoring="neg_root_mean_squared_error",
                cv=inner_cv,
                n_jobs=-1,
                random_state=args.random_state,
                refit=True,
            )
            search.fit(X_train, y_train)
            best_estimators[model_name] = search.best_estimator_
            tuned_rows.append(
                evaluate_model(
                    search.best_estimator_,
                    model_name,
                    "tuned",
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    extra={
                        "best_cv_rmse": -search.best_score_,
                        "best_params": json.dumps(search.best_params_, ensure_ascii=False),
                    },
                )
            )

        if tuned_rows:
            tuned_val_df = pd.DataFrame(tuned_rows).sort_values("val_RMSE").reset_index(drop=True)
            tuned_val_df.to_csv(args.output_dir / "tuned_validation_results.csv", index=False)

    if len(tuned_val_df) > 0:
        best_model_name = tuned_val_df.iloc[0]["model"]
        best_version = "tuned"
        best_estimator = best_estimators[best_model_name]
    else:
        best_model_name = untuned_val_df.iloc[0]["model"]
        best_version = "untuned"
        best_estimator = clone(models_untuned[best_model_name])
        best_estimator.fit(X_train, y_train)

    best_estimator.fit(X_trainval, y_trainval)
    test_pred = best_estimator.predict(X_test)
    test_metrics = regression_metrics(y_test, test_pred)

    final_test_df = pd.DataFrame(
        [
            {
                "model": best_model_name,
                "version": best_version,
                "test_RMSE": test_metrics["RMSE"],
                "test_MAE": test_metrics["MAE"],
                "test_R2": test_metrics["R2"],
                "test_Bias": test_metrics["Bias"],
                "test_NRMSE_mean": test_metrics["NRMSE_mean"],
                "test_NRMSE_range": test_metrics["NRMSE_range"],
            }
        ]
    )
    final_test_df.to_csv(args.output_dir / "final_test_results.csv", index=False)

    test_rows = df_model.loc[X_test.index, [c for c in [args.id_col] if c in df_model.columns]].copy()
    test_rows[args.target_col] = y_test.values
    test_rows["PPFD_pred"] = test_pred
    test_rows["residual"] = test_rows["PPFD_pred"] - test_rows[args.target_col]
    test_rows.to_csv(args.output_dir / "test_predictions.csv", index=False)

    package = {
        "pipeline": best_estimator,
        "feature_columns": SPECTRAL_COLS,
        "model_name": best_model_name,
        "model_version": best_version,
        "target_col": args.target_col,
        "input_csv": str(args.input_csv),
        "n_rows": int(len(df_model)),
        "split_config": {
            "test_size": args.test_size,
            "val_size_within_trainval": args.val_size,
            "random_state": args.random_state,
        },
        "test_metrics": test_metrics,
    }
    joblib.dump(package, args.output_dir / "best_model_package.joblib")

    metadata = {
        "model_name": best_model_name,
        "model_version": best_version,
        "feature_columns": SPECTRAL_COLS,
        "target_col": args.target_col,
        "input_csv": str(args.input_csv),
        "n_rows": int(len(df_model)),
        "test_metrics": test_metrics,
        "used_tuning": not args.skip_tuning,
        "available_models": list(models_untuned.keys()),
    }
    (args.output_dir / "best_model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Input CSV:", args.input_csv)
    print("Output dir:", args.output_dir)
    print("Rows used:", len(df_model))
    print("Feature columns:", SPECTRAL_COLS)
    print("Best model:", best_model_name, "| version:", best_version)
    print(
        "Test metrics:",
        json.dumps(test_metrics, indent=2, ensure_ascii=False),
    )
    print("Saved package to:", args.output_dir / "best_model_package.joblib")


if __name__ == "__main__":
    main()
