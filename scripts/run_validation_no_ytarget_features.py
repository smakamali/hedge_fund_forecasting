"""
Validation experiment: train and evaluate WITHOUT any features derived from y_target.
No lags, no rolling, no global/sub_category mean, no horizon_x_subcat.
Only: 86 features + indicators + zero flags + winsorize + log Type C + horizon_numeric.
Run: from project root: python scripts/run_validation_no_ytarget_features.py
"""
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import pandas as pd
import lightgbm as lgb
from src.evaluation import temporal_train_test_split, weighted_rmse_score
from src.preprocessing import (
    FEATURE_COLS,
    TYPE_C_FEATURES,
    ZERO_INFLATED_FEATURES,
    temporal_impute_missing,
    create_missing_indicators,
    apply_imputation,
    winsorize_features,
    apply_winsorize_bounds,
    log_transform_type_c,
    create_zero_inflation_flags,
)

ENTITY_COLS = ["code", "sub_code", "sub_category", "horizon"]
TS_COL = "ts_index"
TARGET_COL = "y_target"


def build_train_features_no_ytarget(train_df):
    """Train features without any y_target-derived columns (no lags, rolling, agg, horizon_x_subcat)."""
    train_imputed, impute_values = temporal_impute_missing(train_df, FEATURE_COLS, method="median")
    train_imputed, indicator_cols = create_missing_indicators(
        train_imputed, FEATURE_COLS, missing_threshold=0.01
    )

    numeric_to_clip = [c for c in FEATURE_COLS if c in train_imputed.columns]
    train_imputed, winsor_bounds = winsorize_features(
        train_imputed, numeric_to_clip, quantiles=(0.01, 0.99), fit_df=train_imputed
    )
    type_c_present = [c for c in TYPE_C_FEATURES if c in train_imputed.columns]
    train_imputed = log_transform_type_c(train_imputed, type_c_present)
    train_imputed, zero_flag_cols = create_zero_inflation_flags(
        train_imputed, ZERO_INFLATED_FEATURES
    )

    train_imputed["horizon_numeric"] = train_imputed["horizon"].astype(float)

    base_feature_cols = (
        [c for c in FEATURE_COLS if c in train_imputed.columns]
        + indicator_cols
        + zero_flag_cols
    )
    all_feature_cols = base_feature_cols + ["horizon_numeric"]
    all_feature_cols = [c for c in all_feature_cols if c in train_imputed.columns]

    artifacts = {
        "impute_values": impute_values,
        "indicator_cols": indicator_cols,
        "winsor_bounds": winsor_bounds,
    }
    return train_imputed, all_feature_cols, artifacts


def build_val_features_no_ytarget(val_df, artifacts):
    """Val features without any y_target-derived columns."""
    val_imputed = apply_imputation(val_df, artifacts["impute_values"])
    val_imputed, _ = create_missing_indicators(val_imputed, FEATURE_COLS, missing_threshold=0.01)
    val_imputed = apply_winsorize_bounds(val_imputed, artifacts["winsor_bounds"])
    type_c_present = [c for c in TYPE_C_FEATURES if c in val_imputed.columns]
    val_imputed = log_transform_type_c(val_imputed, type_c_present)
    val_imputed, _ = create_zero_inflation_flags(val_imputed, ZERO_INFLATED_FEATURES)
    val_imputed["horizon_numeric"] = val_imputed["horizon"].astype(float)
    return val_imputed


def main():
    _data_dir = os.environ.get("DATA_DIR", "data")
    train_path = os.path.join(_project_root, _data_dir, "train.parquet")
    print("Loading train...")
    train_df = pd.read_parquet(train_path)
    train_part, val_df, cutoff = temporal_train_test_split(
        train_df, ts_col=TS_COL, test_size=0.2
    )

    print("Building train features (no y_target-derived features)...")
    train_fe, all_feature_cols, artifacts = build_train_features_no_ytarget(train_part)

    X_train = train_fe[all_feature_cols].fillna(0).to_numpy()
    y_train = train_fe["y_target"].to_numpy()
    w_train = train_fe["weight"].to_numpy()

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "min_data_in_leaf": 100,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": 42,
    }
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    print("Training model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(0)],
    )

    val_fe = build_val_features_no_ytarget(val_df, artifacts)
    for c in all_feature_cols:
        if c not in val_fe.columns:
            val_fe[c] = 0
    X_val = val_fe[all_feature_cols].fillna(0).to_numpy()
    y_val = val_df["y_target"].to_numpy()
    w_val = val_df["weight"].to_numpy()

    pred_val = model.predict(X_val)
    score = weighted_rmse_score(y_val, pred_val, w_val)
    print(f"Validation skill score (no y_target-derived features): {score:.6f}")
    return score


if __name__ == "__main__":
    score = main()
