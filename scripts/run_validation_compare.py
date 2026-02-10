"""
Compare fallback vs sequential inference on a temporal validation split (last 20% of train).
Reports competition metric (weighted skill score) for both approaches.
Run: from project root: python scripts/run_validation_compare.py
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
from run_phase6_submit import (
    build_train_features,
    build_test_features,
    ENTITY_COLS,
    TS_COL,
    TARGET_COL,
    LAGS,
    WINDOWS,
)
from run_phase6_sequential_submit import (
    precompute_train_state,
    compute_block_temporal_features,
    _entity_tuple,
)
from src.preprocessing import (
    FEATURE_COLS,
    TYPE_C_FEATURES,
    ZERO_INFLATED_FEATURES,
    apply_imputation,
    create_missing_indicators,
    apply_winsorize_bounds,
    log_transform_type_c,
    create_zero_inflation_flags,
)


def main():
    _data_dir = os.environ.get("DATA_DIR", "data")
    train_path = os.path.join(_project_root, _data_dir, "train.parquet")
    print("Loading train...")
    train_df = pd.read_parquet(train_path)
    train_part, val_df, cutoff = temporal_train_test_split(
        train_df, ts_col=TS_COL, test_size=0.2
    )

    print("Building train features (on train part only)...")
    train_fe, all_feature_cols, artifacts = build_train_features(train_part)

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

    y_val = val_df["y_target"].to_numpy()
    w_val = val_df["weight"].to_numpy()

    # ---- Experiment 1: Fallback (one-shot val features) ----
    print("\n--- Experiment 1: Fallback (zeros / train means for temporal features) ---")
    val_fallback = build_test_features(val_df, artifacts)
    for c in all_feature_cols:
        if c not in val_fallback.columns:
            val_fallback[c] = 0
    X_val_fallback = val_fallback[all_feature_cols].fillna(0).to_numpy()
    pred_fallback = model.predict(X_val_fallback)
    score_fallback = weighted_rmse_score(y_val, pred_fallback, w_val)
    print(f"Validation skill score (fallback): {score_fallback:.6f}")

    # ---- Experiment 2: Sequential inference on val ----
    print("\n--- Experiment 2: Sequential inference (train y + past val predictions) ---")
    (
        entity_history,
        running_global_sum,
        running_global_count,
        running_subcat_sum,
        running_subcat_count,
        train_entity_count,
        global_mean_fallback,
        subcat_mean_fallback,
    ) = precompute_train_state(train_part)

    val_base = apply_imputation(val_df, artifacts["impute_values"])
    val_base, _ = create_missing_indicators(val_base, FEATURE_COLS, missing_threshold=0.01)
    val_base = apply_winsorize_bounds(val_base, artifacts["winsor_bounds"])
    type_c_present = [c for c in TYPE_C_FEATURES if c in val_base.columns]
    val_base = log_transform_type_c(val_base, type_c_present)
    val_base, _ = create_zero_inflation_flags(val_base, ZERO_INFLATED_FEATURES)

    lag_cols = artifacts["lag_cols"]
    rolling_cols = artifacts["rolling_cols"]
    entity_count_col = artifacts["entity_count_col"]

    val_ts_values = sorted(val_df[TS_COL].unique())
    val_predictions = np.full(len(val_df), np.nan, dtype=float)

    for T in val_ts_values:
        mask = val_df[TS_COL] == T
        block_df = val_df.loc[mask]
        block_base = val_base.loc[mask].copy()
        if len(block_df) == 0:
            continue

        block_temporal = compute_block_temporal_features(
            block_df,
            entity_history,
            running_global_sum,
            running_global_count,
            running_subcat_sum,
            running_subcat_count,
            T,
            lag_cols,
            rolling_cols,
            LAGS,
            WINDOWS,
            global_mean_fallback,
            subcat_mean_fallback,
            entity_count_col,
        )

        block_base["horizon_numeric"] = block_base["horizon"].astype(float)
        block_base["horizon_x_subcat"] = (
            block_base["horizon_numeric"].values
            * block_temporal["y_target_sub_category_mean"].values
        )
        for c in block_temporal.columns:
            block_base[c] = block_temporal[c].values

        for c in all_feature_cols:
            if c not in block_base.columns:
                block_base[c] = 0
        X_T = block_base[all_feature_cols].fillna(0).to_numpy()
        pred_T = model.predict(X_T)

        for j, idx in enumerate(block_df.index):
            val_predictions[val_df.index.get_loc(idx)] = pred_T[j]

        for j, idx in enumerate(block_df.index):
            row = block_df.loc[idx]
            ent = _entity_tuple(row)
            if ent not in entity_history:
                entity_history[ent] = []
            entity_history[ent].append((T, float(pred_T[j])))
            entity_history[ent].sort(key=lambda x: x[0])

        running_global_sum += float(np.sum(pred_T))
        running_global_count += len(pred_T)
        for subcat in block_df["sub_category"].unique():
            pos = (block_df["sub_category"].values == subcat).nonzero()[0]
            running_subcat_sum[subcat] = running_subcat_sum.get(subcat, 0) + float(
                np.sum(pred_T[pos])
            )
            running_subcat_count[subcat] = running_subcat_count.get(subcat, 0) + len(pos)

    assert np.isfinite(val_predictions).all(), "NaN/Inf in sequential val predictions"
    score_sequential = weighted_rmse_score(y_val, val_predictions, w_val)
    print(f"Validation skill score (sequential): {score_sequential:.6f}")

    # Return for optional doc writing
    return {
        "score_fallback": score_fallback,
        "score_sequential": score_sequential,
        "val_size": len(val_df),
        "train_size": len(train_part),
        "cutoff": cutoff,
    }


if __name__ == "__main__":
    results = main()
    print("\n--- Summary ---")
    print(f"Fallback   skill score: {results['score_fallback']:.6f}")
    print(f"Sequential skill score: {results['score_sequential']:.6f}")
    print(f"Difference (sequential - fallback): {results['score_sequential'] - results['score_fallback']:.6f}")
