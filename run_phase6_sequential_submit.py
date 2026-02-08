"""
Phase 6 (sequential): Train on full data, then run sequential inference on test:
at each ts_index T, build lag/rolling/aggregate features from train y_target + 
previous test predictions, predict, then update state.
Writes final_submission_sequential.csv. Reference: run_phase6_submit.py (unchanged).
Run: conda run -n forecast_fund python run_phase6_sequential_submit.py
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from run_phase6_submit import (
    build_train_features,
    ENTITY_COLS,
    TS_COL,
    TARGET_COL,
    LAGS,
    WINDOWS,
)
from preprocessing import (
    FEATURE_COLS,
    TYPE_C_FEATURES,
    ZERO_INFLATED_FEATURES,
    apply_imputation,
    create_missing_indicators,
    apply_winsorize_bounds,
    log_transform_type_c,
    create_zero_inflation_flags,
)


def _entity_tuple(row):
    return tuple(row[c] for c in ENTITY_COLS)


def precompute_train_state(train_df):
    """
    Precompute per-entity history (ts_index, y_target) and running global/subcat stats.
    Returns (entity_history, running_global_sum, running_global_count, running_subcat_sum,
             running_subcat_count, train_entity_count, global_mean_fallback, subcat_mean_fallback).
    """
    entity_history = {}
    train_sorted = train_df.sort_values(ENTITY_COLS + [TS_COL])
    for ent, grp in train_sorted.groupby(ENTITY_COLS):
        entity_history[ent] = list(zip(grp[TS_COL].tolist(), grp[TARGET_COL].tolist()))
        entity_history[ent].sort(key=lambda x: x[0])

    running_global_sum = float(train_df[TARGET_COL].sum())
    running_global_count = int(len(train_df))

    running_subcat_sum = {}
    running_subcat_count = {}
    for cat, grp in train_df.groupby("sub_category"):
        running_subcat_sum[cat] = float(grp[TARGET_COL].sum())
        running_subcat_count[cat] = int(len(grp))

    train_entity_count = train_df.groupby(ENTITY_COLS).size().to_dict()
    global_mean_fallback = train_df[TARGET_COL].mean()
    subcat_mean_fallback = train_df.groupby("sub_category")[TARGET_COL].mean().to_dict()

    return (
        entity_history,
        running_global_sum,
        running_global_count,
        running_subcat_sum,
        running_subcat_count,
        train_entity_count,
        global_mean_fallback,
        subcat_mean_fallback,
    )


def compute_block_temporal_features(
    block: pd.DataFrame,
    entity_history: dict,
    running_global_sum: float,
    running_global_count: int,
    running_subcat_sum: dict,
    running_subcat_count: dict,
    T: int,
    lag_cols: list,
    rolling_cols: list,
    LAGS: list,
    WINDOWS: list,
    global_mean_fallback: float,
    subcat_mean_fallback: dict,
    entity_count_col: str,
) -> pd.DataFrame:
    """
    For test rows at ts_index == T, compute lag/rolling/agg/entity_obs_count from current state.
    State is for ts_index < T only. Returns DataFrame with columns lag_cols + rolling_cols + agg_cols + entity_count_col.
    """
    TARGET_COL = "y_target"
    agg_cols = [f"{TARGET_COL}_global_mean", f"{TARGET_COL}_sub_category_mean"]

    n = len(block)
    out = {}

    for lag, col in zip(LAGS, lag_cols):
        out[col] = np.zeros(n, dtype=float)
    for w in WINDOWS:
        out[f"{TARGET_COL}_rolling_mean_{w}"] = np.zeros(n, dtype=float)
        out[f"{TARGET_COL}_rolling_std_{w}"] = np.zeros(n, dtype=float)
    out[f"{TARGET_COL}_global_mean"] = np.zeros(n, dtype=float)
    out[f"{TARGET_COL}_sub_category_mean"] = np.zeros(n, dtype=float)
    out[entity_count_col] = np.zeros(n, dtype=float)

    global_mean_T = (
        running_global_sum / running_global_count
        if running_global_count > 0
        else global_mean_fallback
    )

    for i in range(n):
        row = block.iloc[i]
        ent = _entity_tuple(row)
        subcat = row["sub_category"]
        hist = entity_history.get(ent, [])
        past = [(t, y) for t, y in hist if t < T]
        past_ys = [y for _, y in past]

        for lag, col in zip(LAGS, lag_cols):
            if len(past_ys) >= lag:
                out[col][i] = past_ys[-lag]
            else:
                out[col][i] = 0.0

        for w in WINDOWS:
            take = past_ys[-w:] if len(past_ys) >= 1 else []
            if len(take) >= 1:
                out[f"{TARGET_COL}_rolling_mean_{w}"][i] = np.mean(take)
                out[f"{TARGET_COL}_rolling_std_{w}"][i] = (
                    np.std(take) if len(take) > 1 else 0.0
                )
            else:
                out[f"{TARGET_COL}_rolling_mean_{w}"][i] = 0.0
                out[f"{TARGET_COL}_rolling_std_{w}"][i] = 0.0

        out[f"{TARGET_COL}_global_mean"][i] = global_mean_T
        if running_subcat_count.get(subcat, 0) > 0:
            out[f"{TARGET_COL}_sub_category_mean"][i] = (
                running_subcat_sum[subcat] / running_subcat_count[subcat]
            )
        else:
            out[f"{TARGET_COL}_sub_category_mean"][i] = subcat_mean_fallback.get(
                subcat, global_mean_fallback
            )
        out[entity_count_col][i] = len(past)

    return pd.DataFrame(out)


def main():
    print("Loading train...")
    train_df = pd.read_parquet("train.parquet")
    print("Building train features...")
    train_fe, all_feature_cols, artifacts = build_train_features(train_df)

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
    print("Training final model on full train...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(100)],
    )

    print("Precomputing train state for sequential inference...")
    (
        entity_history,
        running_global_sum,
        running_global_count,
        running_subcat_sum,
        running_subcat_count,
        train_entity_count,
        global_mean_fallback,
        subcat_mean_fallback,
    ) = precompute_train_state(train_df)

    print("Loading test...")
    test_df = pd.read_parquet("test.parquet")

    test_base = apply_imputation(test_df, artifacts["impute_values"])
    test_base, _ = create_missing_indicators(
        test_base, FEATURE_COLS, missing_threshold=0.01
    )
    test_base = apply_winsorize_bounds(test_base, artifacts["winsor_bounds"])
    type_c_present = [c for c in TYPE_C_FEATURES if c in test_base.columns]
    test_base = log_transform_type_c(test_base, type_c_present)
    test_base, _ = create_zero_inflation_flags(test_base, ZERO_INFLATED_FEATURES)

    lag_cols = artifacts["lag_cols"]
    rolling_cols = artifacts["rolling_cols"]
    agg_cols = artifacts["agg_cols"]
    entity_count_col = artifacts["entity_count_col"]

    test_ts_values = sorted(test_df[TS_COL].unique())
    id_order = test_df["id"].tolist()
    id_to_idx = {id_: i for i, id_ in enumerate(id_order)}
    predictions = [None] * len(id_order)

    print("Sequential inference over test ts_index...")
    for T in test_ts_values:
        mask = test_df[TS_COL] == T
        block_df = test_df.loc[mask]
        block_base = test_base.loc[mask].copy()
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
            id_val = test_df.loc[idx, "id"]
            predictions[id_to_idx[id_val]] = pred_T[j]

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
            running_subcat_count[subcat] = running_subcat_count.get(subcat, 0) + len(
                pos
            )

    pred_arr = np.array(predictions, dtype=float)
    assert np.isfinite(pred_arr).all(), "NaN or Inf in predictions"
    out = pd.DataFrame({"id": id_order, "prediction": pred_arr})
    out.to_csv("final_submission_sequential.csv", index=False)
    print("Saved final_submission_sequential.csv with", len(out), "rows.")


if __name__ == "__main__":
    main()
