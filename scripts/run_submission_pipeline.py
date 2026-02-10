"""
Three-step submission pipeline: v1 (Model A sequential), v2 (noise-robust A2), v3 (Model B + A').
Run from project root: python scripts/run_submission_pipeline.py [--step 1|2|3|all] [--config path] [--experiment NAME]
Requires data/train.parquet and data/test.parquet. Writes to output/. Logs metadata and MLflow runs when --experiment is set.
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from functools import lru_cache

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow

from src.config_loader import load_config
from src.evaluation import temporal_train_test_split, weighted_rmse_score
from src.preprocessing import (
    FEATURE_COLS,
    TYPE_C_FEATURES,
    ZERO_INFLATED_FEATURES,
    ENTITY_CATEGORICAL_COLS,
    encode_entity_categoricals,
    fit_target_transform,
    transform_target,
    inverse_transform_target,
    temporal_impute_missing,
    create_missing_indicators,
    apply_imputation,
    fit_min_max_bounds,
    apply_min_max_scale,
    create_lag_features,
    create_rolling_features,
    create_aggregate_features_t1,
    create_entity_count,
    winsorize_features,
    apply_winsorize_bounds,
    log_transform_type_c,
    create_zero_inflation_flags,
    select_input_lags,
    create_input_lag_features,
)

from run_validation_lagged_features import (
    _build_train_features,
    _build_val_features,
    prepare_dataset_for_lag_config,
    train_and_evaluate,
    ENTITY_COLS,
    ENTITY_CAT_FEATURE_NAMES,
    TS_COL,
    TARGET_COL,
    Y_LAGS,
    WINDOWS,
    make_skill_feval,
)


logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Paths and config
# -----------------------------------------------------------------------------

def _get_paths(cfg):
    data_dir = cfg.get("data_dir", "data")
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(_project_root, data_dir)
    out_dir = os.path.join(_project_root, "output")
    train_path = os.path.join(data_dir, "train.parquet")
    test_path = os.path.join(data_dir, "test.parquet")
    return {"train_path": train_path, "test_path": test_path, "out_dir": out_dir, "data_dir": data_dir}


def _build_metadata(step_name, metrics, cfg):
    """Build metadata dict with metrics, input_lags, target_lags, lightgbm params."""
    il = cfg.get("input_lags", {})
    tl = cfg.get("target_lags", {})
    lb = cfg.get("lightgbm", {})
    return {
        "step": step_name,
        "metrics": metrics,
        "input_lags": il,
        "target_lags": tl,
        "lightgbm": lb,
    }


def _metadata_path_for_submission(submission_path):
    """e.g. output/submission_v1.csv -> output/submission_v1_metadata.json"""
    base, _ = os.path.splitext(submission_path)
    return base + "_metadata.json"


def _write_metadata(metadata, submission_path, out_dir):
    """Write metadata JSON next to submission CSV (same base name + _metadata.json)."""
    dir_part = os.path.dirname(submission_path) or out_dir
    base = os.path.splitext(os.path.basename(submission_path))[0]
    meta_path = os.path.join(dir_part, base + "_metadata.json")
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", meta_path)
    return meta_path


def _log_run_mlflow(experiment_name, run_name, metadata, submission_path, out_dir):
    """Log run to MLflow: set experiment, log params/metrics, log metadata and submission as artifacts."""
    def _to_param(val):
        if isinstance(val, (list, dict)):
            return json.dumps(val)
        return val

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        # Log params (flattened for MLflow; lists/dicts as JSON string)
        for key, val in metadata.get("input_lags", {}).items():
            mlflow.log_param("input_lags." + key, _to_param(val))
        for key, val in metadata.get("target_lags", {}).items():
            mlflow.log_param("target_lags." + key, _to_param(val))
        for key, val in metadata.get("lightgbm", {}).items():
            mlflow.log_param("lightgbm." + key, _to_param(val))
        for key, val in metadata.get("metrics", {}).items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(key, float(val))
        dir_part = os.path.dirname(submission_path) or out_dir
        base = os.path.splitext(os.path.basename(submission_path))[0]
        meta_path = os.path.join(dir_part, base + "_metadata.json")
        if os.path.isfile(meta_path):
            mlflow.log_artifact(meta_path)
        if os.path.isfile(submission_path):
            mlflow.log_artifact(submission_path)


def _entity_tuple(row):
    return tuple(row[c] for c in ENTITY_COLS)


def _max_lag_from_spec(lag_spec):
    """Max lag value from lag_spec (list or dict of feature -> lags). Used to take train tail for test input lags."""
    if lag_spec is None:
        return 1
    if isinstance(lag_spec, list):
        return max(lag_spec) if lag_spec else 1
    return max(lag for lags in lag_spec.values() for lag in lags) if lag_spec else 1


@lru_cache(maxsize=2)
def _load_parquet_cached(path):
    """Cached parquet loader to avoid re-reading large train/test files from disk."""
    return pd.read_parquet(path)


def _warn_and_fill_missing_features(df, feature_cols, critical_features=None, prefix=""):
    """
    Ensure all feature_cols exist in df, filling missing ones with 0.0 and emitting a warning.
    Optionally assert that all critical_features are present.
    """
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        msg_prefix = f"{prefix}: " if prefix else ""
        logger.warning(
            "%sWARNING: %d features missing in data; filling with 0.0. First up to 10: %s",
            msg_prefix,
            len(missing_features),
            missing_features[:10],
        )
        if len(missing_features) > 10:
            logger.warning("%s... and %d more", msg_prefix, len(missing_features) - 10)
        for c in missing_features:
            df[c] = 0.0
    if critical_features:
        missing_critical = [c for c in critical_features if c not in df.columns]
        if missing_critical:
            raise AssertionError(f"{prefix}: critical features missing: {missing_critical}")
    return df


# -----------------------------------------------------------------------------
# Sequential prediction state and block temporal features (validation artifacts)
# -----------------------------------------------------------------------------

def _precompute_train_state(train_df):
    """Per-entity history (ts_index, y) and running global/subcat stats from train."""
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
    global_mean_fallback = train_df[TARGET_COL].mean()
    subcat_mean_fallback = train_df.groupby("sub_category")[TARGET_COL].mean().to_dict()
    return (
        entity_history,
        running_global_sum,
        running_global_count,
        running_subcat_sum,
        running_subcat_count,
        global_mean_fallback,
        subcat_mean_fallback,
    )


def _compute_block_temporal_features(block, entity_history, running_global_sum, running_global_count,
                                     running_subcat_sum, running_subcat_count,
                                     T, lag_cols, y_lags, rolling_cols, windows,
                                     global_mean_fallback, subcat_mean_fallback, entity_count_col):
    """Target-derived features for rows at ts_index T from current state."""
    n = len(block)
    out = {}
    for lag, col in zip(y_lags, lag_cols):
        out[col] = np.zeros(n, dtype=float)
    for w in windows:
        out[f"{TARGET_COL}_rolling_mean_{w}"] = np.zeros(n, dtype=float)
        out[f"{TARGET_COL}_rolling_std_{w}"] = np.zeros(n, dtype=float)
    out[f"{TARGET_COL}_global_mean"] = np.zeros(n, dtype=float)
    out[f"{TARGET_COL}_sub_category_mean"] = np.zeros(n, dtype=float)
    out[entity_count_col] = np.zeros(n, dtype=float)

    global_mean_T = running_global_sum / running_global_count if running_global_count > 0 else global_mean_fallback

    for i in range(n):
        row = block.iloc[i]
        ent = _entity_tuple(row)
        subcat = row["sub_category"]
        hist = entity_history.get(ent, [])
        past = [(t, y) for t, y in hist if t < T]
        past_ys = [y for _, y in past]

        for lag, col in zip(y_lags, lag_cols):
            out[col][i] = past_ys[-lag] if len(past_ys) >= lag else 0.0
        for w in windows:
            take = past_ys[-w:] if len(past_ys) >= 1 else []
            if len(take) >= 1:
                out[f"{TARGET_COL}_rolling_mean_{w}"][i] = np.mean(take)
                out[f"{TARGET_COL}_rolling_std_{w}"][i] = np.std(take) if len(take) > 1 else 0.0
            else:
                out[f"{TARGET_COL}_rolling_std_{w}"][i] = 0.0
        out[f"{TARGET_COL}_global_mean"][i] = global_mean_T
        if running_subcat_count.get(subcat, 0) > 0:
            out[f"{TARGET_COL}_sub_category_mean"][i] = running_subcat_sum[subcat] / running_subcat_count[subcat]
        else:
            out[f"{TARGET_COL}_sub_category_mean"][i] = subcat_mean_fallback.get(subcat, global_mean_fallback)
        out[entity_count_col][i] = len(past)

    return pd.DataFrame(out)


def _sequential_predict(df, train_fe, model, artifacts, feature_cols_final, target_transform):
    """
    Run sequential prediction on df (val or test). train_fe is the feature-built train used for state and input_lags.
    df must be sorted by ENTITY_COLS + [TS_COL]. Returns predictions array in same order as df.
    """
    lag_cols = artifacts["lag_cols"]
    rolling_cols = artifacts["rolling_cols"]
    y_lags = artifacts.get("y_lags", Y_LAGS)
    windows = artifacts.get("windows", WINDOWS)
    entity_count_col = artifacts["entity_count_col"]
    lag_spec = artifacts.get("lag_spec")
    use_float16 = artifacts.get("use_float16_when_large", False)

    entity_history, running_global_sum, running_global_count, running_subcat_sum, running_subcat_count, global_mean_fb, subcat_mean_fb = _precompute_train_state(
        train_fe[ENTITY_COLS + [TS_COL] + [TARGET_COL]].copy()
    )

    # Base + input_lag for df: combine train tail (last max_lag rows per entity) with df to reduce memory
    max_lag = _max_lag_from_spec(lag_spec)
    train_base = train_fe.groupby(ENTITY_COLS, group_keys=False).tail(max_lag)[ENTITY_COLS + [TS_COL] + FEATURE_COLS].copy()
    train_base["_ord"] = -1
    min_max_bounds = artifacts.get("input_feature_min_max", {})
    if min_max_bounds:
        df_prep = apply_imputation(df.copy(), artifacts["impute_values"])
        df_prep = apply_min_max_scale(df_prep, min_max_bounds)
        df_base = df_prep[ENTITY_COLS + [TS_COL] + [c for c in FEATURE_COLS if c in df_prep.columns]].copy()
    else:
        df_base = df[ENTITY_COLS + [TS_COL] + [c for c in FEATURE_COLS if c in df.columns]].copy()
    df_base["_ord"] = np.arange(len(df_base))
    combined = pd.concat([train_base, df_base], ignore_index=True)
    combined = combined.sort_values(ENTITY_COLS + [TS_COL])
    combined, _ = create_input_lag_features(
        combined, ENTITY_COLS, TS_COL, FEATURE_COLS, lag_spec, use_float16_when_large=use_float16
    )
    df_mask = combined["_ord"] >= 0
    df_input_lags = combined.loc[df_mask].sort_values("_ord").drop(columns=["_ord"])
    input_lag_cols = artifacts.get("input_lag_cols", [])
    df_base_with_lags = df.copy()
    for c in input_lag_cols:
        if c in df_input_lags.columns:
            df_base_with_lags[c] = df_input_lags[c].values

    # Apply same preprocessing as val: imputation, winsorize, encode, entity merge (from artifacts)
    impute_values = artifacts["impute_values"]
    winsor_bounds = artifacts["winsor_bounds"]
    entity_encodings = artifacts.get("entity_encodings", {})
    entity_mean_from_train = artifacts["entity_mean_from_train"]
    entity_std_from_train = artifacts["entity_std_from_train"]
    entity_count_from_train = artifacts["entity_count_from_train"]
    global_mean = artifacts["global_mean"]
    global_std = artifacts["global_std"]

    df_work = apply_imputation(df_base_with_lags, impute_values)
    missing_threshold = artifacts.get("missing_threshold", 0.01)
    df_work, _ = create_missing_indicators(df_work, FEATURE_COLS, missing_threshold=missing_threshold)
    df_work = apply_winsorize_bounds(df_work, winsor_bounds)
    type_c_present = [c for c in TYPE_C_FEATURES if c in df_work.columns]
    df_work = log_transform_type_c(df_work, type_c_present)
    df_work, _ = create_zero_inflation_flags(df_work, ZERO_INFLATED_FEATURES)
    df_work, _ = encode_entity_categoricals(df_work, ENTITY_CATEGORICAL_COLS, encodings=entity_encodings)
    df_work = df_work.merge(entity_mean_from_train, on=ENTITY_COLS, how="left")
    df_work["entity_mean"] = df_work["entity_mean"].fillna(global_mean)
    df_work = df_work.merge(entity_std_from_train, on=ENTITY_COLS, how="left")
    df_work["entity_std"] = df_work["entity_std"].fillna(global_std)
    df_work = df_work.merge(entity_count_from_train, on=ENTITY_COLS, how="left")
    df_work["entity_obs_count"] = df_work["entity_obs_count"].fillna(0)

    ts_values = sorted(df[TS_COL].unique())
    predictions = [None] * len(df)
    idx_counter = 0

    for T in ts_values:
        mask = df[TS_COL].values == T
        block_df = df.loc[mask]
        if len(block_df) == 0:
            continue
        block_temporal = _compute_block_temporal_features(
            block_df, entity_history, running_global_sum, running_global_count,
            running_subcat_sum, running_subcat_count, T,
            lag_cols, y_lags, rolling_cols, windows,
            global_mean_fb, subcat_mean_fb, entity_count_col,
        )
        block_work = df_work.loc[mask].copy()
        horizon_numeric = block_work["horizon"].astype(float)
        horizon_x_subcat = horizon_numeric * block_temporal["y_target_sub_category_mean"].values
        block_work["horizon_numeric"] = horizon_numeric
        block_work["horizon_x_subcat"] = horizon_x_subcat
        for c in block_temporal.columns:
            block_work[c] = block_temporal[c].values
        critical_features = [c for c in ENTITY_CAT_FEATURE_NAMES if c in feature_cols_final]
        block_work = _warn_and_fill_missing_features(
            block_work, feature_cols_final, critical_features=critical_features, prefix="_sequential_predict"
        )
        X_block = block_work[feature_cols_final].fillna(0).to_numpy()
        pred_t = model.predict(X_block)
        pred = inverse_transform_target(pred_t, target_transform) if target_transform is not None else pred_t

        for j in range(len(block_df)):
            predictions[idx_counter] = pred[j]
            idx_counter += 1

        for j, idx in enumerate(block_df.index):
            ent = _entity_tuple(block_df.loc[idx])
            if ent not in entity_history:
                entity_history[ent] = []
            entity_history[ent].append((T, float(pred[j])))
            entity_history[ent].sort(key=lambda x: x[0])
        running_global_sum += float(np.sum(pred))
        running_global_count += len(pred)
        for subcat in block_df["sub_category"].unique():
            pos = (block_df["sub_category"].values == subcat).nonzero()[0]
            running_subcat_sum[subcat] = running_subcat_sum.get(subcat, 0) + float(np.sum(pred[pos]))
            running_subcat_count[subcat] = running_subcat_count.get(subcat, 0) + len(pos)

    pred_arr = np.array([p if p is not None else np.nan for p in predictions], dtype=float)
    return pred_arr


def _write_submission(test_df, pred, out_path, id_col="id"):
    """Write submission CSV. Build id from entity+ts if missing."""
    if id_col in test_df.columns:
        ids = test_df[id_col].values
    else:
        ids = (
            test_df["code"].astype(str) + "__"
            + test_df["sub_code"].astype(str) + "__"
            + test_df["sub_category"].astype(str) + "__"
            + test_df["horizon"].astype(str) + "__"
            + test_df[TS_COL].astype(str)
        ).values
    out = pd.DataFrame({"id": ids, "prediction": pred})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("Saved %s with %d rows.", out_path, len(out))
    return out_path


# -----------------------------------------------------------------------------
# Step 1: Model A -> val errors + submission_v1
# -----------------------------------------------------------------------------

def _run_step1(cfg, paths, experiment_name=None):
    train_path = paths["train_path"]
    test_path = paths["test_path"]
    out_dir = paths["out_dir"]
    data_dir = paths.get("data_dir", os.path.join(_project_root, "data"))

    il = cfg["input_lags"]
    tl = cfg["target_lags"]
    lb = cfg["lightgbm"]

    logger.info("Step 1: loading train data from %s", train_path)
    train_df = _load_parquet_cached(train_path).copy()
    train_part, val_df, _ = temporal_train_test_split(train_df, ts_col=TS_COL, test_size=0.1)

    train_fe, all_feature_cols, artifacts = _build_train_features(
        train_part,
        use_target_lags=tl["use_target_lags"],
        use_rolling=tl["use_rolling"],
        use_aggregates=tl["use_aggregates"],
        use_input_lags=True,
        use_target_transform=tl["use_target_transform"],
        lags_max=il["lags_max"],
        top_k_per_feature=il["top_k_per_feature"],
        use_global_lags=il["use_global_lags"],
        global_lags=il["global_lags"],
        y_lags=tl.get("y_lags", Y_LAGS),
        use_float16_when_large=il.get("use_float16_when_large", False),
    )
    artifacts["missing_threshold"] = 0.01
    feature_cols_final = all_feature_cols

    X_train = train_fe[feature_cols_final].fillna(0).to_numpy()
    y_train = train_fe[TARGET_COL].to_numpy()
    w_train = train_fe["weight"].to_numpy()
    target_transform = artifacts.get("target_transform")
    entity_cat = [c for c in ENTITY_CAT_FEATURE_NAMES if c in feature_cols_final]

    logger.info("Step 1: building validation features")
    val_fe = _build_val_features(val_df, artifacts, train_fe=train_fe)
    critical_val_features = [c for c in ENTITY_CAT_FEATURE_NAMES if c in feature_cols_final]
    val_fe = _warn_and_fill_missing_features(
        val_fe, feature_cols_final, critical_features=critical_val_features, prefix="step1_val"
    )
    X_val = val_fe[feature_cols_final].fillna(0).to_numpy()
    y_val_orig = val_df[TARGET_COL].to_numpy()
    w_val = val_df["weight"].to_numpy()

    params = {
        "objective": "regression", "metric": "None", "boosting_type": "gbdt",
        "num_leaves": lb["num_leaves"], "min_data_in_leaf": lb["min_data_in_leaf"],
        "learning_rate": lb["learning_rate"], "verbose": -1, "seed": lb["seed"],
    }
    if lb.get("max_depth", -1) > 0:
        params["max_depth"] = lb["max_depth"]
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train, feature_name=feature_cols_final, categorical_feature=entity_cat)
    y_val_model = transform_target(y_val_orig, target_transform) if target_transform else y_val_orig
    val_data = lgb.Dataset(X_val, label=y_val_model, weight=w_val, reference=train_data)
    feval = make_skill_feval(y_val_orig, w_val, target_transform)
    logger.info("Step 1: training Model A with 80/10 split")
    model_a_80 = lgb.train(
        params, train_data,
        num_boost_round=lb.get("num_boost_round", 1000),
        valid_sets=[val_data], valid_names=["val"], feval=feval,
        callbacks=[lgb.log_evaluation(0), lgb.early_stopping(stopping_rounds=lb.get("early_stopping_rounds", 50), verbose=False)],
    )

    # Train/val metrics (in original space)
    pred_train_t = model_a_80.predict(X_train)
    pred_train_orig = inverse_transform_target(pred_train_t, target_transform) if target_transform else pred_train_t
    y_train_orig = inverse_transform_target(y_train, target_transform) if target_transform else train_fe[TARGET_COL].to_numpy()
    train_skill = weighted_rmse_score(y_train_orig, pred_train_orig, w_train)
    train_rmse = np.sqrt(np.mean((y_train_orig - pred_train_orig) ** 2))

    # Sequential prediction on val to get errors
    val_df_sorted = val_df.sort_values(ENTITY_COLS + [TS_COL]).reset_index(drop=True)
    logger.info("Step 1: sequential validation prediction to compute errors")
    pred_val = _sequential_predict(val_df_sorted, train_fe, model_a_80, artifacts, feature_cols_final, target_transform)
    y_val = val_df_sorted[TARGET_COL].to_numpy()
    w_val_sorted = val_df_sorted["weight"].to_numpy()
    val_skill = weighted_rmse_score(y_val, pred_val, w_val_sorted)
    val_rmse = np.sqrt(np.mean((y_val - pred_val) ** 2))
    errors_val = y_val - pred_val
    errors_val_transformed = errors_val
    if target_transform is not None:
        # Also compute validation errors in transformed target space for use in Step 2 noise injection
        y_val_t = transform_target(y_val, target_transform)
        pred_val_t = transform_target(pred_val, target_transform)
        errors_val_transformed = y_val_t - pred_val_t
    # Also persist validation errors as artifacts (with and without transform).
    errors_path = os.path.join(out_dir, "validation_errors.npy")
    errors_path_trans = os.path.join(out_dir, "validation_errors_transformed.npy")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    errors_path_ts = os.path.join(out_dir, f"validation_errors_{ts}.npy")
    errors_path_trans_ts = os.path.join(out_dir, f"validation_errors_transformed_{ts}.npy")
    os.makedirs(out_dir, exist_ok=True)
    np.save(errors_path, errors_val)
    np.save(errors_path_trans, errors_val_transformed)
    # Timestamped copies for easier tracking across runs.
    np.save(errors_path_ts, errors_val)
    np.save(errors_path_trans_ts, errors_val_transformed)
    logger.info(
        "Saved validation errors to %s, %s and timestamped copies %s, %s",
        errors_path,
        errors_path_trans,
        errors_path_ts,
        errors_path_trans_ts,
    )
    step1_metrics = {
        "train_skill": float(train_skill), "train_rmse": float(train_rmse),
        "val_skill": float(val_skill), "val_rmse": float(val_rmse),
        "best_iteration": int(getattr(model_a_80, "best_iteration", 0) or 0),
    }

    # Train Model A on full train
    logger.info("Step 1: training Model A on full train")
    train_full_df = _load_parquet_cached(train_path).copy()
    train_fe_full, _, artifacts_full = _build_train_features(
        train_full_df,
        use_target_lags=tl["use_target_lags"],
        use_rolling=tl["use_rolling"],
        use_aggregates=tl["use_aggregates"],
        use_input_lags=True,
        use_target_transform=tl["use_target_transform"],
        lags_max=il["lags_max"],
        top_k_per_feature=il["top_k_per_feature"],
        use_global_lags=il["use_global_lags"],
        global_lags=il["global_lags"],
        y_lags=tl.get("y_lags", Y_LAGS),
        use_float16_when_large=il.get("use_float16_when_large", False),
    )
    feature_cols_final_full = all_feature_cols
    X_train_full = train_fe_full[feature_cols_final_full].fillna(0).to_numpy()
    y_train_full = train_fe_full[TARGET_COL].to_numpy()
    w_train_full = train_fe_full["weight"].to_numpy()
    train_data_full = lgb.Dataset(X_train_full, label=y_train_full, weight=w_train_full, feature_name=feature_cols_final_full, categorical_feature=entity_cat)
    num_rounds = getattr(model_a_80, "best_iteration", None) or lb.get("num_boost_round", 1000)
    model_a_full = lgb.train(
        params, train_data_full, num_boost_round=num_rounds,
        callbacks=[lgb.log_evaluation(50)],
    )

    logger.info("Step 1: predicting on test set")
    test_df = _load_parquet_cached(test_path).copy()
    test_sorted = test_df.sort_values(ENTITY_COLS + [TS_COL]).reset_index(drop=True)
    pred_test = _sequential_predict(
        test_sorted, train_fe_full, model_a_full, artifacts_full, feature_cols_final_full, artifacts_full.get("target_transform"),
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"submission_v1_{ts}.csv")
    _write_submission(test_sorted, pred_test, out_path)
    metadata = _build_metadata("step1", step1_metrics, cfg)
    _write_metadata(metadata, out_path, out_dir)
    if experiment_name:
        _log_run_mlflow(experiment_name, "step1", metadata, out_path, out_dir)
    return out_path


# -----------------------------------------------------------------------------
# Step 2: Noisy target lags -> Model A2 -> submission_v2
# -----------------------------------------------------------------------------

def _run_step2(cfg, paths, experiment_name=None):
    out_dir = paths["out_dir"]
    errors_path = os.path.join(out_dir, "validation_errors.npy")
    errors_path_trans = os.path.join(out_dir, "validation_errors_transformed.npy")
    if not os.path.isfile(errors_path):
        raise FileNotFoundError("Run step 1 first to generate " + errors_path)
    il = cfg["input_lags"]
    tl = cfg["target_lags"]
    lb = cfg["lightgbm"]
    train_path = paths["train_path"]
    test_path = paths["test_path"]

    logger.info("Step 2: loading train data from %s", train_path)
    train_df = _load_parquet_cached(train_path).copy()
    train_imputed, impute_values = temporal_impute_missing(train_df, FEATURE_COLS, method="median")
    train_imputed, indicator_cols = create_missing_indicators(train_imputed, FEATURE_COLS, missing_threshold=0.01)
    train_imputed = train_imputed.sort_values(ENTITY_COLS + [TS_COL])
    target_transform = fit_target_transform(train_df[TARGET_COL].to_numpy()) if tl["use_target_transform"] else None
    if target_transform:
        logger.info("Step 2: applying target transform before noise injection")
        train_imputed[TARGET_COL] = transform_target(train_imputed[TARGET_COL].to_numpy(), target_transform)

    # Noisy target: y_noisy = y_true + sample from errors_val
    if target_transform is not None and os.path.isfile(errors_path_trans):
        logger.info("Step 2: using transformed-space validation errors for noise")
        errors_val_for_noise = np.load(errors_path_trans)
    else:
        logger.info("Step 2: using original-space validation errors for noise")
        errors_val_for_noise = np.load(errors_path)
    np.random.seed(lb.get("seed", 42))
    noise = np.random.choice(errors_val_for_noise, size=len(train_imputed), replace=True)
    train_imputed = train_imputed.copy()
    train_imputed["y_noisy"] = train_imputed[TARGET_COL].values + noise
    y_col = "y_noisy"
    train_imputed, lag_cols = create_lag_features(
        train_imputed, ENTITY_COLS, y_col, TS_COL, lags=tl.get("y_lags", Y_LAGS)
    )
    # Rolling/aggregate features are created from y_noisy for training, but we want
    # the feature *names* to follow the y_target_* convention so they align with
    # _compute_block_temporal_features and _sequential_predict at validation/test time.
    train_imputed, rolling_cols_raw = create_rolling_features(
        train_imputed, ENTITY_COLS, y_col, TS_COL, windows=tl.get("windows", WINDOWS)
    )
    train_imputed, agg_cols_raw = create_aggregate_features_t1(
        train_imputed, y_col, TS_COL, group_col="sub_category"
    )
    # Alias aggregate columns to y_target_* names and use those in the feature list.
    train_imputed["y_target_global_mean"] = train_imputed[f"{y_col}_global_mean"]
    train_imputed["y_target_sub_category_mean"] = train_imputed[f"{y_col}_sub_category_mean"]
    agg_cols = ["y_target_global_mean", "y_target_sub_category_mean"]
    # For rolling features, create y_target_* aliases and use them as rolling_cols.
    rolling_cols = []
    for c in rolling_cols_raw:
        if "_rolling_mean_" in c:
            suffix = c.split("_rolling_mean_", 1)[1]
            new_name = f"{TARGET_COL}_rolling_mean_{suffix}"
        elif "_rolling_std_" in c:
            suffix = c.split("_rolling_std_", 1)[1]
            new_name = f"{TARGET_COL}_rolling_std_{suffix}"
        else:
            # Fallback: keep original name if it does not match the expected pattern.
            new_name = c
        train_imputed[new_name] = train_imputed[c]
        rolling_cols.append(new_name)
    train_imputed, entity_count_col = create_entity_count(train_imputed, ENTITY_COLS, TS_COL)
    numeric_to_clip = [c for c in FEATURE_COLS if c in train_imputed.columns]
    train_imputed, winsor_bounds = winsorize_features(train_imputed, numeric_to_clip, quantiles=(0.01, 0.99), fit_df=train_imputed)
    type_c_present = [c for c in TYPE_C_FEATURES if c in train_imputed.columns]
    train_imputed = log_transform_type_c(train_imputed, type_c_present)
    train_imputed, zero_flag_cols = create_zero_inflation_flags(train_imputed, ZERO_INFLATED_FEATURES)
    train_imputed, entity_encodings = encode_entity_categoricals(train_imputed, ENTITY_CATEGORICAL_COLS, encodings=None)
    lag_spec = select_input_lags(
        train_imputed, FEATURE_COLS, ENTITY_COLS, TS_COL, TARGET_COL,
        lags_max=il["lags_max"], top_k_per_feature=il["top_k_per_feature"],
        use_global_lags=il["use_global_lags"], global_lags=il["global_lags"],
    )
    input_feature_min_max_a2 = {}
    if il.get("use_float16_when_large", False):
        scale_cols = [c for c in FEATURE_COLS if c in train_imputed.columns and pd.api.types.is_numeric_dtype(train_imputed[c])]
        if scale_cols:
            input_feature_min_max_a2 = fit_min_max_bounds(train_imputed, scale_cols)
            train_imputed = apply_min_max_scale(train_imputed, input_feature_min_max_a2)
    train_imputed, input_lag_cols = create_input_lag_features(train_imputed, ENTITY_COLS, TS_COL, FEATURE_COLS, lag_spec, use_float16_when_large=il.get("use_float16_when_large", False))
    global_mean = float(train_imputed[TARGET_COL].mean())
    global_std = float(train_imputed[TARGET_COL].std()) or 1.0
    horizon_numeric = train_imputed["horizon"].astype(float)
    horizon_x_subcat = horizon_numeric * train_imputed["y_target_sub_category_mean"]
    train_imputed["horizon_numeric"] = horizon_numeric
    train_imputed["horizon_x_subcat"] = horizon_x_subcat
    entity_feature_cols = [c + "_enc" for c in ENTITY_CATEGORICAL_COLS]
    base_feature_cols = entity_feature_cols + [c for c in FEATURE_COLS if c in train_imputed.columns] + indicator_cols + zero_flag_cols
    all_feature_cols = base_feature_cols + ["horizon_numeric", "horizon_x_subcat"] + lag_cols + rolling_cols + agg_cols + [entity_count_col] + input_lag_cols
    all_feature_cols = [c for c in all_feature_cols if c in train_imputed.columns]
    # Entity stats for sequential prediction artifacts: use only the 90% temporal training portion to avoid leakage.
    artifacts_a2 = {
        "impute_values": impute_values, "indicator_cols": indicator_cols, "winsor_bounds": winsor_bounds,
        "input_feature_min_max": input_feature_min_max_a2,
        "lag_cols": lag_cols, "rolling_cols": rolling_cols, "agg_cols": agg_cols, "entity_count_col": entity_count_col,
        "input_lag_cols": input_lag_cols, "lag_spec": lag_spec, "y_lags": tl.get("y_lags", Y_LAGS), "windows": tl.get("windows", WINDOWS),
        "global_mean": global_mean, "global_std": global_std, "entity_count_from_train": train_df.groupby(ENTITY_COLS).size().reset_index(name="entity_obs_count"),
        "entity_encodings": entity_encodings, "target_transform": target_transform,
    }
    X_train = train_imputed[all_feature_cols].fillna(0).to_numpy()
    y_train = train_imputed[TARGET_COL].to_numpy()
    w_train = train_imputed["weight"].to_numpy()
    entity_cat = [c for c in ENTITY_CAT_FEATURE_NAMES if c in all_feature_cols]

    # 10% temporal validation for early stopping and num_rounds
    unique_ts = sorted(train_imputed[TS_COL].unique())
    split_idx = int(len(unique_ts) * 0.9)
    cutoff = unique_ts[split_idx]
    mask_90 = train_imputed[TS_COL].values <= cutoff
    mask_10 = ~mask_90
    # Entity-level statistics for artifacts: compute from 90%% training slice only (avoid validation leakage).
    entity_mean_from_train = train_imputed.loc[mask_90].groupby(ENTITY_COLS)[TARGET_COL].mean().reset_index(name="entity_mean")
    entity_std_from_train = train_imputed.loc[mask_90].groupby(ENTITY_COLS)[TARGET_COL].std().reset_index(name="entity_std")
    entity_std_from_train["entity_std"] = entity_std_from_train["entity_std"].fillna(global_std)
    artifacts_a2["entity_mean_from_train"] = entity_mean_from_train
    artifacts_a2["entity_std_from_train"] = entity_std_from_train
    artifacts_a2["missing_threshold"] = 0.01
    X_train_90 = train_imputed.loc[mask_90, all_feature_cols].fillna(0).to_numpy()
    y_train_90 = train_imputed.loc[mask_90, TARGET_COL].to_numpy()
    w_train_90 = train_imputed.loc[mask_90, "weight"].to_numpy()
    X_val = train_imputed.loc[mask_10, all_feature_cols].fillna(0).to_numpy()
    y_val_orig = train_imputed.loc[mask_10, TARGET_COL].to_numpy()
    w_val = train_imputed.loc[mask_10, "weight"].to_numpy()
    y_val_model = transform_target(y_val_orig, target_transform) if target_transform else y_val_orig

    params = {
        "objective": "regression", "metric": "None", "boosting_type": "gbdt",
        "num_leaves": lb["num_leaves"], "min_data_in_leaf": lb["min_data_in_leaf"],
        "learning_rate": lb["learning_rate"], "verbose": -1, "seed": lb["seed"],
    }
    if lb.get("max_depth", -1) > 0:
        params["max_depth"] = lb["max_depth"]
    train_data_90 = lgb.Dataset(X_train_90, label=y_train_90, weight=w_train_90, feature_name=all_feature_cols, categorical_feature=entity_cat)
    val_data = lgb.Dataset(X_val, label=y_val_model, weight=w_val, reference=train_data_90)
    feval = make_skill_feval(y_val_orig, w_val, target_transform)
    logger.info("Step 2: training Model A2 with 90/10 temporal split")
    model_a2_90 = lgb.train(
        params, train_data_90,
        num_boost_round=lb.get("num_boost_round", 1000),
        valid_sets=[val_data], valid_names=["val"], feval=feval,
        callbacks=[lgb.log_evaluation(0), lgb.early_stopping(stopping_rounds=lb.get("early_stopping_rounds", 50), verbose=False)],
    )
    num_rounds = getattr(model_a2_90, "best_iteration", None) or lb.get("num_boost_round", 1000)
    pred_val_t = model_a2_90.predict(X_val)
    pred_val_orig = inverse_transform_target(pred_val_t, target_transform) if target_transform else pred_val_t
    val_rmse = np.sqrt(np.mean((y_val_orig - pred_val_orig) ** 2))
    val_skill = weighted_rmse_score(y_val_orig, pred_val_orig, w_val)

    # Train final model on full train with num_rounds from early stopping
    train_data_full = lgb.Dataset(X_train, label=y_train, weight=w_train, feature_name=all_feature_cols, categorical_feature=entity_cat)
    logger.info("Step 2: training final Model A2 on full train")
    model_a2 = lgb.train(
        params, train_data_full, num_boost_round=num_rounds,
        callbacks=[lgb.log_evaluation(100)],
    )
    pred_train = model_a2.predict(X_train)
    pred_train_orig = inverse_transform_target(pred_train, target_transform) if target_transform else pred_train
    y_train_orig = inverse_transform_target(y_train, target_transform) if target_transform else y_train
    train_rmse = np.sqrt(np.mean((y_train_orig - pred_train_orig) ** 2))
    train_skill = weighted_rmse_score(y_train_orig, pred_train_orig, w_train)
    step2_metrics = {
        "train_skill": float(train_skill), "train_rmse": float(train_rmse),
        "val_skill": float(val_skill), "val_rmse": float(val_rmse),
        "best_iteration": int(num_rounds),
    }

    logger.info("Step 2: loading test data from %s", test_path)
    test_df = _load_parquet_cached(test_path).copy()
    train_fe_for_state = train_imputed.copy()
    train_fe_for_state = train_fe_for_state.rename(columns={})  # already has lag_cols etc.
    test_sorted = test_df.sort_values(ENTITY_COLS + [TS_COL]).reset_index(drop=True)
    pred_test = _sequential_predict(test_sorted, train_fe_for_state, model_a2, artifacts_a2, all_feature_cols, target_transform)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"submission_v2_{ts}.csv")
    _write_submission(test_sorted, pred_test, out_path)
    metadata = _build_metadata("step2", step2_metrics, cfg)
    _write_metadata(metadata, out_path, out_dir)
    if experiment_name:
        _log_run_mlflow(experiment_name, "step2", metadata, out_path, out_dir)
    return out_path


# -----------------------------------------------------------------------------
# Step 3: Model B + Model A' -> submission_v3
# -----------------------------------------------------------------------------

def _run_step3(cfg, paths, experiment_name=None):
    il = cfg["input_lags"]
    tl = cfg["target_lags"]
    lb = cfg["lightgbm"]
    train_path = paths["train_path"]
    test_path = paths["test_path"]
    out_dir = paths["out_dir"]

    # --- Load train data ---
    logger.info("Step 3: loading train data from %s", train_path)
    train_df = _load_parquet_cached(train_path).copy()

    # --- Model B: base + input lags only (no target lags/rolling/aggregates) ---
    train_fe_b, all_feature_cols_b, artifacts_b = _build_train_features(
        train_df,
        use_target_lags=False,
        use_rolling=False,
        use_aggregates=False,
        use_input_lags=True,
        use_target_transform=False,
        lags_max=il["lags_max"],
        top_k_per_feature=il["top_k_per_feature"],
        use_global_lags=il["use_global_lags"],
        global_lags=il["global_lags"],
        use_float16_when_large=il.get("use_float16_when_large", False),
    )
    feature_cols_b = [c for c in all_feature_cols_b if c in train_fe_b.columns]
    X_train_b = train_fe_b[feature_cols_b].fillna(0).to_numpy()
    y_train_b = train_fe_b[TARGET_COL].to_numpy()
    w_train_b = train_fe_b["weight"].to_numpy()
    entity_cat = [c for c in ENTITY_CAT_FEATURE_NAMES if c in feature_cols_b]

    # --- Model B: 10% temporal validation for early stopping, then train on full ---
    unique_ts_b = sorted(train_fe_b[TS_COL].unique())
    split_idx_b = int(len(unique_ts_b) * 0.9)
    cutoff_b = unique_ts_b[split_idx_b]
    mask_b_90 = (train_fe_b[TS_COL].values <= cutoff_b)
    mask_b_10 = ~mask_b_90
    X_b_90 = train_fe_b.loc[mask_b_90, feature_cols_b].fillna(0).to_numpy()
    y_b_90 = train_fe_b.loc[mask_b_90, TARGET_COL].to_numpy()
    w_b_90 = train_fe_b.loc[mask_b_90, "weight"].to_numpy()
    X_b_val = train_fe_b.loc[mask_b_10, feature_cols_b].fillna(0).to_numpy()
    y_b_val = train_fe_b.loc[mask_b_10, TARGET_COL].to_numpy()
    w_b_val = train_fe_b.loc[mask_b_10, "weight"].to_numpy()
    params_b = {
        "objective": "regression", "metric": "None", "boosting_type": "gbdt",
        "num_leaves": lb["num_leaves"], "min_data_in_leaf": lb["min_data_in_leaf"],
        "learning_rate": lb["learning_rate"], "verbose": -1, "seed": lb["seed"],
    }
    if lb.get("max_depth", -1) > 0:
        params_b["max_depth"] = lb["max_depth"]
    train_data_b_90 = lgb.Dataset(X_b_90, label=y_b_90, weight=w_b_90, feature_name=feature_cols_b, categorical_feature=entity_cat)
    val_data_b = lgb.Dataset(X_b_val, label=y_b_val, weight=w_b_val, reference=train_data_b_90)
    feval_b = make_skill_feval(y_b_val, w_b_val, None)
    logger.info("Step 3: training Model B with 90/10 temporal split")
    model_b_90 = lgb.train(
        params_b, train_data_b_90,
        num_boost_round=lb.get("num_boost_round", 500),
        valid_sets=[val_data_b], valid_names=["val"], feval=feval_b,
        callbacks=[lgb.log_evaluation(0), lgb.early_stopping(stopping_rounds=lb.get("early_stopping_rounds", 50), verbose=False)],
    )
    num_rounds_b = getattr(model_b_90, "best_iteration", None) or lb.get("num_boost_round", 500)
    train_data_b = lgb.Dataset(X_train_b, label=y_train_b, weight=w_train_b, feature_name=feature_cols_b, categorical_feature=entity_cat)
    logger.info("Step 3: training final Model B on full train")
    model_b = lgb.train(params_b, train_data_b, num_boost_round=num_rounds_b, callbacks=[lgb.log_evaluation(100)])
    pred_train_b = model_b.predict(X_train_b)
    train_rmse_b = np.sqrt(np.mean((y_train_b - pred_train_b) ** 2))
    train_skill_b = weighted_rmse_score(y_train_b, pred_train_b, w_train_b)

    # --- Pseudo target for Model A': Model B predictions on full train (one pass) ---
    train_df = train_df.copy()
    pred_b_train = model_b.predict(train_fe_b[feature_cols_b].fillna(0).to_numpy())
    # Align by index: train_fe_b is sorted by entity+ts, train_df may be in read order
    train_df["y_pred_B_train"] = pd.Series(pred_b_train, index=train_fe_b.index).reindex(train_df.index).values
    train_pred_skill_b = weighted_rmse_score(train_df[TARGET_COL].to_numpy(), train_df["y_pred_B_train"].to_numpy(), train_df["weight"].to_numpy())

    # --- Build training data for Model A': base + pseudo target lags from y_pred_B_train ---
    train_imputed, impute_values = temporal_impute_missing(train_df, FEATURE_COLS, method="median")
    train_imputed, indicator_cols = create_missing_indicators(train_imputed, FEATURE_COLS, missing_threshold=0.01)
    train_imputed = train_imputed.sort_values(ENTITY_COLS + [TS_COL])
    # Pseudo target-derived features from OOF predictions (lags, rolling, aggregates)
    train_imputed, lag_cols_pseudo = create_lag_features(train_imputed, ENTITY_COLS, "y_pred_B_train", TS_COL, lags=tl.get("y_lags", Y_LAGS))
    train_imputed, rolling_cols_pseudo = create_rolling_features(train_imputed, ENTITY_COLS, "y_pred_B_train", TS_COL, windows=tl.get("windows", WINDOWS))
    train_imputed, agg_cols_pseudo = create_aggregate_features_t1(train_imputed, "y_pred_B_train", TS_COL, group_col="sub_category")
    train_imputed["y_target_global_mean"] = train_imputed["y_pred_B_train_global_mean"]
    train_imputed["y_target_sub_category_mean"] = train_imputed["y_pred_B_train_sub_category_mean"]
    train_imputed, entity_count_col = create_entity_count(train_imputed, ENTITY_COLS, TS_COL)
    # Same preprocessing as elsewhere: winsorize, log type-C, zero flags, encode
    numeric_to_clip = [c for c in FEATURE_COLS if c in train_imputed.columns]
    train_imputed, winsor_bounds = winsorize_features(train_imputed, numeric_to_clip, quantiles=(0.01, 0.99), fit_df=train_imputed)
    type_c_present = [c for c in TYPE_C_FEATURES if c in train_imputed.columns]
    train_imputed = log_transform_type_c(train_imputed, type_c_present)
    train_imputed, zero_flag_cols = create_zero_inflation_flags(train_imputed, ZERO_INFLATED_FEATURES)
    train_imputed, entity_encodings = encode_entity_categoricals(train_imputed, ENTITY_CATEGORICAL_COLS, encodings=None)
    # Input lags + optional min-max for float16
    lag_spec = select_input_lags(train_imputed, FEATURE_COLS, ENTITY_COLS, TS_COL, TARGET_COL, lags_max=il["lags_max"], top_k_per_feature=il["top_k_per_feature"], use_global_lags=il["use_global_lags"], global_lags=il["global_lags"])
    input_feature_min_max_ap = {}
    if il.get("use_float16_when_large", False):
        scale_cols = [c for c in FEATURE_COLS if c in train_imputed.columns and pd.api.types.is_numeric_dtype(train_imputed[c])]
        if scale_cols:
            input_feature_min_max_ap = fit_min_max_bounds(train_imputed, scale_cols)
            train_imputed = apply_min_max_scale(train_imputed, input_feature_min_max_ap)
    train_imputed, input_lag_cols = create_input_lag_features(train_imputed, ENTITY_COLS, TS_COL, FEATURE_COLS, lag_spec, use_float16_when_large=il.get("use_float16_when_large", False))
    global_mean = float(train_imputed[TARGET_COL].mean())
    global_std = float(train_imputed[TARGET_COL].std()) or 1.0
    horizon_numeric = train_imputed["horizon"].astype(float)
    horizon_x_subcat = horizon_numeric * train_imputed["y_target_sub_category_mean"]
    train_imputed["horizon_numeric"] = horizon_numeric
    train_imputed["horizon_x_subcat"] = horizon_x_subcat
    # Feature list for Model A' (same shape as step 1/2 but with pseudo target columns)
    entity_feature_cols = [c + "_enc" for c in ENTITY_CATEGORICAL_COLS]
    base_feature_cols = entity_feature_cols + [c for c in FEATURE_COLS if c in train_imputed.columns] + indicator_cols + zero_flag_cols
    all_feature_cols_ap = base_feature_cols + ["horizon_numeric", "horizon_x_subcat"] + lag_cols_pseudo + rolling_cols_pseudo + agg_cols_pseudo + [entity_count_col] + input_lag_cols
    all_feature_cols_ap = [c for c in all_feature_cols_ap if c in train_imputed.columns]
    # Artifacts for test-time: imputation, bounds, encodings, lag_spec, etc.
    artifacts_ap = {
        "impute_values": impute_values, "indicator_cols": indicator_cols, "winsor_bounds": winsor_bounds,
        "input_feature_min_max": input_feature_min_max_ap,
        "lag_cols": lag_cols_pseudo, "rolling_cols": rolling_cols_pseudo, "agg_cols": agg_cols_pseudo, "entity_count_col": entity_count_col,
        "input_lag_cols": input_lag_cols, "lag_spec": lag_spec, "y_lags": tl.get("y_lags", Y_LAGS), "windows": tl.get("windows", WINDOWS),
        "global_mean": global_mean, "global_std": global_std or 1.0, "entity_count_from_train": train_df.groupby(ENTITY_COLS).size().reset_index(name="entity_obs_count"),
        "entity_encodings": entity_encodings, "target_transform": None,
    }
    # --- Model A': 10% temporal validation for early stopping, then train on full ---
    X_train_ap = train_imputed[all_feature_cols_ap].fillna(0).to_numpy()
    y_train_ap = train_imputed[TARGET_COL].to_numpy()
    w_train_ap = train_imputed["weight"].to_numpy()
    entity_cat_ap = [c for c in ENTITY_CAT_FEATURE_NAMES if c in all_feature_cols_ap]
    unique_ts_ap = sorted(train_imputed[TS_COL].unique())
    split_idx_ap = int(len(unique_ts_ap) * 0.9)
    cutoff_ap = unique_ts_ap[split_idx_ap]
    mask_ap_90 = (train_imputed[TS_COL].values <= cutoff_ap)
    mask_ap_10 = ~mask_ap_90
    # Entity statistics for sequential-style artifacts: compute from 90%% temporal training slice only.
    entity_mean_from_train_ap = train_imputed.loc[mask_ap_90].groupby(ENTITY_COLS)[TARGET_COL].mean().reset_index(name="entity_mean")
    entity_std_from_train_ap = train_imputed.loc[mask_ap_90].groupby(ENTITY_COLS)[TARGET_COL].std().reset_index(name="entity_std")
    entity_std_from_train_ap["entity_std"] = entity_std_from_train_ap["entity_std"].fillna(global_std or 1.0)
    artifacts_ap["entity_mean_from_train"] = entity_mean_from_train_ap
    artifacts_ap["entity_std_from_train"] = entity_std_from_train_ap
    artifacts_ap["missing_threshold"] = 0.01
    X_ap_90 = train_imputed.loc[mask_ap_90, all_feature_cols_ap].fillna(0).to_numpy()
    y_ap_90 = train_imputed.loc[mask_ap_90, TARGET_COL].to_numpy()
    w_ap_90 = train_imputed.loc[mask_ap_90, "weight"].to_numpy()
    X_ap_val = train_imputed.loc[mask_ap_10, all_feature_cols_ap].fillna(0).to_numpy()
    y_ap_val = train_imputed.loc[mask_ap_10, TARGET_COL].to_numpy()
    w_ap_val = train_imputed.loc[mask_ap_10, "weight"].to_numpy()
    params_ap = {
        "objective": "regression", "metric": "None", "boosting_type": "gbdt",
        "num_leaves": lb["num_leaves"], "min_data_in_leaf": lb["min_data_in_leaf"],
        "learning_rate": lb["learning_rate"], "verbose": -1, "seed": lb["seed"],
    }
    if lb.get("max_depth", -1) > 0:
        params_ap["max_depth"] = lb["max_depth"]
    train_data_ap_90 = lgb.Dataset(X_ap_90, label=y_ap_90, weight=w_ap_90, feature_name=all_feature_cols_ap, categorical_feature=entity_cat_ap)
    val_data_ap = lgb.Dataset(X_ap_val, label=y_ap_val, weight=w_ap_val, reference=train_data_ap_90)
    feval_ap = make_skill_feval(y_ap_val, w_ap_val, None)
    logger.info("Step 3: training Model A' with 90/10 temporal split")
    model_ap_90 = lgb.train(
        params_ap, train_data_ap_90,
        num_boost_round=lb.get("num_boost_round", 500),
        valid_sets=[val_data_ap], valid_names=["val"], feval=feval_ap,
        callbacks=[lgb.log_evaluation(0), lgb.early_stopping(stopping_rounds=lb.get("early_stopping_rounds", 50), verbose=False)],
    )
    num_rounds_ap = getattr(model_ap_90, "best_iteration", None) or lb.get("num_boost_round", 500)
    pred_ap_val = model_ap_90.predict(X_ap_val)
    val_rmse_ap = np.sqrt(np.mean((y_ap_val - pred_ap_val) ** 2))
    val_skill_ap = weighted_rmse_score(y_ap_val, pred_ap_val, w_ap_val)
    train_data_ap = lgb.Dataset(X_train_ap, label=y_train_ap, weight=w_train_ap, feature_name=all_feature_cols_ap, categorical_feature=entity_cat_ap)
    logger.info("Step 3: training final Model A' on full train")
    model_ap = lgb.train(params_ap, train_data_ap, num_boost_round=num_rounds_ap, callbacks=[lgb.log_evaluation(100)])
    pred_train_ap = model_ap.predict(X_train_ap)
    train_rmse_ap = np.sqrt(np.mean((y_train_ap - pred_train_ap) ** 2))
    train_skill_ap = weighted_rmse_score(y_train_ap, pred_train_ap, w_train_ap)
    # Keys aligned with step1/step2 for primary model (A'); Model B metrics prefixed for MLflow consistency
    step3_metrics = {
        "train_skill": float(train_skill_ap), "train_rmse": float(train_rmse_ap),
        "val_skill": float(val_skill_ap), "val_rmse": float(val_rmse_ap),
        "best_iteration": int(num_rounds_ap),
        "model_b_train_skill": float(train_skill_b), "model_b_train_rmse": float(train_rmse_b),
        "model_b_train_pred_skill": float(train_pred_skill_b), "model_b_best_iteration": int(num_rounds_b),
    }

    # --- Test-time: predict with B, build pseudo lags from B's predictions, then predict with A' ---
    logger.info("Step 3: loading test data from %s", test_path)
    test_df = _load_parquet_cached(test_path).copy()
    # Model B preprocessing on test: use B-specific artifacts where applicable
    test_base = apply_imputation(test_df, artifacts_b.get("impute_values", impute_values))
    test_base, _ = create_missing_indicators(test_base, FEATURE_COLS, missing_threshold=artifacts_ap.get("missing_threshold", 0.01))
    test_base = apply_winsorize_bounds(test_base, winsor_bounds)
    type_c_present = [c for c in TYPE_C_FEATURES if c in test_base.columns]
    test_base = log_transform_type_c(test_base, type_c_present)
    test_base, _ = create_zero_inflation_flags(test_base, ZERO_INFLATED_FEATURES)
    test_base, _ = encode_entity_categoricals(test_base, ENTITY_CATEGORICAL_COLS, encodings=entity_encodings)
    test_base = test_base.merge(artifacts_ap["entity_mean_from_train"], on=ENTITY_COLS, how="left")
    test_base["entity_mean"] = test_base["entity_mean"].fillna(global_mean)
    test_base = test_base.merge(artifacts_ap["entity_std_from_train"], on=ENTITY_COLS, how="left")
    test_base = test_base.merge(artifacts_ap["entity_count_from_train"], on=ENTITY_COLS, how="left")
    test_base["entity_obs_count"] = test_base["entity_obs_count"].fillna(0)
    # Build input lags and horizon columns for Model B so test_base has feature_cols_b before predict
    max_lag_b = _max_lag_from_spec(artifacts_b["lag_spec"])
    train_tail_b = train_fe_b.groupby(ENTITY_COLS, group_keys=False).tail(max_lag_b)[ENTITY_COLS + [TS_COL] + [c for c in FEATURE_COLS if c in train_fe_b.columns]].copy()
    train_tail_b["_ord"] = -1
    test_input_b = test_base[ENTITY_COLS + [TS_COL] + [c for c in FEATURE_COLS if c in test_base.columns]].copy()
    test_input_b["_ord"] = np.arange(len(test_input_b))
    combined_b = pd.concat([train_tail_b, test_input_b], ignore_index=True)
    combined_b = combined_b.sort_values(ENTITY_COLS + [TS_COL])
    combined_b, _ = create_input_lag_features(combined_b, ENTITY_COLS, TS_COL, FEATURE_COLS, artifacts_b["lag_spec"], use_float16_when_large=il.get("use_float16_when_large", False))
    test_mask_b = combined_b["_ord"] >= 0
    test_input_lags_b = combined_b.loc[test_mask_b].sort_values("_ord")[artifacts_b["input_lag_cols"]]
    for c in artifacts_b["input_lag_cols"]:
        test_base[c] = test_input_lags_b[c].values
    test_base["horizon_numeric"] = test_base["horizon"].astype(float)
    test_base["horizon_x_subcat"] = test_base["horizon_numeric"] * global_mean  # B trained with use_aggregates=False
    # Model B predictions on test, then build lags/rolling/agg from them (no sequential loop)
    pred_b_test = model_b.predict(test_base[feature_cols_b].fillna(0).to_numpy())
    test_base["y_pred_B_test"] = pred_b_test
    test_base = test_base.sort_values(ENTITY_COLS + [TS_COL])
    test_base, _ = create_lag_features(test_base, ENTITY_COLS, "y_pred_B_test", TS_COL, lags=tl.get("y_lags", Y_LAGS))
    test_base, _ = create_rolling_features(test_base, ENTITY_COLS, "y_pred_B_test", TS_COL, windows=tl.get("windows", WINDOWS))
    test_base, _ = create_aggregate_features_t1(test_base, "y_pred_B_test", TS_COL, group_col="sub_category")
    # Alias test columns to match training feature names (y_pred_B_train_* -> same names as train)
    for c in lag_cols_pseudo:
        c_test = c.replace("y_pred_B_train", "y_pred_B_test")
        if c_test in test_base.columns:
            test_base[c] = test_base[c_test]
    for c in rolling_cols_pseudo:
        c_test = c.replace("y_pred_B_train", "y_pred_B_test")
        if c_test in test_base.columns:
            test_base[c] = test_base[c_test]
    test_base["y_pred_B_train_global_mean"] = test_base["y_pred_B_test_global_mean"]
    test_base["y_pred_B_train_sub_category_mean"] = test_base["y_pred_B_test_sub_category_mean"]
    test_base["y_target_global_mean"] = test_base["y_pred_B_test_global_mean"]
    test_base["y_target_sub_category_mean"] = test_base["y_pred_B_test_sub_category_mean"]
    # Apply min-max scaling on base features before creating input lags to mirror training.
    if input_feature_min_max_ap:
        test_base = apply_min_max_scale(test_base, input_feature_min_max_ap)
    # Input lags for test: only last max_lag rows per entity from train (not full train) to reduce memory
    max_lag = _max_lag_from_spec(lag_spec)
    train_tail = train_imputed.groupby(ENTITY_COLS, group_keys=False).tail(max_lag)[ENTITY_COLS + [TS_COL] + [c for c in FEATURE_COLS if c in train_imputed.columns]].copy()
    train_tail["_ord"] = -1
    test_input = test_base[ENTITY_COLS + [TS_COL] + [c for c in FEATURE_COLS if c in test_base.columns]].copy()
    test_input["_ord"] = np.arange(len(test_input))
    combined = pd.concat([train_tail, test_input], ignore_index=True)
    combined = combined.sort_values(ENTITY_COLS + [TS_COL])
    combined, _ = create_input_lag_features(combined, ENTITY_COLS, TS_COL, FEATURE_COLS, lag_spec, use_float16_when_large=il.get("use_float16_when_large", False))
    test_mask = combined["_ord"] >= 0
    test_input_lags = combined.loc[test_mask].sort_values("_ord")[input_lag_cols]
    for c in input_lag_cols:
        test_base[c] = test_input_lags[c].values
    test_base["horizon_numeric"] = test_base["horizon"].astype(float)
    test_base["horizon_x_subcat"] = test_base["horizon_numeric"] * test_base["y_target_sub_category_mean"]
    for c in all_feature_cols_ap:
        if c not in test_base.columns:
            test_base[c] = 0.0
    X_test_ap = test_base[all_feature_cols_ap].fillna(0).to_numpy()
    pred_v3 = model_ap.predict(X_test_ap)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"submission_v3_{ts}.csv")
    _write_submission(test_base, pred_v3, out_path)
    metadata = _build_metadata("step3", step3_metrics, cfg)
    _write_metadata(metadata, out_path, out_dir)
    if experiment_name:
        _log_run_mlflow(experiment_name, "step3", metadata, out_path, out_dir)
    return out_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Three-step submission pipeline (v1, v2, v3).")
    parser.add_argument("--step", type=str, default="all", choices=["1", "2", "3", "all"], help="Run step 1, 2, 3, or all.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML. Default: project_root/config.yaml.")
    parser.add_argument("--experiment", type=str, default=None, help="MLflow experiment name. If set, log metadata and artifacts for each step.")
    args = parser.parse_args()
    config_path = args.config or os.path.join(_project_root, "config.yaml")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    logger.info("Loading config from %s", config_path)
    cfg = load_config(config_path)
    paths = _get_paths(cfg)
    paths["data_dir"] = os.path.join(_project_root, cfg.get("data_dir", "data"))
    os.makedirs(paths["out_dir"], exist_ok=True)
    experiment_name = args.experiment

    if args.step in ("1", "all"):
        logger.info("Starting Step 1")
        _run_step1(cfg, paths, experiment_name=experiment_name)
    if args.step in ("2", "all"):
        logger.info("Starting Step 2")
        _run_step2(cfg, paths, experiment_name=experiment_name)
    if args.step in ("3", "all"):
        logger.info("Starting Step 3")
        _run_step3(cfg, paths, experiment_name=experiment_name)
    logger.info("All requested steps completed.")


if __name__ == "__main__":
    main()
