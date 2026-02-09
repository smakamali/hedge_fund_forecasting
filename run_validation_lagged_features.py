"""
Validation experiment: base features plus y_target lags/rolling/aggregates (autoregressive)
and optional input lag features. Self-contained; only imports from preprocessing and evaluation.
Run: conda run -n forecast_fund python run_validation_lagged_features.py
Optional flags: --no-input-lags, --no-target-lags, --no-rolling, --no-aggregates
Config: lag params (lags_max, top_k_per_feature, use_global_lags, global_lags) loaded from config.yaml or --config path.
To see step-by-step debug diagnostics, set logging level to DEBUG.
MLflow tracking is enabled by default; use --no-mlflow to disable.
"""
import argparse
import json
import logging
import os
import tempfile

import mlflow
import numpy as np
import pandas as pd
import lightgbm as lgb
from evaluation import temporal_train_test_split, weighted_rmse_score, make_skill_feval
from preprocessing import (
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

ENTITY_COLS = ["code", "sub_code", "sub_category", "horizon"]
ENTITY_CAT_FEATURE_NAMES = [c + "_enc" for c in ENTITY_CATEGORICAL_COLS]
TS_COL = "ts_index"
TARGET_COL = "y_target"
Y_LAGS = [1, 2, 3, 5, 10, 20]
WINDOWS = [5, 20]

DEFAULT_LAGS_MAX = 5
DEFAULT_TOP_K_PER_FEATURE = 2
DEFAULT_USE_GLOBAL_LAGS = False
DEFAULT_GLOBAL_LAGS = None

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load config from YAML or JSON file. Returns dict with input_lags section. Uses defaults if file missing."""
    defaults = {
        "input_lags": {
            "lags_max": DEFAULT_LAGS_MAX,
            "top_k_per_feature": DEFAULT_TOP_K_PER_FEATURE,
            "use_global_lags": DEFAULT_USE_GLOBAL_LAGS,
            "global_lags": DEFAULT_GLOBAL_LAGS,
        }
    }
    if not os.path.isfile(path):
        return defaults
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    try:
        if path.endswith(".yaml") or path.endswith(".yml"):
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                raise ImportError(
                    "YAML config requires pyyaml. Install with: pip install pyyaml. Or use .json config."
                )
        else:
            data = json.loads(content)
        if data is None:
            return defaults
        il = data.get("input_lags", {})
        merged = defaults["input_lags"].copy()
        if "lags_max" in il:
            merged["lags_max"] = int(il["lags_max"])
        if "top_k_per_feature" in il:
            merged["top_k_per_feature"] = int(il["top_k_per_feature"])
        if "use_global_lags" in il:
            merged["use_global_lags"] = bool(il["use_global_lags"])
        if "global_lags" in il:
            gl = il["global_lags"]
            merged["global_lags"] = gl if gl is None else [int(x) for x in gl]
        return {"input_lags": merged}
    except Exception as e:
        logger.warning("Failed to parse config %s: %s. Using defaults.", path, e)
        return defaults


def _build_train_features(
    train_df,
    use_target_lags=True,
    use_rolling=True,
    use_aggregates=True,
    use_input_lags=True,
    use_target_transform=False,
    lags_max=5,
    top_k_per_feature=2,
    use_global_lags=False,
    global_lags=None,
    use_float16_when_large=False,
    y_lags=None,
):
    """Full feature pipeline on train (has y_target). Returns (train_fe, all_feature_cols, artifacts)."""
    train_imputed, impute_values = temporal_impute_missing(train_df, FEATURE_COLS, method="median")
    train_imputed, indicator_cols = create_missing_indicators(train_imputed, FEATURE_COLS, missing_threshold=0.01)

    train_imputed = train_imputed.sort_values(ENTITY_COLS + [TS_COL])

    # Optional target transform (MinMax -> log1p -> MinMax) fit on train, applied before target-derived features
    if use_target_transform:
        target_transform = fit_target_transform(train_df[TARGET_COL].to_numpy())
        train_imputed[TARGET_COL] = transform_target(train_imputed[TARGET_COL].to_numpy(), target_transform)
    else:
        target_transform = None

    lag_cols = []
    if use_target_lags:
        lags = y_lags if y_lags is not None else Y_LAGS
        train_imputed, lag_cols = create_lag_features(
            train_imputed, ENTITY_COLS, TARGET_COL, TS_COL, lags=lags
        )
    rolling_cols = []
    if use_rolling:
        train_imputed, rolling_cols = create_rolling_features(
            train_imputed, ENTITY_COLS, TARGET_COL, TS_COL, windows=WINDOWS
        )
    agg_cols = []
    if use_aggregates:
        train_imputed, agg_cols = create_aggregate_features_t1(
            train_imputed, TARGET_COL, TS_COL, group_col="sub_category"
        )
    train_imputed, entity_count_col = create_entity_count(train_imputed, ENTITY_COLS, TS_COL)

    numeric_to_clip = [c for c in FEATURE_COLS if c in train_imputed.columns]
    train_imputed, winsor_bounds = winsorize_features(
        train_imputed, numeric_to_clip, quantiles=(0.01, 0.99), fit_df=train_imputed
    )
    type_c_present = [c for c in TYPE_C_FEATURES if c in train_imputed.columns]
    train_imputed = log_transform_type_c(train_imputed, type_c_present)
    train_imputed, zero_flag_cols = create_zero_inflation_flags(train_imputed, ZERO_INFLATED_FEATURES)
    train_imputed, entity_encodings = encode_entity_categoricals(
        train_imputed, ENTITY_CATEGORICAL_COLS, encodings=None
    )

    input_lag_cols = []
    lag_spec = None
    if use_input_lags:
        lag_spec = select_input_lags(
            train_imputed, FEATURE_COLS, ENTITY_COLS, TS_COL, TARGET_COL,
            lags_max=lags_max,
            top_k_per_feature=top_k_per_feature,
            use_global_lags=use_global_lags,
            global_lags=global_lags,
        )
        train_imputed, input_lag_cols = create_input_lag_features(
            train_imputed, ENTITY_COLS, TS_COL, FEATURE_COLS, lag_spec,
            use_float16_when_large=use_float16_when_large,
        )
    extra_feature_cols = lag_cols + rolling_cols + agg_cols + [entity_count_col] + input_lag_cols

    # Entity/global stats (needed for interaction fallback when aggregates disabled, and for artifacts)
    global_mean = float(train_imputed[TARGET_COL].mean())
    global_std = float(train_imputed[TARGET_COL].std())
    if pd.isna(global_std) or global_std == 0:
        global_std = 1.0

    # Add interaction columns via concat to avoid fragmented-DataFrame PerformanceWarning
    horizon_numeric = train_imputed["horizon"].astype(float)
    if use_aggregates and agg_cols:
        horizon_x_subcat = horizon_numeric * train_imputed["y_target_sub_category_mean"]
    else:
        horizon_x_subcat = horizon_numeric * global_mean
    train_imputed = pd.concat(
        [
            train_imputed,
            pd.DataFrame({"horizon_numeric": horizon_numeric, "horizon_x_subcat": horizon_x_subcat}, index=train_imputed.index),
        ],
        axis=1,
    )
    interaction_cols = ["horizon_numeric", "horizon_x_subcat"]
    entity_feature_cols = [c + "_enc" for c in ENTITY_CATEGORICAL_COLS]

    base_feature_cols = (
        entity_feature_cols
        + [c for c in FEATURE_COLS if c in train_imputed.columns]
        + indicator_cols
        + zero_flag_cols
    )
    all_feature_cols = base_feature_cols + interaction_cols + extra_feature_cols
    all_feature_cols = [c for c in all_feature_cols if c in train_imputed.columns]

    # Entity-level stats from train (for fallbacks)
    entity_mean_from_train = (
        train_imputed.groupby(ENTITY_COLS)[TARGET_COL].mean().reset_index(name="entity_mean")
    )
    entity_std_from_train = (
        train_imputed.groupby(ENTITY_COLS)[TARGET_COL].std().reset_index(name="entity_std")
    )
    entity_std_from_train["entity_std"] = entity_std_from_train["entity_std"].fillna(global_std)
    entity_std_from_train.loc[entity_std_from_train["entity_std"] == 0, "entity_std"] = global_std

    artifacts = {
        "impute_values": impute_values,
        "indicator_cols": indicator_cols,
        "winsor_bounds": winsor_bounds,
        "lag_cols": lag_cols,
        "rolling_cols": rolling_cols,
        "agg_cols": agg_cols,
        "entity_count_col": entity_count_col,
        "input_lag_cols": input_lag_cols,
        "lag_spec": lag_spec,
        "use_float16_when_large": use_float16_when_large,
        "y_lags": (y_lags if y_lags is not None else Y_LAGS) if use_target_lags else [],
        "windows": WINDOWS if use_rolling else [],
        "global_mean": global_mean,
        "global_std": global_std,
        "entity_mean_from_train": entity_mean_from_train,
        "entity_std_from_train": entity_std_from_train,
        "entity_count_from_train": train_df.groupby(ENTITY_COLS).size().reset_index(name="entity_obs_count"),
        "entity_encodings": entity_encodings,
        "target_transform": target_transform,
    }
    return train_imputed, all_feature_cols, artifacts


def _build_val_features(val_df, artifacts, train_fe=None):
    """Build val feature matrix.
    Input lag features: if train_fe is provided, compute from combined train+val (real values); else fallback to 0.
    Target-derived features (lags, rolling, agg): if train_fe is provided, compute from combined train+val (real values);
    otherwise use per-entity train mean/std fallbacks."""
    impute_values = artifacts["impute_values"]
    winsor_bounds = artifacts["winsor_bounds"]
    lag_cols = artifacts["lag_cols"]
    rolling_cols = artifacts["rolling_cols"]
    input_lag_cols = artifacts.get("input_lag_cols", [])
    lag_spec = artifacts.get("lag_spec")
    use_float16_when_large = artifacts.get("use_float16_when_large", False)
    global_mean = artifacts["global_mean"]
    global_std = artifacts["global_std"]
    entity_mean_from_train = artifacts["entity_mean_from_train"]
    entity_std_from_train = artifacts["entity_std_from_train"]
    entity_count_from_train = artifacts["entity_count_from_train"]
    entity_encodings = artifacts.get("entity_encodings", {})

    val_imputed = apply_imputation(val_df, impute_values)
    val_imputed, _ = create_missing_indicators(val_imputed, FEATURE_COLS, missing_threshold=0.01)
    val_imputed = apply_winsorize_bounds(val_imputed, winsor_bounds)
    type_c_present = [c for c in TYPE_C_FEATURES if c in val_imputed.columns]
    val_imputed = log_transform_type_c(val_imputed, type_c_present)
    val_imputed, _ = create_zero_inflation_flags(val_imputed, ZERO_INFLATED_FEATURES)
    val_imputed, _ = encode_entity_categoricals(
        val_imputed, ENTITY_CATEGORICAL_COLS, encodings=entity_encodings
    )

    # Per-entity fallbacks: merge entity mean/std from train (fallback to global)
    val_imputed = val_imputed.merge(entity_mean_from_train, on=ENTITY_COLS, how="left")
    val_imputed["entity_mean"] = val_imputed["entity_mean"].fillna(global_mean)
    val_imputed = val_imputed.merge(entity_std_from_train, on=ENTITY_COLS, how="left")
    val_imputed["entity_std"] = val_imputed["entity_std"].fillna(global_std)

    # Input lags: compute from train+val when we have train_fe (inputs are known for val)
    if train_fe is not None and input_lag_cols and lag_spec is not None:
        train_base = train_fe[ENTITY_COLS + [TS_COL] + FEATURE_COLS].copy()
        train_base["_val_ord"] = -1
        val_base = val_imputed[ENTITY_COLS + [TS_COL] + FEATURE_COLS].copy()
        val_base["_val_ord"] = np.arange(len(val_base))
        combined = pd.concat([train_base, val_base], ignore_index=True)
        combined = combined.sort_values(ENTITY_COLS + [TS_COL])
        combined, _ = create_input_lag_features(
            combined, ENTITY_COLS, TS_COL, FEATURE_COLS, lag_spec,
            use_float16_when_large=use_float16_when_large,
        )
        val_mask = combined["_val_ord"] >= 0
        val_input_lags = combined.loc[val_mask].sort_values("_val_ord")[input_lag_cols]
        val_input_lags_df = pd.DataFrame(
            val_input_lags.values, index=val_imputed.index, columns=input_lag_cols
        )
        val_imputed = pd.concat([val_imputed, val_input_lags_df], axis=1)
    else:
        val_input_lags = None

    # Target-derived features: compute from combined train+val when train_fe provided (real values)
    agg_cols = artifacts.get("agg_cols", [])
    target_derived_cols = lag_cols + rolling_cols + agg_cols
    target_transform = artifacts.get("target_transform")

    if train_fe is not None and target_derived_cols:
        train_base = train_fe[ENTITY_COLS + [TS_COL, TARGET_COL]].copy()
        train_base["_val_ord"] = -1
        val_base = val_imputed[ENTITY_COLS + [TS_COL, TARGET_COL]].copy()
        if target_transform is not None:
            val_base[TARGET_COL] = transform_target(val_df[TARGET_COL].to_numpy(), target_transform)
        val_base["_val_ord"] = np.arange(len(val_base))
        combined = pd.concat([train_base, val_base], ignore_index=True)
        combined = combined.sort_values(ENTITY_COLS + [TS_COL])
        y_lags_list = artifacts.get("y_lags", [])
        windows_list = artifacts.get("windows", WINDOWS)
        if lag_cols and y_lags_list:
            combined, _ = create_lag_features(
                combined, ENTITY_COLS, TARGET_COL, TS_COL, lags=y_lags_list
            )
        if rolling_cols and windows_list:
            combined, _ = create_rolling_features(
                combined, ENTITY_COLS, TARGET_COL, TS_COL, windows=windows_list
            )
        if agg_cols:
            combined, _ = create_aggregate_features_t1(
                combined, TARGET_COL, TS_COL, group_col="sub_category"
            )
        val_mask = combined["_val_ord"] >= 0
        val_target_derived = combined.loc[val_mask].sort_values("_val_ord")[target_derived_cols]
        val_target_derived_df = pd.DataFrame(
            val_target_derived.values, index=val_imputed.index, columns=target_derived_cols
        )
        val_imputed = pd.concat([val_imputed, val_target_derived_df], axis=1)
    else:
        extra_cols = {}
        for c in lag_cols:
            extra_cols[c] = val_imputed["entity_mean"].values
        for c in rolling_cols:
            if "rolling_std" in c:
                extra_cols[c] = 0.0
            else:
                extra_cols[c] = val_imputed["entity_mean"].values
        if agg_cols:
            extra_cols["y_target_global_mean"] = val_imputed["entity_mean"].values
            extra_cols["y_target_sub_category_mean"] = val_imputed["entity_mean"].values
        if val_input_lags is None and input_lag_cols:
            for c in input_lag_cols:
                extra_cols[c] = 0.0
        if extra_cols:
            extra = pd.DataFrame(extra_cols, index=val_imputed.index)
            val_imputed = pd.concat([val_imputed, extra], axis=1)

    horizon_numeric = val_imputed["horizon"].astype(float)

    val_imputed = val_imputed.merge(entity_count_from_train, on=ENTITY_COLS, how="left")
    val_imputed["entity_obs_count"] = val_imputed["entity_obs_count"].fillna(0)

    horizon_x_subcat = horizon_numeric * val_imputed["entity_mean"].values
    val_imputed = pd.concat(
        [val_imputed, pd.DataFrame({"horizon_numeric": horizon_numeric, "horizon_x_subcat": horizon_x_subcat}, index=val_imputed.index)],
        axis=1,
    )
    return val_imputed


def main(
    use_input_lags=True,
    use_target_lags=True,
    use_rolling=True,
    use_aggregates=True,
    use_target_transform=False,
    use_mlflow=True,
    lags_max=5,
    top_k_per_feature=2,
    use_global_lags=False,
    global_lags=None,
    experiment_name="hedge_fund_forecasting_default",
    run_name=None,
):
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    if use_mlflow:
        mlflow.set_experiment(experiment_name)
        if run_name is None:
            run_name_parts = []
            if use_target_lags:
                run_name_parts.append("ylag")
            if use_rolling:
                run_name_parts.append("roll")
            if use_aggregates:
                run_name_parts.append("agg")
            if use_input_lags:
                run_name_parts.append("xlag")
            if use_target_transform:
                run_name_parts.append("ytrans")
            run_name = "+".join(run_name_parts) if run_name_parts else "base"

        mlflow.start_run(run_name=run_name)
        params = {
            "use_input_lags": use_input_lags,
            "use_target_lags": use_target_lags,
            "use_rolling": use_rolling,
            "use_aggregates": use_aggregates,
            "use_target_transform": use_target_transform,
            "lags_max": lags_max,
            "top_k_per_feature": top_k_per_feature,
            "use_global_lags": use_global_lags,
            "global_lags": ",".join(map(str, global_lags)) if global_lags else "null",
        }
        mlflow.log_params({k: str(v) for k, v in params.items()})
    try:
        return _run_validation(
            use_input_lags=use_input_lags,
            use_target_lags=use_target_lags,
            use_rolling=use_rolling,
            use_aggregates=use_aggregates,
            use_target_transform=use_target_transform,
            use_mlflow=use_mlflow,
            lags_max=lags_max,
            top_k_per_feature=top_k_per_feature,
            use_global_lags=use_global_lags,
            global_lags=global_lags,
        )
    finally:
        if use_mlflow:
            mlflow.end_run()


def _run_validation(
    use_input_lags=True,
    use_target_lags=True,
    use_rolling=True,
    use_aggregates=True,
    use_target_transform=False,
    use_mlflow=True,
    lags_max=5,
    top_k_per_feature=2,
    use_global_lags=False,
    global_lags=None,
):
    logger.info("Loading train...")
    dataset = prepare_dataset_for_lag_config(
        use_input_lags=use_input_lags,
        use_target_lags=use_target_lags,
        use_rolling=use_rolling,
        use_aggregates=use_aggregates,
        use_target_transform=use_target_transform,
        lags_max=lags_max,
        top_k_per_feature=top_k_per_feature,
        use_global_lags=use_global_lags,
        global_lags=global_lags,
    )
    if not use_input_lags:
        logger.info("Excluding input lag features (--no-input-lags).")

    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    w_train = dataset["w_train"]
    X_val = dataset["X_val"]
    y_val_orig = dataset["y_val_orig"]
    w_val = dataset["w_val"]
    feature_cols_final = dataset["feature_cols_final"]
    target_transform = dataset["target_transform"]

    # --- Debug Step 1: targets and weights ---
    logger.debug("--- Debug Step 1: targets and weights ---")
    logger.debug(
        "y_train: shape=%s, nan=%s, inf=%s, min=%.4f, max=%.4f, mean=%.4f",
        y_train.shape, np.isnan(y_train).sum(), np.isinf(y_train).sum(),
        np.nanmin(y_train), np.nanmax(y_train), np.nanmean(y_train),
    )
    logger.debug(
        "w_train: shape=%s, nan=%s, inf=%s, <=0=%s, min=%.4f, max=%.4f",
        w_train.shape, np.isnan(w_train).sum(), np.isinf(w_train).sum(), (w_train <= 0).sum(),
        np.nanmin(w_train), np.nanmax(w_train),
    )
    logger.debug(
        "y_val:   shape=%s, nan=%s, inf=%s, min=%.4f, max=%.4f",
        y_val_orig.shape, np.isnan(y_val_orig).sum(), np.isinf(y_val_orig).sum(), np.nanmin(y_val_orig), np.nanmax(y_val_orig),
    )
    logger.debug("w_val:   nan=%s, inf=%s, <=0=%s", np.isnan(w_val).sum(), np.isinf(w_val).sum(), (w_val <= 0).sum())

    # --- Debug Step 2: feature matrices finite? ---
    logger.debug("--- Debug Step 2: feature matrices ---")
    X_train_finite = np.isfinite(X_train)
    X_val_finite = np.isfinite(X_val)
    logger.debug(
        "X_train: shape=%s, all_finite=%s, nan_count=%s, inf_count=%s",
        X_train.shape, X_train_finite.all(), np.isnan(X_train).sum(), np.isinf(X_train).sum(),
    )
    logger.debug(
        "X_val:   shape=%s, all_finite=%s, nan_count=%s, inf_count=%s",
        X_val.shape, X_val_finite.all(), np.isnan(X_val).sum(), np.isinf(X_val).sum(),
    )
    if not X_train_finite.all() or not X_val_finite.all():
        for j in range(X_train.shape[1]):
            if not np.isfinite(X_train[:, j]).all() or not np.isfinite(X_val[:, j]).all():
                logger.debug(
                    "  first non-finite column index: %s (name: %s)",
                    j, feature_cols_final[j] if j < len(feature_cols_final) else "?",
                )

    params = {
        "objective": "regression",
        "metric": "None",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "verbose": -1,
        "min_data_in_leaf": 20,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "seed": 42,
    }
    if use_mlflow:
        mlflow.log_params({k: str(v) for k, v in params.items()})
    parts = ["base"]
    if use_target_lags:
        parts.append("target_lags")
    if use_rolling:
        parts.append("rolling")
    if use_aggregates:
        parts.append("aggregates")
    if use_input_lags:
        parts.append("input_lags")
    logger.info("Training model (%s)...", " + ".join(parts))
    val_skill, train_skill, val_rmse, train_rmse, n_rounds, model = train_and_evaluate(
        dataset,
        num_leaves=31,
        min_data_in_leaf=20,
        max_depth=-1,
        use_skill_feval=True,
        return_model=True,
    )
    pred_val_t = model.predict(X_val)
    pred_val = (
        inverse_transform_target(pred_val_t, target_transform)
        if target_transform is not None
        else pred_val_t
    )
    logger.info(
        "Train:      RMSE=%.6f  skill=%.6f",
        train_rmse, train_skill,
    )
    logger.info(
        "Validation: RMSE=%.6f  skill=%.6f",
        val_rmse, val_skill,
    )

    if use_mlflow:
        mlflow.log_metrics({
            "train_rmse": train_rmse,
            "train_skill": train_skill,
            "val_rmse": val_rmse,
            "val_skill": val_skill,
            "num_boost_round_actual": model.best_iteration,
        })
        mlflow.lightgbm.log_model(model, "model")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(feature_cols_final))
            feat_path = f.name
        mlflow.log_artifact(feat_path, "features")
        os.unlink(feat_path)

    # --- Debug Step 3: prediction distribution ---
    logger.debug("--- Debug Step 3: validation predictions ---")
    logger.debug(
        "pred_val: min=%.4f, max=%.4f, mean=%.4f, std=%.4f, n_unique=%s",
        pred_val.min(), pred_val.max(), pred_val.mean(), pred_val.std(),
        len(np.unique(pred_val.round(decimals=6))),
    )
    if np.isfinite(pred_val).all():
        logger.debug("pred_val: all finite")
    else:
        logger.debug("pred_val: nan=%s, inf=%s", np.isnan(pred_val).sum(), np.isinf(pred_val).sum())

    # --- Debug Step 4: why is skill 0? (ratio = weighted_MSE / sum(w*y^2); skill = sqrt(1 - clip(ratio,0,1))) ---
    denom = np.sum(w_val * y_val_orig ** 2)
    weighted_mse = np.sum(w_val * (y_val_orig - pred_val) ** 2)
    ratio = weighted_mse / denom if denom > 0 else float("nan")
    logger.debug("--- Debug Step 4: skill score components ---")
    logger.debug("sum(w_val * y_val^2) = %.6e", denom)
    logger.debug("sum(w_val * (y_val - pred_val)^2) = %.6e", weighted_mse)
    logger.debug("ratio (weighted_MSE / denom) = %.6f  (skill=0 when ratio >= 1)", ratio)

    # --- Debug Step 5: weight concentration (high-weight rows can dominate and push ratio > 1) ---
    logger.debug("--- Debug Step 5: weight concentration ---")
    w_pos = np.maximum(w_val, 0.0)
    denom_pos = np.sum(w_pos * y_val_orig ** 2)
    mse_pos = np.sum(w_pos * (y_val_orig - pred_val) ** 2)
    idx_top = np.argsort(w_pos)[-max(1, len(w_val) // 100):]  # top 1% by weight
    pct_denom_top = 100.0 * np.sum(w_pos[idx_top] * y_val_orig[idx_top] ** 2) / denom_pos if denom_pos > 0 else 0
    pct_mse_top = 100.0 * np.sum(w_pos[idx_top] * (y_val_orig[idx_top] - pred_val[idx_top]) ** 2) / mse_pos if mse_pos > 0 else 0
    logger.debug(
        "Top 1%% of rows by weight: contribute %.1f%% of denom, %.1f%% of weighted MSE",
        pct_denom_top, pct_mse_top,
    )
    logger.debug("(If most of the metric is in few rows, improving those predictions will bring ratio below 1.)")

    return val_skill


def prepare_dataset_for_lag_config(
    use_input_lags=True,
    use_target_lags=False,
    use_rolling=False,
    use_aggregates=False,
    use_target_transform=False,
    lags_max=5,
    top_k_per_feature=2,
    use_global_lags=False,
    global_lags=None,
    use_float16_when_large=False,
    y_lags=None,
):
    """
    Load data, build features, and extract arrays for a given lag config.
    Used by run_tune_lgb_params to avoid repeating data generation per param combo.
    Returns dict with X_train, y_train, w_train, X_val, y_val_orig, w_val,
    feature_cols_final, target_transform, y_train_orig, entity_cat_feature_names.
    """
    train_df = pd.read_parquet("train.parquet")
    train_part, val_df, cutoff = temporal_train_test_split(
        train_df, ts_col=TS_COL, test_size=0.2
    )
    train_fe, all_feature_cols, artifacts = _build_train_features(
        train_part,
        use_target_lags=use_target_lags,
        use_rolling=use_rolling,
        use_aggregates=use_aggregates,
        use_input_lags=use_input_lags,
        use_target_transform=use_target_transform,
        lags_max=lags_max,
        top_k_per_feature=top_k_per_feature,
        use_global_lags=use_global_lags,
        global_lags=global_lags,
        use_float16_when_large=use_float16_when_large,
        y_lags=y_lags,
    )
    needs_train_fe = use_input_lags or use_target_lags or use_rolling or use_aggregates
    val_fe = _build_val_features(
        val_df, artifacts, train_fe=train_fe if needs_train_fe else None
    )
    if use_input_lags:
        feature_cols_final = all_feature_cols
    else:
        input_lag_cols = set(artifacts.get("input_lag_cols", []))
        feature_cols_final = [c for c in all_feature_cols if c not in input_lag_cols]
    for c in feature_cols_final:
        if c not in val_fe.columns:
            val_fe[c] = 0.0
    X_train = train_fe[feature_cols_final].fillna(0).to_numpy()
    y_train = train_fe[TARGET_COL].to_numpy()
    w_train = train_fe["weight"].to_numpy()
    X_val = val_fe[feature_cols_final].fillna(0).to_numpy()
    y_val_orig = val_df[TARGET_COL].to_numpy()
    w_val = val_df["weight"].to_numpy()
    _keys = ENTITY_COLS + [TS_COL]
    _right = train_part[_keys + [TARGET_COL]].drop_duplicates(subset=_keys, keep="first")
    _merged = train_fe[_keys].merge(_right, on=_keys, how="left")
    y_train_orig = _merged[TARGET_COL].to_numpy()
    target_transform = artifacts.get("target_transform")
    entity_cat_feature_names = [c for c in ENTITY_CAT_FEATURE_NAMES if c in feature_cols_final]
    return {
        "X_train": X_train,
        "y_train": y_train,
        "w_train": w_train,
        "X_val": X_val,
        "y_val_orig": y_val_orig,
        "w_val": w_val,
        "feature_cols_final": feature_cols_final,
        "target_transform": target_transform,
        "y_train_orig": y_train_orig,
        "entity_cat_feature_names": entity_cat_feature_names,
    }


def train_and_evaluate(
    dataset,
    num_leaves=31,
    min_data_in_leaf=20,
    max_depth=-1,
    use_skill_feval=True,
    return_model=False,
):
    """
    Train LightGBM with given tree params and return metrics. No MLflow.
    dataset: dict from prepare_dataset_for_lag_config.
    use_skill_feval: if True, use make_skill_feval for early stopping; else use RMSE.
    return_model: if True, return model as 6th element (for MLflow logging).
    Returns (val_skill, train_skill, val_rmse, train_rmse, num_boost_round_actual[, model]).
    """
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    w_train = dataset["w_train"]
    X_val = dataset["X_val"]
    y_val_orig = dataset["y_val_orig"]
    w_val = dataset["w_val"]
    feature_cols_final = dataset["feature_cols_final"]
    target_transform = dataset["target_transform"]
    y_train_orig = dataset["y_train_orig"]
    entity_cat_feature_names = dataset["entity_cat_feature_names"]
    y_val_model = (
        transform_target(y_val_orig, target_transform)
        if target_transform is not None
        else y_val_orig.copy()
    )
    params = {
        "objective": "regression",
        "metric": "None" if use_skill_feval else "rmse",
        "boosting_type": "gbdt",
        "num_leaves": num_leaves,
        "learning_rate": 0.1,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "verbose": -1,
        "min_data_in_leaf": min_data_in_leaf,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "seed": 42,
    }
    if max_depth > 0:
        params["max_depth"] = max_depth
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        weight=w_train,
        feature_name=feature_cols_final,
        categorical_feature=entity_cat_feature_names,
    )
    val_data = lgb.Dataset(
        X_val,
        label=y_val_model,
        weight=w_val,
        reference=train_data,
    )
    callbacks = [
        lgb.log_evaluation(0),
        lgb.early_stopping(stopping_rounds=50, verbose=False),
    ]
    feval = make_skill_feval(y_val_orig, w_val, target_transform) if use_skill_feval else None
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        valid_names=["val"],
        feval=feval,
        callbacks=callbacks,
    )
    pred_train_t = model.predict(X_train)
    pred_val_t = model.predict(X_val)
    if target_transform is not None:
        pred_train = inverse_transform_target(pred_train_t, target_transform)
        pred_val = inverse_transform_target(pred_val_t, target_transform)
    else:
        pred_train = pred_train_t
        pred_val = pred_val_t
    train_rmse = np.sqrt(np.mean((y_train_orig - pred_train) ** 2))
    train_skill = weighted_rmse_score(y_train_orig, pred_train, w_train)
    val_rmse = np.sqrt(np.mean((y_val_orig - pred_val) ** 2))
    val_skill = weighted_rmse_score(y_val_orig, pred_val, w_val)
    out = (val_skill, train_skill, val_rmse, train_rmse, model.best_iteration)
    if return_model:
        out = out + (model,)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validation: base + optional target lags, rolling, aggregates, input lags."
    )
    parser.add_argument(
        "--no-input-lags",
        action="store_true",
        help="Exclude input lag features (lagged 86 features).",
    )
    parser.add_argument(
        "--no-target-lags",
        action="store_true",
        help="Exclude y_target lag features (e.g. y_target_lag_1, ...).",
    )
    parser.add_argument(
        "--no-rolling",
        action="store_true",
        help="Exclude y_target rolling mean/std features.",
    )
    parser.add_argument(
        "--no-aggregates",
        action="store_true",
        help="Exclude y_target aggregate features (global mean, sub_category mean).",
    )
    parser.add_argument(
        "--target-transform",
        action="store_true",
        help="Enable target transformation (MinMax -> log1p -> MinMax). Default is no transform.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow experiment tracking.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (YAML or JSON). Default: config.yaml in project root.",
    )
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.yaml")
    cfg = load_config(config_path)
    il = cfg["input_lags"]

    score = main(
        use_input_lags=not args.no_input_lags,
        use_target_lags=not args.no_target_lags,
        use_rolling=not args.no_rolling,
        use_aggregates=not args.no_aggregates,
        use_target_transform=args.target_transform,
        use_mlflow=not args.no_mlflow,
        lags_max=il["lags_max"],
        top_k_per_feature=il["top_k_per_feature"],
        use_global_lags=il["use_global_lags"],
        global_lags=il["global_lags"],
    )
