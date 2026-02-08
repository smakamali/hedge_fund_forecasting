"""
Phase 6: Train on full data, build test features (no y_target), predict and save final_submission.csv.
Run: conda run -n forecast_fund python run_phase6_submit.py
Optional: --target-transform to enable target transformation (default is no transform).
"""
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from evaluation import evaluate_predictions
from preprocessing import (
    FEATURE_COLS, TYPE_C_FEATURES, ZERO_INFLATED_FEATURES,
    ENTITY_CATEGORICAL_COLS, encode_entity_categoricals,
    fit_target_transform, transform_target, inverse_transform_target,
    temporal_impute_missing, create_missing_indicators, apply_imputation,
    create_lag_features, create_rolling_features, create_aggregate_features_t1, create_entity_count,
    winsorize_features, apply_winsorize_bounds, log_transform_type_c, create_zero_inflation_flags,
    select_input_lags, create_input_lag_features,
)

ENTITY_COLS = ['code', 'sub_code', 'sub_category', 'horizon']
ENTITY_CAT_FEATURE_NAMES = [c + '_enc' for c in ENTITY_CATEGORICAL_COLS]
TS_COL = 'ts_index'
TARGET_COL = 'y_target'
LAGS = [1, 2, 3, 5, 10, 20]
WINDOWS = [5, 20]


def build_train_features(train_df, use_target_transform=False):
    """Full feature pipeline on train (has y_target). Returns (train_fe, all_feature_cols, artifacts)."""
    train_imputed, impute_values = temporal_impute_missing(train_df, FEATURE_COLS, method='median')
    train_imputed, indicator_cols = create_missing_indicators(train_imputed, FEATURE_COLS, missing_threshold=0.01)

    train_imputed = train_imputed.sort_values(ENTITY_COLS + [TS_COL])

    # Optional target transform (MinMax -> log1p -> MinMax) fit on train, applied before target-derived features
    if use_target_transform:
        target_transform = fit_target_transform(train_df[TARGET_COL].to_numpy())
        train_imputed[TARGET_COL] = transform_target(train_imputed[TARGET_COL].to_numpy(), target_transform)
    else:
        target_transform = None

    train_imputed, rolling_cols = create_rolling_features(train_imputed, ENTITY_COLS, TARGET_COL, TS_COL, windows=WINDOWS)
    train_imputed, agg_cols = create_aggregate_features_t1(train_imputed, TARGET_COL, TS_COL, group_col='sub_category')
    train_imputed, entity_count_col = create_entity_count(train_imputed, ENTITY_COLS, TS_COL)

    numeric_to_clip = [c for c in FEATURE_COLS if c in train_imputed.columns]
    train_imputed, winsor_bounds = winsorize_features(train_imputed, numeric_to_clip, quantiles=(0.01, 0.99), fit_df=train_imputed)
    type_c_present = [c for c in TYPE_C_FEATURES if c in train_imputed.columns]
    train_imputed = log_transform_type_c(train_imputed, type_c_present)
    train_imputed, zero_flag_cols = create_zero_inflation_flags(train_imputed, ZERO_INFLATED_FEATURES)
    train_imputed, entity_encodings = encode_entity_categoricals(
        train_imputed, ENTITY_CATEGORICAL_COLS, encodings=None
    )

    lag_spec = select_input_lags(
        train_imputed, FEATURE_COLS, ENTITY_COLS, TS_COL, TARGET_COL,
        lags_max=5, top_k_per_feature=2,
    )
    train_imputed, input_lag_cols = create_input_lag_features(
        train_imputed, ENTITY_COLS, TS_COL, FEATURE_COLS, lag_spec,
    )
    extra_feature_cols = lag_cols + rolling_cols + agg_cols + [entity_count_col] + input_lag_cols

    train_imputed['horizon_numeric'] = train_imputed['horizon'].astype(float)
    train_imputed['horizon_x_subcat'] = train_imputed['horizon_numeric'] * train_imputed['y_target_sub_category_mean']
    interaction_cols = ['horizon_numeric', 'horizon_x_subcat']
    entity_feature_cols = [c + '_enc' for c in ENTITY_CATEGORICAL_COLS]
    base_feature_cols = (
        entity_feature_cols
        + [c for c in FEATURE_COLS if c in train_imputed.columns]
        + indicator_cols
        + zero_flag_cols
    )
    all_feature_cols = base_feature_cols + interaction_cols + extra_feature_cols
    all_feature_cols = [c for c in all_feature_cols if c in train_imputed.columns]

    # Entity/global stats in transformed target space (for fallbacks)
    global_mean = float(train_imputed[TARGET_COL].mean())
    global_std = float(train_imputed[TARGET_COL].std())
    if pd.isna(global_std) or global_std == 0:
        global_std = 1.0
    entity_mean_from_train = (
        train_imputed.groupby(ENTITY_COLS)[TARGET_COL].mean().reset_index(name='entity_mean')
    )
    entity_std_from_train = (
        train_imputed.groupby(ENTITY_COLS)[TARGET_COL].std().reset_index(name='entity_std')
    )
    entity_std_from_train['entity_std'] = entity_std_from_train['entity_std'].fillna(global_std)
    entity_std_from_train.loc[entity_std_from_train['entity_std'] == 0, 'entity_std'] = global_std

    artifacts = {
        'impute_values': impute_values,
        'indicator_cols': indicator_cols,
        'winsor_bounds': winsor_bounds,
        'lag_cols': lag_cols,
        'rolling_cols': rolling_cols,
        'agg_cols': agg_cols,
        'entity_count_col': entity_count_col,
        'input_lag_cols': input_lag_cols,
        'lag_spec': lag_spec,
        'global_mean': global_mean,
        'global_std': global_std,
        'entity_mean_from_train': entity_mean_from_train,
        'entity_std_from_train': entity_std_from_train,
        'subcat_mean': train_imputed.groupby('sub_category')[TARGET_COL].mean().to_dict(),
        'entity_count_from_train': train_df.groupby(ENTITY_COLS).size().reset_index(name='entity_obs_count'),
        'entity_encodings': entity_encodings,
        'target_transform': target_transform,
    }
    return train_imputed, all_feature_cols, artifacts


def build_test_features(test_df, artifacts):
    """Build test feature matrix (no y_target): use per-entity train mean/std fallbacks."""
    impute_values = artifacts['impute_values']
    indicator_cols = artifacts['indicator_cols']
    winsor_bounds = artifacts['winsor_bounds']
    lag_cols = artifacts['lag_cols']
    rolling_cols = artifacts['rolling_cols']
    agg_cols = artifacts['agg_cols']
    entity_count_col = artifacts['entity_count_col']
    input_lag_cols = artifacts.get('input_lag_cols', [])
    global_mean = artifacts['global_mean']
    global_std = artifacts['global_std']
    entity_mean_from_train = artifacts['entity_mean_from_train']
    entity_std_from_train = artifacts['entity_std_from_train']
    entity_count_from_train = artifacts['entity_count_from_train']
    entity_encodings = artifacts.get('entity_encodings', {})

    test_imputed = apply_imputation(test_df, impute_values)
    test_imputed, _ = create_missing_indicators(test_imputed, FEATURE_COLS, missing_threshold=0.01)
    test_imputed = apply_winsorize_bounds(test_imputed, winsor_bounds)
    type_c_present = [c for c in TYPE_C_FEATURES if c in test_imputed.columns]
    test_imputed = log_transform_type_c(test_imputed, type_c_present)
    test_imputed, zero_flag_cols = create_zero_inflation_flags(test_imputed, ZERO_INFLATED_FEATURES)
    test_imputed, _ = encode_entity_categoricals(
        test_imputed, ENTITY_CATEGORICAL_COLS, encodings=entity_encodings
    )

    test_imputed = test_imputed.merge(entity_mean_from_train, on=ENTITY_COLS, how='left')
    test_imputed['entity_mean'] = test_imputed['entity_mean'].fillna(global_mean)
    test_imputed = test_imputed.merge(entity_std_from_train, on=ENTITY_COLS, how='left')
    test_imputed['entity_std'] = test_imputed['entity_std'].fillna(global_std)

    for c in lag_cols:
        test_imputed[c] = test_imputed['entity_mean'].values
    for c in rolling_cols:
        if 'rolling_std' in c:
            test_imputed[c] = 0.0
        else:
            test_imputed[c] = test_imputed['entity_mean'].values
    for c in input_lag_cols:
        test_imputed[c] = 0.0
    test_imputed['y_target_global_mean'] = test_imputed['entity_mean'].values
    test_imputed['y_target_sub_category_mean'] = test_imputed['entity_mean'].values
    test_imputed = test_imputed.merge(
        entity_count_from_train, on=ENTITY_COLS, how='left'
    )
    test_imputed['entity_obs_count'] = test_imputed['entity_obs_count'].fillna(0)

    test_imputed['horizon_numeric'] = test_imputed['horizon'].astype(float)
    test_imputed['horizon_x_subcat'] = test_imputed['horizon_numeric'] * test_imputed['entity_mean'].values

    return test_imputed


def main():
    parser = argparse.ArgumentParser(description="Phase 6: train on full data, predict test, save submission.")
    parser.add_argument(
        "--target-transform",
        action="store_true",
        help="Enable target transformation (MinMax -> log1p -> MinMax). Default is no transform.",
    )
    args = parser.parse_args()
    use_target_transform = args.target_transform

    print("Loading train...")
    train_df = pd.read_parquet('train.parquet')
    print("Building train features (target_transform=%s)..." % use_target_transform)
    train_fe, all_feature_cols, artifacts = build_train_features(train_df, use_target_transform=use_target_transform)

    X_train = train_fe[all_feature_cols].fillna(0).to_numpy()
    y_train = train_fe['y_target'].to_numpy()
    w_train = train_fe['weight'].to_numpy()

    params = {
        'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 31,
        'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'verbose': -1, 'min_data_in_leaf': 100, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'seed': 42,
    }
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        weight=w_train,
        feature_name=all_feature_cols,
        categorical_feature=[c for c in ENTITY_CAT_FEATURE_NAMES if c in all_feature_cols],
    )
    print("Training final model on full train...")
    model = lgb.train(params, train_data, num_boost_round=500, callbacks=[lgb.log_evaluation(100)])

    print("Loading test...")
    test_df = pd.read_parquet('test.parquet')
    print("Building test features...")
    test_fe = build_test_features(test_df, artifacts)

    for c in all_feature_cols:
        if c not in test_fe.columns:
            test_fe[c] = 0
    X_test = test_fe[all_feature_cols].fillna(0).to_numpy()

    pred_t = model.predict(X_test)
    target_transform = artifacts.get('target_transform')
    if target_transform is not None:
        pred = inverse_transform_target(pred_t, target_transform)
    else:
        pred = pred_t
    out = pd.DataFrame({'id': test_df['id'].values, 'prediction': pred})
    out.to_csv('final_submission.csv', index=False)
    print("Saved final_submission.csv with", len(out), "rows.")
    assert out['prediction'].notna().all(), "NaN in predictions"
    assert out['id'].notna().all(), "NaN in id"


if __name__ == '__main__':
    main()
