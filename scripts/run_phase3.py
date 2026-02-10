"""Run Phase 3 LightGBM pipeline. Run from project root: python scripts/run_phase3.py"""
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
from src.evaluation import temporal_train_test_split, evaluate_predictions
from src.preprocessing import (
    FEATURE_COLS, TYPE_C_FEATURES, ZERO_INFLATED_FEATURES,
    temporal_impute_missing, create_missing_indicators, apply_imputation,
    create_lag_features, create_rolling_features, create_aggregate_features_t1, create_entity_count,
    winsorize_features, apply_winsorize_bounds, log_transform_type_c, create_zero_inflation_flags,
)

ENTITY_COLS = ['code', 'sub_code', 'sub_category', 'horizon']
TS_COL = 'ts_index'
TARGET_COL = 'y_target'

_data_dir = os.environ.get("DATA_DIR", "data")
train_path = os.path.join(_project_root, _data_dir, "train.parquet")
df = pd.read_parquet(train_path)
train_df, val_df, cutoff = temporal_train_test_split(df, ts_col=TS_COL, test_size=0.2)
train_imputed, impute_values = temporal_impute_missing(train_df, FEATURE_COLS, method='median')
val_imputed = apply_imputation(val_df, impute_values)
train_imputed, indicator_cols = create_missing_indicators(train_imputed, FEATURE_COLS, missing_threshold=0.01)
val_imputed, _ = create_missing_indicators(val_imputed, FEATURE_COLS, missing_threshold=0.01)

combined = pd.concat([train_imputed, val_imputed], ignore_index=True)
combined = combined.sort_values(ENTITY_COLS + [TS_COL])
combined, lag_cols = create_lag_features(combined, ENTITY_COLS, TARGET_COL, TS_COL, lags=[1, 2, 3, 5, 10, 20])
combined, rolling_cols = create_rolling_features(combined, ENTITY_COLS, TARGET_COL, TS_COL, windows=[5, 20])
combined, agg_cols = create_aggregate_features_t1(combined, TARGET_COL, TS_COL, group_col='sub_category')
combined, entity_count_col = create_entity_count(combined, ENTITY_COLS, TS_COL)
extra_feature_cols = lag_cols + rolling_cols + agg_cols + [entity_count_col]

train_fe = combined[combined[TS_COL] <= cutoff].copy()
val_fe = combined[combined[TS_COL] > cutoff].copy()

numeric_to_clip = [c for c in FEATURE_COLS if c in train_fe.columns]
train_fe, winsor_bounds = winsorize_features(train_fe, numeric_to_clip, quantiles=(0.01, 0.99), fit_df=train_fe)
val_fe = apply_winsorize_bounds(val_fe, winsor_bounds)
type_c_present = [c for c in TYPE_C_FEATURES if c in train_fe.columns]
train_fe = log_transform_type_c(train_fe, type_c_present)
val_fe = log_transform_type_c(val_fe, type_c_present)
train_fe, zero_flag_cols = create_zero_inflation_flags(train_fe, ZERO_INFLATED_FEATURES)
val_fe, _ = create_zero_inflation_flags(val_fe, ZERO_INFLATED_FEATURES)

train_fe['horizon_numeric'] = train_fe['horizon'].astype(float)
val_fe['horizon_numeric'] = val_fe['horizon'].astype(float)
train_fe['horizon_x_subcat'] = train_fe['horizon_numeric'] * train_fe['y_target_sub_category_mean']
val_fe['horizon_x_subcat'] = val_fe['horizon_numeric'] * val_fe['y_target_sub_category_mean']
interaction_cols = ['horizon_numeric', 'horizon_x_subcat']

base_feature_cols = [c for c in FEATURE_COLS if c in train_fe.columns] + indicator_cols + zero_flag_cols
all_feature_cols = base_feature_cols + interaction_cols + extra_feature_cols
all_feature_cols = [c for c in all_feature_cols if c in train_fe.columns and c in val_fe.columns]

X_train = train_fe[all_feature_cols].fillna(0).to_numpy()
y_train = train_fe['y_target'].to_numpy()
w_train = train_fe['weight'].to_numpy()
X_val = val_fe[all_feature_cols].fillna(0).to_numpy()
y_val = val_fe['y_target'].to_numpy()
w_val = val_fe['weight'].to_numpy()

params = {
    'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 31,
    'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'verbose': -1, 'min_data_in_leaf': 100, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'seed': 42,
}
train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
val_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data)
model_lgb = lgb.train(
    params, train_data, num_boost_round=300, valid_sets=[val_data], valid_names=['val'],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
)
y_pred_val = model_lgb.predict(X_val)
metrics_val = evaluate_predictions(y_val, y_pred_val, w_val, 'Validation')
print('Phase 3 Validation Skill Score:', metrics_val['skill_score'])
