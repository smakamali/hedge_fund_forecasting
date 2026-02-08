"""
Preprocessing utilities for time series forecasting competition.
All functions maintain temporal discipline: no future data leakage.
Rolling/aggregate features use only data up to ts_index t-1 when predicting at t.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Any, Union, Optional, Tuple

# 86 anonymized feature columns (from basic-example.ipynb)
FEATURE_COLS = [
    'feature_a', 'feature_b', 'feature_c', 'feature_d', 'feature_e', 'feature_f', 'feature_g',
    'feature_h', 'feature_i', 'feature_j', 'feature_k', 'feature_l', 'feature_m', 'feature_n',
    'feature_o', 'feature_p', 'feature_q', 'feature_r', 'feature_s', 'feature_t', 'feature_u',
    'feature_v', 'feature_w', 'feature_x', 'feature_y', 'feature_z', 'feature_aa', 'feature_ab',
    'feature_ac', 'feature_ad', 'feature_ae', 'feature_af', 'feature_ag', 'feature_ah', 'feature_ai',
    'feature_aj', 'feature_ak', 'feature_al', 'feature_am', 'feature_an', 'feature_ao', 'feature_ap',
    'feature_aq', 'feature_ar', 'feature_as', 'feature_at', 'feature_au', 'feature_av', 'feature_aw',
    'feature_ax', 'feature_ay', 'feature_az', 'feature_ba', 'feature_bb', 'feature_bc', 'feature_bd',
    'feature_be', 'feature_bf', 'feature_bg', 'feature_bh', 'feature_bi', 'feature_bj', 'feature_bk',
    'feature_bl', 'feature_bm', 'feature_bn', 'feature_bo', 'feature_bp', 'feature_bq', 'feature_br',
    'feature_bs', 'feature_bt', 'feature_bu', 'feature_bv', 'feature_bw', 'feature_bx', 'feature_by',
    'feature_bz', 'feature_ca', 'feature_cb', 'feature_cc', 'feature_cd', 'feature_ce', 'feature_cf',
    'feature_cg', 'feature_ch'
]

# Type C: large-scale features for log transform (EDA_OBSERVATIONS.md 9.3)
TYPE_C_FEATURES = [
    'feature_at', 'feature_au', 'feature_av', 'feature_aw', 'feature_ax', 'feature_ay',
    'feature_ba', 'feature_bb', 'feature_bc', 'feature_bd', 'feature_be', 'feature_bf',
    'feature_bh', 'feature_bj', 'feature_bk'
]

# Zero-inflated features for binary flags (examples from EDA)
ZERO_INFLATED_FEATURES = [
    'feature_o', 'feature_ag', 'feature_by', 'feature_h', 'feature_i', 'feature_j',
    'feature_k', 'feature_p', 'feature_q', 'feature_aa', 'feature_ab', 'feature_ac', 'feature_ae'
]

# Entity columns used as categorical (label-encoded) or continuous (horizon) features
ENTITY_CATEGORICAL_COLS = ["code", "sub_code", "sub_category"]


def encode_entity_categoricals(
    df: pd.DataFrame,
    cat_cols: List[str],
    encodings: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Label-encode categorical entity columns for model input.
    Fit (encodings=None) or transform (encodings provided). Unseen categories -> -1 (LightGBM treats as missing).
    Returns (df with *_enc columns, encodings dict).
    """
    df_out = df.copy()
    if encodings is None:
        encodings = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        if col not in encodings:
            uniques = df_out[col].astype(str).dropna().unique()
            encodings[col] = {v: i for i, v in enumerate(sorted(uniques))}
        mapping = encodings[col]
        df_out[col + "_enc"] = df_out[col].astype(str).map(mapping).fillna(-1).astype(np.int32)
    return df_out, encodings


def fit_target_transform(y_train: np.ndarray) -> Dict[str, Any]:
    """
    Fit target transform: MinMax -> log1p -> MinMax (all fit on train only).
    Returns a dict with 'minmax1' and 'minmax2' scalers for use with transform_target / inverse_transform_target.
    """
    y = np.asarray(y_train, dtype=float).reshape(-1, 1)
    minmax1 = MinMaxScaler()
    y1 = minmax1.fit_transform(y).ravel()
    y2 = np.log1p(y1)
    minmax2 = MinMaxScaler()
    minmax2.fit(y2.reshape(-1, 1))
    return {"minmax1": minmax1, "minmax2": minmax2}


def transform_target(y: np.ndarray, target_transform: Dict[str, Any]) -> np.ndarray:
    """Apply transform: MinMax(1) -> log1p -> MinMax(2)."""
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    minmax1 = target_transform["minmax1"]
    minmax2 = target_transform["minmax2"]
    y1 = minmax1.transform(y).ravel()
    y2 = np.log1p(y1)
    y3 = minmax2.transform(y2.reshape(-1, 1)).ravel()
    return y3


def inverse_transform_target(y_transformed: np.ndarray, target_transform: Dict[str, Any]) -> np.ndarray:
    """Apply inverse: MinMax(2)^{-1} -> expm1 -> MinMax(1)^{-1}."""
    y3 = np.asarray(y_transformed, dtype=float).reshape(-1, 1)
    minmax1 = target_transform["minmax1"]
    minmax2 = target_transform["minmax2"]
    y2 = minmax2.inverse_transform(y3).ravel()
    y1 = np.expm1(y2)
    y = minmax1.inverse_transform(y1.reshape(-1, 1)).ravel()
    return y


def temporal_impute_missing(df: pd.DataFrame, feature_cols: List[str], method: str = 'median') -> tuple:
    """
    Impute missing values using statistics from df (train set).
    Returns (df_imputed, impute_values dict).
    """
    df_imputed = df.copy()
    impute_values = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        if df[col].isna().any():
            if method == 'median':
                impute_val = df[col].median()
            elif method == 'mean':
                impute_val = df[col].mean()
            else:
                impute_val = 0.0
            impute_values[col] = impute_val
            df_imputed[col] = df_imputed[col].fillna(impute_val)
    return df_imputed, impute_values


def apply_imputation(df: pd.DataFrame, impute_values: Dict[str, float]) -> pd.DataFrame:
    """Apply pre-computed imputation values to a dataframe."""
    df_out = df.copy()
    for col, val in impute_values.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(val)
    return df_out


def create_missing_indicators(df: pd.DataFrame, feature_cols: List[str], missing_threshold: float = 0.01) -> tuple:
    """Create binary is_missing columns for features with missing rate > threshold. Returns (df, indicator_cols)."""
    df_out = df.copy()
    indicator_cols = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        missing_rate = df[col].isna().mean()
        if missing_rate > missing_threshold:
            indicator_col = f"{col}_is_missing"
            df_out[indicator_col] = df[col].isna().astype(int)
            indicator_cols.append(indicator_col)
    return df_out, indicator_cols


def create_lag_features(
    df: pd.DataFrame,
    entity_cols: List[str],
    target_col: str,
    ts_col: str,
    lags: List[int] = [1, 2, 3, 5, 10, 20]
) -> tuple:
    """
    Create lag features per entity. Uses only past data (shift by lag).
    Returns (df_with_lags, lag_cols).
    """
    df_out = df.sort_values(entity_cols + [ts_col]).copy()
    lag_cols = []
    for lag in lags:
        lag_col = f"{target_col}_lag_{lag}"
        df_out[lag_col] = df_out.groupby(entity_cols)[target_col].shift(lag)
        lag_cols.append(lag_col)
    return df_out, lag_cols


def create_rolling_features(
    df: pd.DataFrame,
    entity_cols: List[str],
    target_col: str,
    ts_col: str,
    windows: List[int] = [5, 20]
) -> tuple:
    """
    Create rolling mean/std of target per entity using only data up to t-1.
    Shift by 1 first, then rolling, so at row t we use t-1, t-2, ... only.
    Returns (df_with_rolling, rolling_cols).
    """
    df_out = df.sort_values(entity_cols + [ts_col]).copy()
    rolling_cols = []
    for window in windows:
        mean_col = f"{target_col}_rolling_mean_{window}"
        std_col = f"{target_col}_rolling_std_{window}"
        df_out[mean_col] = df_out.groupby(entity_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df_out[std_col] = df_out.groupby(entity_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
        rolling_cols.extend([mean_col, std_col])
    return df_out, rolling_cols


def create_aggregate_features_t1(
    df: pd.DataFrame,
    target_col: str,
    ts_col: str,
    group_col: str = 'sub_category'
) -> tuple:
    """
    Global and per-group mean of y_target using only ts_index < t (t-1 discipline).
    For each ts_index t, we use mean of all rows with ts_index < t.
    """
    df_out = df.copy()
    agg_cols = []
    by_ts = df.groupby(ts_col)[target_col].agg(['sum', 'count']).sort_index()
    cumsum = by_ts['sum'].cumsum().shift(1)
    cumcount = by_ts['count'].cumsum().shift(1)
    global_mean_series = cumsum / cumcount
    global_mean_series = global_mean_series.fillna(df[target_col].mean())
    df_out[f"{target_col}_global_mean"] = df_out[ts_col].map(global_mean_series)
    agg_cols.append(f"{target_col}_global_mean")

    by_grp_ts = df.groupby([group_col, ts_col])[target_col].agg(['sum', 'count']).sort_index()
    grp_cumsum = by_grp_ts.groupby(level=0)['sum'].transform(lambda x: x.cumsum().shift(1))
    grp_cumcount = by_grp_ts.groupby(level=0)['count'].transform(lambda x: x.cumsum().shift(1))
    grp_mean_series = (grp_cumsum / grp_cumcount).fillna(df[target_col].mean())
    grp_df = grp_mean_series.reset_index()
    grp_df.columns = [group_col, ts_col, f"{target_col}_{group_col}_mean"]
    df_out = df_out.merge(grp_df, on=[group_col, ts_col], how='left')
    df_out[f"{target_col}_{group_col}_mean"] = df_out[f"{target_col}_{group_col}_mean"].fillna(df[target_col].mean())
    agg_cols.append(f"{target_col}_{group_col}_mean")
    return df_out, agg_cols


def winsorize_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    quantiles: tuple = (0.01, 0.99),
    fit_df: pd.DataFrame = None
) -> tuple:
    """
    Clip features at quantiles (e.g. Q01/Q99). Fit on fit_df (e.g. train), apply to df.
    Returns (df_winsorized, bounds_dict).
    """
    fit_df = fit_df if fit_df is not None else df
    df_out = df.copy()
    bounds = {}
    for col in feature_cols:
        if col not in fit_df.columns:
            continue
        q_low, q_high = fit_df[col].quantile([quantiles[0], quantiles[1]]).values
        bounds[col] = (float(q_low), float(q_high))
        if col in df_out.columns:
            df_out[col] = df_out[col].clip(lower=q_low, upper=q_high)
    return df_out, bounds


def apply_winsorize_bounds(df: pd.DataFrame, bounds: Dict[str, tuple]) -> pd.DataFrame:
    """Apply pre-computed winsorize bounds to df."""
    df_out = df.copy()
    for col, (q_low, q_high) in bounds.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].clip(lower=q_low, upper=q_high)
    return df_out


def log_transform_type_c(df: pd.DataFrame, type_c_cols: List[str]) -> pd.DataFrame:
    """Apply log(1+x) to Type C (large-scale) features."""
    df_out = df.copy()
    for col in type_c_cols:
        if col in df_out.columns:
            df_out[col] = np.log1p(df_out[col].clip(lower=0))
    return df_out


def create_zero_inflation_flags(df: pd.DataFrame, zero_inflated_cols: List[str]) -> tuple:
    """Add binary is_zero columns for given features. Returns (df, flag_cols)."""
    df_out = df.copy()
    flag_cols = []
    for col in zero_inflated_cols:
        if col not in df.columns:
            continue
        flag_col = f"{col}_is_zero"
        df_out[flag_col] = (df[col] == 0).astype(int)
        flag_cols.append(flag_col)
    return df_out, flag_cols


# Fixed lags 1-4 for "all features" time-delay embedding (used by run_validation_lagged_features)
ALL_FEATURE_LAGS = [1, 2, 3]


def create_all_feature_lag_features(
    df: pd.DataFrame,
    entity_cols: List[str],
    ts_col: str,
    feature_cols: List[str],
    lags: List[int] = None,
) -> tuple:
    """
    Create 1-, 2-, 3-, 4-, and 5-step lagged features for all given features, per entity.
    Uses only past data (shift by lag). Same as create_input_lag_features with lag_spec=[1,2,3,4,5].
    Returns (df_with_columns, lag_cols).
    """
    lags = lags if lags is not None else ALL_FEATURE_LAGS
    return create_input_lag_features(df, entity_cols, ts_col, feature_cols, lag_spec=lags)


def select_input_lags(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    entity_cols: List[str],
    ts_col: str,
    target_col: str,
    lags_max: int = 5,
    top_k_per_feature: int = 2,
    min_abs_corr: float = 0.0,
    use_global_lags: bool = False,
    global_lags: List[int] = None,
) -> Union[Dict[str, List[int]], List[int]]:
    """
    Correlation-based lag selection for input features (train-only, no leakage).
    For each feature and lag in 1..lags_max, compute correlation between feature(t-lag) and y_target(t);
    keep top_k lags per feature by absolute correlation, or use global_lags for all.
    Returns lag_spec: dict feature -> list of lags, or list of lags if use_global_lags.
    """
    if use_global_lags and global_lags is not None:
        return global_lags

    train_sorted = train_df.sort_values(entity_cols + [ts_col]).copy()
    lag_spec = {}

    for f in feature_cols:
        if f not in train_sorted.columns:
            continue
        grp = train_sorted.groupby(entity_cols)[f]
        corrs = []
        for lag in range(1, lags_max + 1):
            shifted = grp.shift(lag)
            valid = shifted.notna() & train_sorted[target_col].notna()
            if valid.sum() < 100:
                corrs.append((lag, 0.0))
                continue
            x = shifted[valid].values
            y = train_sorted.loc[valid, target_col].values
            c = np.corrcoef(x, y)[0, 1]
            if np.isnan(c):
                c = 0.0
            corrs.append((lag, abs(c)))
        corrs.sort(key=lambda x: -x[1])
        chosen = [lag for lag, _ in corrs[:top_k_per_feature] if _ >= min_abs_corr]
        if not chosen:
            chosen = [1]
        lag_spec[f] = chosen

    return lag_spec


def create_input_lag_features(
    df: pd.DataFrame,
    entity_cols: List[str],
    ts_col: str,
    feature_cols: List[str],
    lag_spec: Union[Dict[str, List[int]], List[int]],
    entity_chunk_size: Optional[int] = None,
    lag_dtype: type = np.float32,
    batch_size: int = 20,
) -> tuple:
    """
    Time-delay-embedding style: add lagged input features per entity (past only).
    lag_spec: list [1,2,3] = same lags for all features, or dict {feature: [1,2], ...}.
    Returns (df_with_columns, input_lag_cols).

    Memory optimizations:
    - entity_chunk_size: process entities in chunks (default: 500 when rows > 500k, else None).
    - lag_dtype: np.float32 to halve memory vs float64 (default: float32).
    - batch_size: add lag columns in batches to avoid peak memory from one large concat.
    """
    df_out = df.sort_values(entity_cols + [ts_col]).copy()
    n_rows = len(df_out)

    # Build list of (col_name, feature, lag) to compute
    to_compute: List[Tuple[str, str, int]] = []
    for f in feature_cols:
        if f not in df_out.columns:
            continue
        lags = lag_spec if isinstance(lag_spec, list) else lag_spec.get(f, [1])
        for lag in lags:
            to_compute.append((f"{f}_lag_{lag}", f, lag))
    input_lag_cols = [c for c, _, _ in to_compute]

    if not to_compute:
        return df_out, []

    # Use entity chunking for large datasets to reduce peak memory
    if entity_chunk_size is None and n_rows > 500_000:
        entity_chunk_size = 500
    if entity_chunk_size is not None and entity_chunk_size <= 0:
        entity_chunk_size = None

    _lag_dtype = lag_dtype

    if entity_chunk_size is not None:
        # Process by entity chunks: lower peak memory
        entity_id = df_out.groupby(entity_cols).ngroup()
        n_entities = int(entity_id.max()) + 1
        df_out["_row_order"] = np.arange(n_rows, dtype=np.int32)

        chunk_results = []
        for chunk_start in range(0, n_entities, entity_chunk_size):
            chunk_ids = set(range(chunk_start, min(chunk_start + entity_chunk_size, n_entities)))
            mask = entity_id.isin(chunk_ids)
            df_chunk = df_out.loc[mask].copy()
            df_chunk = df_chunk.sort_values(entity_cols + [ts_col])

            for batch_start in range(0, len(to_compute), batch_size):
                batch = to_compute[batch_start : batch_start + batch_size]
                series_list = []
                for col_name, feat, lag in batch:
                    ser = df_chunk.groupby(entity_cols)[feat].shift(lag)
                    ser = ser.astype(_lag_dtype)
                    ser.name = col_name
                    series_list.append(ser)
                batch_df = pd.concat(series_list, axis=1)
                df_chunk = pd.concat([df_chunk, batch_df], axis=1)

            chunk_results.append(df_chunk)
            del df_chunk

        df_out = pd.concat(chunk_results, axis=0, ignore_index=False)
        df_out = df_out.sort_values("_row_order").drop(columns=["_row_order"])
    else:
        # No entity chunking: add lag columns in batches
        for batch_start in range(0, len(to_compute), batch_size):
            batch = to_compute[batch_start : batch_start + batch_size]
            series_list = []
            for col_name, feat, lag in batch:
                ser = df_out.groupby(entity_cols)[feat].shift(lag)
                ser = ser.astype(_lag_dtype)
                ser.name = col_name
                series_list.append(ser)
            batch_df = pd.concat(series_list, axis=1)
            df_out = pd.concat([df_out, batch_df], axis=1)
            del series_list, batch_df

    return df_out, input_lag_cols


def create_entity_count(df: pd.DataFrame, entity_cols: List[str], ts_col: str) -> tuple:
    """
    Count of observations per entity up to current ts_index (coverage indicator).
    Uses expanding count per entity (no y_target, so no t-1 constraint).
    Returns (df_with_count, count_col_name).
    """
    df_out = df.sort_values(entity_cols + [ts_col]).copy()
    count_col = 'entity_obs_count'
    df_out[count_col] = df_out.groupby(entity_cols)[ts_col].cumcount() + 1
    return df_out, count_col
