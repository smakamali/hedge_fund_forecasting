"""
Evaluation utilities for time series forecasting competition.
Implements the competition metric and validation helpers.
"""

import numpy as np

from preprocessing import inverse_transform_target


def _clip01(x: float) -> float:
    """Clip value to [0, 1] range."""
    return float(np.minimum(np.maximum(x, 0.0), 1.0))


def weighted_rmse_score(y_target, y_pred, w) -> float:
    """
    Competition metric: skill score based on weighted RMSE.

    Formula: sqrt(1 - clip((weighted_MSE / weighted_var_y), [0, 1]))
    Higher is better. Perfect prediction = 1.0.
    """
    denom = np.sum(w * y_target ** 2)
    if denom == 0:
        return 0.0
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = _clip01(ratio)
    val = 1.0 - clipped
    return float(np.sqrt(val))


def temporal_train_test_split(df, ts_col='ts_index', test_size=0.2):
    """
    Split data temporally: train on earlier timestamps, validate on later.
    Returns train_df, val_df, cutoff.
    """
    unique_ts = sorted(df[ts_col].unique())
    n_ts = len(unique_ts)
    split_idx = int(n_ts * (1 - test_size))
    cutoff = unique_ts[split_idx]

    train_df = df[df[ts_col] <= cutoff].copy()
    val_df = df[df[ts_col] > cutoff].copy()

    print(f"Temporal split at {ts_col} = {cutoff}")
    print(f"Train: {len(train_df):,} rows, {ts_col} in [{train_df[ts_col].min()}, {train_df[ts_col].max()}]")
    print(f"Val:   {len(val_df):,} rows, {ts_col} in [{val_df[ts_col].min()}, {val_df[ts_col].max()}]")

    return train_df, val_df, cutoff


def make_skill_feval(y_orig, w, target_transform):
    """
    Build a LightGBM feval that computes val_skill (weighted_rmse_score) in original space.
    Returns (eval_name, eval_result, is_higher_better) for early stopping to maximize val_skill.
    """
    def feval(preds, train_data):
        if target_transform is not None:
            preds_orig = inverse_transform_target(preds, target_transform)
        else:
            preds_orig = preds
        score = weighted_rmse_score(y_orig, preds_orig, w)
        return ("val_skill", score, True)  # True = higher is better
    return feval


def evaluate_predictions(y_true, y_pred, weights, dataset_name='Dataset'):
    """Evaluate predictions and print metrics. Returns dict of metrics."""
    skill_score = weighted_rmse_score(y_true, y_pred, weights)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    weighted_mae = np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

    metrics = {'skill_score': skill_score, 'mae': mae, 'rmse': rmse, 'weighted_mae': weighted_mae}

    print(f"\n{dataset_name} Metrics:")
    print(f"  Skill Score (competition metric): {skill_score:.6f}")
    print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, Weighted MAE: {weighted_mae:.4f}")

    return metrics
