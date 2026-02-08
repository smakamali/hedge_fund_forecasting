# Validation Results: Fallback vs Sequential Inference

## Setup

- **Data**: `train.parquet` with a temporal 80/20 split.
- **Train part**: `ts_index <= 2881` (4,123,556 rows).
- **Validation part**: `ts_index > 2881` (1,213,858 rows).
- **Model**: LightGBM for experiments 2–4; LinearRegression for Phase 1 baseline (see below).
- **Metric**: Competition skill score = `sqrt(1 - clip(weighted_MSE / weighted_var_y))`; higher is better.

## Experiments

1. **Phase 1 baseline (linear)**  
   From `modeling_v1_baseline.ipynb`: temporal 80/20 split, **LinearRegression** with `sample_weight=weight`, 86 features + median imputation + missing indicators (no lags, rolling, or aggregates). Same split logic as above.

2. **Fallback (one-shot)**  
   Validation features use **zeros** for lag/rolling and **train-part means** for global/sub_category aggregates and entity counts (no sequential use of past predictions). Same logic as `run_phase6_submit.py` for test.

3. **Sequential inference**  
   For each validation `ts_index` T, temporal features (lags, rolling, aggregates, entity_obs_count) are built from train-part `y_target` plus **predictions already made** for validation timestamps < T. Same logic as `run_phase6_sequential_submit.py` for test.

4. **No y_target-derived features**  
   Train and validate with only the 86 anonymized features + missing indicators + zero-inflation flags + winsorize + log(Type C) + `horizon_numeric`. No lags, rolling, global/sub_category mean, or `horizon_x_subcat`. Validation only (no test submission).

## Validation Results (single run)

| Experiment                          | Validation skill score  |
|-------------------------------------|-------------------------|
| Phase 1 baseline (linear)           | 0.0334                  |
| target lags - fallback              | **0.0361**              |
| target lags - fallback + input lags | 0.01984                 |
| target lags - Sequential            | 0.0000                  |
| No y_target-derived features        | 0.0000                  |

- **Phase 1 baseline (linear)**: From `modeling_v1_baseline.ipynb`; validation skill score 0.0334. Linear model on 86 features + imputation + missing indicators only.
- **Fallback**: Modest but positive skill on this temporal holdout.
- **Sequential**: Score 0.0000 on this run (metric clips when ratio ≥ 1). Suggests either a bug in the validation sequential path or a large distribution shift when feeding model-predicted lags/aggregates instead of zeros/constants; worth checking before relying on sequential for submission.
- **No y_target-derived features**: Training and validation use only the 86 features + indicators + zero flags + winsorize + log(Type C) + `horizon_numeric` (no lags, rolling, global/sub_category mean, or `horizon_x_subcat`). Score 0.0000; the fallback’s skill comes from the y_target-derived temporal features.

- **Fallback + input lags**: As of the lagged-input-features implementation, `build_train_features` adds correlation-selected lagged input features (e.g. `feature_a_lag_1`, `feature_a_lag_2`) via `select_input_lags` and `create_input_lag_features`.

## How to reproduce

```bash
# Fallback vs sequential
conda run -n forecast_fund python run_validation_compare.py

# No y_target-derived features (validation only)
conda run -n forecast_fund python run_validation_no_ytarget_features.py
```

## Full-data submission runs

- **Fallback submission**: `conda run -n forecast_fund python run_phase6_submit.py` → `final_submission.csv`
- **Sequential submission**: `conda run -n forecast_fund python run_phase6_sequential_submit.py` → `final_submission_sequential.csv`

Test set has no labels, so these cannot be scored locally; only the validation comparison above is available for comparing the two approaches.


INFO: Training model (base + target_lags + rolling + aggregates + input_lags + no target transform)
INFO: Train:      RMSE=19.128919  skill=0.852587
INFO: Validation: RMSE=27.451037  skill=0.000000

INFO: Training model (base + target_lags + rolling + aggregates + input_lags + target transform)
INFO: Train:      RMSE=20.928353  skill=0.850135
INFO: Validation: RMSE=27.520798  skill=0.000000

