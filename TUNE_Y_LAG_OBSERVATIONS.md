# Experiment Observations: hedge_fund_forecasting_tune_y_lag

Review of MLflow experiment `hedge_fund_forecasting_tune_y_lag` — target lag tuning for **xlag_gl1** config, with full LightGBM param grid per ylag configuration.

**Script:** `run_tune_target_lags.py`  
**Review:** `python review_mlflow_results.py --ylag-only` (in `forecast_fund` env) — configs ranked by **average of top 5 runs** (by val_skill) per config.  
**Base config:** xlag_gl1 (global_lags=[1]) + target lags, use_rolling=False, use_aggregates=False.

---

## Results Overview

### Configs Sorted by Avg of Top 5 Runs (val_skill)

From `review_mlflow_results.py --ylag-only`:

| Rank | Ylag Config     | n_runs | avg_val_skill | avg_val_rmse | avg_train_skill | avg_train_rmse | avg_num_boost_round |
|------|-----------------|--------|---------------|--------------|----------------|----------------|---------------------|
| 1    | **ylag_1_2_3_4_5** | 5   | **0.843563**  | 14.300719    | 0.856641       | 19.528644      | 191.4               |
| 2    | ylag_1_2_3_4    | 5      | 0.843304      | 14.337571    | 0.855276       | 19.881047      | 195.2               |
| 3    | ylag_1_2_3      | 5      | 0.843256      | 14.376349    | 0.855107       | 19.998239      | 180.4               |
| 4    | ylag_1_2        | 5      | 0.842930      | 14.392936    | 0.853869       | 19.950698      | 143.8               |
| 5    | ylag_1_2_3_4_5_10 | 5   | 0.842789      | 14.359660    | 0.855195       | 19.807066      | 163.4               |
| 6    | ylag_1          | 5      | 0.842777      | 14.535435    | 0.855305       | 20.082984      | 284.0               |

**Overall best (by avg of top 5):** `ylag_1_2_3_4_5` (avg_val_skill **0.843563**).

### Single-Best Run per Config (for reference)

| Ylag Config   | Best Run                              | val_skill  |
|---------------|----------------------------------------|------------|
| ylag_1        | ylag_1_nl70_md200_dunbounded           | 0.843935   |
| ylag_1_2      | ylag_1_2_nl70_md100_d7                 | 0.844630   |
| ylag_1_2_3    | ylag_1_2_3_nl50_md50_d7                | 0.844253   |
| ylag_1_2_3_4  | ylag_1_2_3_4_nl31_md200_dunbounded     | **0.844875** (highest single run) |
| ylag_1_2_3_4_5  | ylag_1_2_3_4_5_nl70_md200_d9         | 0.844111   |
| ylag_1_2_3_4_5_10 | ylag_1_2_3_4_5_10_nl70_md200_d9    | 0.843126   |

---

## Key Observations

### 1. Ranking by top-5 average vs single best

- **By average of top 5 runs:** **ylag_1_2_3_4_5** is best (0.843563), then ylag_1_2_3_4 (0.843304), ylag_1_2_3 (0.843256).
- **By single best run:** ylag_1_2_3_4 has the highest single val_skill (0.844875).
- **Recommendation:** Prefer **y_lags = [1, 2, 3, 4, 5]** for robustness (best avg of top 5); use **y_lags = [1, 2, 3, 4]** if optimizing for peak single-run performance.

### 2. Validation skill and RMSE

- Val skill is in the **~0.84–0.85** range; avg val_rmse **~14.3–14.5**.
- Best avg val_rmse among configs is ylag_1_2_3_4_5 (14.30); ylag_1 has the worst avg val_rmse (14.54).

### 3. Train vs validation

- avg_train_skill is ~0.85–0.86; avg_val_skill ~0.84, so some overfitting but stable.
- ylag_1 has the highest avg num_boost_round (284); other configs are in the ~144–195 range.

### 4. More lags: 5 vs 10

- **ylag_1_2_3_4_5** (avg 0.843563) beats **ylag_1_2_3_4_5_10** (0.842789). Adding the 10th lag hurts the top-5 average.

### 5. Reproducing results

- Run in `forecast_fund`: `conda run -n forecast_fund python review_mlflow_results.py --ylag-only`
- Full metrics and per-run details are printed; summary table is sorted by avg val_skill over top 5 runs per config.

---

## Experiment Setup (for reference)

- **Ylag configs:** [1], [1,2], [1,2,3], [1,2,3,4], [1,2,3,4,5], [1,2,3,4,5,10]
- **LGB param grid:** num_leaves ∈ {15, 31, 50, 70}, min_data_in_leaf ∈ {20, 50, 100, 200}, max_depth ∈ {5, 7, 9, -1}  
  (56 valid combos per ylag config)
- **MLflow experiment:** hedge_fund_forecasting_tune_y_lag
