# Experiment Observations: hedge_fund_forecasting_tune_xlag

Review of MLflow experiment `hedge_fund_forecasting_tune_xlag` — input lag tuning for the xlag-only (base + input lags) configuration.

## Results Overview

### Best Runs (by val_skill)

| Rank | Run Name | val_skill | val_rmse | train_skill | num_boost_round | Config |
|------|----------|-----------|----------|-------------|-----------------|--------|
| 1 | xlag_gl1 | **0.1279** | 27.31 | 0.278 | 24 | global_lags=[1] |
| 2 | xlag_gl1_2_3 | 0.1242 | 27.45 | 0.288 | 28 | global_lags=[1,2,3] |
| 3 | xlag_lm10_tk2 | 0.1227 | 27.36 | 0.308 | 36 | lags_max=10, top_k=2 |
| 4 | xlag_lm15_tk3 | 0.1218 | 27.36 | 0.280 | 25 | lags_max=15, top_k=3 |
| 5 | xlag_lm10_tk5 | 0.1193 | 27.39 | 0.275 | 24 | lags_max=10, top_k=5 |
| 6 | xlag_lm10_tk3 | 0.1192 | 27.39 | 0.292 | 29 | lags_max=10, top_k=3 |
| 7 | xlag_lm5_tk3 | 0.1191 | 27.37 | 0.282 | 27 | lags_max=5, top_k=3 |

### Experiment 1: Global Lags (xlag_gl*)

| Run Name | val_skill | global_lags | num_boost_round |
|----------|-----------|-------------|-----------------|
| xlag_gl1 | **0.1279** | [1] | 24 |
| xlag_gl1_2_3 | 0.1242 | [1,2,3] | 28 |
| xlag_gl1_2_3_5_10 | 0.1145 | [1,2,3,5,10] | 26 |
| xlag_gl1_2_3_5 | 0.1130 | [1,2,3,5] | 16 |
| xlag_gl1_2 | 0.1022 | [1,2] | 13 |
| xlag_gl1_2_3_5_10_15 | 0.1006 | [1,2,3,5,10,15] | 12 |

### Experiment 2: Correlation-Based (lags_max + top_k)

| Run Name | val_skill | lags_max | top_k | num_boost_round |
|----------|-----------|----------|-------|-----------------|
| xlag_lm10_tk2 | **0.1227** | 10 | 2 | 36 |
| xlag_lm15_tk3 | 0.1218 | 15 | 3 | 25 |
| xlag_lm10_tk5 | 0.1193 | 10 | 5 | 24 |
| xlag_lm10_tk3 | 0.1192 | 10 | 3 | 29 |
| xlag_lm5_tk3 | 0.1191 | 5 | 3 | 27 |
| xlag_lm5_tk2 | 0.1168 | 5 | 2 | 25 |
| ... | ... | ... | ... | ... |
| xlag_lm20_tk4 | 0.0952 | 20 | 4 | 11 |
| xlag_lm20_tk6 | NaN | 20 | 6 | NaN |

---

## Key Observations

### 1. Simpler lag configurations perform better

- **Best overall**: `xlag_gl1` (global_lags=[1]) with val_skill 0.1279
- Using a single lag outperforms using multiple lags in this setup

### 2. Experiment 1: Global lags — more lags reduce performance

There is a clear downward trend as more lags are added:

- [1] → 0.1279
- [1,2,3] → 0.1242
- [1,2,3,5,10] → 0.1145
- [1,2,3,5] → 0.1130
- [1,2] → 0.1022
- [1,2,3,5,10,15] → 0.1006

Longer horizons (more lags) generally worsen validation skill.

### 3. Experiment 2: Correlation-based selection

- **Best Exp2 run**: `xlag_lm10_tk2` (lags_max=10, top_k=2) with val_skill 0.1227
- lags_max=5, 10, 15 all give competitive results
- lags_max=20 performs worse, with val_skill around 0.095–0.107
- For a given lags_max, moderate top_k (2–5) tends to work best

### 4. Early stopping behavior

- num_boost_round_actual ranges from 11 to 36
- Runs with more lags (e.g. global [1,2,3,5,10,15], lags_max=20) often stop earlier
- Suggests that richer lag features can overfit sooner

### 5. Train vs validation gap

- train_skill is typically 0.23–0.31
- val_skill is 0.095–0.128
- Overfitting is present but moderate

### 6. Failed or incomplete run

- `xlag_lm20_tk6` has NaN for all metrics and num_boost_round
- Likely failed during training or evaluation
- Worth re-running or investigating logs

### 7. Validation RMSE

- Val RMSE clusters around 27.3–27.5 for most runs
- val_skill separates runs better than val_rmse

---

## Recommendations

1. **Preferred configuration**: Use **global_lags=[1]** or **lags_max=10, top_k=2** for production or further experiments
2. **Avoid long horizons**: Do not use lags beyond ~10 for this model and data
3. **Avoid large top_k with large lags_max**: (20, 6) led to a failed run; combinations like (20, 4–6) perform poorly
4. **Next steps**: Re-run or debug `xlag_lm20_tk6` if that configuration is important; otherwise deprioritize lags_max=20
