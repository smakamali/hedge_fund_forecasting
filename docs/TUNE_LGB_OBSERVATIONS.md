# Experiment Observations: hedge_fund_forecasting_tune_lgb

Review of MLflow experiment `hedge_fund_forecasting_tune_lgb` — LightGBM parameter tuning (num_leaves, min_data_in_leaf, max_depth) for the top xlag configurations from `hedge_fund_forecasting_tune_xlag` (including xlag_gl1_2, added later).

**Experiment run:** Feb 2025 (post xlag tuning)

---

## Results Overview

### Best Runs per Config (by val_skill)

| Rank | Config | Best Run | val_skill | val_rmse | num_leaves | min_data_in_leaf | max_depth |
|------|--------|----------|-----------|----------|------------|------------------|-----------|
| 1 | xlag_gl1 | xlag_gl1_nl70_md20_d9 | **0.1386** | 27.23 | 70 | 20 | 9 |
| 2 | xlag_gl1_2 | xlag_gl1_2_nl50_md50_dunbounded | 0.1344 | 27.25 | 50 | 50 | unbounded |
| 3 | xlag_lm10_tk2 | xlag_lm10_tk2_nl70_md100_dunbounded | 0.1326 | 27.41 | 70 | 100 | unbounded |
| 4 | xlag_lm10_tk5 | xlag_lm10_tk5_nl70_md100_dunbounded | 0.1304 | 27.49 | 70 | 100 | unbounded |
| 5 | xlag_lm5_tk3 | xlag_lm5_tk3_nl50_md200_d9 | 0.1301 | 27.41 | 50 | 200 | 9 |
| 6 | xlag_lm10_tk3 | xlag_lm10_tk3_nl70_md100_dunbounded | 0.1290 | 27.29 | 70 | 100 | unbounded |
| 7 | xlag_gl1_2_3 | xlag_gl1_2_3_nl50_md50_dunbounded | 0.1286 | 27.35 | 50 | 50 | unbounded |
| 8 | xlag_lm15_tk3 | xlag_lm15_tk3_nl70_md100_d7 | 0.1275 | 27.39 | 70 | 100 | 7 |

### Full Metrics for Best Runs

| Config | val_skill | val_rmse | train_skill | train_rmse | num_boost_round_actual |
|--------|-----------|----------|-------------|------------|------------------------|
| xlag_gl1 | 0.1386 | 27.23 | 0.3155 | 32.58 | 20 |
| xlag_gl1_2 | 0.1344 | 27.25 | 0.2989 | 32.60 | 20 |
| xlag_lm10_tk2 | 0.1326 | 27.41 | 0.3198 | 32.71 | 20 |
| xlag_lm10_tk5 | 0.1304 | 27.49 | 0.3019 | 33.23 | 17 |
| xlag_lm5_tk3 | 0.1301 | 27.41 | 0.2885 | 33.28 | 20 |
| xlag_lm10_tk3 | 0.1290 | 27.29 | 0.2940 | 32.67 | 14 |
| xlag_gl1_2_3 | 0.1286 | 27.35 | 0.3000 | 32.72 | 20 |
| xlag_lm15_tk3 | 0.1275 | 27.39 | 0.3295 | 33.23 | 34 |

---

## Key Observations

### 1. Overall best configuration

- **xlag_gl1** remains the best-performing lag config after LGB tuning (val_skill **0.1386**).
- The same config ranked #1 in the xlag-only experiment; tuning LightGBM params boosted it further.

### 2. Impact of LGB tuning vs. default params (from tune_xlag)

| Config | Before (default LGB) | After (tuned LGB) | Gain |
|--------|----------------------|-------------------|------|
| xlag_gl1 | 0.1279 | 0.1386 | +0.0107 (+8.4%) |
| xlag_gl1_2 | 0.1022 | 0.1344 | +0.0322 (+31.5%) |
| xlag_lm10_tk2 | 0.1227 | 0.1326 | +0.0099 (+8.1%) |
| xlag_lm10_tk5 | 0.1193 | 0.1304 | +0.0111 (+9.3%) |
| xlag_lm10_tk3 | 0.1192 | 0.1290 | +0.0098 (+8.2%) |
| xlag_lm5_tk3 | 0.1191 | 0.1301 | +0.0110 (+9.2%) |
| xlag_gl1_2_3 | 0.1242 | 0.1286 | +0.0044 (+3.5%) |
| xlag_lm15_tk3 | 0.1218 | 0.1275 | +0.0057 (+4.7%) |

- All configs improve with tuned params; gains range from ~4% to ~31% relative.
- **xlag_gl1_2** shows the largest gain (+31.5%): it was weak with default LGB (0.1022) but jumps to #2 overall (0.1344) with tuned params (nl=50, md=50, unbounded depth).
- xlag_gl1_2_3 gains least (~3.5%), possibly close to optimum with default params.

### 3. Parameter patterns among best runs

- **num_leaves**: 50 or 70 in all best runs; 70 dominates (5/8 configs).
- **min_data_in_leaf**: varies—20 (xlag_gl1), 50 (xlag_gl1_2, xlag_gl1_2_3), 100 (four configs), 200 (xlag_lm5_tk3).
- **max_depth**: mix of unbounded (5 configs) and bounded 7 or 9 (3 configs). Best overall (xlag_gl1) uses max_depth=9.
- xlag_gl1_2 and xlag_gl1_2_3 share similar best params (nl=50, md=50, unbounded depth).
- No single parameter set is best for all configs; tuning per lag config is worthwhile.

### 4. Early stopping behavior

- **num_boost_round_actual** is low (14–34), so early stopping stops early.
- Typical best iteration: ~20. Models converge quickly with skill-based early stopping.
- xlag_lm15_tk3 reaches 34 rounds; xlag_lm10_tk3 stops at 14.

### 5. Train vs. validation gap

- Train skill (≈0.29–0.33) is ~2× validation skill (≈0.13), indicating overfitting.
- Despite that, tuned models generalize better than defaults; higher val_skill without markedly worse overfitting.

### 6. Recommended production config

- **Primary**: xlag_gl1 (global_lags=[1]) with num_leaves=70, min_data_in_leaf=20, max_depth=9 → val_skill ~0.1386, val_rmse ~27.23.
- **Strong alternative**: xlag_gl1_2 (global_lags=[1,2]) with num_leaves=50, min_data_in_leaf=50, max_depth unbounded → val_skill ~0.1344, val_rmse ~27.25. LGB tuning improved this config most (+31.5%); worth validating as a backup.

---

## Suggested Next Steps

1. Run a validation experiment with the recommended config to confirm stability.
2. Optionally try higher `num_boost_round` with longer patience to see if early stopping is too aggressive.
3. Explore `min_data_in_leaf` in [20, 50] for xlag_gl1 to check for further gains.
4. Re-run the pipeline with xlag_gl1 + best LGB params for final submission.
