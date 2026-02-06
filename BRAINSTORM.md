# Brainstorm: Preprocessing & Modeling for Time Series Forecasting

## 1. Problem Statement Recap

| Aspect | Description |
|--------|-------------|
| **Objective** | Predict a **continuous numerical value** (`y_target`) for each (code, sub_code, sub_category, horizon) at a given `ts_index`. |
| **Constraint** | At prediction time `ts_index = t`, use **only** data with `ts_index ≤ t`. No look-ahead. |
| **Metric** | Skill-score style: `sqrt(1 - clip(weighted_MSE / weighted_var_y))`. Higher is better. Weights `w` are row-level; **do not use `weight` as a feature.** |
| **Evaluation** | Public LB ≈ 25% of test; private LB = remaining 75%. Avoid overfitting to public. |

---

## 2. Dataset Definitions

### Columns

- **Identifiers:** `id` (code__sub_code__sub_category__horizon__ts_index), `code`, `sub_code`, `sub_category`
- **Temporal:** `ts_index` (integer “time”)
- **Task:** `horizon` (1 = short, 3 = medium, 10 = long, 25 = extra long — categorical, not a ts_index delta)
- **Target:** `y_target` (train only)
- **Metric:** `weight` (train only; use only in evaluation, **not as a feature**)
- **Features:** 86 anonymized columns `feature_a` … `feature_ch`

### Data shape (from notebook)

- **Train:** ~5.3M rows × 94 columns (id, code, sub_code, sub_category, horizon, ts_index, 86 features, y_target, weight)
- **Test:** same schema but **no** `y_target` or `weight`; need to output `id` and `prediction`

### Known data quirks

- **Missing values:** Some features have fewer non-null counts (e.g. `feature_h`, `feature_cd`); imputation must be **temporal** (e.g. rolling mean up to current `ts_index`).
- **Cardinality:** 23 codes, ~180 sub_codes, 5 sub_categories; horizons 1, 3, 10, 25.
- **Scale:** Features and `y_target` vary widely (e.g. y_target std ~32, weight up to 1e13); scaling/robust loss may help.

---

## 3. Preprocessing Ideas

### 3.1 Missing values

- **Rule:** Any imputation must use only past data (ts_index ≤ t) to avoid leakage.
- **Options:**
  - **Forward-fill / expanding mean** per (code, sub_code, sub_category) or globally, computed up to current `ts_index`.
  - **Simple global median/mean** computed on train data with `ts_index ≤ t` in a temporal split or expanding window.
  - **Explicit missing indicator** binary feature + imputed value so the model can downweight imputed cells.

### 3.2 Feature scaling

- **Recommendation:** Scale features (e.g. StandardScaler, RobustScaler) to improve gradient-based and distance-based models.
- **Leakage-safe:** Fit scaler only on data with `ts_index ≤ t` when predicting at `t`, or use a temporal validation split and fit on “past” fold only.

### 3.3 Categoricals: code, sub_code, sub_category, horizon

- **Encode for tree models:** Label encoding or target encoding (with strict temporal discipline: encode using only past data).
- **Encode for linear/GBDT:** One-hot (if cardinality manageable) or target/impact encoding; again, compute statistics only from past.
- **Hierarchy:** Competition tips say “similarities within sub_category, differences for same sub_code across codes.” Consider:
  - Group-level aggregates (e.g. mean `y_target` by sub_category up to `t`) as features.
  - Separate models or weights per (sub_category, horizon) and blend.

### 3.4 Temporal features

- **ts_index:** Use as a feature (possibly scaled or binned) so the model can adapt to trend/non-stationarity.
- **Recency:** e.g. “time since last observation” per entity, or weighting recent points more in loss (see Tips: “weighting the most recent periods”).

### 3.5 Feature engineering (all leakage-safe)

- **Rolling / expanding stats** per entity (code, sub_code, sub_category) over `ts_index`: mean, std, min, max of `y_target` or key features, using only past data.
- **Lag features:** For each entity, lags of `y_target` (and maybe key features) at previous `ts_index`; requires alignment of history.
- **Cross-section:** Within same `ts_index`, aggregates across codes/sub_codes (e.g. global or category mean) as extra features.
- **Horizon-specific:** Different lags or windows per `horizon` (short vs long horizon may use different lookbacks).

### 3.6 Target transformation

- **Metric:** The official metric is based on weighted MSE and weighted variance of `y`. A model that minimizes weighted MSE is consistent.
- **Optional:** Log or Box–Cox on `y_target` if distribution is skewed; predict in transformed space and invert for submission. Can help stability and heteroscedasticity.

---

## 4. Modeling Approaches

### 4.1 Baseline (current example)

- **Linear regression** on raw (or minimally processed) features; one global model. Good baseline to beat.

### 4.2 Regression models that respect the metric

- **Weighted regression:** Use sample_weight = `weight` when fitting so the training objective aligns with weighted MSE.
- **Linear / Ridge / Lasso:** With scaling and optional categorical encoding; easy to keep temporal discipline.
- **Gradient Boosting (e.g. LightGBM, XGBoost, CatBoost):** Often best for tabular + mixed types; support `weight`; handle missing values natively with temporal imputation or indicators.

### 4.3 Horizon-specific models

- **Separate model per horizon** (1, 3, 10, 25): Different dynamics for short vs long horizon; tips suggest horizon is important.
- **Single model with horizon as feature:** Simpler; may need strong interactions (e.g. horizon × features) to capture differences.

### 4.4 Hierarchy-aware modeling

- **Per (sub_category, horizon) or (code, horizon) models:** More parameters, better fit to local structure; risk of overfitting where data is sparse.
- **Mixed / hierarchical:** e.g. global model + residual model per group, or blend of global and group-level predictions.
- **Target encoding:** Encode code/sub_code/sub_category by past mean `y_target` (and optionally variance) so the model gets group-level signal.

### 4.5 Time-aware training

- **Temporal train/validation split:** e.g. train on `ts_index ≤ T1`, validate on `ts_index in (T1, T2]` to mimic test (future only).
- **Expanding window:** Retrain or refit using data up to increasing `ts_index`; final model(s) trained on full train set for submission.
- **Sample weighting:** Give higher weight to recent `ts_index` in the loss (as suggested in tips) to emphasize recent dynamics.

### 4.6 Regularization and robustness

- **Low signal-to-noise:** Prefer regularized models (Ridge, Lasso, strong regularization in GBDT); avoid overfitting.
- **Non-stationarity:** Emphasize recent data (weights or recent-only validation); consider simple trend/season features from `ts_index`.

### 4.7 Ensembles

- **Blend** different model types (e.g. linear + GBDT) or different horizons/group models.
- **Stacking:** Meta-model on out-of-fold predictions; base models and meta-model must be trained with strict temporal splits to avoid leakage.

---

## 5. Validation & Avoiding Leakage

- **Strict temporal split:** Validation set must have `ts_index` strictly after training set. No shuffling across time.
- **Metrics:** Use the **same** weighted skill score as the competition (using `weight`) on the validation set.
- **Preprocessing:** All scaling, imputation, target encoding, and rolling stats must be computed using only data with `ts_index ≤` current prediction time (or ≤ train end for a single validation split).
- **Cross-validation:** Time-based folds (e.g. 3–5 folds by `ts_index`); each fold: train on past, validate on next period.

---

## 6. Suggested Pipeline (Ordered)

1. **Load train/test;** ensure test has no `y_target`/`weight` and that `id` is preserved for submission.
2. **Temporal split:** e.g. last 20% of `ts_index` in train as validation; report weighted skill score.
3. **Missing values:** Implement expanding/windowed imputation (e.g. median or mean up to `t`); add missing indicator if useful.
4. **Scaling:** Fit on train (or past part of train) and transform train/validation/test.
5. **Categoricals:** Target encode or label encode (with temporal discipline); optionally add group aggregates.
6. **Features:** Add `ts_index`, optional lags/rolling stats, horizon and group aggregates; keep all operations leakage-safe.
7. **Train:** Start with **weighted** Ridge or LightGBM (sample_weight = `weight`); try horizon-specific or single model.
8. **Validate:** Evaluate with `weighted_rmse_score` on validation set.
9. **Retrain:** Fit final model(s) on full training data (with same preprocessing logic) and predict on test.
10. **Submit:** CSV with `id` and `prediction`; ensure row order matches test.

---

## 7. Quick Wins to Try First

- Use **sample_weight = weight** in all sklearn/GBDT fits.
- **Temporal train/val split** and report the official metric.
- **RobustScaler** or **StandardScaler** on features (fit on past only).
- **LightGBM** or **XGBoost** with weight, default handling of missing values, and optional categorical columns.
- **One model per horizon** and compare to a single model with horizon as feature.
- **Target encoding** for code/sub_code/sub_category (with temporal discipline) and/or simple group means as features.

---

## 8. References

- **COMPETITION.md:** Full rules, metric, submission format, tips (temporal split, hierarchy, low SNR, non-stationarity).
- **basic-example.ipynb:** Data load, metric implementation, simple linear baseline; good starting point for structure.
- **example-of-an-incorrect-modelling.ipynb:** Illustrates what **not** to do (likely look-ahead or wrong split).
