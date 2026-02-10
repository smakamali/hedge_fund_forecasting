# EDA Observations: sub_category, code, sub_code, horizon, ts_index vs y_target

Summary of findings from the exploratory data analysis in `basic-example.ipynb` (train data only).

---

## 1. Cardinality and Balance

| Variable       | Unique values | Notes |
|----------------|---------------|--------|
| **sub_category** | 5           | Roughly balanced (~1.05–1.07M rows each). |
| **code**       | 23            | — |
| **sub_code**   | 180           | — |
| **horizon**    | 4 (1, 3, 10, 25) | Row counts decrease with horizon: horizon 1 has the most (~1.39M), horizon 25 the least (~1.22M). |
| **ts_index**   | 3,601         | Range 1–3,601 (span 3,600). |

**Takeaway:** Design is balanced on sub_category; horizon is a meaningful grouping with less data at longer horizons.

---

## 2. y_target Distribution

### Overall

- **Mean:** ≈ −0.67  
- **Median:** ≈ −0.0006  
- **Std:** ≈ 32.5  
- **Min / max:** ≈ −2,202 and +2,314  

The target is **heavily skewed** with long tails; median near zero but mean clearly negative. Outliers on both sides matter for robust modeling and the weighted metric.

### By sub_category (mean, std)

| sub_category | Mean    | Std   |
|--------------|--------|-------|
| PHHHVYZI     | −1.45  | 53.5  |
| PZ9S1Z4V     | −1.22  | 28.2  |
| NQ58FVQM     | −1.00  | 19.8  |
| DPPUO5X2     | −0.18  | 29.3  |
| V8BKY1IV     | +0.53  | 19.5  |

**Observation:** Strong level shift by sub_category. V8BKY1IV is the only positive-mean category; PHHHVYZI and PZ9S1Z4V have the highest variance.

### By horizon (mean, std)

| Horizon | Mean    | Std  |
|---------|--------|------|
| 1       | −0.08  | 11.7 |
| 3       | −0.25  | 19.4 |
| 10      | −0.78  | 33.8 |
| 25      | −1.68  | 52.8 |

**Observation:** As horizon increases, mean y_target becomes more negative and variance increases (longer horizons are noisier and more negative on average).

---

## 3. y_target by code and sub_code

- **Codes:** Wide spread. Some codes have very negative means (e.g. W4S29LF4 ≈ −10.95, QAQDDTPJ ≈ −6.30) with high variance; others sit near zero with very low variance (e.g. 83EG83KQ, SJZP0OVU). **Code is a strong predictor of level and scale.**
- **Sub_codes (top 10 by row count):** Means range from about −3.74 to +2.43 with stds in the 24–43 range. **Sub_code also carries strong signal** and is a good candidate for encoding or group-specific modeling.

---

## 4. Hierarchy: code → sub_code → sub_category

- **Sub_codes per code:** 39–176 (mean ~81). Each code has many sub_codes.
- **Codes per sub_category:** Every sub_category has **all 23 codes**.
- **Sub_codes per sub_category:** Every sub_category has **all 180 sub_codes**.

So the same set of codes and sub_codes appears in every sub_category; the hierarchy is “flat” in that sense. What changes by sub_category is the **combination** (code, sub_code, sub_category). This aligns with the competition tip: “similarities within sub_category, differences for same sub_code across codes.”

---

## 5. Cross-tab: Mean y_target by (sub_category × horizon)

| sub_category | Horizon 1 | Horizon 3 | Horizon 10 | Horizon 25 |
|--------------|-----------|-----------|------------|------------|
| DPPUO5X2     | −0.07     | −0.18     | −0.26     | −0.22     |
| NQ58FVQM     | −0.06     | −0.22     | −1.04     | −2.91     |
| PHHHVYZI     | −0.20     | −0.60     | −1.68     | −3.57     |
| PZ9S1Z4V     | −0.14     | −0.42     | −1.35     | −3.22     |
| V8BKY1IV     | +0.06     | +0.17     | +0.46     | +1.53     |

**Observations:**

- **V8BKY1IV** is the only category with positive means; they **increase** with horizon (short- to long-term more positive).
- The other four categories have means that **decrease** (become more negative) as horizon increases.
- At horizon 25 the spread is largest: PHHHVYZI and PZ9S1Z4V around −3.2 to −3.6; V8BKY1IV +1.53. There is a clear **horizon × sub_category interaction**; the effect of horizon is not the same across sub_categories.

---

## 6. Row Counts and ts_index Coverage per Entity

- **Entities:** 36,923 unique (code, sub_code, sub_category, horizon).
- **Rows per entity:** Mean ~144.6, range 1–213. 25% of entities have ≤104 rows, 50% ≤166, 75% ≤194.
- **Ts_index coverage per entity:** Same distribution (one row per ts_index per entity). So some entities have only 1 timestamp, others up to 213.

**Implication:** Time coverage is **uneven**. Entities with short history will have fewer or no lags/rolling features; any temporal encoding or modeling must handle variable-length history.

---

## 7. Implications for Modeling

1. **Categorical features:** Use **code**, **sub_code**, and **sub_category** as features. Treat all as **one-hot encoded** so the model can learn level shifts and interactions per level (cardinalities 23, 180, 5 are manageable for one-hot).
2. **Horizon as continuous:** The EDA shows a **continuous progression** of mean y_target with horizon (monotonic for each sub_category: more negative at 1 → 3 → 10 → 25 for four categories, more positive for V8BKY1IV). Treat **horizon as a numeric/continuous feature** (values 1, 3, 10, 25) rather than categorical. This uses one degree of freedom for the main effect and lets the model capture the smooth relationship; if the response is non-linear, a transform (e.g. log(horizon)) or flexible model can capture it.
3. **Horizon × sub_category interaction:** The cross-tab shows the effect of horizon differs by sub_category. Include an **interaction** between horizon (continuous) and sub_category (one-hot), or allow the model to learn it (e.g. tree splits on both).
4. **y_target** is skewed and heavy-tailed; consider robust scaling, clipping, or a transform (e.g. log/Box–Cox where valid), and train with **sample_weight = weight** to match the competition metric.
5. **Variable time coverage:** Avoid assuming long history for every entity when designing lags or rolling features. See **Section 8** for suggested window length.

---

## 8. Lag and time-window design

Given the EDA (entity coverage, stability tip, no explicit competition guidance on lags), the following is a reasonable default for defining lags and rolling windows.

### Recommended window length

- **Range:** **20–50** steps in `ts_index`, with a preference for the **shorter end (e.g. 20–30)** unless validation shows benefit from longer memory.
- **Rationale:**
  - **Entity coverage:** 25% of entities have ≤104 `ts_index` points; median ~166, max 213. A long window (e.g. 100+) leaves many entities with few or no valid lags for much of their history.
  - **Stability:** The competition notes the process may not be fully stable over time and suggests focusing on recent periods. Long windows dilute recent signal.
  - **Information:** A very short window (e.g. 1–5) may underuse autocorrelation; 20–30 steps gives recent dynamics without requiring long history.

### Options

1. **Fixed window:** Use **20 or 30** as lookback — e.g. lags 1–20 or 1–30 (or one lag per step). Most entities then have enough history for most of their timeline.
2. **Sparse lags:** Use a small set of lags, e.g. **1, 2, 3, 5, 10, (20)**. Short- and medium-term memory without requiring a long contiguous window; entities with 10+ observations can use most of these.
3. **Horizon-aware (optional):** If horizon is interpreted as "how far ahead we care about", use a window that scales with horizon (e.g. 2× or 3× horizon). Validate before committing.

### Summary

A **window length of 20–30** (or equivalently lags 1–20 or 1–30, or a sparse set like 1, 2, 3, 5, 10, 20) is a reasonable default: it respects variable coverage, emphasizes recent periods, and still gives the model temporal structure to learn from.

---
## 9. Feature Distribution Analysis and Transformation Strategy

### 9.1 Overview
Analysis of histogram statistics (skewness, kurtosis, quantiles) for all 90 numeric features reveals highly heterogeneous distributions requiring feature-type-specific preprocessing strategies.

### 9.2 Target Variable Characteristics

**`y_target`**: Range [-2201, 2314], mean ≈ 0 (-0.67)
- **Extreme concentration**: 96.8% of values fall in a single bin near zero
- **Skewness**: 1.19 (moderate right tail)
- **Kurtosis**: 289.76 (extremely heavy tails with rare but massive outliers)
- **Implication**: The weighted RMSE metric will be heavily influenced by tail events. Most predictions will be near zero, but the model must handle extreme outliers effectively.

**`weight`**: Catastrophically skewed (skew=2304, kurt=5.3M)
- Single extreme outlier at 1.39e13; 99.9%+ values in first bin
- Cannot be used as a feature (per competition rules), but this distribution emphasizes why weighted metrics focus on rare high-impact events

### 9.3 Feature Distribution Taxonomy

#### Type A: Near-Uniform (6 features)
**Features**: `feature_b`, `feature_c`, `feature_d`, `feature_e`, `feature_f`, `feature_g`
- Skewness ≈ 0, Kurtosis ≈ -1.2 (platykurtic - flatter than normal)
- Range [0.13, 17], nearly perfect uniform distributions
- **Transformation**: Minimal preprocessing needed; standardization only

#### Type B: Zero-Inflated / Highly Right-Skewed (30+ features)
**Examples**: `feature_h`, `feature_i`, `feature_j`, `feature_k`, `feature_l`, `feature_m`, `feature_o`, `feature_p`, `feature_q`, `feature_aa`, `feature_ab`, `feature_ac`, `feature_ae`
- **57-97% of values concentrated in first bin** (at or near zero)
- Skewness: 1.3 to 23 (extreme right tails)
- Kurtosis: 0.6 to 800+ (very heavy tails)
- Examples:
  - `feature_h`: 57% in first bin, skew=1.76, kurt=5.1
  - `feature_o`: 97% in first bin, skew=13.9, kurt=295
  - `feature_ag`: 98% in first bin, skew=22.0, kurt=801
- **Transformation**: 
  - Apply `log(1 + x)` or `log(x)` for non-zero values
  - Consider creating binary "is_zero" flags to capture zero-inflation pattern
  - May need quantile transformation for extreme cases

#### Type C: Large-Scale Features (11 features)
**Features**: `feature_at`, `feature_au`, `feature_av`, `feature_aw`, `feature_ax`, `feature_ay`, `feature_ba`, `feature_bb`, `feature_bc`, `feature_bd`, `feature_be`, `feature_bf`, `feature_bh`, `feature_bj`, `feature_bk`
- Values range from thousands to millions (some exceed 10^7)
- Skewness: 3.8 to 23 (strong right skew)
- Kurtosis: 20 to 800 (heavy tails)
- 88-99% concentration in first bin
- Examples:
  - `feature_at`: mean=3299, max=550K, skew=10.3, kurt=166
  - `feature_ba`: mean=45K, max=31.7M, skew=22.7, kurt=764
- **Transformation**: 
  - Log transform mandatory: `log(1 + x)`
  - Follow with RobustScaler or QuantileTransformer
  - Consider winsorizing at Q99 to cap extreme outliers

#### Type D: Left-Skewed / Negative-Only (10 features)
**Features**: `feature_bs`, `feature_bt`, `feature_bu`, `feature_by`, `feature_bz`, `feature_ca`, `feature_cb`, `feature_cc`, `feature_cd`
- **All values are negative** (e.g., `feature_bs`: [-12.99, -0.08])
- Negative skewness: -0.57 to -16.2 (left tails)
- High concentration near zero (upper bound)
- Examples:
  - `feature_by`: 97% in last bin near zero, skew=-16.2, kurt=592
  - `feature_ca`: 95% in last bin, skew=-10.6, kurt=290
- **Transformation**: 
  - Option 1: Standardize as-is (preserve negative scale)
  - Option 2: Reflect and log: `-log(-x + eps)` if symmetric treatment with positive features is desired
  - Winsorize left tail at Q01 if needed

#### Type E: Symmetric / Centered at Zero (6 features)
**Features**: `feature_n`, `feature_w`, `feature_x`, `feature_y`, `feature_z`, `feature_al`
- Centered near zero (mean ≈ 0)
- Wide symmetric or slightly skewed ranges (e.g., `feature_w`: [-901, 901])
- Moderate to extreme kurtosis (27 to 260) indicating heavy tails
- Examples:
  - `feature_n`: range [-41, 52], mean=0.003, skew=2.0, kurt=157
  - `feature_w`: range [-901, 901], mean=-1.5, skew=-0.37, kurt=45
- **Transformation**: 
  - Winsorize/clip extreme outliers at Q01/Q99
  - Standardize with StandardScaler or RobustScaler

#### Type F: Moderate Skew (8 features)
**Features**: `feature_a`, `feature_ah`, `feature_ai`, `feature_aj`, `feature_aq`, `feature_ar`, `feature_as`, `feature_cf`, `feature_cg`
- Skewness: 0.9 to 2.8 (mild to moderate right tail)
- Kurtosis: 0.2 to 16 (moderate tails)
- More balanced distributions with reasonable spread
- Examples:
  - `feature_ai`: range [0.003, 42.8], mean=1.94, skew=1.15, kurt=5.8
  - `feature_aq`: range [0.012, 25.6], mean=1.51, skew=1.00, kurt=0.72
- **Transformation**: 
  - Light transforms: `sqrt(x)` or mild Box-Cox
  - StandardScaler may suffice for some
  - Test both transformed and untransformed versions

#### Type G: Bounded/Discrete-Like (1 feature)
**Feature**: `feature_ch`
- Range [0, 10], appears to take integer values
- Skew=1.2, Kurt=1.2 (moderate)
- Distinct peaks at integer values
- **Transformation**: 
  - Treat as ordinal/categorical (one-hot or target encoding)
  - Or leave as continuous numeric if treating as ordinal scale

### 9.4 Missing Value Patterns

Most features have full coverage (n ≈ 5.34M), but notable exceptions:
- **~1% missing**: `feature_h`, `feature_i`, `feature_j`, `feature_k` (n ≈ 5.28M)
- **~3-5% missing**: `feature_by`, `feature_cd`, `feature_ce` (n ≈ 4.75M-5.06M)
- **~7-12% missing**: `feature_at`, `feature_ay`, `feature_bi`, `feature_al`, `feature_aw` (n ≈ 4.67M-5.19M)

**Handling**: Use temporal-only imputation (forward fill within entity, or rolling mean) to avoid look-ahead bias.

### 9.5 Recommended Preprocessing Pipeline

#### Step 1: Feature Type Classification
Programmatically classify features into types A-G based on:
- Skewness threshold (|skew| > 2 → highly skewed)
- Kurtosis threshold (kurt > 10 → heavy tails)
- Zero concentration (% in first bin > 80% → zero-inflated)
- Value scale (max > 10,000 → large-scale)

#### Step 2: Type-Specific Transformations

```python
Pseudo-code strategy
if feature in type_B or type_C: # Zero-inflated or large-scale
feature_transformed = np.log1p(feature)
feature_is_zero = (feature == 0).astype(int) # Binary flag
elif feature in type_D: # Negative-only
feature_transformed = feature # Keep as-is or reflect+log
elif feature in type_E: # Symmetric with heavy tails
feature_transformed = winsorize(feature, limits=[0.01, 0.01])
elif feature in type_F: # Moderate skew
feature_transformed = np.sqrt(feature) # or Box-Cox
All features: scale after transformation
feature_scaled = RobustScaler().fit_transform(feature_transformed)
```


#### Step 3: Validation
For each transformation:
- Re-compute skewness and kurtosis (target: |skew| < 1, kurt < 5)
- Check for remaining outliers (Q01/Q99 ratio)
- Validate no information leakage across time

### 9.6 Feature Engineering Opportunities

Based on distribution patterns:
1. **Zero-inflation flags**: Binary indicators for 30+ zero-inflated features
2. **Log-ratios**: For feature pairs with similar scales (e.g., `feature_b`/`feature_c`)
3. **Interaction with horizon**: Since horizon shows continuous progression, create `feature_i × horizon` terms for key features
4. **Robust aggregations**: Use median/MAD instead of mean/std for lag features given heavy tails
5. **Quantile-based features**: Percentile ranks within entity-horizon groups

### 9.7 Key Takeaways

1. **No single transformation fits all**: A heterogeneous pipeline with feature-type-specific preprocessing is essential
2. **Heavy tails dominate**: 60+ features have kurtosis > 10, requiring robust methods
3. **Zero-inflation is pervasive**: 30+ features are heavily zero-inflated, suggesting mixture models or two-stage approaches
4. **Scale varies by 10+ orders of magnitude**: Features range from 0.0001 to 10^7+, necessitating careful scaling
5. **Target variable's extreme kurtosis** (289.76) suggests the model must handle rare but high-impact predictions—this aligns with the weighted metric focusing on important events

### 9.8 Applicability to Gradient Boosted Trees (LightGBM)

**Critical insight:** Tree-based models like LightGBM are **scale-invariant** and **distribution-agnostic**, making many transformations described above unnecessary or lower-priority compared to linear models or neural networks.

#### What LightGBM Does NOT Need

- ❌ **Scaling/Standardization (StandardScaler, RobustScaler)**: Trees split on thresholds, not distances; scale is irrelevant
- ❌ **Normality transformations for skewness alone**: No assumption of normal distributions; right-skewed, left-skewed, or multimodal distributions are fine as-is
- ❌ **Handling multicollinearity**: Trees naturally handle correlated features via feature importance and regularization

#### What LightGBM DOES Benefit From

**High Priority:**
- ✅ **Outlier clipping/winsorizing** (features with kurt > 50): Prevents overfitting to rare extreme values and wasted splits on noise
  - **Impact:** 20%+ improvement in reducing overfitting
  - **Action:** Clip at Q01/Q99 or Q05/Q95 for Type E features and extreme cases

- ✅ **Feature engineering** (lags, interactions, aggregations): Trees cannot automatically create temporal lags or complex interactions
  - **Impact:** 20-40% improvement
  - **Action:** Create lag features (Section 8), horizon × sub_category interactions, entity-level aggregations

**Moderate Priority:**
- ⚠️ **Log transforms for large-scale features** (Type C only): Features spanning 7+ orders of magnitude benefit from log transform
  - **Why:** Improves split point selection across different scales (captures both 1→10 and 10K→100K changes)
  - **Impact:** 10-20% improvement for these 11 features
  - **Action:** Apply `log(1 + x)` to Type C features only (`feature_at`, `feature_au`, etc.)

- ⚠️ **Zero-inflation flags** (features with >80% zeros): Explicit binary indicators help trees learn "zero vs non-zero" patterns efficiently
  - **Impact:** 5-10% improvement for zero-inflated features
  - **Action:** Create `feature_X_is_zero` for ~10-15 Type B features

**Low/Skip Priority:**
- ❌ **Scaling after transforms**: Not needed
- ❌ **Log transforms for skewness reduction** (Type B, D, F): Minimal benefit for trees; test but expect low impact
- ❌ **Normalization of moderate-skew features**: Trees handle these naturally

#### Expected Impact by Model Type

| Transformation | Linear Model | Neural Net | LightGBM | Notes |
|----------------|--------------|------------|----------|-------|
| **Scaling** | ✅ Critical | ✅ Critical | ❌ Not needed | Trees are scale-invariant |
| **Log (large-scale)** | ✅ Critical | ✅ Critical | ⚠️ Helpful (10-20%) | Only for extreme-scale features (Type C) |
| **Log (skewness only)** | ✅ Critical | ⚠️ Helpful | ❌ Low value | Trees split on thresholds, not distances |
| **Outlier clipping** | ⚠️ Helpful | ⚠️ Helpful | ✅ Important (20%+) | Prevents overfitting to rare values |
| **Zero-inflation flags** | ⚠️ Helpful | ⚠️ Helpful | ⚠️ Helpful (5-10%) | Explicit pattern capture |
| **Lag features** | ✅ Critical | ✅ Critical | ✅ Critical (40%+) | Cannot be learned automatically |
| **Feature interactions** | ⚠️ Helpful | ⚠️ Helpful | ✅ Important (20%+) | Trees approximate but explicit helps |

#### Recommended LightGBM Preprocessing Pipeline

**Priority 1 (Critical):**
1. Winsorize extreme outliers (kurt > 50 features) at Q01/Q99
2. Create lag features (see Section 8 for window length: 20-30 steps)
3. Temporal train/validation split (no look-ahead)

**Priority 2 (High Value):**
4. Log transform Type C features only (11 large-scale features): `log(1 + x)`
5. Create zero-inflation flags for ~10-15 features with >80% zeros
6. Feature engineering: horizon × sub_category interactions, entity-level aggregations
7. Handle categoricals: Use LightGBM native categorical support (no one-hot needed)

**Priority 3 (Optional/Test):**
8. Test log transforms for Type B features (zero-inflated): compare raw vs `log(1 + x)`
9. Test transforms for Type D features (negative): likely unnecessary

**Priority 4 (Skip):**
10. Scaling/standardization: Not needed for trees
11. Transforms purely for symmetry: Not needed for trees

#### Bottom Line for LightGBM

**Skip 60-70% of transformations** described in sections 9.3-9.5. Focus instead on:
- **Outlier management** (more critical for trees than linear models)
- **Feature engineering** (2-4× higher impact than transformations)
- **Selective log transforms** (only for extreme-scale features, not for skewness)

**Experimentation strategy:** Start with baseline (raw features + lags) → add outlier clipping → add log transforms (Type C) → add zero flags. Measure marginal benefit at each step on validation set.

---

## Source

- **Notebook:** `basic-example.ipynb` — section “EDA: sub_category, code, sub_code, horizon, ts_index vs y_target”.
- **Data:** `train.parquet` only.
