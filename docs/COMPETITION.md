# Kaggle Competition: Time Series Forecasting

## Overview

Participants receive an **integer-indexed time series dataset** (`ts_index` column). Each record is identified by:

- **code**
- **sub-code**
- **sub-category**
- **forecast horizon**

**Objective:** Train a model that generalizes robustly out-of-sample and accurately predicts future values for each combination (code, sub-code, sub-category, horizon).

**Constraint:** Your forecast must not use any data whose `ts_index` is greater than the `ts_index` of the forecast data.

**Ranking:** Submissions are ranked according to an aggregate out-of-sample metric calculated for all combinations.

---

## Description

- You predict a **continuous numerical value**.
- Scoring uses a measure inspired by the **skill score**.
- The test set is **partially hidden (75%)** throughout the process to ensure a true out-of-sample evaluation.

---

## Dataset Description

The dataset is a **time-series / tabular** dataset with the following columns:

| Column | Description |
|--------|-------------|
| **id** | A unique key constructed by concatenating `code`, `sub_code`, `sub_category`, `horizon`, and `ts_index` with a double underscore (`__`). This ensures each row is distinctly identifiable. |
| **code** | A unique identifier for the entity. |
| **sub_code** | A categorical attribute grouping entities into sub-families or segments. |
| **sub_category** | A categorical label describing the broad category to which the entity belongs. |
| **ts_index** | Integer timestamp of the observation: indicates when the features were recorded. |
| **horizon** | A categorical forecast-horizon group. Typical codes: **1** = short-term, **3** = medium-term, **10** = long-term, **25** = extra long-term. These codes do not represent the difference between `ts_index`. |
| **weight** | A numeric weight for each row, used in the evaluation metric. **DO NOT USE AS A FEATURE.** These weights correspond to \( w \) in the loss function formula. |
| **feature_a**, **feature_b**, …, **feature_ch** | A set of **86 anonymized features**. |

### Data shape

Each row represents one forecast instance for a particular combination of (`code`, `sub_code`, `sub_category`, `ts_index`, `horizon`) along with its associated feature values. All features can be fed directly into typical regression models.

---

## Evaluation

- Submissions are evaluated using the metric below.
- **Public leaderboard:** ~25% of the test data.
- **Private leaderboard (final):** remaining 75%.
- Final ranking may differ from the public ranking—avoid overfitting to the public set.

### Metric formula

The metric is computed over a set of lines \( I \) (25% for public, 75% for private):

\[
\sqrt{1 - \text{clip}_{[0,1]}\left( \frac{\sum_{i \in I} w_i (y_i - \hat{y}_i)^2}{\sum_{i \in I} w_i y_i^2} \right)}
\]

### Reference implementation

```python
def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))

def weighted_rmse_score(y_target, y_pred, w) -> float:
    denom = np.sum(w * y_target ** 2)
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = _clip01(ratio)
    val = 1.0 - clipped
    return float(np.sqrt(val))
```

### Reproducibility

- This is **not** code-only: you can submit predictions as CSV files.
- **Recommended:** provide a Kaggle notebook for full reproducibility.
- **Winners:** must submit executable code (exact dependency versions, Python version) so results can be reproduced and checked for data leakage. Non-reproducible solutions may be disqualified from monetary prize eligibility.

### Data leakage rules

To avoid look-forward bias, your code must:

- Predict at `ts_index` \( t \) using **only** data from `ts_index` \( 0 \) to \( t \).
- Process all data **strictly sequentially**.

---

## Submission

- **Format:** CSV with columns `id` and `prediction`.
- **Scope:** Predictions must be made on the test file (`test.parquet`).

### Example

```csv
id,prediction
W2MW3G2L__STALY73S__9ZI8OAJB__1__2991,5.764190326788755
83EG83KQ__R571RU17__PHHHVYZI__1__3353,5.764190326788755
W2MW3G2L__STALY73S__Q101PRO5__3__2991,5225125.5454
83EG83KQ__R571RU17__PZ9S1Z4V__1__3353,4545.4545
```

### Recommended notebook structure

1. **Imports**
2. **Functions or classes** used
3. **Code that fits the model**
4. **Prediction code**

---

## Tips

- **Temporal split:** Exact times are hidden in the test data, but it comes from a period **after** the training data. Consider focusing on a specific window (e.g. weighting the most recent periods) if it helps.
- **Hierarchy:** Code, sub-code, and sub-category have both similarities and differences (e.g. similarities within the same sub-category, differences for the same sub-code across codes). Use an appropriate weighting between in-category and out-of-category data.
- **Challenges:**
  - Low **signal-to-noise ratio**.
  - Underlying process may not be fully **stable** over `ts_index`/time.
- **Data:** You may **only** use the provided data. External data is **not allowed** (risk of look-forward bias).

### Suggested directions

1. **Data mining and feature analysis**
2. **Advanced modeling techniques**
