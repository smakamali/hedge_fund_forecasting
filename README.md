# Hedge Fund Forecasting

Time series forecasting pipeline for a Kaggle-style competition: predict a continuous target (`y_target`) per entity and time index using 86 anonymized features, with **strict no-future-data leakage** and a weighted skill-score metric.

---

## Overview

- **Goal:** Predict future values for each combination of `code`, `sub_code`, `sub_category`, `horizon`, and `ts_index`.
- **Constraint:** At prediction time `t`, only data with `ts_index ≤ t` may be used (sequential, no look-ahead).
- **Metric:** Weighted skill score (higher is better); see [docs/COMPETITION.md](docs/COMPETITION.md) for the formula and rules.

This repo provides:

- **Shared library** (`src/`): preprocessing, evaluation, config loading — all temporally safe.
- **Validation scripts**: temporal train/val splits, LightGBM with skill-based early stopping.
- **Three-step submission pipeline**: Model A (sequential), noise-robust A2, and B + A' (two-stage) producing `submission_v1`, `submission_v2`, and `submission_v3`.

---

## Repository structure

```
hedge_fund_forecasting/
├── config.yaml           # Input/target lags, LightGBM params, data_dir
├── requirements.txt     # Python dependencies
├── data/                # train.parquet, test.parquet (from competition)
├── output/              # Submission CSVs, metadata JSONs, validation errors
├── src/                 # Shared library
│   ├── __init__.py
│   ├── config_loader.py # YAML/JSON config with defaults
│   ├── constants.py     # Entity cols, target, lags, feature lists
│   ├── evaluation.py    # Skill score, temporal split, LightGBM feval
│   └── preprocessing.py # Leak-free feature engineering (lags, rolling, etc.)
├── scripts/             # Entry points
│   ├── run_submission_pipeline.py   # Main: 3-step pipeline → v1/v2/v3
│   ├── run_validation_lagged_features.py  # Core validation + feature build
│   ├── run_validation_no_ytarget_features.py
│   ├── run_validation_compare.py
│   ├── run_tune_target_lags.py
│   ├── run_tune_input_lags_exp1.py
│   ├── run_tune_input_lags_exp2.py
│   ├── run_tune_lgb_params.py
│   ├── run_phase3.py
│   ├── run_phase6_submit.py
│   ├── run_phase6_sequential_submit.py
│   ├── run_experiments.py
│   └── review_mlflow_results.py
├── docs/                # Competition rules, EDA, tuning notes
│   ├── COMPETITION.md
│   ├── EDA_OBSERVATIONS.md
│   ├── TUNE_*.md
│   └── ...
├── notebook/            # basic-example, eda-notebook
└── basic-example.ipynb
```

---

## Setup

### Requirements

- Python 3.8+
- Dependencies: see `requirements.txt`

```bash
pip install -r requirements.txt
```

Key packages: `numpy`, `pandas`, `scikit-learn`, `pyarrow`, `lightgbm`, `mlflow`, `pyyaml`. Optional: `polars`, `matplotlib`, `seaborn`, `kaggle`, `jupyter`.

### Data

Place the competition data in the configured data directory (default `data/`):

- `data/train.parquet` — training set (includes `y_target`, `weight`, entity columns, `ts_index`, 86 features).
- `data/test.parquet` — test set (no `y_target`; same schema otherwise).

You can override the path via `config.yaml` → `data_dir`.

---

## Configuration

`config.yaml` (or a custom path via `--config`) controls:

| Section        | Purpose |
|----------------|--------|
| **input_lags** | `lags_max`, `top_k_per_feature`, `use_global_lags`, `global_lags`, `use_float16_when_large` — which input-feature lags to create and optional scaling for float16. |
| **target_lags**| `use_target_lags`, `y_lags`, `use_rolling`, `use_aggregates`, `use_target_transform` — target-derived autoregressive features. |
| **lightgbm**   | `num_leaves`, `min_data_in_leaf`, `max_depth`, `learning_rate`, `num_boost_round`, `early_stopping_rounds`, etc. |
| **data_dir**   | Directory containing `train.parquet` and `test.parquet` (relative to project root or absolute). |

Defaults are defined in `src/config_loader.py` and merged with the file.

---

## Quick start: submission pipeline

Run from the **project root**:

```bash
# All three steps (v1, v2, v3)
python scripts/run_submission_pipeline.py

# Single step
python scripts/run_submission_pipeline.py --step 1
python scripts/run_submission_pipeline.py --step 2
python scripts/run_submission_pipeline.py --step 3

# Custom config and MLflow experiment
python scripts/run_submission_pipeline.py --config config.yaml --experiment my_experiment
```

- **Step 1:** Model A (full features, target lags + input lags) → sequential validation → saves validation errors → trains on full train → **submission_v1_\<timestamp\>.csv**.
- **Step 2:** Uses Step 1’s validation errors to create a noisy target; trains Model A2 → **submission_v2_\<timestamp\>.csv**.
- **Step 3:** Model B (no target lags) + Model A' (pseudo target from B) → **submission_v3_\<timestamp\>.csv**.

Outputs go to `output/`: CSV submissions, `*_metadata.json` next to each CSV, and (from Step 1) `validation_errors.npy` / `validation_errors_transformed.npy`.

---

## Three-step pipeline in detail

### Step 1 — Model A (submission_v1)

- Temporal split (e.g. 80% train / 10% val).
- **Features:** base (86 + entity encoding, missing indicators, winsorize, log type-C, zero flags) + **target lags/rolling/aggregates** + **input lags** (correlation-selected); optional target transform (MinMax → log1p → MinMax).
- Validation uses **sequential prediction**: for each `ts_index`, target-derived features use only past predictions and running stats (no future leakage).
- Validation errors (original and optionally transformed) are saved for Step 2.
- Final model is trained on **full train** (rounds from early stopping); test predictions are produced sequentially → `submission_v1_<ts>.csv`.

### Step 2 — Model A2 (submission_v2)

- **Requires** Step 1 (reads `output/validation_errors.npy` or `validation_errors_transformed.npy`).
- Builds a **noisy target** `y_noisy = y_target + random_sample(validation_errors)` and the same target-derived feature set (lags/rolling/aggregates) from `y_noisy`, keeping names compatible with inference.
- Trains **Model A2** (90/10 temporal split, then full train) for robustness to noise.
- Sequential prediction on test → `submission_v2_<ts>.csv`.

### Step 3 — Model B + Model A' (submission_v3)

- **Model B:** Base + **input lags only** (no target lags/rolling/aggregates). Trained on 90% of time; early stopping on 10%.
- **Pseudo-target:** B’s predictions on **full** train (`y_pred_B_train`); the 10% used for early stopping is out-of-fold, so no direct leakage.
- **Model A':** Same feature layout as A, but target-derived features are built from `y_pred_B_train` instead of `y_target`. Trains on full train; rounds from 90/10 early stopping.
- **Test:** Predict with B (full-train) → build lags/rolling/aggregates from B’s test predictions → feed into A' → **submission_v3_<ts>.csv**.

---

## Other scripts

| Script | Purpose |
|--------|--------|
| `run_validation_lagged_features.py` | Core validation: build train/val features (with/without input lags, target lags, rolling, aggregates), train LightGBM with temporal split and skill-based early stopping. The submission pipeline imports its feature-building and training helpers. |
| `run_validation_no_ytarget_features.py` | Validation without target-derived features (base + input lags only). |
| `run_validation_compare.py` | Compare validation setups. |
| `run_tune_target_lags.py`, `run_tune_input_lags_exp1.py`, `run_tune_input_lags_exp2.py`, `run_tune_lgb_params.py` | Hyperparameter and lag tuning (use `config.yaml` and optionally MLflow). |
| `run_phase3.py`, `run_phase6_submit.py`, `run_phase6_sequential_submit.py` | Phased experiment and submission variants. |
| `run_experiments.py` | Batch experiments. |
| `review_mlflow_results.py` | Inspect MLflow runs (e.g. after `--experiment NAME`). |

Run validation from project root, e.g.:

```bash
python scripts/run_validation_lagged_features.py
python scripts/run_validation_lagged_features.py --config config.yaml --no-mlflow
```

---

## Key concepts

- **No leakage:** All feature engineering uses only past data (e.g. lags/rolling/aggregates at time `t` use data with `ts_index < t`). See `src/preprocessing.py` and the sequential prediction logic in `run_submission_pipeline.py`.
- **Sequential prediction:** For validation and test, predictions are made one `ts_index` at a time; target-derived features for the next step use only those past predictions and running global/sub_category stats.
- **Skill score:** The competition metric is implemented in `src/evaluation.py` as `weighted_rmse_score(y_target, y_pred, w)`. LightGBM early stopping uses a custom feval that maximizes this score on the validation set.

---

## Outputs and MLflow

- **output/**  
  - `submission_v1_<timestamp>.csv`, `submission_v2_<timestamp>.csv`, `submission_v3_<timestamp>.csv` — submission files.  
  - `submission_v*_<timestamp>_metadata.json` — metrics and config used for that run.  
  - `validation_errors.npy`, `validation_errors_transformed.npy` (and timestamped copies) — produced by Step 1 for Step 2.

- **MLflow:** If you pass `--experiment NAME`, each pipeline step logs params, metrics, and artifacts (metadata + submission CSV) to MLflow. Use `review_mlflow_results.py` to inspect runs.

---

## Documentation

- [docs/COMPETITION.md](docs/COMPETITION.md) — Rules, metric, dataset, submission format, leakage rules.
- [docs/EDA_OBSERVATIONS.md](docs/EDA_OBSERVATIONS.md) — Exploratory analysis notes.
- [docs/TUNE_*.md](docs/TUNE_*.md) — Tuning observations (target lags, input lags, LightGBM).
- [docs/REVIEW_COMMENTS.MD](docs/REVIEW_COMMENTS.MD) — Code review and design notes.

---

## License

See the repository for license information.
