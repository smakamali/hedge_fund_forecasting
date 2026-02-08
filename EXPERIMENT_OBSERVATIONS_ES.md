# Experiment Observations: hedge_fund_forecasting_es

Summary of the MLflow experiment "hedge_fund_forecasting_es" (early stopping, 1000 max rounds, patience 50).

## Results Overview

| Run Name | val_skill | val_rmse | train_skill | num_boost_round_actual | Features |
|----------|-----------|----------|-------------|------------------------|----------|
| **base** | **0.1197** | 27.34 | 0.306 | 87 | none |
| **xlag** | **0.1176** | 27.40 | 0.264 | 40 | input lags only |
| **agg** | 0.0833 | 27.46 | 0.249 | 28 | aggregates only |
| ylag+roll+agg+xlag_trans | 0.0252 | 27.55 | 0.436 | 4 | full + target transform |
| roll+agg | 0.0000 | 27.55 | 0.217 | 1 | rolling + aggregates |
| ylag+roll+agg | 0.0000 | 27.55 | 0.244 | 1 | ylag + rolling + aggregates |
| ylag+roll+agg+xlag | 0.0000 | 27.55 | 0.244 | 1 | full |
| ylag+xlag | 0.0000 | 27.55 | 0.214 | 1 | ylag + xlag |
| ylag+roll | 0.0000 | 27.55 | 0.244 | 1 | ylag + rolling |
| ylag+agg | 0.0000 | 27.55 | 0.244 | 1 | ylag + aggregates |
| roll | 0.0000 | 27.55 | 0.217 | 1 | rolling only |
| ylag | 0.0000 | 27.55 | 0.244 | 1 | target lags only |

---

## Key Observations

### 1. Best-performing configurations
- **base** (no target lags, rolling, aggregates, or input lags) achieves the highest validation skill (0.120).
- **xlag** (input lags only) is close (0.118), with slightly lower train skill.
- **agg** (aggregates only) is third (0.083), with fewer boosting rounds (28).

### 2. Early stopping behavior
- **base**: 87 rounds (early stop from 1000).
- **xlag**: 40 rounds.
- **agg**: 28 rounds.
- **full+trans**: 4 rounds only.
- **9 of 12 runs** stopped at **1 boosting round**, so the best iteration was effectively the initial model.

### 3. Autoregressive features and overfitting
Configurations that use **target lags** (ylag) or **rolling** (roll) almost always:
- Stop at 1 round.
- Get val_skill = 0.

This suggests validation loss degrades as soon as training begins. Possible causes:
- Information leakage (future or validation-period information in features).
- Train/validation distribution shift.
- Overfitting on highly predictive but non-generalizing features.

### 4. Input lags (xlag) vs target lags (ylag)
- **xlag only**: strong validation performance (0.118), stable training (40 rounds).
- **ylag** (with or without other features): poor validation, 1-round stop.

So lagged input features generalize better than lagged target features in this setup.

### 5. Train vs validation gap
- **base**: train_skill 0.31 vs val_skill 0.12.
- **xlag**: train 0.26 vs val 0.12.

Models that train for many rounds show a clear gap; 1-round runs have minimal training and therefore smaller gaps, but worse validation skill.

### 6. Target transform
- **ylag+roll+agg+xlag_trans**: stops at 4 rounds, val_skill 0.025.
- Train skill 0.44 is much higher than val_skill 0.025, indicating strong overfitting once training progresses.

### 7. Validation RMSE
Val RMSE is similar across runs (27.3–27.6). The skill score separates configurations more clearly than RMSE.

---

## Recommendations
1. Prefer **base** or **xlag** for production.
2. Treat **target lags** and **rolling** with caution; investigate leakage and distribution shift.
3. Consider increasing early-stopping patience or lowering learning rate to allow more rounds before stopping.
4. Inspect why validation loss worsens immediately for ylag/roll configs (e.g., temporal overlap, validation set composition).
