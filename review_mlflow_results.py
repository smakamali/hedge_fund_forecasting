"""Review MLflow experiment results for hedge_fund_forecasting_es."""
import mlflow
import pandas as pd

experiments = mlflow.search_experiments()
for exp in experiments:
    print("Experiment:", exp.name, "(id=", exp.experiment_id, ")")

runs = mlflow.search_runs(
    experiment_names=["hedge_fund_forecasting_tune_xlag"],
    max_results=50,
)
print("\n--- Runs (hedge_fund_forecasting_tune_xlag) ---")
cols = [
    "run_id",
    "tags.mlflow.runName",
    "metrics.val_skill",
    "metrics.val_rmse",
    "metrics.train_skill",
    "metrics.train_rmse",
    "metrics.num_boost_round_actual",
    "params.use_input_lags",
    "params.use_target_lags",
    "params.use_rolling",
    "params.use_aggregates",
    "params.use_target_transform",
    "params.lags_max",
    "params.top_k_per_feature",
    "params.use_global_lags",
    "params.global_lags",
]
available = [c for c in cols if c in runs.columns]
df = runs[available].sort_values("metrics.val_skill", ascending=False)
print(df.to_string())
