"""Review MLflow experiment results.

- hedge_fund_forecasting_tune_xlag: xlag config sweep
- hedge_fund_forecasting_tune_lgb: LightGBM param tuning per TOP_7_CONFIGS
- hedge_fund_forecasting_tune_y_lag: target lag tuning for xlag_gl1 + LGB params

Usage (from project root):
  python scripts/review_mlflow_results.py           # all sections
  python scripts/review_mlflow_results.py --ylag-only   # only tune_y_lag (avoids listing experiments)
"""
import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import mlflow
import pandas as pd

# Config names we care about (run_name_base in run_tune_lgb_params)
TOP_7_CONFIG_NAMES = [
    "xlag_gl1",
    "xlag_gl1_2",
    "xlag_gl1_2_3",
    "xlag_lm10_tk2",
    "xlag_lm15_tk3",
    "xlag_lm10_tk5",
    "xlag_lm10_tk3",
    "xlag_lm5_tk3",
]


def main():
    experiments = mlflow.search_experiments()
    for exp in experiments:
        print("Experiment:", exp.name, "(id=", exp.experiment_id, ")")

    # --- LGB tuning: best run per config (TOP_7_CONFIGS) ---
    print("\n" + "=" * 80)
    print("hedge_fund_forecasting_tune_lgb — best parameter set and metrics per config")
    print("=" * 80)

    runs = mlflow.search_runs(
        experiment_names=["hedge_fund_forecasting_tune_lgb"],
        max_results=500,
    )
    if runs.empty:
        print("No runs found for hedge_fund_forecasting_tune_lgb.")
    else:
        # Run names are like xlag_gl1_nl15_md20_d5 or xlag_gl1_nl31_md20_dunbounded
        def config_from_run_name(name):
            if pd.isna(name):
                return None
            for base in TOP_7_CONFIG_NAMES:
                if name == base or name.startswith(base + "_nl"):
                    return base
            return None

        runs["config"] = runs["tags.mlflow.runName"].map(config_from_run_name)
        runs = runs.dropna(subset=["config"])
        runs = runs[runs["metrics.val_skill"].notna()]

        cols = [
            "tags.mlflow.runName",
            "config",
            "metrics.val_skill",
            "metrics.val_rmse",
            "metrics.train_skill",
            "metrics.train_rmse",
            "metrics.num_boost_round_actual",
            "params.num_leaves",
            "params.min_data_in_leaf",
            "params.max_depth",
        ]
        available = [c for c in cols if c in runs.columns]
        df = runs[available].copy()
        df = df.sort_values("metrics.val_skill", ascending=False)

        best_per_config = []
        for config in TOP_7_CONFIG_NAMES:
            subset = df[df["config"] == config]
            if subset.empty:
                print(f"\n{config}: no runs")
                best_per_config.append({"config": config, "run_name": None})
                continue
            best = subset.iloc[0]
            best_per_config.append(
                {
                    "config": config,
                    "run_name": best["tags.mlflow.runName"],
                    "val_skill": best["metrics.val_skill"],
                    "val_rmse": best["metrics.val_rmse"],
                    "train_skill": best["metrics.train_skill"],
                    "train_rmse": best["metrics.train_rmse"],
                    "num_boost_round_actual": best["metrics.num_boost_round_actual"],
                    "num_leaves": best.get("params.num_leaves"),
                    "min_data_in_leaf": best.get("params.min_data_in_leaf"),
                    "max_depth": best.get("params.max_depth"),
                }
            )
            print(f"\n--- {config} ---")
            print(f"  Best run: {best['tags.mlflow.runName']}")
            print(f"  Params:   num_leaves={best.get('params.num_leaves')}, "
                  f"min_data_in_leaf={best.get('params.min_data_in_leaf')}, "
                  f"max_depth={best.get('params.max_depth')}")
            print(f"  Metrics:  val_skill={best['metrics.val_skill']:.6f}, val_rmse={best['metrics.val_rmse']:.6f}, "
                  f"train_skill={best['metrics.train_skill']:.6f}, train_rmse={best['metrics.train_rmse']:.6f}, "
                  f"num_boost_round_actual={best['metrics.num_boost_round_actual']}")

        # Summary table: config -> best val_skill
        print("\n" + "-" * 80)
        print("Summary (best val_skill per config)")
        print("-" * 80)
        summary = pd.DataFrame(best_per_config)
        summary_with_skill = summary.dropna(subset=["val_skill"])
        if not summary_with_skill.empty:
            summary_with_skill = summary_with_skill.sort_values("val_skill", ascending=False)
            disp_cols = ["config", "run_name", "val_skill", "val_rmse", "num_leaves", "min_data_in_leaf", "max_depth"]
            disp_cols = [c for c in disp_cols if c in summary_with_skill.columns]
            print(summary_with_skill[disp_cols].to_string(index=False))
            overall_best = summary_with_skill.iloc[0]
            print(f"\nOverall best: {overall_best['config']} (val_skill={overall_best['val_skill']:.6f})")
        else:
            print(summary.to_string(index=False))

    _print_ylag_results()

    # --- Optional: tune_xlag runs (legacy) ---
    print("\n" + "=" * 80)
    print("hedge_fund_forecasting_tune_xlag (last 50 runs)")
    print("=" * 80)
    xlag_runs = mlflow.search_runs(
        experiment_names=["hedge_fund_forecasting_tune_xlag"],
        max_results=50,
    )
    if not xlag_runs.empty:
        xlag_cols = [
            "tags.mlflow.runName",
            "metrics.val_skill",
            "metrics.val_rmse",
            "metrics.train_skill",
            "metrics.train_rmse",
            "params.lags_max",
            "params.top_k_per_feature",
            "params.use_global_lags",
            "params.global_lags",
        ]
        xlag_available = [c for c in xlag_cols if c in xlag_runs.columns]
        print(xlag_runs[xlag_available].sort_values("metrics.val_skill", ascending=False).to_string())
    else:
        print("No runs found.")


# Number of top runs to average per ylag config (for ranking)
YLAG_TOP_N_AVG = 5


def _print_ylag_results():
    """Print ylag configs ranked by average of top 5 runs (by val_skill) per config."""
    YLAG_EXPERIMENT_NAMES = ["hedge_fund_forecasting_tune_y_lag", "hedge_fund_forecasting_tune_ylag"]
    print("\n" + "=" * 80)
    print("hedge_fund_forecasting_tune_y_lag — avg of top {} runs per ylag config (sorted by avg val_skill)".format(YLAG_TOP_N_AVG))
    print("=" * 80)
    ylag_runs = None
    for name in YLAG_EXPERIMENT_NAMES:
        try:
            runs = mlflow.search_runs(
                experiment_names=[name],
                max_results=500,
            )
            if not runs.empty:
                ylag_runs = runs
                break
        except Exception:
            continue
    if ylag_runs is not None and not ylag_runs.empty:
        # Run names like ylag_1_nl70_md20_d9, ylag_1_2_3_nl50_md100_dunbounded
        def ylag_config_from_run_name_simple(name):
            if pd.isna(name):
                return None
            if name.startswith("ylag_") and "_nl" in name:
                return name.split("_nl")[0]  # ylag_1, ylag_1_2, ylag_1_2_3, etc.
            return None
        ylag_runs["ylag_config"] = ylag_runs["tags.mlflow.runName"].map(ylag_config_from_run_name_simple)
        ylag_runs = ylag_runs.dropna(subset=["ylag_config"])
        ylag_runs = ylag_runs[ylag_runs["metrics.val_skill"].notna()]
        ylag_configs = ylag_runs["ylag_config"].unique().tolist()
        ylag_configs.sort(key=lambda s: [int(x) for x in s.replace("ylag_", "").split("_") if x])

        # For each config: take top N by val_skill, average metrics, then sort configs by avg val_skill
        avg_per_ylag = []
        for yc in ylag_configs:
            subset = ylag_runs[ylag_runs["ylag_config"] == yc].copy()
            subset = subset.sort_values("metrics.val_skill", ascending=False)
            top_n = subset.head(YLAG_TOP_N_AVG)
            n_used = len(top_n)
            avg_per_ylag.append({
                "ylag_config": yc,
                "n_runs": n_used,
                "avg_val_skill": top_n["metrics.val_skill"].mean(),
                "avg_val_rmse": top_n["metrics.val_rmse"].mean(),
                "avg_train_skill": top_n["metrics.train_skill"].mean(),
                "avg_train_rmse": top_n["metrics.train_rmse"].mean(),
                "avg_num_boost_round": top_n["metrics.num_boost_round_actual"].mean(),
            })
            print(f"\n--- {yc} (top {n_used} runs averaged) ---")
            for i, (_, row) in enumerate(top_n.iterrows(), 1):
                print(f"  #{i} {row['tags.mlflow.runName']}: val_skill={row['metrics.val_skill']:.6f}, val_rmse={row['metrics.val_rmse']:.6f}")
            print(f"  -> avg val_skill={top_n['metrics.val_skill'].mean():.6f}, avg val_rmse={top_n['metrics.val_rmse'].mean():.6f}")

        ylag_summary = pd.DataFrame(avg_per_ylag).sort_values("avg_val_skill", ascending=False)
        print("\n" + "-" * 80)
        print("Summary (configs sorted by avg val_skill over top {} runs)".format(YLAG_TOP_N_AVG))
        print("-" * 80)
        disp_cols = ["ylag_config", "n_runs", "avg_val_skill", "avg_val_rmse", "avg_train_skill", "avg_train_rmse", "avg_num_boost_round"]
        print(ylag_summary[disp_cols].to_string(index=False))
        overall = ylag_summary.iloc[0]
        print(f"\nOverall best (by avg of top {YLAG_TOP_N_AVG}): {overall['ylag_config']} (avg_val_skill={overall['avg_val_skill']:.6f})")
    else:
        print("No runs found for hedge_fund_forecasting_tune_y_lag (or experiment not found).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review MLflow experiment results.")
    parser.add_argument("--ylag-only", action="store_true", help="Only print hedge_fund_forecasting_tune_y_lag section")
    args = parser.parse_args()
    if args.ylag_only:
        _print_ylag_results()
    else:
        main()
