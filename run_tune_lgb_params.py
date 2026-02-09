"""
LightGBM parameter tuning for top 7 xlag configurations.
Data generation is performed once per lag config; only LightGBM params vary.
Tunes num_leaves, min_data_in_leaf, max_depth per LightGBM docs.
Usage: conda run -n forecast_fund python run_tune_lgb_params.py
"""
import itertools
import logging

import mlflow

from run_validation_lagged_features import (
    prepare_dataset_for_lag_config,
    train_and_evaluate,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("run_validation_lagged_features").setLevel(logging.INFO)

# Top 7 lag configs by val_skill from hedge_fund_forecasting_tune_xlag
TOP_7_CONFIGS = [
    {"run_name": "xlag_gl1", "use_global_lags": True, "global_lags": [1]},
    {"run_name": "xlag_gl1_2", "use_global_lags": True, "global_lags": [1, 2]},
    {"run_name": "xlag_gl1_2_3", "use_global_lags": True, "global_lags": [1, 2, 3]},
    {"run_name": "xlag_lm10_tk2", "lags_max": 10, "top_k_per_feature": 2},
    {"run_name": "xlag_lm15_tk3", "lags_max": 15, "top_k_per_feature": 3},
    {"run_name": "xlag_lm10_tk5", "lags_max": 10, "top_k_per_feature": 5, "use_float16_when_large": True},
    {"run_name": "xlag_lm10_tk3", "lags_max": 10, "top_k_per_feature": 3},
    {"run_name": "xlag_lm5_tk3", "lags_max": 5, "top_k_per_feature": 3},
]

# Base lag params for all top 7 (xlag-only)
BASE_LAG_PARAMS = {
    "use_input_lags": True,
    "use_target_lags": False,
    "use_rolling": False,
    "use_aggregates": False,
    "use_target_transform": False,
}

# Param grid: num_leaves, min_data_in_leaf, max_depth
# Constraint: num_leaves <= 2^max_depth when max_depth > 0
NUM_LEAVES_VALUES = [15, 31, 50, 70]
MIN_DATA_IN_LEAF_VALUES = [20, 50, 100, 200]
MAX_DEPTH_VALUES = [5, 7, 9, -1]


def build_param_grid():
    """Build valid (num_leaves, min_data_in_leaf, max_depth) combos."""
    grid = []
    for num_leaves, min_data_in_leaf, max_depth in itertools.product(
        NUM_LEAVES_VALUES,
        MIN_DATA_IN_LEAF_VALUES,
        MAX_DEPTH_VALUES,
    ):
        if max_depth > 0 and num_leaves > 2 ** max_depth:
            continue
        grid.append((num_leaves, min_data_in_leaf, max_depth))
    return grid


PARAM_GRID = build_param_grid()


def run_all():
    mlflow.set_experiment("hedge_fund_forecasting_tune_lgb")
    all_results = []
    for i, config in enumerate(TOP_7_CONFIGS, 1):
        run_name_base = config["run_name"]
        lag_params = {**BASE_LAG_PARAMS}
        if "use_global_lags" in config:
            lag_params["use_global_lags"] = config["use_global_lags"]
            lag_params["global_lags"] = config["global_lags"]
            lag_params["lags_max"] = 5
            lag_params["top_k_per_feature"] = 2
        else:
            lag_params["use_global_lags"] = False
            lag_params["global_lags"] = None
            lag_params["lags_max"] = config["lags_max"]
            lag_params["top_k_per_feature"] = config["top_k_per_feature"]
        if "use_float16_when_large" in config:
            lag_params["use_float16_when_large"] = config["use_float16_when_large"]

        logging.info("=" * 60)
        logging.info("Lag config %d/7: %s (preparing data once)", i, run_name_base)
        logging.info("=" * 60)
        try:
            dataset = prepare_dataset_for_lag_config(**lag_params)
        except (MemoryError, OSError) as e:
            logging.exception("Skipping %s: data prep failed (OOM or resource error): %s", run_name_base, e)
            continue

        config_results = []
        for j, (nl, md, depth) in enumerate(PARAM_GRID, 1):
            depth_str = str(depth) if depth > 0 else "unbounded"
            run_name = f"{run_name_base}_nl{nl}_md{md}_d{depth_str}"
            try:
                val_skill, train_skill, val_rmse, train_rmse, n_rounds = train_and_evaluate(
                    dataset,
                    num_leaves=nl,
                    min_data_in_leaf=md,
                    max_depth=depth if depth > 0 else -1,
                )
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_params({
                        "use_input_lags": str(lag_params["use_input_lags"]),
                        "use_target_lags": str(lag_params["use_target_lags"]),
                        "use_rolling": str(lag_params["use_rolling"]),
                        "use_aggregates": str(lag_params["use_aggregates"]),
                        "lags_max": str(lag_params["lags_max"]),
                        "top_k_per_feature": str(lag_params["top_k_per_feature"]),
                        "use_global_lags": str(lag_params["use_global_lags"]),
                        "global_lags": (
                            ",".join(map(str, lag_params["global_lags"]))
                            if lag_params.get("global_lags") else "null"
                        ),
                        "use_float16_when_large": str(lag_params.get("use_float16_when_large", False)),
                        "num_leaves": str(nl),
                        "min_data_in_leaf": str(md),
                        "max_depth": str(depth) if depth > 0 else "-1",
                    })
                    mlflow.log_metrics({
                        "val_skill": val_skill,
                        "train_skill": train_skill,
                        "val_rmse": val_rmse,
                        "train_rmse": train_rmse,
                        "num_boost_round_actual": n_rounds,
                    })
                config_results.append((run_name, val_skill, None))
                if j % 10 == 0 or j == len(PARAM_GRID):
                    logging.info("  %d/%d param combos done, best so far: %.6f", j, len(PARAM_GRID),
                                 max(r[1] for r in config_results))
            except Exception as e:
                logging.exception("Run %s failed: %s", run_name, e)
                config_results.append((run_name, None, str(e)))

        best = max((r for r in config_results if r[1] is not None), key=lambda x: x[1], default=None)
        if best:
            logging.info("Best for %s: %s (val_skill=%.6f)", run_name_base, best[0], best[1])
            all_results.append((run_name_base, best[0], best[1]))

    logging.info("")
    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    for lag_name, best_run, skill in all_results:
        logging.info("  %s: %s (val_skill=%.6f)", lag_name, best_run, skill)
    overall = max(all_results, key=lambda x: x[2], default=None)
    if overall:
        logging.info("Overall best: %s / %s (val_skill=%.6f)", overall[0], overall[1], overall[2])


if __name__ == "__main__":
    run_all()
