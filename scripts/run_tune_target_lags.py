"""
Target lag tuning for xlag_gl1 config.
Base: xlag_gl1 (global_lags=[1]) + target lags.
Searches over y_lags configurations, each with full LightGBM param grid.
Usage: from project root: python scripts/run_tune_target_lags.py
"""
import itertools
import logging
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import mlflow

from run_validation_lagged_features import (
    prepare_dataset_for_lag_config,
    train_and_evaluate,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("run_validation_lagged_features").setLevel(logging.INFO)

# Target lag configs to search
YLAG_CONFIGS = [
    [1],
    [1, 2],
    [1, 2, 3],
    [1, 2, 3, 4],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 10],
]

# Base: xlag_gl1 (best input lag config from tune_lgb)
BASE_PARAMS = {
    "use_input_lags": True,
    "use_target_lags": True,
    "use_rolling": False,
    "use_aggregates": False,
    "use_target_transform": False,
    "lags_max": 5,
    "top_k_per_feature": 2,
    "use_global_lags": True,
    "global_lags": [1],
}

# LightGBM param grid (same as run_tune_lgb_params)
NUM_LEAVES_VALUES = [15, 31, 50, 70]
MIN_DATA_IN_LEAF_VALUES = [20, 50, 100, 200]
MAX_DEPTH_VALUES = [5, 7, 9, -1]


def ylag_run_name(y_lags):
    """e.g. [1,2,3] -> ylag_1_2_3"""
    return "ylag_" + "_".join(map(str, y_lags))


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
    mlflow.set_experiment("hedge_fund_forecasting_tune_y_lag")
    all_results = []
    for i, y_lags in enumerate(YLAG_CONFIGS, 1):
        run_name_base = ylag_run_name(y_lags)
        params = {**BASE_PARAMS, "y_lags": y_lags}

        logging.info("=" * 60)
        logging.info("Ylag config %d/%d: %s (preparing data once)", i, len(YLAG_CONFIGS), run_name_base)
        logging.info("=" * 60)
        try:
            dataset = prepare_dataset_for_lag_config(**params)
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
                    use_skill_feval=False,
                )
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_params({
                        "use_input_lags": str(params["use_input_lags"]),
                        "use_target_lags": str(params["use_target_lags"]),
                        "y_lags": ",".join(map(str, y_lags)),
                        "use_rolling": str(params["use_rolling"]),
                        "use_aggregates": str(params["use_aggregates"]),
                        "use_global_lags": str(params["use_global_lags"]),
                        "global_lags": ",".join(map(str, params["global_lags"])),
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
    for ylag_name, best_run, skill in all_results:
        logging.info("  %s: %s (val_skill=%.6f)", ylag_name, best_run, skill)
    overall = max(all_results, key=lambda x: x[2], default=None)
    if overall:
        logging.info("Overall best: %s / %s (val_skill=%.6f)", overall[0], overall[1], overall[2])


if __name__ == "__main__":
    run_all()
