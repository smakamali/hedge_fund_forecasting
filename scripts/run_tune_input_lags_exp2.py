"""
Experiment 2: Tune lags_max and top_k_per_feature for xlag-only config.
Uses only valid combinations where top_k_per_feature <= lags_max.
All runs logged to MLflow experiment hedge_fund_forecasting_tune_xlag.
Usage: from project root: python scripts/run_tune_input_lags_exp2.py
"""
import logging
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from run_validation_lagged_features import main

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("run_validation_lagged_features").setLevel(logging.INFO)

LAGS_MAX_VALUES = [15, 20]
TOP_K_VALUES = [2, 3, 4, 5, 6] # 7, 10]

CONFIGS = [
    (lags_max, top_k)
    for lags_max in LAGS_MAX_VALUES
    for top_k in TOP_K_VALUES
    if top_k <= lags_max
]


def run_all():
    results = []
    for i, (lm, tk) in enumerate(CONFIGS, 1):
        name = f"xlag_lm{lm}_tk{tk}"
        logging.info("=" * 60)
        logging.info("Experiment 2 %d/%d: %s", i, len(CONFIGS), name)
        logging.info("=" * 60)
        try:
            score = main(
                use_input_lags=True,
                use_target_lags=False,
                use_rolling=False,
                use_aggregates=False,
                lags_max=lm,
                top_k_per_feature=tk,
                use_mlflow=True,
                experiment_name="hedge_fund_forecasting_tune_xlag",
                run_name=name,
            )
            results.append((name, score, None))
            logging.info(">>> %s: val_skill=%.6f", name, score)
        except Exception as e:
            logging.exception("Experiment %s failed: %s", name, e)
            results.append((name, None, str(e)))

    logging.info("")
    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    for name, score, err in results:
        if err:
            logging.info("  %s: FAILED (%s)", name, err)
        else:
            logging.info("  %s: val_skill=%.6f", name, score)
    best = max((r for r in results if r[1] is not None), key=lambda r: r[1], default=None)
    if best:
        logging.info("Best: %s (val_skill=%.6f)", best[0], best[1])


if __name__ == "__main__":
    run_all()
