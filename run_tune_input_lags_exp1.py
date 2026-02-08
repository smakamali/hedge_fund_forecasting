"""
Experiment 1: Tune global_lags for xlag-only config.
Runs 6 configurations with use_global_lags=True and different global_lags.
All runs logged to MLflow experiment hedge_fund_forecasting_tune_xlag.
Usage: conda run -n forecast_fund python run_tune_input_lags_exp1.py
"""
import logging

from run_validation_lagged_features import main

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("run_validation_lagged_features").setLevel(logging.INFO)

GLOBAL_LAGS_CONFIGS = [
    [1],
    [1, 2],
    [1, 2, 3],
    [1, 2, 3, 5],
    [1, 2, 3, 5, 10],
    [1, 2, 3, 5, 10, 15],
]

RUN_NAMES = [
    "xlag_gl1",
    "xlag_gl1_2",
    "xlag_gl1_2_3",
    "xlag_gl1_2_3_5",
    "xlag_gl1_2_3_5_10",
    "xlag_gl1_2_3_5_10_15",
]


def run_all():
    results = []
    for i, (config, name) in enumerate(zip(GLOBAL_LAGS_CONFIGS, RUN_NAMES), 1):
        logging.info("=" * 60)
        logging.info("Experiment 1 %d/%d: %s", i, len(GLOBAL_LAGS_CONFIGS), name)
        logging.info("=" * 60)
        try:
            score = main(
                use_input_lags=True,
                use_target_lags=False,
                use_rolling=False,
                use_aggregates=False,
                use_global_lags=True,
                global_lags=config,
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
