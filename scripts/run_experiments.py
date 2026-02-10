"""
Run validation experiments with different feature-flag combinations sequentially.
Each run is tracked as a separate MLflow experiment run.
Usage: from project root: python scripts/run_experiments.py
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

# Configure logging: INFO for experiment progress, suppress verbose DEBUG from validation
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("run_validation_lagged_features").setLevel(logging.INFO)

# Experiment configurations: (name, use_input_lags, use_target_lags, use_rolling, use_aggregates, use_target_transform)
EXPERIMENTS = [
    ("base", False, False, False, False, False),
    ("ylag", False, True, False, False, False),
    ("roll", False, False, True, False, False),
    ("agg", False, False, False, True, False),
    ("xlag", True, False, False, False, False),
    ("ylag+roll", False, True, True, False, False),
    ("ylag+agg", False, True, False, True, False),
    ("ylag+xlag", True, True, False, False, False),
    ("roll+agg", False, False, True, True, False),
    ("ylag+roll+agg", False, True, True, True, False),
    ("ylag+roll+agg+xlag", True, True, True, True, False),
    ("full+trans", True, True, True, True, True),
]


def run_all():
    results = []
    for i, (name, use_input_lags, use_target_lags, use_rolling, use_aggregates, use_target_transform) in enumerate(
        EXPERIMENTS, 1
    ):
        logging.info("=" * 60)
        logging.info("Experiment %d/%d: %s", i, len(EXPERIMENTS), name)
        logging.info("=" * 60)
        try:
            score = main(
                use_input_lags=use_input_lags,
                use_target_lags=use_target_lags,
                use_rolling=use_rolling,
                use_aggregates=use_aggregates,
                use_target_transform=use_target_transform,
                use_mlflow=True,
            )
            results.append((name, score, None))
            logging.info(">>> %s: val_skill=%.6f", name, score)
        except Exception as e:
            logging.exception("Experiment %s failed: %s", name, e)
            results.append((name, None, str(e)))

    # Summary
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
