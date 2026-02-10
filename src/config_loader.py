"""
Load and merge YAML/JSON config with defaults.
Used by validation, tuning, and submission pipeline.
Supports: input_lags, target_lags, lightgbm, data_dir.
"""

import json
import os

DEFAULT_LAGS_MAX = 5
DEFAULT_TOP_K_PER_FEATURE = 2
DEFAULT_USE_GLOBAL_LAGS = False
DEFAULT_GLOBAL_LAGS = None

DEFAULT_Y_LAGS = [1, 2, 3, 5, 10, 20]
DEFAULT_USE_TARGET_LAGS = True
DEFAULT_USE_ROLLING = True
DEFAULT_USE_AGGREGATES = True
DEFAULT_USE_TARGET_TRANSFORM = False

DEFAULT_NUM_LEAVES = 31
DEFAULT_MIN_DATA_IN_LEAF = 20
DEFAULT_MAX_DEPTH = -1
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_NUM_BOOST_ROUND = 1000
DEFAULT_EARLY_STOPPING_ROUNDS = 50


def load_config(path: str) -> dict:
    """Load config from YAML or JSON. Returns merged dict with input_lags, target_lags, lightgbm, data_dir."""
    defaults = {
        "input_lags": {
            "lags_max": DEFAULT_LAGS_MAX,
            "top_k_per_feature": DEFAULT_TOP_K_PER_FEATURE,
            "use_global_lags": DEFAULT_USE_GLOBAL_LAGS,
            "global_lags": DEFAULT_GLOBAL_LAGS,
            "use_float16_when_large": False,
        },
        "target_lags": {
            "use_target_lags": DEFAULT_USE_TARGET_LAGS,
            "y_lags": DEFAULT_Y_LAGS,
            "use_rolling": DEFAULT_USE_ROLLING,
            "use_aggregates": DEFAULT_USE_AGGREGATES,
            "use_target_transform": DEFAULT_USE_TARGET_TRANSFORM,
        },
        "lightgbm": {
            "num_leaves": DEFAULT_NUM_LEAVES,
            "min_data_in_leaf": DEFAULT_MIN_DATA_IN_LEAF,
            "max_depth": DEFAULT_MAX_DEPTH,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "seed": 42,
            "num_boost_round": DEFAULT_NUM_BOOST_ROUND,
            "early_stopping_rounds": DEFAULT_EARLY_STOPPING_ROUNDS,
        },
        "data_dir": "data",
    }
    if not os.path.isfile(path):
        return defaults
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    try:
        if path.endswith(".yaml") or path.endswith(".yml"):
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                raise ImportError(
                    "YAML config requires pyyaml. Install with: pip install pyyaml. Or use .json config."
                )
        else:
            data = json.loads(content)
        if data is None:
            return defaults

        # input_lags
        il = data.get("input_lags", {})
        merged_il = defaults["input_lags"].copy()
        if "lags_max" in il:
            merged_il["lags_max"] = int(il["lags_max"])
        if "top_k_per_feature" in il:
            merged_il["top_k_per_feature"] = int(il["top_k_per_feature"])
        if "use_global_lags" in il:
            merged_il["use_global_lags"] = bool(il["use_global_lags"])
        if "global_lags" in il:
            gl = il["global_lags"]
            merged_il["global_lags"] = gl if gl is None else [int(x) for x in gl]
        if "use_float16_when_large" in il:
            merged_il["use_float16_when_large"] = bool(il["use_float16_when_large"])

        # target_lags
        tl = data.get("target_lags", {})
        merged_tl = defaults["target_lags"].copy()
        if "use_target_lags" in tl:
            merged_tl["use_target_lags"] = bool(tl["use_target_lags"])
        if "y_lags" in tl:
            merged_tl["y_lags"] = [int(x) for x in tl["y_lags"]]
        if "use_rolling" in tl:
            merged_tl["use_rolling"] = bool(tl["use_rolling"])
        if "use_aggregates" in tl:
            merged_tl["use_aggregates"] = bool(tl["use_aggregates"])
        if "use_target_transform" in tl:
            merged_tl["use_target_transform"] = bool(tl["use_target_transform"])

        # lightgbm
        lb = data.get("lightgbm", {})
        merged_lb = defaults["lightgbm"].copy()
        for key in ["num_leaves", "min_data_in_leaf", "max_depth", "num_boost_round", "early_stopping_rounds"]:
            if key in lb:
                merged_lb[key] = int(lb[key])
        for key in ["learning_rate", "feature_fraction", "bagging_fraction", "lambda_l1", "lambda_l2"]:
            if key in lb:
                merged_lb[key] = float(lb[key])
        if "bagging_freq" in lb:
            merged_lb["bagging_freq"] = int(lb["bagging_freq"])
        if "seed" in lb:
            merged_lb["seed"] = int(lb["seed"])

        # data_dir
        data_dir = data.get("data_dir", defaults["data_dir"])
        if isinstance(data_dir, str):
            data_dir = str(data_dir).strip() or "data"

        return {
            "input_lags": merged_il,
            "target_lags": merged_tl,
            "lightgbm": merged_lb,
            "data_dir": data_dir,
        }
    except Exception as e:
        import logging
        logging.warning("Failed to parse config %s: %s. Using defaults.", path, e)
        return defaults
