"""
Shared constants for entity columns, target, time index, and default lags/windows.
Re-exports from preprocessing for a single place scripts can import from.
"""

from .preprocessing import (
    FEATURE_COLS,
    TYPE_C_FEATURES,
    ZERO_INFLATED_FEATURES,
    ENTITY_CATEGORICAL_COLS,
)

ENTITY_COLS = ["code", "sub_code", "sub_category", "horizon"]
TS_COL = "ts_index"
TARGET_COL = "y_target"
Y_LAGS = [1, 2, 3, 5, 10, 20]
WINDOWS = [5, 20]

__all__ = [
    "ENTITY_COLS",
    "TS_COL",
    "TARGET_COL",
    "Y_LAGS",
    "WINDOWS",
    "FEATURE_COLS",
    "TYPE_C_FEATURES",
    "ZERO_INFLATED_FEATURES",
    "ENTITY_CATEGORICAL_COLS",
]
