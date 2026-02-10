# Code Review: Submission Pipeline - Issues and Fixes

**File:** `scripts/run_submission_pipeline.py`  
**Date:** 2026-02-09  
**Status:** Issues identified, fixes required before production use

---

## Critical Issues (High Priority)

### Issue #2: Entity Statistics Computed on Validation Set (Information Leakage)

**Severity:** HIGH - Validation metrics will be optimistically biased

**Location:**
- Step 1: Line 559 (within `entity_mean_from_train` computation)
- Step 2: Lines 559-561
- Step 3: Lines 758-760

**Problem:**
Entity-level statistics (mean, std) are computed from the FULL training dataset, including the temporal validation portion. When validating models (e.g., Step 1's 80/10 split), the validation set predictions use entity statistics that were computed including the validation period itself.

**Code Example (Current - Step 2):**
```python
# Line 559-561 - Leaks validation information
entity_mean_from_train = train_imputed.groupby(ENTITY_COLS)[TARGET_COL].mean().reset_index(name="entity_mean")
entity_std_from_train = train_imputed.groupby(ENTITY_COLS)[TARGET_COL].std().reset_index(name="entity_std")
entity_std_from_train["entity_std"] = entity_std_from_train["entity_std"].fillna(global_std)
```

**Impact on Generalization:**
Validation performance will be optimistically biased. The model may appear to generalize better than it actually does, leading to poor performance on truly unseen test data. This affects model selection and hyperparameter tuning decisions.

**Recommended Fix:**

For Step 1 (in `_build_train_features` or before validation):
```python
# Compute entity stats ONLY from training portion (not validation)
entity_mean_from_train = train_part.groupby(ENTITY_COLS)[TARGET_COL].mean().reset_index(name="entity_mean")
entity_std_from_train = train_part.groupby(ENTITY_COLS)[TARGET_COL].std().reset_index(name="entity_std")
entity_std_from_train["entity_std"] = entity_std_from_train["entity_std"].fillna(global_std)

# Store in artifacts for validation use
artifacts["entity_mean_from_train"] = entity_mean_from_train
artifacts["entity_std_from_train"] = entity_std_from_train
```

For Steps 2 & 3, when doing 90/10 temporal split:
```python
# Compute entity stats ONLY from the 90% training split
train_90_part = train_imputed.loc[mask_90]
entity_mean_from_train = train_90_part.groupby(ENTITY_COLS)[TARGET_COL].mean().reset_index(name="entity_mean")
entity_std_from_train = train_90_part.groupby(ENTITY_COLS)[TARGET_COL].std().reset_index(name="entity_std")
entity_std_from_train["entity_std"] = entity_std_from_train["entity_std"].fillna(global_std)

# For final model on full train, recompute with all data
```

---

### Issue #3: Step 2 Noise Injection Target Space Mismatch

**Severity:** HIGH - Model trained on incorrectly scaled noise

**Location:** Lines 515-523

**Problem:**
1. `errors_val` from Step 1 (line 443) are computed in **ORIGINAL target space**: `errors_val = y_val - pred_val`
2. In Step 2, if `use_target_transform=True`, the target is transformed (line 517)
3. Then **original-space errors** are added to **transformed-space targets** (line 523)

**Code Example (Current - Incorrect):**
```python
# Line 515-523
target_transform = fit_target_transform(train_df[TARGET_COL].to_numpy()) if tl["use_target_transform"] else None
if target_transform:
    train_imputed[TARGET_COL] = transform_target(train_imputed[TARGET_COL].to_numpy(), target_transform)

# Noise is in original space, but target is now in transformed space!
np.random.seed(lb.get("seed", 42))
noise = np.random.choice(errors_val, size=len(train_imputed), replace=True)
train_imputed["y_noisy"] = train_imputed[TARGET_COL].values + noise  # WRONG: mixing spaces
```

**Impact on Generalization:**
The noise distribution doesn't match the actual error distribution in the transformed space. The model trains on incorrectly scaled noise, making it poorly calibrated for test-time error propagation.

**Recommended Fix (Option 1 - Transform errors):**
```python
# In Step 1, save errors in BOTH spaces
# Line 443 (Step 1 - modify):
errors_val = y_val - pred_val
errors_val_transformed = errors_val  # original space
if target_transform is not None:
    # Also compute errors in transformed space
    y_val_t = transform_target(y_val, target_transform)
    pred_val_t = transform_target(pred_val, target_transform)
    errors_val_transformed = y_val_t - pred_val_t

# Save both versions
np.save(os.path.join(out_dir, "validation_errors.npy"), errors_val)
np.save(os.path.join(out_dir, "validation_errors_transformed.npy"), errors_val_transformed)

# In Step 2 - use appropriate error space:
target_transform = fit_target_transform(train_df[TARGET_COL].to_numpy()) if tl["use_target_transform"] else None
if target_transform:
    train_imputed[TARGET_COL] = transform_target(train_imputed[TARGET_COL].to_numpy(), target_transform)
    # Use transformed errors
    errors_path = os.path.join(out_dir, "validation_errors_transformed.npy")
else:
    # Use original errors
    errors_path = os.path.join(out_dir, "validation_errors.npy")

errors_val = np.load(errors_path)
np.random.seed(lb.get("seed", 42))
noise = np.random.choice(errors_val, size=len(train_imputed), replace=True)
train_imputed["y_noisy"] = train_imputed[TARGET_COL].values + noise
```

**Recommended Fix (Option 2 - Simpler):**
```python
# Disable target transform for Step 2 only
# In _run_step2, line 515:
target_transform = None  # Force no transform for noise-robust model
# This ensures errors and targets are in same space
```

---

### Issue #4: Inconsistent Preprocessing Order - Input Feature Scaling

**Severity:** HIGH - Train/test distribution mismatch

**Location:**
- Training (Steps 2 & 3): Lines 543-547, 741-745
- Test prediction (Step 3): Line 871

**Problem:**
In training, min-max scaling is applied BEFORE creating input lag features. In Step 3 test-time (line 871), scaling happens AFTER input lags are created, causing a mismatch.

**Code Example (Current - Incorrect):**
```python
# TRAINING (Step 3, lines 741-746) - CORRECT ORDER:
if il.get("use_float16_when_large", False):
    scale_cols = [c for c in FEATURE_COLS if c in train_imputed.columns and pd.api.types.is_numeric_dtype(train_imputed[c])]
    if scale_cols:
        input_feature_min_max_ap = fit_min_max_bounds(train_imputed, scale_cols)
        train_imputed = apply_min_max_scale(train_imputed, input_feature_min_max_ap)  # BEFORE lag creation
train_imputed, input_lag_cols = create_input_lag_features(...)  # Line 746

# TEST (Step 3, lines 870-884) - WRONG ORDER:
if input_feature_min_max_ap:
    test_base = apply_min_max_scale(test_base, input_feature_min_max_ap)  # Line 871 - AFTER lags!
# But input lags were already created on lines 874-884 BEFORE this scaling
```

**Impact on Generalization:**
Input lag features will have different scales between train and test, potentially degrading model performance significantly.

**Recommended Fix:**
```python
# In Step 3 test prediction, move scaling BEFORE input lag creation:

# Lines 870-884 - REORDER:
# 1. Scale base features FIRST (if needed)
if input_feature_min_max_ap:
    # Scale only the base FEATURE_COLS, not lag features
    scale_cols = [c for c in FEATURE_COLS if c in test_base.columns]
    test_base[scale_cols] = apply_min_max_scale(test_base[scale_cols], input_feature_min_max_ap)

# 2. THEN create input lags from scaled features
max_lag = _max_lag_from_spec(lag_spec)
train_tail = train_imputed.groupby(ENTITY_COLS, group_keys=False).tail(max_lag)[ENTITY_COLS + [TS_COL] + [c for c in FEATURE_COLS if c in train_imputed.columns]].copy()
train_tail["_ord"] = -1
test_input = test_base[ENTITY_COLS + [TS_COL] + [c for c in FEATURE_COLS if c in test_base.columns]].copy()
test_input["_ord"] = np.arange(len(test_input))
combined = pd.concat([train_tail, test_input], ignore_index=True)
combined = combined.sort_values(ENTITY_COLS + [TS_COL])
combined, _ = create_input_lag_features(combined, ENTITY_COLS, TS_COL, FEATURE_COLS, lag_spec, use_float16_when_large=il.get("use_float16_when_large", False))
test_mask = combined["_ord"] >= 0
test_input_lags = combined.loc[test_mask].sort_values("_ord")[input_lag_cols]
for c in input_lag_cols:
    test_base[c] = test_input_lags[c].values
```

---

## Major Issues (Medium-High Priority)

### Issue #5: Missing Input Lag Preprocessing in Step 3 Model B Training

**Severity:** MEDIUM-HIGH - Distribution shift between train/test for Model B

**Location:** Lines 835-849 (Step 3 test-time Model B preparation)

**Problem:**
When preparing test data for Model B, input lags are created from features that may have different preprocessing than during Model B training. Need to verify that Model B's training data went through the same preprocessing pipeline.

**Code Example (Current):**
```python
# Lines 835-849 - Model B test preparation
test_base = apply_imputation(test_df, impute_values)
test_base, _ = create_missing_indicators(test_base, FEATURE_COLS, missing_threshold=0.01)
test_base = apply_winsorize_bounds(test_base, winsor_bounds)
# ... more preprocessing ...
# Then create input lags for Model B (line 843)
combined_b, _ = create_input_lag_features(combined_b, ENTITY_COLS, TS_COL, FEATURE_COLS, artifacts_b["lag_spec"], ...)
```

**Issue:**
Model B was trained with `_build_train_features` (lines 659-671), which may apply different imputation/winsorization than what's used here. The preprocessing artifacts come from Model A' training, not Model B training.

**Recommended Fix:**
```python
# Save Model B's preprocessing artifacts separately during training
artifacts_b = {
    "impute_values": impute_values_b,  # From Model B's training data
    "indicator_cols": indicator_cols_b,
    "winsor_bounds": winsor_bounds_b,
    "entity_encodings": entity_encodings_b,
    # ... other B-specific artifacts
}

# At test time, use Model B's artifacts for preprocessing before Model B prediction:
test_base_b = apply_imputation(test_df, artifacts_b["impute_values"])
test_base_b, _ = create_missing_indicators(test_base_b, FEATURE_COLS, missing_threshold=0.01)
test_base_b = apply_winsorize_bounds(test_base_b, artifacts_b["winsor_bounds"])
# ... continue with Model B's preprocessing pipeline
```

---

### Issue #6: Horizon Interaction Feature Inconsistency in Step 3

**Severity:** MEDIUM - Potential train/test mismatch for Model B

**Location:** Line 849 vs Line 886

**Problem:**
The `horizon_x_subcat` feature is computed differently for Model B test input (line 849) vs Model A' test input (line 886). Need to verify what value was used during Model B training.

**Code Example (Current):**
```python
# Line 849 - Model B test input:
test_base["horizon_x_subcat"] = test_base["horizon_numeric"] * global_mean  # Uses global_mean

# Line 886 - Model A' test input:
test_base["horizon_x_subcat"] = test_base["horizon_numeric"] * test_base["y_target_sub_category_mean"]  # Uses subcat mean
```

**Question:**
During Model B training (lines 659-671), `use_aggregates=False` is set, but the code may still create `horizon_x_subcat`. What value was used then?

**Recommended Fix:**
```python
# Option 1: Verify Model B training code and ensure consistency
# In _build_train_features when use_aggregates=False:
# Check if horizon_x_subcat is created and with what value

# Option 2: Explicitly handle in test code
# Line 849 - Match whatever Model B training used:
if "horizon_x_subcat" in feature_cols_b:
    # Model B was trained with this feature - need to match training computation
    # If Model B used global_mean during training, use it here:
    test_base["horizon_x_subcat"] = test_base["horizon_numeric"] * global_mean
else:
    # Model B doesn't use this feature - no need to compute
    pass
```

---

### Issue #7: Target Transform Inconsistency Between Steps

**Severity:** MEDIUM - Predictions in different scales if ensembling

**Location:** 
- Step 1: Uses `tl["use_target_transform"]` from config
- Step 2: Uses `tl["use_target_transform"]` from config
- Step 3: Hardcoded `use_target_transform=False` (line 665)

**Problem:**
If the config specifies `use_target_transform=True`, Steps 1 and 2 will use transformed targets, but Step 3 won't. This creates inconsistency if you were to ensemble predictions across steps.

**Code Example (Current):**
```python
# Step 1, line 386-387:
use_target_transform=tl["use_target_transform"],  # From config

# Step 2, line 515:
target_transform = fit_target_transform(train_df[TARGET_COL].to_numpy()) if tl["use_target_transform"] else None

# Step 3, line 665:
use_target_transform=False,  # HARDCODED!
```

**Impact:**
If you ensemble v1, v2, v3 predictions, they'll be in different scales if target transform is enabled.

**Recommended Fix:**
```python
# Option 1: Make Step 3 consistent with config
# Line 665:
use_target_transform=tl["use_target_transform"],  # Use config value

# And line 769:
"target_transform": artifacts_ap.get("target_transform"),  # Propagate correctly

# Option 2: Document that steps are independent and shouldn't be ensembled
# Add comment:
# NOTE: Step 3 uses no target transform regardless of config, as it trains on 
# pseudo-targets from Model B. Do not ensemble with Steps 1-2 predictions.
```

---

## Minor Issues (Lower Priority)

### Issue #8: Memory Inefficiency - Full Training Data Multiple Times

**Severity:** LOW - Performance issue, not generalization issue

**Location:** Steps 1-3, multiple locations

**Problem:**
Each step reloads and fully processes the training data independently. For large datasets, this is memory-intensive and slow.

**Impact:**
Not a generalization issue, but could cause OOM errors or slow execution on large datasets.

**Recommended Fix:**
```python
# Consider refactoring to share preprocessing:
def _load_and_preprocess_train(train_path, cfg):
    """Shared preprocessing for all steps."""
    train_df = pd.read_parquet(train_path)
    # Common preprocessing
    return train_df

# Or add caching:
@functools.lru_cache(maxsize=1)
def _load_train_cached(train_path):
    return pd.read_parquet(train_path)
```

---

### Issue #9: Hardcoded Constants in Sequential Prediction

**Severity:** LOW - Potential train/test inconsistency

**Location:** Line 282 (`_sequential_predict`)

**Problem:**
Missing indicator threshold is hardcoded to `0.01` in `_sequential_predict` but should come from artifacts to ensure train/test consistency.

**Code Example (Current):**
```python
# Line 282:
df_work, _ = create_missing_indicators(df_work, FEATURE_COLS, missing_threshold=0.01)  # HARDCODED
```

**Recommended Fix:**
```python
# Add to artifacts during training:
artifacts["missing_threshold"] = 0.01  # Or from config

# In _sequential_predict:
missing_threshold = artifacts.get("missing_threshold", 0.01)
df_work, _ = create_missing_indicators(df_work, FEATURE_COLS, missing_threshold=missing_threshold)
```

---

### Issue #10: No Validation of Feature Alignment

**Severity:** LOW - Silent failures possible

**Location:** Lines 317-319, 404-406, 887-889

**Problem:**
Missing features are filled with 0.0 without warning. If feature engineering produces different columns between train and test (e.g., due to missing entities or edge cases), the model silently gets zeros.

**Code Example (Current):**
```python
# Lines 317-319:
for c in feature_cols_final:
    if c not in block_work.columns:
        block_work[c] = 0.0  # Silent fill
```

**Impact:**
Silent distribution shift. The model will underperform but you won't know why.

**Recommended Fix:**
```python
# Add validation and warnings:
missing_features = [c for c in feature_cols_final if c not in block_work.columns]
if missing_features:
    print(f"WARNING: {len(missing_features)} features missing in test data, filling with 0.0:")
    print(f"  {missing_features[:10]}")  # Show first 10
    if len(missing_features) > 10:
        print(f"  ... and {len(missing_features) - 10} more")
    for c in missing_features:
        block_work[c] = 0.0

# Or add assertions for critical features:
critical_features = [c for c in ENTITY_CAT_FEATURE_NAMES if c in feature_cols_final]
missing_critical = [c for c in critical_features if c not in block_work.columns]
assert len(missing_critical) == 0, f"Critical features missing: {missing_critical}"
```

---

## Summary and Priority

**Must fix before production:**
1. ✅ Issue #2: Entity statistics validation leakage
2. ✅ Issue #3: Step 2 noise injection space mismatch
3. ✅ Issue #4: Preprocessing order consistency

**Should fix for robustness:**
4. ✅ Issue #5: Model B preprocessing alignment
5. ✅ Issue #6: Horizon feature consistency check
6. ✅ Issue #7: Target transform consistency

**Nice to have:**
7. ✅ Issue #8: Memory optimization
8. ✅ Issue #9: Hardcoded constants in artifacts
9. ✅ Issue #10: Feature alignment validation

---

## Testing Recommendations

1. **Unit tests for preprocessing consistency:**
   - Verify same features generated in train vs test
   - Verify feature values in same scale

2. **Integration tests:**
   - Small synthetic dataset with known properties
   - Verify sequential prediction maintains temporal causality

3. **Validation:**
   - Compare validation scores with/without fixes
   - Expect validation scores to decrease slightly (less optimistic) after fixing leakage

---

**Next Steps:**
1. Fix critical issues (#2, #3, #4)
2. Run validation to verify fixes don't break functionality
3. Compare metrics before/after fixes
4. Address medium-priority issues
5. Add unit tests for consistency checks
