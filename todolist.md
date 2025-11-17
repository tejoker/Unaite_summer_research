# TODO List

## Dataset Preparation

### Simplified Two-Dataset Approach

Process only two datasets (not 105 separate runs):
1. **Golden baseline**: 4,308 timesteps × 2,890 variables (normal operation)
2. **Full test timeline**: 8,640 timesteps × 2,890 variables (all 105 anomalies)

Benefits:
- 40x faster: ~3 hours instead of 35-40 hours
- Natural multi-anomaly handling
- Can detect cascade effects
- Enables root cause analysis

### Data Preparation Script

Location: `telemanom/prepare_datasets.py`

Applies NaN handling to both datasets:
```bash
cd telemanom
python3 prepare_datasets.py
```

### Running the Pipeline

Location: `run_tucker_cam_benchmark.sh`

```bash
bash run_tucker_cam_benchmark.sh
```

Output:
- `results/golden_baseline/weights/weights_enhanced.csv`
- `results/test_timeline/weights/weights_enhanced.csv`

## NaN Handling

### Solution 2: Drop first row after differencing (IMPLEMENTED)

Location: `executable/final_pipeline/preprocessing_no_mi.py:168`

```python
# After differencing
df_diff = df_diff.iloc[1:].reset_index(drop=True)
```

This is standard practice and preserves causality.

### Additional fix needed: ffill + dropna BEFORE differencing

Current order has a problem - NaN in raw data propagate through log/diff.

Required change in preprocessing_no_mi.py around line 160:

```python
# BEFORE differencing loop
df_raw = df_raw.ffill()
df_raw = df_raw.dropna()

# Then existing differencing code
for col in df_raw.columns:
    series_log = np.log1p(df_raw[col])
    series_diff = series_log.diff()
    df_diff[f"{col}_diff"] = series_diff

df_diff = df_diff.iloc[1:]  # Solution 2
```

## Limitations

### Cannot make pipeline fully NaN-agnostic

- Causality constraint: Cannot use backward fill
- Leading NaN must be dropped
- DAG constraint prevents masked loss approach

## Anomaly Detection: Multiple Anomalies and Cascade Disambiguation

### Problem

Golden-only comparison `distance(G_t, G_golden)` cannot distinguish:
- New anomaly onset
- Cascade/remnant effects from previous anomaly
- Recovery fluctuations

Example ambiguity:
```
t=100: Anomaly 1 starts → abs_score = 0.8 (HIGH)
t=110: Cascade from anomaly 1 → abs_score = 0.6 (HIGH) - still flagged
t=130: Anomaly 2 starts → abs_score = 0.9 (HIGH) - can't tell it's NEW
```

### Solution: Dual-Metric with Trend Analysis

Compute three metrics per window:

```python
# 1. Absolute deviation from normal
abs_score[t] = distance(G_t, G_golden)

# 2. Rate of change (leverages 90% overlap)
change_score[t] = distance(G_t, G_{t-1})

# 3. Trend (getting better or worse?)
lookback = 5
abs_trend[t] = abs_score[t] - abs_score[t - lookback]
```

### Decision Logic

```python
# Thresholds (use adaptive thresholding - see below)
threshold_normal = 0.05
threshold_change = 0.15
threshold_trend = 0.1

if abs_score[t] < threshold_normal:
    status = "NORMAL"
elif change_score[t] > threshold_change and abs_trend[t] > threshold_trend:
    status = "NEW_ANOMALY_ONSET"  # Getting worse + changing
elif change_score[t] > threshold_change and abs_trend[t] < -threshold_trend:
    status = "RECOVERY_FLUCTUATION"  # Getting better + changing
else:
    status = "CASCADE_OR_PERSISTENT"  # Abnormal but stable
```

### Adaptive Thresholding (Hundman NPDT approach)

Instead of fixed thresholds, use rolling statistics from normal windows:

```python
# Maintain rolling window of recent NORMAL periods
normal_windows = [abs_score[i] for i in recent_normal_indices]

# Set threshold as mean + 3*std
threshold_normal = np.mean(normal_windows) + 3 * np.std(normal_windows)

# Same for change_score
normal_changes = [change_score[i] for i in recent_normal_indices]
threshold_change = np.mean(normal_changes) + 3 * np.std(normal_changes)
```

### Expected Outcome

This approach should distinguish:
- **New onset**: High change + positive trend (system degrading)
- **Cascade**: High abs but low change (stable-abnormal)
- **Recovery noise**: High change + negative trend (system improving)

No labels needed, only golden baseline.

## Technical Details

### Tucker-CAM Configuration (IMPLEMENTED)

The pipeline uses **Tucker-decomposed Fast CAM-DAG** for memory-efficient nonlinear causal discovery.

**Architecture:**
```
Rolling Window → DynoTEARS Framework → CAM Model (P-splines) → Tucker Decomposition
```

**Tucker Factorization:**
- W (contemporaneous): Core[r,r,r] + U1[d,r] + U2[d,r] + U3[K,r]
- A (lagged): Core[r,r,r,r] + U1[d,r] + U2[d,r] + U3[p,r] + U4[K,r]
- Memory reduction: ~4500× (875M → 192K parameters with r=20)

**Files:**
- `executable/final_pipeline/cam_model_tucker.py` - Tucker-decomposed CAM model
- `executable/final_pipeline/dynotears_tucker_cam.py` - Tucker-CAM optimizer
- `executable/final_pipeline/dbn_dynotears_tucker_cam.py` - Rolling window wrapper
- `executable/launcher.py` - Detects USE_TUCKER_CAM environment variable

**Parameters (in `run_tucker_cam_benchmark.sh`):**
- Tucker ranks: r_w=20, r_a=10
- P-splines: n_knots=5, lambda_smooth=0.01
- Option D: lambda_w=0.0, lambda_a=0.0
- Top-K: 10,000 edges/window (post-hoc sparsification)

### Rolling Windows

- Window size: 100 timesteps
- Stride: 10 timesteps (90% overlap)
- Overlap enables change detection between consecutive windows

### Dataset Statistics

- **Total variables**: 2,890 (82 channels × 25-55 features each)
- **Golden timesteps**: 4,308
- **Test timesteps**: 8,640
- **Total anomalies**: 105 (69 SMAP + 36 MSL)
- **Anomaly types**: Point (59%) and Contextual (41%)

## Root Cause Analysis

Causal graphs enable tracing anomalies to root causes via:
1. Topological ordering of the DAG
2. Identifying source nodes (no parents) with high anomaly scores
3. Tracing causal paths from root to affected variables

This is a key advantage over LSTM approaches which lack explicit causal structure.
