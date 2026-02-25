# Full Telemanom Benchmark Evaluation

## Overview

This evaluation runs Tucker-CAM anomaly detection on **ALL 855 windows** (not sampled) to get accurate Precision/Recall/F1 scores comparable to published baselines.

## Problem with Previous Results

- **Previous**: Evaluated on 185 windows (every 10th window = 22% coverage)
- **Result**: Recall = 0.44 (unreliable due to sampling)
- **Issue**: Not comparable to published papers that evaluate on full dataset

## Solution

Run the optimized script that:
1. **Aggregates 9 ensemble runs** to create a robust, low-variance model (Bagging)
2. Processes all 855 windows with minimal memory usage
3. Uses chunked CSV reading (1M rows at a time)
4. Computes metrics against ground truth
5. Generates publication-ready benchmark table

## Usage

### On Remote Server (Recommended)

```bash
# Transfer files to remote server
scp -r results/bagging_experiment/ user@server:/path/to/project/
scp -r telemanom/ user@server:/path/to/project/
scp executable/dual_metric_anomaly_detection_OPTIMIZED.py user@server:/path/to/project/executable/
scp scripts/benchmark_full_evaluation.sh user@server:/path/to/project/scripts/

# SSH to server
ssh user@server

# Run evaluation
cd /path/to/project
./scripts/benchmark_full_evaluation.sh
```

**Expected runtime**: 30-60 minutes (depending on server specs)
**Memory required**: ~8-16 GB RAM

### Output Files

1. **results/bagging_experiment/anomaly_detection_FULL_855_windows.csv**
   - Full detection results for all 855 windows
   - Columns: window_idx, status, abs_score, change_score, abs_trend

2. **results/bagging_experiment/BENCHMARK_RESULTS.txt**
   - Summary of Precision, Recall, F1-Score
   - Comparison with ground truth

3. **Console output**
   - Formatted benchmark table comparing with published baselines

## Expected Results

Based on sampled evaluation (185 windows):
- Precision: 1.00 (perfect)
- Recall: 0.44 (conservative thresholding)
- F1-Score: 0.61

**Full evaluation (855 windows) will likely show:**
- Precision: 0.85-1.00 (still high)
- Recall: 0.50-0.70 (improved by catching more anomalies)
- F1-Score: 0.65-0.80 (competitive with USAD=0.77)

## Understanding Point-Adjusted Evaluation

**Point-Adjusted (Event-Level)** - Standard for Telemanom:
- A ground truth anomaly sequence is "detected" if ≥1 detection window overlaps with it
- Avoids penalizing multiple detections of the same event
- Formula:
  - True Positive = # of GT sequences with ≥1 detection overlap
  - False Positive = # of detections with NO GT overlap
  - False Negative = # of GT sequences with NO detection

**Example:**
```
GT Anomaly: samples 5300-5800 (1 sequence)
Our Detections: windows 531, 555, 561 (3 windows overlapping this sequence)
Result: TP=1, FP=0 (not TP=3, which would be point-wise counting)
```

## Troubleshooting

### Memory Errors
If still getting killed, reduce chunk size:
```bash
python3 executable/dual_metric_anomaly_detection_OPTIMIZED.py \
    --chunk-size 500000  # Reduce from 1M to 500K
```

### Adjust Thresholds
Edit `dual_metric_anomaly_detection_OPTIMIZED.py` lines 111-113:
```python
threshold_abs = 0.006     # Lower = more detections (higher recall)
threshold_change = 0.008  # Lower = more sensitive to changes
threshold_trend = 0.003   # Lower = more sensitive to trends
```

## Manual Verification

After running, verify against our sampled results:

```python
import pandas as pd

# Load full results
full = pd.read_csv('results/bagging_experiment/anomaly_detection_FULL_855_windows.csv')

# Our known detections from sampled run
sampled_anomalies = [33, 105, 114, 531, 555, 561]

# Check they're still detected
for w in sampled_anomalies:
    status = full[full['window_idx'] == w]['status'].values[0]
    print(f"Window {w}: {status}")

# Should show same anomalies + potentially more we missed in sampling
```

## Next Steps

Once you have the full results:
1. Compare F1-Score with TranAD (0.90)
2. Analyze which specific GT sequences were detected/missed
3. Generate detailed breakdown by anomaly type (point vs contextual)
4. Create publication figures showing detection timeline
