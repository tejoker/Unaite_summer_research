# Quick Start - Command Line Reference

## Complete Workflow (Recommended)

### One-Line Complete Workflow
```bash
./run_complete_workflow.sh data/baseline.csv data/anomaly.csv results/my_analysis
```

## Individual Components

### If You Already Have DynoTEARS Weights

```bash
# Direct anomaly detection + reconstruction
python executable/final_pipeline/end_to_end_pipeline.py \
    --baseline-weights results/baseline/weights.csv \
    --anomaly-weights results/anomaly/weights.csv \
    --original-data data/anomaly_period.csv \
    --output-dir results/reconstruction/
```

### If You Need to Run Full Pipeline

#### Step 1: Preprocess Baseline
```bash
export INPUT_CSV_FILE=data/baseline.csv
export IS_GOLDEN_DATA=true
export DYNOTEARS_WINDOW_SIZE=1000
export RESULT_DIR=results/baseline/preprocessing

python executable/final_pipeline/preprocessing.py
```

#### Step 2: Preprocess Anomaly
```bash
export INPUT_CSV_FILE=data/anomaly.csv
export IS_GOLDEN_DATA=false
export DYNOTEARS_WINDOW_SIZE=1000
export RESULT_DIR=results/anomaly/preprocessing

python executable/final_pipeline/preprocessing.py
```

#### Step 3: Run DynoTEARS (on separate GPU cluster)
```bash
# Transfer preprocessed files to cluster and run dbn_dynotears.py
# Then transfer weights back
```

#### Step 4: Detect + Reconstruct
```bash
python executable/final_pipeline/end_to_end_pipeline.py \
    --baseline-weights results/baseline/weights.csv \
    --anomaly-weights results/anomaly/weights.csv \
    --original-data data/anomaly.csv \
    --output-dir results/reconstruction/
```

## Common Variations

### Different Correction Strategies
```bash
# Conservative (zero out anomalies)
--correction-strategy zero

# Partial correction (70% baseline, 30% current)
--correction-strategy soft_correction

# Full replacement (default)
--correction-strategy replace_with_baseline
```

### Sensitivity Adjustment
```bash
# More sensitive
--correction-threshold 0.05

# Less sensitive
--correction-threshold 0.2
```

### Specify Variable Names
```bash
--variable-names Temp1 Temp2 Pressure Flow Speed Torque
```

## Output Files

All results saved to `--output-dir`:
- `anomaly_detection_results.json` - Full detection details
- `corrected_weights.csv` - Corrected weight matrix
- `reconstructed_time_series.csv` - Reconstructed series
- `original_vs_reconstructed.csv` - Comparison
- `PIPELINE_SUMMARY.txt` - Summary report

## Expected Results

### If Anomaly Detected
```
Anomaly Detected: YES
Anomaly Type: spike
Top Contributing Edges:
  1. Temp1 -> Temp3: change=0.234
  2. Press -> Flow: change=0.187

Corrected 8 edges
RMSE: 0.123
```

### If No Anomaly
```
Anomaly Detected: NO
No reconstruction needed
```

## Quick Validation

```bash
# Check if pipeline completed
cat results/reconstruction/PIPELINE_SUMMARY.txt

# View detection results
cat results/reconstruction/anomaly_detection_results.json | python -m json.tool

# Quick plot comparison (if you have matplotlib)
python -c "
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('results/reconstruction/original_vs_reconstructed.csv', index_col=0)
df.plot(figsize=(15,8))
plt.savefig('comparison.png')
"
```

## Troubleshooting

```bash
# Check logs
cat results/reconstruction/pipeline_execution.log

# Verify weights format
head -20 results/baseline/weights.csv

# Test imports
python -c "from test.anomaly_detection_suite.anomaly_detection_suite import UnifiedAnomalyDetectionSuite; print('OK')"
```
