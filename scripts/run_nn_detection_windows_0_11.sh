#!/bin/bash
# Use ONLY windows 0-11 as golden baseline (non-collapsed weights)
cd ~/program_internship_paul_wurth

# First, extract windows 0-11 from golden baseline
python3 << 'PYEOF'
import polars as pl

df_golden = pl.read_csv('results/golden_baseline/weights/weights_enhanced.csv')
df_golden_early = df_golden.filter(pl.col('window_idx') < 12)
df_golden_early.write_csv('results/golden_baseline_windows_0_11.csv')
print(f"Extracted {df_golden_early.height} rows from windows 0-11")
PYEOF

# Now run detection with these 12 windows only
python executable/chunked_nn_detector.py \
  --golden results/golden_baseline_windows_0_11.csv \
  --test results/test_timeline/weights/weights_enhanced.csv \
  --output results/nn_detection_windows_0_11.csv \
  --chunk-size 50 \
  --sample-rate 1 \
  --lag 0 \
  2>&1 | tee logs/nn_detection_windows_0_11.log

echo "Done! Check results/nn_detection_windows_0_11.csv"
