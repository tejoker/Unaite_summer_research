# Telemanom Anomaly Analysis Pipeline

Tucker-CAM causal discovery analysis on NASA Telemanom spacecraft anomaly dataset.

## Overview

This pipeline analyzes causal relationships in spacecraft telemetry data:
- **Golden baseline**: 4308 timesteps of clean data from multiple spacecraft
- **105 anomalies**: Individual point and contextual anomalies from SMAP/MSL missions

## Quick Start

### Run Complete Pipeline (35-40 hours)

```bash
# On remote server
ssh nicolas_b@172.30.100.108
cd ~/program_internship_paul_wurth

# Run full pipeline (extraction + Golden + 105 anomalies)
nohup bash run_telemanom_pipeline.sh > pipeline.log 2>&1 &

# Monitor progress
tail -f pipeline.log

# Check completion status (should reach 105)
find results/anomaly_signatures -name 'weights_enhanced.csv' | wc -l
```

### Pipeline Stages

1. **Extract anomalies** (30 seconds)
   - Re-extracts 105 anomalies with correct class labels (point/contextual)
   - Output: `telemanom/individual_anomalies/anomaly_*.csv`

2. **Golden baseline** (~20 minutes)
   - Preprocessing + Tucker-CAM on clean data
   - Output: `results/telemanom_golden_baseline/weights/weights_enhanced.csv`

3. **All 105 anomalies** (~35-38 hours)
   - Preprocessing + Tucker-CAM on each anomaly
   - Output: `results/anomaly_signatures/anomaly_*/weights/weights_enhanced.csv`

### After Pipeline Completes

```bash
# Build comparison catalog
python3 build_anomaly_signature_catalog.py

# Generate analysis plots
python3 analyze_anomaly_signatures.py

# Sync results to local machine
rsync -avz -e "ssh -p 22" nicolas_b@172.30.100.108:/home/nicolas_b/program_internship_paul_wurth/results/ ~/program_internship_paul_wurth/results/
```

## Project Structure

```
├── run_telemanom_pipeline.sh       # Main pipeline script
├── extract_individual_anomalies.py # Anomaly extraction
├── build_anomaly_signature_catalog.py
├── analyze_anomaly_signatures.py
├── executable/
│   ├── launcher.py                 # Tucker-CAM launcher
│   └── final_pipeline/             # Preprocessing + Tucker-CAM
├── telemanom/
│   ├── golden_period_dataset_clean.csv
│   ├── test_dataset_merged_clean.csv
│   ├── labeled_anomalies.csv
│   └── individual_anomalies/       # 105 extracted anomalies
├── results/
│   ├── telemanom_golden_baseline/
│   └── anomaly_signatures/
└── logs/
```

## Requirements

- Python 3.10+
- PyTorch
- pandas, numpy, scipy
- statsmodels

## Notes

- Pipeline has **resume capability**: re-running skips completed anomalies
- Average processing time: ~20 minutes per anomaly
- GPU recommended but not required (CPU fallback available)

## Method Overview

### Tucker-CAM Pipeline

1. **Preprocessing**: Forward-fill → differencing → standardization
2. **Causal Discovery**: Rolling windows + Tucker decomposition + DynoTEARS
3. **Sparsification**: Post-hoc Top-K=10,000 edges per window
4. **Analysis**: Compare Golden vs Anomaly causal patterns

### Key Features

- Scalable to 2,890 variables (NASA Telemanom)
- Temporal causal structure evolution detection
- Stable edge counts (CV=0%) via Top-K selection
- Resume capability for long-running analyses
