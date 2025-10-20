# Telemanom Anomaly Detection via Causal Graph Changes

Detects anomalies in multivariate time series by learning Dynamic Bayesian Networks (DBNs) and tracking structural changes in the causal graph. Uses DynoTEARS for causal discovery and a 4-metric ensemble for anomaly classification.

This was developed for the Telemanom dataset (NASA spacecraft telemetry) but works on any multivariate time series where anomalies manifest as changes in causal structure.

## What this does

Given a baseline period (normal behavior) and test windows:

1. **Learn causal graphs** using DynoTEARS (a continuous optimization method for structure learning in DBNs)
2. **Compare graphs** between baseline and test windows across 4 metrics:
   - **SHD** (Structural Hamming Distance): counts edge differences
   - **Frobenius norm**: measures weight magnitude changes  
   - **Spectral radius**: detects stability shifts
   - **Max edge weight**: flags individual strong changes
3. **Detect anomalies** when metrics exceed learned thresholds (voting ensemble)
4. **Classify type**: spike, drift, level shift, trend change, amplitude change
5. **Root cause analysis**: identifies which variables/edges changed


## Quick start

```bash
# Run on Telemanom golden baseline + one anomaly file
./run_end_to_end_pipeline.sh

# Or manually:
python executable/launcher.py \
    --baseline data/Golden/golden_period_dataset_mean_channel.csv \
    --test data/Anomaly/telemanom/isolated_anomaly_001_P-1_seq1.csv \
    --output results/my_run
```

## How it works

### Pipeline stages

**1. Preprocessing** (`preprocessing_no_mi.py`)
- Makes series stationary (ADF/KPSS tests, differencing if needed)
- Finds optimal lag structure via AutoReg AIC
- No mutual information filtering (removed for simplicity)

**2. Causal discovery** (`dbn_dynotears.py`)
- Learns DBN structure using DynoTEARS (Zheng et al. 2020)
- Estimates contemporaneous and lagged effects
- Enforces acyclicity via augmented Lagrangian
- Outputs: adjacency matrices `W` (contemporaneous) and `A` (lagged)

**3. Graph comparison** (`binary_detection_metrics.py`)
- Computes 4 metrics between baseline and test graphs:
  - SHD: structural differences (edges added/removed)
  - Frobenius: sum of squared weight changes
  - Spectral: largest eigenvalue change
  - MaxEdge: maximum single edge change
- Returns binary detection (anomaly/normal) per metric

**4. Anomaly classification** (`anomaly_classification.py`)
- If detected, classifies into 6 types based on temporal patterns
- Uses heuristics on metric trajectories across windows

**5. Root cause** (`root_cause_analysis.py`)
- Identifies which edges changed most
- Ranks variables by contribution to anomaly score

### Detection strategy

We use a **voting ensemble**: if ≥2 of the 4 metrics flag anomaly, we declare it.

Thresholds are learned from golden baseline by fitting each metric's distribution (μ + k·σ, where k is tuned for desired sensitivity).



## Repository layout

```
executable/
├── final_pipeline/
│   ├── preprocessing_no_mi.py       # Stationarity, differencing, lag selection
│   ├── dbn_dynotears.py             # DBN structure learning (main algorithm)
│   ├── dynotears.py                 # Core DynoTEARS optimization
│   ├── structuremodel.py            # Neural net for structure learning
│   ├── transformers.py              # Data transformations
│   └── window_by_window_detection.py # Sliding window controller
├── test/
│   └── anomaly_detection_suite/
│       ├── binary_detection_metrics.py    # 4-metric computation
│       ├── anomaly_detection_suite.py     # Ensemble voting
│       ├── anomaly_classification.py      # Type classification
│       └── root_cause_analysis.py         # Edge importance ranking
├── launcher.py                      # Main orchestrator
└── causal_discovery_benchmark.py    # Performance testing

config/
├── default.yaml                     # Hyperparameters (lambda, thresholds, etc.)
└── config_manager.py                # Config loading

scripts/                             # Quick runners for common tasks
analysis/                            # Visualization and reporting tools
```

Files are organized by function: preprocessing → discovery → detection → classification → analysis.

## Setup

Requires Python 3.9+ and PyTorch. No GPU needed (runs fine on CPU).

```bash
# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (optional, faster for large datasets)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Tested on Linux/macOS. Windows should work but YMMV with shell scripts.

## Usage examples

### Basic detection

```bash
# Run full pipeline on one anomaly file
python executable/launcher.py \
    --baseline data/Golden/golden_period_dataset_mean_channel.csv \
    --test data/Anomaly/telemanom/isolated_anomaly_044_T-1_seq1.csv \
    --output results/T1_seq1
```

### Batch processing

```bash
# Process all Telemanom anomalies in parallel
./run_telemanom_parallel.sh
```

### Window-by-window detection

```bash
# Slide a window over long time series, detect changes
python executable/final_pipeline/window_by_window_detection.py \
    --baseline golden.csv \
    --test long_series.csv \
    --window-size 500 \
    --step 100
```

### Custom config

Edit `config/default.yaml` or pass parameters:

```bash
python executable/launcher.py \
    --lambda-w 0.05 \
    --lambda-a 0.05 \
    --threshold-multiplier 3.0 \
    ...
```

## Key parameters

Most important knobs in `config/default.yaml`:

```yaml
# DynoTEARS sparsity penalties
lambda_w: 0.1          # Contemporaneous edges (higher = sparser)
lambda_a: 0.1          # Lagged edges

# Detection thresholds
threshold_multiplier: 2.5    # For μ + k·σ (higher = less sensitive)
voting_threshold: 2          # Min metrics to vote "anomaly" (2/4 is default)

# Optimization
max_iter: 100          # DynoTEARS iterations
h_tol: 1e-8           # Acyclicity constraint tolerance
lr: 0.001             # Learning rate

# Preprocessing
differencing: auto     # "auto", "force", or "none"
max_lag: 5            # Max lags to consider in AutoReg
```

Tuning advice:
- **High false positives?** Increase `threshold_multiplier` or `voting_threshold`
- **Missing anomalies?** Decrease thresholds, try `voting_threshold: 1`
- **Graphs too dense?** Increase `lambda_w`/`lambda_a`
- **Graphs too sparse?** Decrease them



## Data format

Input CSVs should have:
- **First column**: timestamp (any parseable datetime format, or just integer index)
- **Remaining columns**: numeric variables (sensors, channels, etc.)
- **No missing values** (drop or interpolate beforehand)

Example:
```csv
timestamp,sensor_A,sensor_B,sensor_C
2024-01-01 00:00:00,1.23,4.56,7.89
2024-01-01 00:01:00,1.25,4.52,7.91
...
```

The code handles stationarity and differencing internally. Just provide raw data.

## Output structure

Results go to `results/<experiment_name>/`:

```
results/my_run/
├── preprocessing/
│   ├── differenced_data.csv       # After stationarity transform
│   ├── optimal_lags.csv           # Selected lags per variable
│   └── preprocessing_summary.json
├── causal_discovery/
│   ├── W_matrix.csv               # Contemporaneous weights
│   ├── A_matrix.csv               # Lagged weights
│   └── dynotears_log.txt
├── detection/
│   ├── metric_scores.csv          # SHD, Frobenius, Spectral, MaxEdge
│   ├── detection_result.json      # Binary: anomaly or not
│   └── voting_summary.txt
├── classification/
│   └── anomaly_type.json          # If detected: spike, drift, etc.
└── root_cause/
    └── edge_importance.csv        # Ranked edges by contribution
```

All CSVs are plain text, easy to load in pandas/Excel/whatever.

## Performance

On Telemanom (55-channel spacecraft telemetry, ~8000 timesteps per file):

- **Preprocessing**: ~5-10s
- **Causal discovery**: ~20-60s (CPU), ~10-30s (GPU)
- **Detection**: <1s
- **Total per file**: ~1-2 minutes

Scales roughly O(n·p²) where n = timesteps, p = variables (due to graph learning).

For very long series (>10k timesteps), use window-by-window mode to process in chunks.

## Troubleshooting

**"Singular matrix" or convergence errors**
- Series might be too correlated. Try higher `lambda_w`/`lambda_a`
- Check for constant columns (remove them)

**Detection too sensitive (many false positives)**
- Increase `threshold_multiplier` from 2.5 to 3.0 or 4.0
- Or require 3/4 metrics instead of 2/4: `voting_threshold: 3`

**Missing real anomalies**
- Decrease thresholds
- Try `voting_threshold: 1` (any single metric triggers)
- Check if baseline truly represents normal behavior

**Slow on CPU**
- Install PyTorch with CUDA support
- Or reduce `max_iter` from 100 to 50
- Or subsample long time series

**Out of memory**
- Reduce batch size or number of variables
- Use window-by-window mode

## Method details

This implements the approach from:

> Zheng, X., et al. (2020). "DAGs with NO TEARS: Continuous Optimization for Structure Learning." *NeurIPS 2018*.

Extended to Dynamic Bayesian Networks (DBNs) for time series. The acyclicity constraint uses an augmented Lagrangian to enforce DAG structure without discrete search.

Anomaly detection is based on comparing learned graphs, similar to change-point detection but using structural metrics instead of raw data statistics.

See [METRICS_CLARIFICATION.md](METRICS_CLARIFICATION.md) for why we use 4 metrics and how they complement each other.

## Limitations

- Assumes anomalies manifest as causal structure changes (true for many system failures, but not all anomaly types)
- Requires sufficient data in baseline to learn stable graph (~500+ timesteps recommended)
- Detection thresholds are dataset-specific (tune on validation set)
- Can't detect anomalies in single-variable series (need multivariate dependencies)

## Documentation

- [METRICS_CLARIFICATION.md](METRICS_CLARIFICATION.md): Why 4 metrics?

## Contributing

This was research code so it's rough around the edges. PRs welcome for:
- Better documentation
- More robust error handling  
- Additional anomaly types
- Benchmarks on other datasets

File issues if something breaks.