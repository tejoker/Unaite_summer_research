# Tucker-CAM: Non-linear Causal Discovery for Anomaly Detection

Detects anomalies in multivariate time series by learning causal graphs and tracking structural changes. Uses **Tucker-CAM** (non-linear) or **DynoTEARS** (linear) for causal discovery with a 4-metric ensemble for anomaly detection.

## Features

- **Non-linear causal discovery**: Tucker-CAM combines P-splines with Tucker decomposition
- **Memory-efficient**: Handles 2,889 variables via chunking and low-rank decomposition
- **Rolling window detection**: Tracks causal structure changes over time
- **4-metric ensemble**: SHD, Frobenius norm, spectral radius, max edge weight
- **GPU-accelerated**: PyTorch 2.0+ with torch.compile() support

## Validated on Real-World Data

- **NASA Telemanom**: Spacecraft telemetry with expert-verified anomaly labels
- **Server Machine Dataset (SMD)**: IT operations monitoring with multi-dimensional KPIs

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run on NASA Telemanom data
python executable/launcher.py \
    --baseline data/Golden/golden_baseline.csv \
    --test data/Anomaly/anomaly_001.csv \
    --output results/
```

## How It Works

**Pipeline:**
1. **Preprocessing**: Stationarity (ADF/KPSS tests), standardization, lag selection
2. **Causal Discovery**: Tucker-CAM learns DBN structure (non-linear relationships)
3. **Graph Comparison**: Compute 4 metrics between baseline and test graphs
4. **Detection**: Voting ensemble (≥2 metrics → anomaly)
5. **Root Cause**: Rank variables by contribution to anomaly score

**Tucker-CAM Model:**
- **P-splines**: Flexible basis functions for non-linear transformation
- **Tucker decomposition**: Reduces memory footprint (d × d × p → rank-based)
- **Chunked computation**: Processes variables in batches for GPU efficiency
- **Acyclicity constraint**: Augmented Lagrangian ensures DAG structure

## Repository Structure

```
executable/
├── launcher.py                                 # Main entry point
└── final_pipeline/
    ├── preprocessing_no_mi.py                  # Data preprocessing
    ├── dbn_dynotears_tucker_cam.py            # Tucker-CAM pipeline
    ├── dynotears_tucker_cam.py                # Tucker-CAM optimization
    ├── cam_model_tucker.py                    # P-spline + Tucker model
    └── dbn_dynotears_tucker_cam_restart.py    # Memory-safe restart wrapper

config/
└── default.yaml                                # Hyperparameters

analysis/                                       # Visualization tools
```

## Key Parameters

```yaml
# Tucker-CAM configuration
rank_w: 20              # Tucker rank for contemporaneous weights
rank_a: 10              # Tucker rank for lagged weights
n_knots: 5              # P-spline knots
lambda_smooth: 0.1      # Smoothness penalty
max_iter: 7             # Optimization iterations

# Detection
voting_threshold: 2     # Min metrics to declare anomaly (2/4)
threshold_multiplier: 2.5   # Threshold = μ + 2.5σ
```

## Performance

**NASA Telemanom (2,889 variables, 421 windows):**
- Window time: ~4 min/window (with Phase 1+2 optimizations)
- Memory: <10GB GPU (RTX 3090), 125GB RAM
- Optimizations: torch.compile(), chunking, vectorization, process restarts

**Hardware Requirements:**
- GPU: 24GB+ VRAM recommended (tested on RTX 3090 24GB)
- RAM: 100GB+ recommended (handles accumulation over multiple windows)
- CPU: Multi-core for data loading

## Model Details

**Tucker-CAM** extends DynoTEARS to non-linear relationships:
- Linear baseline: results coming soon
- Non-linear extension via P-splines handles complex sensor dynamics
- Tucker decomposition: O(d²p) → O(d·rank²·p) memory reduction

**Based on:**
- Zheng et al. (2018). "DAGs with NO TEARS" (*NeurIPS*)
- Extended to DBNs with Tucker decomposition for scalability

## Limitations

- Assumes anomalies manifest as causal structure changes
- Requires ≥500 timesteps in baseline for stable graph learning
- Best suited for continuous sensor data (not categorical/discrete)
- Detection thresholds are dataset-specific

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/tucker-cam-anomaly-detection.git
cd tucker-cam-anomaly-detection

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
# Ensure CUDA 11.8+ is installed
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Usage Examples

**Basic detection:**
```bash
python executable/launcher.py \
    --baseline data/baseline.csv \
    --test data/test.csv \
    --output results/
```

**Custom parameters:**
```bash
python executable/launcher.py \
    --baseline data/baseline.csv \
    --test data/test.csv \
    --output results/ \
    --rank-w 15 \
    --rank-a 8 \
    --max-iter 10
```

**GPU selection:**
```bash
CUDA_VISIBLE_DEVICES=0 python executable/launcher.py ...
```

## Output Structure

```
results/
├── preprocessing/
│   ├── differenced_data.csv
│   └── preprocessing_summary.json
├── causal_discovery/
│   ├── edges.csv                   # Learned causal edges
│   └── history/                    # Optimization history
├── detection/
│   ├── metric_scores.csv
│   └── detection_result.json
└── classification/
    └── anomaly_type.json
```

## Citation

If you use this code, please cite:
```bibtex
@misc{tucker-cam-2025,
  title={Tucker-CAM: Scalable Non-linear Causal Discovery for Anomaly Detection},
  author={Nicolas Bigeard},
  year={2025}
}
```

## License

MIT Licence

## Contributing

Issues and PRs welcome! This is research code with room for improvement.
