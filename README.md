# Tucker-CAM: Tucker Decomposition for Causal Anomaly Mining

**Scalable Temporal Causal Discovery for Anomaly Detection in Multivariate Time Series**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

Tucker-CAM learns time-varying causal structures from multivariate time series using Tucker decomposition to enable efficient causal discovery on high-dimensional data. By analyzing how causal relationships evolve over time, the method can detect anomalies that manifest as structural changes in the underlying causal graph.

**Key Innovation:** Tucker decomposition of the weight tensor enables learning causal graphs with thousands of variables while maintaining computational tractability and avoiding memory overflow.

## Features

- 🚀 **Scalable**: Handles 2,890 variables (NASA Telemanom spacecraft data)
- ⚡ **Efficient**: Tucker decomposition + post-hoc Top-K sparsification
- 🎯 **Accurate**: Detects anomalies via temporal causal structure evolution
- 📊 **Interpretable**: Identifies which causal links change during anomalies
- 🔄 **Temporal**: Rolling window analysis captures non-stationary dynamics

## Method

### Tucker-CAM Pipeline

```
Raw Time Series
    ↓
[1] Preprocessing
    • Forward-fill imputation (causality-preserving)
    • Log transform + differencing (stationarity)
    • Standardization (numerical stability)
    ↓
[2] Causal Discovery (Tucker-CAM)
    • Rolling windows (size=100, stride=10)
    • Tucker decomposition of weight tensor
    • DynoTEARS optimization (zero penalties)
    • Post-hoc Top-K sparsification (K=10,000)
    ↓
[3] Anomaly Detection
    • Compare Golden vs Anomaly causal graphs
    • Statistical tests + classification
    ↓
Results: Temporal causal graphs + anomaly scores
```

### Tucker Decomposition

High-dimensional weight tensor **W** (variables × variables × lags) is decomposed:

```
W ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃
```

Where:
- **G**: Core tensor (compressed representation)
- **U₁, U₂, U₃**: Factor matrices (latent dimensions)

**Benefits:**
- Reduces parameters from O(p²L) to O(r²L + rp) where r << p
- Enables learning on 2,890 variables (8.3M possible edges)
- Maintains expressiveness via rank tuning

### Post-hoc Top-K Sparsification (Option D)

1. **Training**: Learn dense graphs with zero penalties (λ₁=0, λ₂=0)
2. **Sparsification**: Select Top-K=10,000 edges by absolute weight per window
3. **Benefits**:
   - Stable edge counts (CV=0%)
   - No hyperparameter tuning (no λ grid search)
   - Captures strongest causal relationships

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/tucker-cam.git
cd tucker-cam

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=1.12.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
scipy>=1.7.0
```

## Usage

### Quick Start: Telemanom Dataset

```bash
# Process Golden (normal) and Anomaly datasets
bash run_telemanom_comparison.sh

# Analyze results
python compare_golden_vs_anomaly.py
```

### Custom Dataset

```python
from executable.launcher import run_pipeline

# Run on your data
run_pipeline(
    input_csv="your_data.csv",
    output_dir="results/your_run",
    tucker_top_k=10000,
    window_size=100,
    stride=10
)
```

### Data Format

Input: CSV file with shape (timesteps, variables)

```csv
timestamp,var1,var2,var3,...
0,1.23,4.56,7.89,...
1,1.25,4.52,7.91,...
```

**Notes:**
- First column can be timestamp (will be used as index)
- NaN values: Use forward-fill for causality preservation
- Recommended: 1000+ timesteps for stable causal learning

## Datasets

### NASA Telemanom (Primary Benchmark)

- **Source**: Spacecraft telemetry from SMAP & MSL missions
- **Variables**: 82 channels × ~35 features = 2,890 dimensions
- **Golden**: 4,308 timesteps (normal operation)
- **Anomaly**: 8,640 timesteps (with labeled anomalies)
- **Reference**: [Hundman et al., KDD 2018](https://arxiv.org/abs/1802.04431)

### Preprocessing Telemanom

```bash
cd telemanom

# Forward-fill imputation (causality-preserving)
python3 << 'EOF'
import pandas as pd

# Golden dataset
df = pd.read_csv('golden_period_dataset.csv')
df.ffill().to_csv('golden_period_dataset_ffill.csv', index=False)

# Anomaly dataset  
df = pd.read_csv('test_dataset_merged.csv')
df.ffill().to_csv('test_dataset_merged_ffill.csv', index=False)
EOF
```

## Configuration

### Tucker-CAM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tucker_top_k` | 10000 | Top-K edges per window (post-hoc sparsification) |
| `window_size` | 100 | Rolling window size (timesteps) |
| `stride` | 10 | Window stride (overlap = window_size - stride) |
| `tucker_rank` | 10 | Tucker decomposition rank |
| `max_iter` | 100 | DynoTEARS optimization iterations |

### Memory Requirements

- **10K edges**: ~19GB RAM (tested on Telemanom 2,890 vars)
- **GPU**: Optional (3% utilization, 0.72GB VRAM)
- **Recommendation**: 20GB+ RAM for large datasets

## Results

### Telemanom Benchmark

Validated on 421 rolling windows:

| Metric | Value |
|--------|-------|
| Edge count stability | CV=0.00% (exactly 10K/window) |
| MSE range | 0.0004 - 0.6 |
| Processing time | 2h 45m (421 windows) |
| Memory peak | 4.15GB RAM, 0.72GB GPU |
| Lagged edges | 99.98% (temporal causality) |

### Output Structure

```
results/
├── golden/
│   ├── weights/
│   │   └── weights_enhanced.csv     # Temporal causal graph
│   └── golden_period_dataset_ffill_differenced_stationary_series.csv
└── anomaly/
    ├── weights/
    │   └── weights_enhanced.csv     # Temporal causal graph
    └── test_dataset_merged_ffill_differenced_stationary_series.csv
```

### Causal Graph Format

```csv
window_id,source,target,lag,weight
0,var1_diff,var2_diff,1,0.234
0,var3_diff,var1_diff,2,0.156
```

## Methodology

### 1. Preprocessing

- **Stationarity**: ADF/KPSS tests + differencing
- **Imputation**: Forward-fill (preserves causality)
- **Normalization**: StandardScaler (zero mean, unit variance)

### 2. Causal Discovery

**Objective:** Learn Dynamic Bayesian Network (DBN) structure

```
X(t) = Σ_{τ=1}^{L} W(τ) · X(t-τ) + noise
```

**Optimization:** DynoTEARS with Tucker decomposition

```
min_{G,U₁,U₂,U₃} ||X(t) - Σ_{τ} (G ×₁ U₁ ×₂ U₂ ×₃ U₃)(τ) · X(t-τ)||²
s.t. DAG constraint (acyclicity)
```

**Sparsification:** Select Top-K edges per window

### 3. Anomaly Detection

Compare Golden vs Anomaly graphs via:
- Edge overlap (Jaccard similarity)
- Weight distribution (KL divergence, Wasserstein distance)
- Random Forest classification on graph features

## Technical Details

### Why Forward Fill Only?

**Causality Preservation:** Forward-fill never uses future information to impute past values.

```python
# CORRECT: Forward fill (t-1 → t)
df.ffill()

# WRONG: Backward fill (t+1 → t) breaks causality!
df.bfill()
```

**Impact on Causal Discovery:**
- Forward-filled segments → constant values → diff = 0
- Low variance → weak causal weights (correct behavior!)
- Active variables still detected with strong weights

### Tucker Decomposition Details

**Rank Selection:** r=10 (empirically validated)
- Too low: Loss of expressiveness
- Too high: Memory overhead, overfitting

**Complexity:**
- Dense: O(p²L) = O(2890² × 20) = 167M parameters
- Tucker: O(r²L + rp) = O(10² × 20 + 10 × 2890) = 31K parameters
- **Reduction: 5,400×**

### Post-hoc Top-K vs In-training Sparsification

| Method | Edges/Window | Stability | Hyperparameters |
|--------|--------------|-----------|-----------------|
| L1 penalty | Variable | CV=15-30% | λ₁ (grid search) |
| Tucker-CAM (Option D) | Stable | **CV=0%** | **None** (just K) |

## Limitations

1. **Linear assumption**: DBN assumes linear structural equations
2. **Stationarity**: Requires preprocessing for non-stationary data
3. **Window size**: Trade-off between temporal resolution and statistical power
4. **Computational**: Scales to ~3K variables (tested), larger needs distributed computing

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{bigeard2025tuckercam,
  title={Tucker-CAM: Scalable Temporal Causal Discovery for Anomaly Detection},
  author={Bigeard, Nicolas and [Co-authors]},
  booktitle={[Conference]},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- **DynoTEARS**: Based on [DYNOTEARS](https://arxiv.org/abs/2002.00498) framework
- **Telemanom**: NASA dataset from [Hundman et al., KDD 2018](https://arxiv.org/abs/1802.04431)
- **Tucker Decomposition**: [Kolda & Bader, 2009](https://doi.org/10.1137/07070111X)

## Contact

- Nicolas Bigeard - [email]
- Project: [GitHub Repository](https://github.com/yourusername/tucker-cam)

## See Also

- [TELEMANOM_PIPELINE_GUIDE.md](TELEMANOM_PIPELINE_GUIDE.md) - Detailed technical guide
- [config/default.yaml](config/default.yaml) - Configuration reference
