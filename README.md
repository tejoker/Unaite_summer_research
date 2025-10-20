# Paul Wurth Time Series Analysis Pipeline

An advanced causal discovery and anomaly detection system for multivariate time series analysis, featuring GPU-optimized DynoTEARS implementation, intelligent weight interpretation, and time series reconstruction capabilities.

## Overview

This repository provides a complete pipeline for analyzing time series data to discover causal relationships, detect anomalies, and reconstruct corrected time series. The system is specifically designed for Paul Wurth's industrial time series analysis requirements with enhanced GPU optimization and comprehensive weight interpretation capabilities.

## NEW: End-to-End Anomaly Detection and Reconstruction

Complete workflow for detecting, classifying, and reconstructing anomalies using only DynoTEARS weights:

```bash
# One-line complete workflow
./run_complete_workflow.sh data/baseline.csv data/anomaly.csv results/my_analysis
```

**Capabilities:**
- Detect anomalies using 4-metric ensemble (+31% F1-score vs single metric)
- Classify anomaly type (spike, drift, structural)
- Identify root causes (which causal edges changed)
- Reconstruct corrected time series

**Quick Start:** See [END_TO_END_GUIDE.md](END_TO_END_GUIDE.md) | [QUICK_START_COMMANDS.md](QUICK_START_COMMANDS.md)

## Architecture

### Key Features

- **GPU-Optimized DynoTEARS**: High-performance causal discovery with mixed precision training and multi-GPU support
- **Weight Interpretation**: Link discovered causal weights to specific time positions in original time series
- **Anomaly Detection**: Automatically detect anomalous causal patterns using statistical and temporal analysis
- **Time Series Reconstruction**: Reconstruct multivariate time series with corrected anomalous weights
- **Interactive Visualizations**: Comprehensive visualization tools for analysis results
- **Experiment Management**: Organized storage and tracking of analysis experiments

## Repository Structure

```
executable/
├── final_pipeline/           # Main analysis pipeline
│   ├── dbn_dynotears.py     # Main pipeline orchestrator
│   ├── dynotears.py         # GPU-optimized DynoTEARS implementation
│   ├── gpu_optimizer.py     # GPU optimization framework
│   ├── weight_interpreter.py # Weight interpretation and analysis
│   ├── ts_reconstructor.py  # Time series reconstruction
│   ├── preprocessing.py     # Data preprocessing utilities
│   ├── anomaly_detector.py  # Anomaly detection algorithms
│   ├── experiment_manager.py # Experiment tracking and storage
│   └── resource_manager.py  # System resource management
├── launcher.py              # Pipeline launcher
└── test/                    # Testing utilities
config/                      # Configuration files
logs/                        # Log files (git-ignored)
results/                     # Analysis results (git-ignored)
```

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n paul_wurth python=3.9
conda activate paul_wurth

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Data Preparation

Prepare your time series data as CSV files with:
- Timestamp index (datetime format)
- Multiple numeric columns (variables)
- No missing values (use preprocessing if needed)

### 3. Basic Usage

```bash
# Run complete pipeline
python executable/launcher.py \
    --input-data your_data.csv \
    --output-dir results/ \
    --gpu-optimization \
    --anomaly-detection

# Run weight interpretation only
python executable/final_pipeline/weight_interpreter.py \
    --weights results/weights/weights_enhanced_*.csv \
    --data your_data.csv \
    --output-dir weight_analysis/

# Run time series reconstruction
python executable/final_pipeline/ts_reconstructor.py \
    --weights results/weights/weights_enhanced_*.csv \
    --data your_data.csv \
    --output-dir reconstruction_results/ \
    --strategy bidirectional \
    --correction-method replace_with_median
```

## Pipeline Workflow

### 1. Preprocessing (`preprocessing.py`)
- **Stationarity Testing**: ADF and KPSS tests
- **Differencing**: Automatic first-differencing for non-stationary series
- **Lag Optimization**: Determine optimal lag structure using AutoReg
- **Mutual Information Masking**: Create dependency masks between variables

### 2. Causal Discovery (`dynotears.py`, `dbn_dynotears.py`)
- **Dynamic Structure Learning**: Learn time-varying causal graphs
- **Neural Network Optimization**: Gradient-based structure learning
- **Constraint Handling**: Acyclicity constraints via augmented Lagrangian
- **GPU Acceleration**: CUDA-optimized tensor operations

### 3. Reconstruction (`reconstruction.py`)
- **Causal Graph Simulation**: Generate synthetic time series from learned structure
- **Anomaly Scoring**: Compare real vs. reconstructed signals
- **Validation**: Statistical tests on reconstruction quality

## Configuration

### Environment Variables

The launcher sets these automatically, but can be overridden:

- `INPUT_CSV_FILE`: Input data path
- `RESULT_DIR`: Output directory for results
- `EXPERIMENT_NAME`: Experiment identifier
- `INPUT_DIFFERENCED_CSV`: Preprocessed data path
- `INPUT_LAGS_CSV`: Optimal lags file
- `INPUT_MI_MASK_CSV`: Mutual information mask

### Key Parameters

#### DynoTears Algorithm
```python
lambda_w: float = 0.1        # Sparsity penalty for contemporaneous effects
lambda_a: float = 0.1        # Sparsity penalty for lagged effects
max_iter: int = 100          # Maximum optimization iterations
h_tol: float = 1e-8          # Acyclicity constraint tolerance
w_threshold: float = 0.0     # Threshold for edge pruning
```

#### Preprocessing
```python
ALPHA_STATIONARITY = 0.05    # P-value threshold for stationarity tests
ALPHA_MI = 0.01              # Significance for mutual information tests
MI_BINS = 5                  # Bins for MI discretization
```

## Sensor Types

The system is optimized for Paul Wurth equipment sensors:

- **Temperatur Druckpfannenlager links/rechts**: Pressure pan bearing temperatures
- **Temperatur Exzenterlager links/rechts**: Eccentric bearing temperatures
- **Temperatur Ständerlager links/rechts**: Stator bearing temperatures
- Custom sensor configurations supported

## Advanced Features

### Parallel Processing
- Multi-core preprocessing with `multiprocessing`
- GPU-accelerated neural network training
- Concurrent experiment execution

### Experiment Management
- Automatic result organization with timestamps
- Hierarchical experiment structure
- Metadata tracking and versioning

### Resource Management
- Memory usage monitoring with `psutil`
- GPU memory optimization
- Automatic cleanup and checkpointing

## Output Structure

Results are organized hierarchically:

```
results/
└── launch_YYYYMMDD_HHMMSS/
    └── experiments/
        ├── causal_discovery/
        │   └── EXPERIMENT_causal_discovery_dataset_v1.0_timestamp/
        ├── preprocessing/
        └── reconstruction/
```

Each experiment folder contains:
- `*.csv`: Processed data and results
- `*.pkl`: Serialized models and graphs
- `*.png`: Visualization plots
- `*.log`: Execution logs
- `metadata.json`: Experiment parameters

## Development

### Adding New Algorithms

1. Create new script in appropriate folder (`final_pipeline/` or `test/`)
2. Follow existing patterns for environment variable usage
3. Update `launcher.py` exclude lists if needed
4. Add to pipeline order in `launcher.py:677`

### Custom Data Types

1. Create new folder in `data/` directory
2. Update `launcher.py` data type detection
3. Configure appropriate preprocessing parameters

## Troubleshooting

### Common Issues

**GPU Memory Errors**
```bash
# Reduce batch size or use CPU fallback
export CUDA_VISIBLE_DEVICES=""
```

**Large Result Sizes**
- Results can grow large; monitor `results/` directory
- Configure cleanup policies in launcher

**Import Errors**
- Some modules are optional (`resource_manager`, `experiment_manager`)
- Check `requirements.txt` for missing dependencies

### Performance Tuning

- Use GPU acceleration when available
- Adjust parallel processing cores based on system
- Monitor memory usage with large datasets
- Consider chunking very large time series

## Citation

If using this platform in research, please cite the DynoTears algorithm and relevant papers on causal discovery in time series.

## License

Industrial research project - check with Paul Wurth for usage permissions.