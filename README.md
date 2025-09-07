# Causal Discovery Pipeline for Industrial Time Series Analysis

![Python](https://img.shields.io/badge/python-%3E=3.9-blue)
![CUDA](https://img.shields.io/badge/CUDA-Supported-green)
![Status](https://img.shields.io/badge/Status-Stable-success)

## Overview
This repository contains a comprehensive pipeline for causal discovery and anomaly detection in industrial time series data, specifically designed for Paul Wurth furnace monitoring systems. The pipeline implements the advanced Dynamic Notears (DYNOTEARS) algorithm to discover causal relationships in temporal data under both normal and anomalous operating conditions.

## Table of Contents
- [Project Structure](#project-structure)
- [Core Pipeline Components](#core-pipeline-components)
- [Performance Analysis](#performance-analysis)
- [Usage Instructions](#usage-instructions)
- [Input Data Format](#input-data-format)
- [Output Structure](#output-structure)
- [Dependencies](#dependencies)
- [Hardware Recommendations](#hardware-recommendations)
- [Troubleshooting](#troubleshooting)
- [Benchmark Results and Analysis](#benchmark-results-and-analysis)
- [DYNOTEARS Variants: Detailed Comparison](#dynotears-variants-detailed-comparison)

## Project Structure
```

program\_internship\_paul\_wurth/
├── executable/
│   ├── launcher.py
│   ├── causal\_discovery\_benchmark.py
│   ├── run\_sota\_benchmark.py
│   │
│   ├── final\_pipeline/
│   │   ├── preprocessing.py
│   │   ├── dynotears.py
│   │   ├── dbn\_dynotears.py
│   │   ├── dynotears\_variants.py
│   │   ├── reconstruction.py
│   │   ├── resource\_manager.py
│   │   └── structuremodel.py
│   │
│   └── test/
│       ├── sota\_anomaly\_methods.py
│       ├── performance\_benchmark.py
│       ├── MCMC\_CPU.py
│       ├── MCMC\_GPU.py
│       └── \[various test files]
│
├── data/
│   ├── Golden/
│   └── Anomaly/
│
├── results/
│   ├── launch2\_\[timestamp]/
│   │   ├── Golden/
│   │   └── Anomaly/
│   └── \[legacy result folders]
│
└── logs/
├── golden\_1\_enhanced.log
├── drift\_enhanced.log
└── performance\_benchmark.log

````

## Core Pipeline Components
### 1. Data Preprocessing (`preprocessing.py`)
- Prepares raw time series data for causal discovery  
- Stationarity testing, differencing, optimal lag calculation  
- Outputs: stationary series, lags, MI masks  

### 2. DYNOTEARS (`dynotears.py`)
- Core causal discovery  
- GPU-accelerated, sparse learning  
- Outputs: adjacency (W), lag matrices (A)  

### 3. DBN DYNOTEARS (`dbn_dynotears.py`)
- Rolling window causal discovery  
- Outputs: time-annotated edges  

### 4. DYNOTEARS Variants (`dynotears_variants.py`)
- Tests `no_mi`, `no_rolling` configurations  

### 5. Reconstruction (`reconstruction.py`)
- Reconstructs data from causal graphs for validation  

### 6. Resource Manager (`resource_manager.py`)
- Manages CPU/GPU load and memory  

## Performance Analysis
- Golden: ~15.2 min  
- Anomaly: ~13.2 min  
- Preprocessing: 2–3 min  
- Discovery: 8–10 min  
- Reconstruction: 2–3 min  

| Method             | Edges | GPU Mem | CPU |
|--------------------|-------|---------|-----|
| Standard DYNOTEARS | 36    | ~8GB    | 64  |
| DBN DYNOTEARS      | Var   | ~12GB   | 8–64|
| No MI Filtering    | 36    | ~6GB    | 32  |
| No Rolling Window  | 36    | ~4GB    | 16  |

## Usage Instructions
### Single Dataset
```bash
cd executable
python launcher.py --csv_file ../data/Golden/chunking/output_of_the_1th_chunk.csv --data_type golden
````

### Batch

```bash
python launcher.py --data_types golden,anomaly --batch_mode
```

### Steps

```bash
python launcher.py --csv_file [file] --scripts preprocessing
python launcher.py --csv_file [file] --scripts dynotears,dbn_dynotears
python launcher.py --csv_file [file] --output_dir ../custom_results
```

### Advanced

```bash
python final_pipeline/performance_benchmark.py
python run_sota_benchmark.py --methods all --datasets golden,anomaly
python final_pipeline/dynotears_variants.py --input [preprocessed_data] --output [results_dir]
```

## Input Data Format

```csv
timestamp,variable1,variable2,...
2023-01-01 00:00:00,1.2,3.4,...
```

* Golden: normal data
* Anomaly: drift, spike, level\_shift, amplitude\_change, variance\_burst, trend\_change, missing\_block

## Output Structure

```
results/launch2_[timestamp]/
├── Golden/
│   ├── preprocessing/
│   ├── weights/
│   ├── history/
│   └── dynotears_variants/
└── Anomaly/
```

## Dependencies

```bash
pip install -r requirements.txt
```

* Core: torch, pandas, numpy, statsmodels, scikit-learn, tqdm, matplotlib, seaborn
* Optional: cupy, numba, joblib

## Hardware Recommendations

* Min: 4 cores, 8 GB RAM, 4 GB GPU
* Optimal: 16+ cores, 32 GB RAM, RTX 3080/4080 or V100+, NVMe SSD

## Troubleshooting

* **CUDA OOM**: lower batch size, or `export CUDA_VISIBLE_DEVICES=""`
* **Stationarity warnings**: non-critical, preprocessing auto-handles
* **Slow runs**: enable GPU, chunk data

## Benchmark Results and Analysis

* **Hardware**: 64 cores, 125 GB RAM, RTX 3090 (23.6 GB)
* **Standard DYNOTEARS**: 36 temporal edges, 0 instantaneous, 13–15 min
* **Variants**: all 36 edges, identical results for no\_mi and no\_rolling
* **Processing Times**: Golden \~15–23 min, anomalies \~13–15 min, Missing Block 3.8 min (fastest)
* **Resource Usage**: GPU <10 MB/batch, 8 CPU workers optimal

## DYNOTEARS Variants: Detailed Comparison

* **No MI Filtering**:

  * All 36 lag-1 edges (fully connected), 0 W
  * Strong self-loops (-0.39 to -0.48), weak cross-edges
  * No filtering: accepts all causal links

* **No Rolling Window**:

  * Identical to no\_mi (VAR on full dataset)
  * 36 edges, 0 W, \~0.005 sec runtime

* **Enhanced Method**:

  * MI + rolling, multi-lag (1–10) analysis
  * 299 edges across lags, \~15% of possible edges retained
  * Sparse, interpretable, \~15 min runtime

| Aspect         | No MI/No Rolling   | Enhanced Method     |
| -------------- | ------------------ | ------------------- |
| Edge Filtering | None               | 85% rejected via MI |
| Temporal Depth | Single lag (lag-1) | Multi-lag (1–10)    |
| Matrix Density | 100% (full)        | \~15% (sparse)      |
| Runtime        | 0.005 sec          | \~15 min            |
| Edge Quality   | All accepted       | Significant only    |

**Takeaway**:

* No MI/No Rolling = fast, exploratory, but spurious
* Enhanced = slower, but high-quality causal discovery
* MI filtering is essential for meaningful results; rolling window adds temporal depth.

