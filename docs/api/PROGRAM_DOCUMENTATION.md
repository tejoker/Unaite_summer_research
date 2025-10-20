# Program and Documentation Repository Map

## Overview

This repository contains a comprehensive time series analysis pipeline for causal discovery and anomaly detection using DynoTEARS. The system processes multivariate time series data to discover causal relationships, detect anomalies, and reconstruct corrected time series.

---

## Documentation Files (.md)

### Core Documentation

#### 1. **README.md**
- **Role**: Main repository overview and quick start guide
- **Purpose**: Entry point for new users
- **Content**: Architecture overview, quick start instructions, repository structure
- **Used by**: New users, developers
- **Dependencies**: None (standalone documentation)

#### 2. **IMPLEMENTATION_SUMMARY.md**
- **Role**: Summary of end-to-end anomaly detection implementation
- **Purpose**: Documents what was added to enable complete anomaly workflow
- **Content**: New files created, integration details, capabilities unlocked
- **Used by**: Developers understanding the system architecture
- **Dependencies**: References other implementation files

#### 3. **END_TO_END_GUIDE.md**
- **Role**: Complete user guide for anomaly detection workflow
- **Purpose**: Step-by-step instructions for running complete pipeline
- **Content**: Detailed workflow instructions, troubleshooting, API reference
- **Used by**: Users running the complete anomaly detection pipeline
- **Dependencies**: References executable scripts and data files

#### 4. **QUICK_START_COMMANDS.md**
- **Role**: Command-line cheat sheet
- **Purpose**: Quick reference for common commands
- **Content**: One-liners, individual component commands, environment setup
- **Used by**: Experienced users, developers
- **Dependencies**: References executable scripts

### Analysis Documentation

#### 5. **ANOMALY_ANALYSIS_REPORT.md**
- **Role**: Analysis of anomalies in dataset (data-focused)
- **Purpose**: Documents direct data analysis of injected anomalies
- **Content**: Anomaly locations, types, magnitudes, data characteristics
- **Used by**: Data analysts, researchers
- **Dependencies**: References raw data files and metadata

#### 6. **FINAL_CONCLUSIONS.md**
- **Role**: Weight-based anomaly detection conclusions
- **Purpose**: Documents findings from weight-based detection experiments
- **Content**: Success/failure analysis, root causes, alternative approaches
- **Used by**: Researchers, developers
- **Dependencies**: References analysis results and weight files

#### 7. **WINDOW_BY_WINDOW_ANALYSIS.md**
- **Role**: Window-by-window detection analysis
- **Purpose**: Detailed analysis of detection performance across windows
- **Content**: Window-level statistics, detection patterns
- **Used by**: Researchers analyzing detection performance
- **Dependencies**: References window analysis results

#### 8. **DRIFT_AND_SPIKE_ANALYSIS.md**
- **Role**: Drift and spike detection analysis
- **Purpose**: Specific analysis of drift vs spike detection performance
- **Content**: Comparative analysis of different anomaly types
- **Used by**: Researchers focusing on specific anomaly types
- **Dependencies**: References analysis results

### Technical Documentation

#### 9. **GRADUAL_ANOMALY_DETECTION.md**
- **Role**: Methods for detecting gradual anomalies
- **Purpose**: Documents alternative approaches for gradual anomaly detection
- **Content**: Proposed methods, implementation details, expected behavior
- **Used by**: Developers implementing new detection methods
- **Dependencies**: References detection algorithms

#### 10. **ADAPTATION_PARADOX.md**
- **Role**: Why gradual changes aren't detected
- **Purpose**: Explains the theoretical limitations of current approach
- **Content**: Mathematical analysis of why gradual anomalies fail
- **Used by**: Researchers understanding detection limitations
- **Dependencies**: References analysis results

#### 11. **VALIDATION_CHECKLIST.md**
- **Role**: Validation procedures
- **Purpose**: Ensures system reliability and correctness
- **Content**: Validation steps, quality checks
- **Used by**: Developers, testers
- **Dependencies**: References test scripts

#### 12. **GPU_OPTIMIZATION_GUIDE.md**
- **Role**: GPU optimization guide
- **Purpose**: Performance optimization instructions
- **Content**: GPU setup, optimization techniques
- **Used by**: Users running on GPU systems
- **Dependencies**: References GPU-optimized scripts

#### 13. **REPOSITORY_ANALYSIS_SUMMARY.md**
- **Role**: Repository structure analysis
- **Purpose**: Documents the overall system architecture
- **Content**: File organization, component relationships
- **Used by**: Developers understanding system structure
- **Dependencies**: References all repository components

#### 14. **CLAUDE.md**
- **Role**: AI assistant interaction notes
- **Purpose**: Documents AI-assisted development process
- **Content**: Development history, decisions made
- **Used by**: Developers understanding development process
- **Dependencies**: References development artifacts

---

## Python Programs

### Main Pipeline

#### 1. **executable/launcher.py**
- **Role**: Main pipeline launcher
- **Purpose**: Orchestrates the complete analysis pipeline
- **Inputs**: 
  - `--data`: Input CSV file path
  - `--output`: Output directory path
- **Outputs**:
  - Creates output directory structure
  - Calls preprocessing.py
  - Calls dbn_dynotears.py
- **Programs it calls**: 
  - `executable/final_pipeline/preprocessing.py`
  - `executable/final_pipeline/dbn_dynotears.py`
- **Dependencies**: subprocess, pathlib, argparse
- **Used by**: Users running complete pipeline

#### 2. **executable/final_pipeline/preprocessing.py**
- **Role**: Data preprocessing
- **Purpose**: Prepares time series data for DynoTEARS analysis
- **Inputs**:
  - Environment variable `INPUT_CSV_FILE`: Raw CSV data
  - Environment variable `RESULT_DIR`: Output directory
- **Outputs**:
  - `{basename}_differenced_stationary_series.csv`: Differenced data
  - `{basename}_optimal_lags.csv`: Optimal lags for each variable
  - `{basename}_mi_mask_edges.csv`: MI mask (now empty, allows all edges)
- **Programs it calls**: None (standalone)
- **Dependencies**: pandas, numpy, statsmodels, multiprocessing
- **Used by**: launcher.py

#### 3. **executable/final_pipeline/dbn_dynotears.py**
- **Role**: DynoTEARS causal discovery
- **Purpose**: Runs DynoTEARS algorithm on preprocessed data
- **Inputs**:
  - Environment variable `INPUT_DIFFERENCED_CSV`: Differenced data
  - Environment variable `INPUT_LAGS_CSV`: Optimal lags
  - Environment variable `RESULT_DIR`: Output directory
- **Outputs**:
  - `weights/weights_enhanced.csv`: Weight matrices per window
  - `history/rolling_checkpoint.pkl`: Checkpoint data
  - `history/weights_history.csv`: Weight history
- **Programs it calls**: None (uses dynotears library)
- **Dependencies**: dynotears, torch, pandas, numpy
- **Used by**: launcher.py

#### 4. **executable/final_pipeline/end_to_end_pipeline.py**
- **Role**: Complete anomaly workflow
- **Purpose**: Orchestrates anomaly detection, classification, and reconstruction
- **Inputs**:
  - `--baseline-weights`: Baseline weight matrices
  - `--anomaly-weights`: Anomaly weight matrices
  - `--original-data`: Original time series data
  - `--output-dir`: Output directory
- **Outputs**:
  - `anomaly_detection_results.json`: Detection results
  - `corrected_weights.csv`: Corrected weight matrices
  - `reconstructed_time_series.csv`: Reconstructed data
  - `PIPELINE_SUMMARY.txt`: Human-readable summary
- **Programs it calls**: 
  - `executable/test/anomaly_detection_suite/anomaly_detection_suite.py`
  - `executable/final_pipeline/weight_corrector.py`
  - `executable/final_pipeline/reconstruction.py`
- **Dependencies**: anomaly detection suite, weight corrector, reconstruction
- **Used by**: Users running anomaly detection workflow

### Analysis Scripts

#### 5. **compare_all_anomalies.py**
- **Role**: Compare all anomaly types
- **Purpose**: Analyzes weight differences across all anomaly types
- **Inputs**: 
  - Searches for weight files in results directories
  - Golden baseline weights
- **Outputs**:
  - `all_anomalies_comparison.png`: Visualization
  - Console output with analysis results
- **Programs it calls**: None (standalone analysis)
- **Dependencies**: pandas, numpy, matplotlib, pathlib
- **Used by**: Researchers analyzing multiple anomaly types

#### 6. **compare_weights.py**
- **Role**: Weight comparison analysis
- **Purpose**: Compares weight matrices using ratio analysis
- **Inputs**:
  - Command line arguments for weight files
- **Outputs**:
  - Console output with weight differences
  - Detailed analysis of weight changes
- **Programs it calls**: None (standalone analysis)
- **Dependencies**: pandas, numpy, pathlib
- **Used by**: Researchers analyzing weight changes

#### 7. **diagnose_detection_failure.py**
- **Role**: Diagnose detection failures
- **Purpose**: Analyzes why certain anomalies aren't detected
- **Inputs**: 
  - Weight files
  - Metadata files
- **Outputs**:
  - Console output with failure analysis
  - Diagnostic information
- **Programs it calls**: None (standalone analysis)
- **Dependencies**: pandas, numpy, pathlib
- **Used by**: Developers debugging detection issues

#### 8. **detect_gradual_anomalies.py**
- **Role**: Gradual anomaly detection
- **Purpose**: Implements alternative methods for gradual anomaly detection
- **Inputs**:
  - Weight data
  - Configuration parameters
- **Outputs**:
  - Detection results for gradual anomalies
  - Analysis of cumulative changes
- **Programs it calls**: None (standalone detection)
- **Dependencies**: pandas, numpy, scipy
- **Used by**: Researchers testing gradual anomaly detection

### Testing Scripts

#### 9. **test_unified_detector.py**
- **Role**: Test unified anomaly detector
- **Purpose**: Tests the unified anomaly detection suite
- **Inputs**:
  - Test data files
  - Configuration parameters
- **Outputs**:
  - Test results
  - Performance metrics
- **Programs it calls**: 
  - `executable/final_pipeline/anomaly_detection/unified_anomaly_detector.py`
- **Dependencies**: unified anomaly detector
- **Used by**: Developers testing detection algorithms

#### 10. **test_robust_detector.py**
- **Role**: Test robust weight detector
- **Purpose**: Tests robust weight-based detection
- **Inputs**:
  - Weight matrices
  - Test parameters
- **Outputs**:
  - Test results
  - Robustness metrics
- **Programs it calls**: 
  - `executable/final_pipeline/anomaly_detection/robust_weight_detector.py`
- **Dependencies**: robust weight detector
- **Used by**: Developers testing robust detection

#### 11. **test_reproducibility.py**
- **Role**: Test reproducibility
- **Purpose**: Ensures results are reproducible
- **Inputs**:
  - Test data
  - Configuration files
- **Outputs**:
  - Reproducibility test results
  - Consistency metrics
- **Programs it calls**: Various pipeline components
- **Dependencies**: Complete pipeline
- **Used by**: Developers ensuring reproducibility

### Utility Scripts

#### 12. **check_anomaly_positions.py**
- **Role**: Check anomaly positions
- **Purpose**: Verifies ground truth anomaly locations
- **Inputs**:
  - Metadata files with anomaly information
  - Detection results
- **Outputs**:
  - Console output with position verification
  - Accuracy metrics
- **Programs it calls**: None (standalone verification)
- **Dependencies**: pandas, json
- **Used by**: Researchers validating ground truth

#### 13. **explain_spike_detection.py**
- **Role**: Explain spike detection
- **Purpose**: Analyzes why spike detection works
- **Inputs**:
  - Spike data
  - Detection results
- **Outputs**:
  - Console output with explanation
  - Analysis of detection mechanism
- **Programs it calls**: None (standalone analysis)
- **Dependencies**: pandas, numpy
- **Used by**: Researchers understanding spike detection

#### 14. **map_window_to_time.py**
- **Role**: Map window to time
- **Purpose**: Converts window indices to actual timestamps
- **Inputs**:
  - Window indices
  - Time series data
- **Outputs**:
  - Time mappings
  - Timestamp information
- **Programs it calls**: None (standalone utility)
- **Dependencies**: pandas, datetime
- **Used by**: Researchers analyzing temporal aspects

#### 15. **visualize_weight_anomalies.py**
- **Role**: Visualize weight anomalies
- **Purpose**: Creates visualizations of weight changes
- **Inputs**:
  - Weight data
  - Anomaly information
- **Outputs**:
  - Visualization plots
  - PNG files
- **Programs it calls**: None (standalone visualization)
- **Dependencies**: matplotlib, pandas, numpy
- **Used by**: Researchers creating visualizations

#### 16. **visualize_window_comparison.py**
- **Role**: Visualize window comparison
- **Purpose**: Compares windows visually
- **Inputs**:
  - Window data
  - Comparison parameters
- **Outputs**:
  - Comparison plots
  - Visualization files
- **Programs it calls**: None (standalone visualization)
- **Dependencies**: matplotlib, pandas, numpy
- **Used by**: Researchers comparing windows

---

## Data Flow and Relationships

### Main Pipeline Flow
```
Raw Data → preprocessing.py → dbn_dynotears.py → Analysis Scripts
```

### Anomaly Detection Flow
```
Baseline Weights + Anomaly Weights → end_to_end_pipeline.py → Detection Results
```

### Analysis Flow
```
Weight Files → Analysis Scripts → Visualizations/Reports
```

### Key Dependencies
- **launcher.py** orchestrates the main pipeline
- **preprocessing.py** prepares data for DynoTEARS
- **dbn_dynotears.py** performs causal discovery
- **end_to_end_pipeline.py** handles anomaly detection workflow
- **Analysis scripts** provide various analysis capabilities
- **Testing scripts** ensure system reliability

### Input/Output Relationships
- **Raw CSV data** → preprocessing.py → **Differenced data + Lags + MI mask**
- **Preprocessed data** → dbn_dynotears.py → **Weight matrices**
- **Weight matrices** → Analysis scripts → **Analysis results**
- **Baseline + Anomaly weights** → end_to_end_pipeline.py → **Detection + Reconstruction results**

---

## Usage Patterns

### For New Users
1. Start with **README.md**
2. Follow **QUICK_START_COMMANDS.md**
3. Use **launcher.py** for complete pipeline

### For Researchers
1. Use **ANOMALY_ANALYSIS_REPORT.md** for data analysis
2. Use **FINAL_CONCLUSIONS.md** for weight-based analysis
3. Use analysis scripts for specific investigations

### For Developers
1. Read **IMPLEMENTATION_SUMMARY.md** for architecture
2. Use **VALIDATION_CHECKLIST.md** for testing
3. Use testing scripts for validation

### For Advanced Users
1. Use **END_TO_END_GUIDE.md** for complete workflow
2. Use individual scripts for specific tasks
3. Use **GPU_OPTIMIZATION_GUIDE.md** for performance optimization
