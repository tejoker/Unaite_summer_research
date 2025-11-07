# Frobenius Test - Comprehensive Causal Discovery Analysis

A comprehensive, enterprise-grade statistical analysis tool for comparing causal discovery weight matrices with advanced visualization, batch processing, and statistical analysis capabilities.

## 🚀 Features Overview

### Backward Compatibility
- **Fully compatible** with existing command-line usage
- **Enhanced functionality** when using new options
- **Legacy support** for all existing scripts and workflows

### Core Enhancements
- **Memory-Efficient Processing**: Handles datasets of any size with chunked loading
- **Comprehensive Statistical Analysis**: 15+ distance metrics, bootstrap CI, outlier detection
- **Advanced Visualization**: Interactive plots, network visualizations, statistical dashboards
- **Batch Processing**: Compare entire directories, multiple files, configuration-based analysis
- **Perfect Integration**: Works seamlessly with launcher.py's new configuration naming system

### Statistical Metrics (15+ metrics)
- **Distance Metrics**: Frobenius, L1, L2, L∞, Wasserstein
- **Correlation Analysis**: Pearson, Spearman correlations with confidence intervals
- **Distribution Testing**: Kolmogorov-Smirnov tests, empirical CDFs
- **Error Metrics**: RMSE, MAE, R², MSE
- **Outlier Detection**: IQR, Z-score, Modified Z-score methods
- **Bootstrap Analysis**: Confidence intervals for all key metrics

### Visualization Options
- **5 Static Plot Types**: Dashboard, statistical, temporal, network, advanced analysis
- **Interactive Dashboards**: Plotly-based with hover details and zoom capabilities
- **Network Visualizations**: Adjacency matrix heatmaps showing causal structure differences
- **Themes**: Default and dark themes with customizable styling
- **Export Formats**: PNG, HTML, Excel, JSON

## 📦 Installation

### Quick Start
```bash
# Install all dependencies (enhanced features included)
pip install -r requirements.txt
```

### Manual Installation of Enhanced Dependencies
```bash
pip install plotly>=5.0.0 pyyaml>=6.0 kaleido>=0.2.1 openpyxl>=3.0.0
```

## 🔧 Usage Examples

### 1. Basic Usage (Backward Compatible)
```bash
# Existing commands work exactly the same
python frobenius_test.py --file1 weights1.csv --file2 weights2.csv
python frobenius_test.py --file1 weights1.csv --file2 weights2.csv --save-plot
```

### 2. Enhanced Two-File Analysis
```bash
# With advanced statistics and interactive plots
python frobenius_test.py --file1 weights1.csv --file2 weights2.csv --advanced-stats --interactive-plots

# Dark theme with verbose output
python frobenius_test.py --file1 weights1.csv --file2 weights2.csv --theme dark --verbose
```

### 3. Configuration Directory Comparison (NEW!)
Perfect for comparing results from different launcher.py configurations:

```bash
# Compare full pipeline vs no MI masking
python frobenius_test.py --compare-configs \
    full_mi_rolling_20241201_143022/ \
    no_mi_rolling_20241201_144115/

# Compare baseline vs full configuration with all features
python frobenius_test.py --compare-configs \
    baseline_no_mi_no_rolling_20241201_150045/ \
    full_mi_rolling_20241201_143022/ \
    --advanced-stats --interactive-plots
```

### 4. Multiple File Comparison (NEW!)
```bash
# Compare multiple files pairwise (1-2, 2-3, 3-4)
python frobenius_test.py --files weights1.csv weights2.csv weights3.csv

# Compare first file against all others (1-2, 1-3, 1-4)
python frobenius_test.py --files reference.csv test1.csv test2.csv test3.csv --comparison-mode one_to_many

# Compare all possible pairs (1-2, 1-3, 1-4, 2-3, 2-4, 3-4)
python frobenius_test.py --files weights1.csv weights2.csv weights3.csv weights4.csv --comparison-mode all_pairs
```

### 5. Batch Directory Processing (NEW!)
```bash
# Process entire results directory
python frobenius_test.py --batch-dir results/ --file-pattern "*weights*.csv"

# Process with custom pattern and comparison mode
python frobenius_test.py --batch-dir results/ --file-pattern "*final_weights*.csv" --comparison-mode all_pairs
```

### 6. Configuration-Based Analysis (NEW!)
```bash
# Create configuration template
python frobenius_test.py --create-config my_config.yaml

# Use custom configuration
python frobenius_test.py --config my_config.yaml --batch-dir results/

# Override specific settings
python frobenius_test.py --config my_config.yaml --chunk-size 50000 --max-workers 8
```

## ⚙️ Configuration File

### Create Template
```bash
python frobenius_test.py --create-config analysis_config.yaml
```

### Configuration Options
```yaml
analysis_settings:
  chunk_size: 100000                # Memory management for large files
  bootstrap_samples: 1000           # Statistical confidence intervals
  confidence_level: 0.95            # CI confidence level
  outlier_threshold: 3.0             # Outlier detection sensitivity
  visualization_theme: 'default'    # 'default' or 'dark'

comparison_settings:
  significance_threshold: 0.01       # Statistical significance level
  network_edge_threshold: 90         # Percentile for network visualization
  max_workers: 4                     # Parallel processing workers

output_settings:
  save_static_plots: true            # Generate static PNG plots
  save_interactive_plots: true       # Generate interactive HTML plots
  save_detailed_json: true           # Save comprehensive JSON results
  export_to_excel: false             # Export to Excel format
  plot_dpi: 300                      # Plot resolution

batch_processing:
  comparison_mode: 'pairwise'        # 'pairwise', 'one_to_many', 'all_pairs'
  file_patterns: ['*weights*.csv']   # File search patterns
  recursive_search: true             # Recursive directory search
```

## 📊 Output Files

### Generated Files (Automatically Created)
```
results/Test/  # or your custom --output-dir
├── frobenius_weights_file1_vs_file2_20241201_143022.json    # Comprehensive metrics
├── main_dashboard_file1_vs_file2_20241201_143022.png         # Main comparison dashboard
├── statistical_analysis_file1_vs_file2_20241201_143022.png  # Statistical plots
├── temporal_analysis_file1_vs_file2_20241201_143022.png     # Time-based analysis
├── network_analysis_file1_vs_file2_20241201_143022.png      # Causal network heatmaps
├── advanced_stats_file1_vs_file2_20241201_143022.png        # Advanced statistical plots
└── interactive_comparison_file1_vs_file2_20241201_143022.html # Interactive dashboard
```

### Key Output Information
- **JSON Results**: All metrics, metadata, file information, timestamps
- **Static Plots**: 5 different plot types with detailed analysis
- **Interactive Plots**: Zoomable, hoverable Plotly dashboards
- **Batch Results**: Aggregate statistics across multiple comparisons

## Integration with Pipeline

### Perfect Integration with Launcher Configuration Naming
```bash
# Step 1: Run different configurations with launcher.py
python launcher.py --folder final_pipeline --csv-file data.csv
# → Creates: full_mi_rolling_20241201_143022/

python launcher.py --folder final_pipeline --csv-file data.csv --variants no_mi
# → Creates: no_mi_rolling_20241201_144115/

python launcher.py --folder final_pipeline --csv-file data.csv --variants no_rolling
# → Creates: mi_no_rolling_20241201_145230/

python launcher.py --folder final_pipeline --csv-file data.csv --variants no_mi no_rolling
# → Creates: baseline_no_mi_no_rolling_20241201_150045/

# Step 2: Compare configurations automatically
python frobenius_test.py --compare-configs \
    full_mi_rolling_20241201_143022/ \
    no_mi_rolling_20241201_144115/ \
    --advanced-stats --interactive-plots
```

### Automated Workflow
```bash
# Compare all configurations in results directory
python frobenius_test.py --batch-dir results/ --comparison-mode all_pairs

# Compare specific configuration patterns
python frobenius_test.py --batch-dir results/ --file-pattern "*mi_rolling*" --comparison-mode pairwise
```

## Key Metrics Explained

### Basic Metrics (Always Computed)
- **Frobenius Distance**: √(Σ(W₁ᵢⱼ - W₂ᵢⱼ)²) - Overall matrix difference magnitude
- **Normalized Frobenius**: Scale-invariant comparison (0 = identical, higher = more different)
- **Relative Distance (%)**: Percentage difference for easy interpretation

### Advanced Metrics (with --advanced-stats)
- **Pearson/Spearman Correlation**: Relationship strength (-1 to 1)
- **R²**: Explained variance (0 to 1, higher = more similar)
- **RMSE/MAE**: Root mean square error / Mean absolute error
- **KS Statistic**: Distribution difference test (0 = same distribution)
- **Bootstrap CI**: Confidence intervals for robustness assessment

### Outlier Analysis
- **IQR Method**: Interquartile range based outlier detection
- **Z-Score Method**: Standard deviation based outliers
- **Modified Z-Score**: Robust outlier detection using median absolute deviation

## Troubleshooting

### Common Issues

#### Memory Problems with Large Files
```bash
# Reduce chunk size for very large files (>1GB)
python frobenius_test.py --file1 huge_file.csv --file2 huge_file2.csv --chunk-size 10000
```

#### Missing Dependencies
```bash
# Check which enhanced features are available
python -c \"import plotly, yaml; print('All enhanced features available')\"

# Install missing packages individually
pip install plotly pyyaml kaleido openpyxl
```

#### Performance Issues
```bash
# Use parallel processing for batch operations
python frobenius_test.py --batch-dir results/ --max-workers 8

# Skip expensive features for large batch jobs
python frobenius_test.py --batch-dir results/ --comparison-mode pairwise  # (no --advanced-stats)
```

### Error Recovery
- **Graceful handling** of malformed CSV files
- **Partial failure recovery** in batch processing
- **Detailed error logging** in frobenius_analysis.log
- **Progress tracking** for long-running operations

## Real-World Use Cases

### 1. Configuration Impact Analysis
Compare how different DynoTEARS configurations affect causal discovery:
```bash
python frobenius_test.py --compare-configs \
    full_mi_rolling_20241201_143022/ \
    baseline_no_mi_no_rolling_20241201_150045/ \
    --advanced-stats
```

### 2. Temporal Stability Analysis
Analyze how causal relationships evolve over time:
```bash
python frobenius_test.py --files \
    weights_early_period.csv \
    weights_middle_period.csv \
    weights_late_period.csv \
    --comparison-mode all_pairs --interactive-plots
```

### 3. Parameter Sensitivity Study
Compare results across different hyperparameter settings:
```bash
python frobenius_test.py --batch-dir parameter_sweep_results/ \
    --comparison-mode one_to_many --file-pattern \"*final_weights*.csv\"
```

### 4. Quality Assurance & Reproducibility
Validate consistency across multiple runs:
```bash
python frobenius_test.py --files \
    run1_weights.csv run2_weights.csv run3_weights.csv \
    --comparison-mode all_pairs --advanced-stats
```

### 5. Production Monitoring
Monitor changes in causal relationships over time:
```bash
python frobenius_test.py --compare-configs \
    production_baseline_weights/ \
    current_production_weights/ \
    --advanced-stats --export-excel
```

## Backward Compatibility

### All Existing Commands Work
```bash
# Original simple usage (still works exactly the same)
python frobenius_test.py --file1 weights1.csv --file2 weights2.csv --save-plot

# Enhanced version (new capabilities)
python frobenius_test.py --file1 weights1.csv --file2 weights2.csv --advanced-stats --interactive-plots
```

### Legacy Function Support
- `load_weights_csv()`: Now uses enhanced memory-efficient loader
- `compute_weights_frobenius_distance()`: Now returns comprehensive metrics
- `create_weights_comparison_plots()`: Now creates advanced visualization suite

### Migration Path
1. **Phase 1**: Use existing commands (no changes needed)
2. **Phase 2**: Add `--advanced-stats` for enhanced metrics
3. **Phase 3**: Add `--interactive-plots` for interactive visualizations
4. **Phase 4**: Use batch processing for configuration comparisons

## Performance

### Memory Optimization
- **Chunked Loading**: Files >100MB processed in chunks
- **Optimal Data Types**: int16/int32, float32 for memory efficiency
- **Progress Tracking**: Real-time progress bars for long operations

### Processing Speed
- **Parallel Processing**: Multi-worker support for batch operations
- **Vectorized Operations**: NumPy optimizations throughout
- **Smart Caching**: Avoid recomputation where possible

### Scalability
- **Large Dataset Support**: Handles files up to several GB
- **Batch Processing**: Process hundreds of files efficiently
- **Resource Management**: Configurable memory and CPU usage

## Quick Reference

### Common Commands
```bash
# Basic comparison (legacy compatible)
python frobenius_test.py --file1 w1.csv --file2 w2.csv

# Compare launcher.py configurations
python frobenius_test.py --compare-configs config1/ config2/

# Batch process with advanced features
python frobenius_test.py --batch-dir results/ --advanced-stats

# Create and use custom configuration
python frobenius_test.py --create-config my_config.yaml
python frobenius_test.py --config my_config.yaml --batch-dir results/
```

### Output Interpretation
- **Frobenius Distance < 0.01**: Very similar matrices
- **Correlation > 0.95**: Strong linear relationship
- **R² > 0.90**: Good predictive relationship
- **KS p-value > 0.05**: Distributions not significantly different
- **Outlier % < 5%**: Normal variation range

---

**Note**: This enhanced version maintains 100% backward compatibility while providing comprehensive new capabilities for advanced causal discovery analysis. All existing scripts and workflows continue to work unchanged.