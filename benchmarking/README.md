# Benchmarking Utilities

Performance profiling and resource monitoring tools for the Tucker-CAM pipeline.

## Files

### `performance_benchmark.py`
Benchmarks specific functions and operations within the Tucker-CAM pipeline.

**Usage:**
```bash
cd benchmarking/
python performance_benchmark.py
```

**Purpose:** Micro-benchmarks for optimization decisions (e.g., testing different chunking strategies, comparing einsum vs matmul).

### `resource_manager.py`
Resource monitoring and management utilities used by `performance_benchmark.py`.

**Features:**
- CPU/memory tracking
- GPU monitoring (if available)
- Resource statistics collection

## Note

These utilities are for **development and profiling only**, not part of the production pipeline.

For end-to-end benchmarking of the full pipeline, use:
```bash
bash run_tucker_cam_benchmark.sh
```
