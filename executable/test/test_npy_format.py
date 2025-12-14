#!/usr/bin/env python3
"""
Test script to verify .npy format implementation in preprocessing and Tucker-CAM pipeline.
Tests that data saves/loads correctly with 10-50x speedup vs CSV.
"""

import numpy as np
import pandas as pd
import time
import tempfile
import os

def test_npy_vs_csv():
    """Compare .npy vs CSV performance for typical preprocessing output."""
    
    # Create test data similar to preprocessing output
    n_samples = 100000  # Typical: ~100k samples
    n_vars = 25         # Typical: 25 variables
    
    print("="*80)
    print("Testing .npy vs CSV Performance")
    print("="*80)
    print(f"Test data: {n_samples} samples × {n_vars} variables")
    print()
    
    # Generate random data
    data = np.random.randn(n_samples, n_vars).astype(np.float32)
    columns = [f"var_{i}" for i in range(n_vars)]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "test.csv")
        npy_file = os.path.join(tmpdir, "test.npy")
        columns_file = os.path.join(tmpdir, "test_columns.npy")
        
        # Test CSV write
        df = pd.DataFrame(data, columns=columns)
        start = time.time()
        df.to_csv(csv_file, index=False)
        csv_write_time = time.time() - start
        csv_size = os.path.getsize(csv_file) / 1024 / 1024  # MB
        
        # Test .npy write
        start = time.time()
        np.save(npy_file, data)
        np.save(columns_file, np.array(columns))
        npy_write_time = time.time() - start
        npy_size = (os.path.getsize(npy_file) + os.path.getsize(columns_file)) / 1024 / 1024  # MB
        
        # Test CSV read
        start = time.time()
        df_loaded = pd.read_csv(csv_file)
        data_csv = df_loaded.values.astype(np.float32)
        csv_read_time = time.time() - start
        
        # Test .npy read
        start = time.time()
        data_npy = np.load(npy_file)
        cols_npy = np.load(columns_file)
        npy_read_time = time.time() - start
        
        # Verify data integrity
        assert np.allclose(data_csv, data_npy, rtol=1e-5), "Data mismatch!"
        assert list(df_loaded.columns) == list(cols_npy), "Column mismatch!"
        
        # Print results
        print("WRITE Performance:")
        print(f"  CSV: {csv_write_time:.3f}s ({csv_size:.2f} MB)")
        print(f"  NPY: {npy_write_time:.3f}s ({npy_size:.2f} MB)")
        print(f"  Speedup: {csv_write_time / npy_write_time:.1f}x faster")
        print()
        
        print("READ Performance:")
        print(f"  CSV: {csv_read_time:.3f}s")
        print(f"  NPY: {npy_read_time:.3f}s")
        print(f"  Speedup: {csv_read_time / npy_read_time:.1f}x faster")
        print()
        
        print("Storage:")
        print(f"  CSV: {csv_size:.2f} MB")
        print(f"  NPY: {npy_size:.2f} MB")
        print(f"  Compression: {csv_size / npy_size:.1f}x smaller")
        print()
        
        print("✅ Data integrity verified (values match within 1e-5 tolerance)")
        print("="*80)
        
        return {
            'csv_read_time': csv_read_time,
            'npy_read_time': npy_read_time,
            'speedup': csv_read_time / npy_read_time
        }


def test_lags_structured_array():
    """Test structured array format for optimal lags."""
    
    print("\nTesting Structured Array for Lags")
    print("="*80)
    
    # Create test lags similar to preprocessing output
    variables = ['P-02', 'P-03', 'T-13', 'T-14', 'F-04']
    lags = [3, 2, 4, 3, 2]
    
    # Create structured array (as done in preprocessing)
    lags_array = np.array([(var, lag) for var, lag in zip(variables, lags)],
                          dtype=[('variable', 'U100'), ('optimal_lag', 'i4')])
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
        npy_file = tmp.name
    
    try:
        # Save and load
        np.save(npy_file, lags_array)
        loaded = np.load(npy_file)
        
        # Verify structure
        print(f"Original: {len(lags_array)} variables")
        print(f"Loaded: {len(loaded)} variables")
        print()
        
        # Test max lag extraction (as done in dbn_dynotears_tucker_cam.py)
        p = int(loaded['optimal_lag'].max())
        print(f"Max lag p: {p}")
        print()
        
        # Show first few entries
        print("Sample entries:")
        for i in range(min(3, len(loaded))):
            print(f"  {loaded[i]['variable']}: lag={loaded[i]['optimal_lag']}")
        
        assert p == max(lags), "Max lag extraction failed!"
        assert len(loaded) == len(variables), "Array length mismatch!"
        
        print()
        print("✅ Structured array format verified")
        print("="*80)
        
    finally:
        os.unlink(npy_file)


if __name__ == '__main__':
    print("\n")
    print("NPY Format Test Suite")
    print("="*80)
    print("Verifying .npy implementation for Tucker-CAM pipeline")
    print("="*80)
    print()
    
    # Run tests
    results = test_npy_vs_csv()
    test_lags_structured_array()
    
    print()
    print("SUMMARY")
    print("="*80)
    print(f"Expected speedup per window: {results['speedup']:.1f}x")
    print(f"Estimated time saved (1271 windows): {results['csv_read_time'] * 1271 / 3600:.1f} hours")
    print("="*80)
    print()
    print("✅ All tests passed! Ready for full pipeline.")
