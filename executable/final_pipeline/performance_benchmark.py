#!/usr/bin/env python3
"""
Performance Benchmark Script for GPU-Optimized DynoTears Pipeline

This script benchmarks the performance improvements of the GPU-optimized
DynoTears pipeline compared to the original CPU-only implementation.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import psutil
import json

from resource_manager import initialize_resources, shutdown_resources, get_resource_manager
from preprocessing import test_stationarity_parallel, find_optimal_lags_parallel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('benchmark_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Performance benchmark for DynoTears pipeline optimization"""
    
    def __init__(self):
        self.results = {
            'system_info': self._get_system_info(),
            'benchmarks': {}
        }
        
    def _get_system_info(self):
        """Collect system information"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    'name': props.name,
                    'memory_gb': props.total_memory / (1024**3)
                })
            info['gpu_info'] = gpu_info
        
        return info
    
    def benchmark_preprocessing(self, df, n_runs=3):
        """Benchmark preprocessing operations"""
        logger.info("Benchmarking preprocessing operations...")
        
        results = {
            'stationarity_tests': {},
            'optimal_lags': {}
        }
        
        # Benchmark stationarity tests
        logger.info("Benchmarking stationarity tests...")
        
        # Single-threaded baseline
        start_time = time.time()
        for _ in range(n_runs):
            from preprocessing import parallel_stationarity_test
            series_data = [(col, df[col].values) for col in df.columns]
            for data in series_data:
                parallel_stationarity_test(data)
        single_thread_time = (time.time() - start_time) / n_runs
        
        # Multi-threaded optimized
        start_time = time.time()
        for _ in range(n_runs):
            test_stationarity_parallel(df)
        multi_thread_time = (time.time() - start_time) / n_runs
        
        results['stationarity_tests'] = {
            'single_thread_seconds': single_thread_time,
            'multi_thread_seconds': multi_thread_time,
            'speedup': single_thread_time / multi_thread_time,
            'series_count': len(df.columns)
        }
        
        # Benchmark optimal lag calculation
        logger.info("Benchmarking optimal lag calculation...")
        
        # Single-threaded baseline
        start_time = time.time()
        for _ in range(n_runs):
            from preprocessing import find_optimal_lag_single
            series_data = [(col, df[col].values) for col in df.columns[:min(10, len(df.columns))]]  # Limit for speed
            for data in series_data:
                find_optimal_lag_single(data)
        single_thread_lag_time = (time.time() - start_time) / n_runs
        
        # Multi-threaded optimized
        start_time = time.time()
        for _ in range(n_runs):
            find_optimal_lags_parallel(df.iloc[:, :min(10, len(df.columns))])  # Limit for speed
        multi_thread_lag_time = (time.time() - start_time) / n_runs
        
        results['optimal_lags'] = {
            'single_thread_seconds': single_thread_lag_time,
            'multi_thread_seconds': multi_thread_lag_time,
            'speedup': single_thread_lag_time / multi_thread_lag_time,
            'series_count': min(10, len(df.columns))
        }
        
        return results
    
    def benchmark_gpu_memory_management(self, data_sizes):
        """Benchmark GPU memory management and batch sizing"""
        if not torch.cuda.is_available():
            logger.warning("GPU not available, skipping GPU benchmarks")
            return {}
        
        logger.info("Benchmarking GPU memory management...")
        
        from dynotears import gpu_manager
        
        results = {}
        
        for n_samples, n_vars in data_sizes:
            logger.info(f"Testing data size: {n_samples} samples, {n_vars} variables")
            
            # Generate synthetic data
            data = np.random.randn(n_samples, n_vars).astype(np.float32)
            data_tensor = torch.from_numpy(data).cuda()
            
            # Test adaptive batch sizing
            start_time = time.time()
            batch_size = gpu_manager.estimate_batch_size(n_samples, n_vars, 2)  # p=2 lags
            batch_estimation_time = time.time() - start_time
            
            # Test memory monitoring
            initial_memory = gpu_manager.get_available_memory()
            
            # Simulate processing batches
            processed_batches = 0
            start_time = time.time()
            
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_data = data_tensor[i:end_idx]
                
                # Simulate processing (matrix operations)
                result = torch.matmul(batch_data, batch_data.T)
                processed_batches += 1
                
                if processed_batches >= 10:  # Limit for benchmark
                    break
            
            processing_time = time.time() - start_time
            final_memory = gpu_manager.get_available_memory()
            
            results[f"{n_samples}x{n_vars}"] = {
                'estimated_batch_size': batch_size,
                'batch_estimation_time_ms': batch_estimation_time * 1000,
                'processing_time_seconds': processing_time,
                'processed_batches': processed_batches,
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_used_mb': initial_memory - final_memory
            }
            
            # Cleanup
            del data_tensor, result
            torch.cuda.empty_cache()
        
        return results
    
    def benchmark_resource_management(self, data_shape):
        """Benchmark resource management initialization and optimization"""
        logger.info("Benchmarking resource management...")
        
        start_time = time.time()
        resource_config = initialize_resources(data_shape)
        init_time = time.time() - start_time
        
        resource_manager = get_resource_manager()
        
        start_time = time.time()
        stats = resource_manager.get_resource_stats()
        stats_time = time.time() - start_time
        
        start_time = time.time()
        shutdown_resources()
        shutdown_time = time.time() - start_time
        
        return {
            'initialization_time_ms': init_time * 1000,
            'stats_collection_time_ms': stats_time * 1000,
            'shutdown_time_ms': shutdown_time * 1000,
            'config': {
                'cpu_workers': resource_config.cpu_workers,
                'gpu_batch_size': resource_config.gpu_batch_size,
                'use_mixed_precision': resource_config.use_mixed_precision,
                'enable_cuda_streams': resource_config.enable_cuda_streams
            },
            'detected_resources': stats
        }
    
    def run_full_benchmark(self, data_file=None, synthetic_size=(1000, 50)):
        """Run complete benchmark suite"""
        logger.info("Starting full performance benchmark...")
        
        # Load or generate data
        if data_file and os.path.exists(data_file):
            logger.info(f"Loading data from {data_file}")
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        else:
            logger.info(f"Generating synthetic data: {synthetic_size}")
            n_samples, n_vars = synthetic_size
            data = np.random.randn(n_samples, n_vars)
            df = pd.DataFrame(data, columns=[f'var_{i}' for i in range(n_vars)])
        
        logger.info(f"Benchmark data shape: {df.shape}")
        
        # Run benchmarks
        self.results['benchmarks']['preprocessing'] = self.benchmark_preprocessing(df)
        
        self.results['benchmarks']['gpu_memory'] = self.benchmark_gpu_memory_management([
            (500, 20), (1000, 50), (2000, 100)
        ])
        
        self.results['benchmarks']['resource_management'] = self.benchmark_resource_management(df.shape)
        
        # Calculate overall performance improvements
        preprocessing_results = self.results['benchmarks']['preprocessing']
        overall_speedup = (
            preprocessing_results['stationarity_tests']['speedup'] +
            preprocessing_results['optimal_lags']['speedup']
        ) / 2
        
        self.results['summary'] = {
            'overall_preprocessing_speedup': overall_speedup,
            'gpu_available': torch.cuda.is_available(),
            'recommended_settings': self._get_recommendations()
        }
        
        return self.results
    
    def _get_recommendations(self):
        """Generate performance recommendations"""
        recommendations = []
        
        if torch.cuda.is_available():
            recommendations.append("GPU acceleration available - recommended for large datasets")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory > 8:
                recommendations.append("High GPU memory - can handle large batch sizes")
            else:
                recommendations.append("Limited GPU memory - use smaller batch sizes")
        else:
            recommendations.append("No GPU detected - rely on CPU multiprocessing")
        
        cpu_count = psutil.cpu_count()
        if cpu_count >= 8:
            recommendations.append(f"High CPU core count ({cpu_count}) - excellent for parallel preprocessing")
        elif cpu_count >= 4:
            recommendations.append(f"Moderate CPU cores ({cpu_count}) - good parallel performance expected")
        else:
            recommendations.append(f"Limited CPU cores ({cpu_count}) - consider upgrading for better performance")
        
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb >= 16:
            recommendations.append("Sufficient memory for large datasets")
        else:
            recommendations.append("Limited memory - consider smaller batch sizes or data chunking")
        
        return recommendations
    
    def save_results(self, output_file="benchmark_results.json"):
        """Save benchmark results to file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {output_file}")
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("DYNOTEARS PIPELINE PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        
        # System info
        sys_info = self.results['system_info']
        print(f"\nSystem Information:")
        print(f"  CPU Cores: {sys_info['cpu_count']}")
        print(f"  Memory: {sys_info['memory_total_gb']:.1f} GB")
        print(f"  GPU Available: {sys_info['gpu_available']}")
        if sys_info['gpu_available']:
            for i, gpu in enumerate(sys_info['gpu_info']):
                print(f"  GPU {i}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
        
        # Preprocessing results
        if 'preprocessing' in self.results['benchmarks']:
            prep = self.results['benchmarks']['preprocessing']
            print(f"\nPreprocessing Performance:")
            print(f"  Stationarity Tests:")
            print(f"    Single-threaded: {prep['stationarity_tests']['single_thread_seconds']:.3f}s")
            print(f"    Multi-threaded: {prep['stationarity_tests']['multi_thread_seconds']:.3f}s")
            print(f"    Speedup: {prep['stationarity_tests']['speedup']:.2f}x")
            print(f"  Optimal Lag Calculation:")
            print(f"    Single-threaded: {prep['optimal_lags']['single_thread_seconds']:.3f}s")
            print(f"    Multi-threaded: {prep['optimal_lags']['multi_thread_seconds']:.3f}s")
            print(f"    Speedup: {prep['optimal_lags']['speedup']:.2f}x")
        
        # Summary
        if 'summary' in self.results:
            summary = self.results['summary']
            print(f"\nOverall Performance:")
            print(f"  Preprocessing Speedup: {summary['overall_preprocessing_speedup']:.2f}x")
            print(f"\nRecommendations:")
            for rec in summary['recommended_settings']:
                print(f"  • {rec}")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='DynoTears Pipeline Performance Benchmark')
    parser.add_argument('--data', type=str, help='Path to data file for benchmarking')
    parser.add_argument('--synthetic-size', nargs=2, type=int, default=[1000, 50],
                       help='Size of synthetic data (samples, variables)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs for averaging')
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark()
    
    try:
        results = benchmark.run_full_benchmark(
            data_file=args.data,
            synthetic_size=tuple(args.synthetic_size)
        )
        
        benchmark.save_results(args.output)
        benchmark.print_summary()
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())