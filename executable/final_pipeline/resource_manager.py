#!/usr/bin/env python3
"""
Resource Manager for GPU/CPU Coordination in DynoTears Pipeline

This module manages system resources, monitors performance, and coordinates
workload distribution between CPU cores and GPU devices for optimal performance.
"""

import os
import time
import logging
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue

import numpy as np
import torch
import psutil

logger = logging.getLogger(__name__)

@dataclass
class SystemResources:
    """System resource information"""
    cpu_count: int
    cpu_percent: float
    memory_total: int  # MB
    memory_available: int  # MB
    gpu_count: int
    gpu_memory_total: List[int]  # MB per GPU
    gpu_memory_available: List[int]  # MB per GPU
    gpu_utilization: List[float]  # % utilization per GPU

@dataclass
class WorkloadConfig:
    """Configuration for workload distribution"""
    cpu_workers: int
    gpu_batch_size: int
    use_mixed_precision: bool
    enable_cuda_streams: bool
    memory_limit_mb: int

class ResourceMonitor:
    """Monitor system resources in real-time"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._current_resources = None
        self._resource_history = []
        
    def start(self):
        """Start resource monitoring thread"""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            logger.info("Resource monitoring started")
    
    def stop(self):
        """Stop resource monitoring thread"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._stop_event.set()
            self._monitor_thread.join()
            logger.info("Resource monitoring stopped")
    
    def get_current_resources(self) -> Optional[SystemResources]:
        """Get current system resource status"""
        return self._current_resources
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                resources = self._collect_resources()
                self._current_resources = resources
                self._resource_history.append((time.time(), resources))
                
                # Keep only last 100 readings
                if len(self._resource_history) > 100:
                    self._resource_history.pop(0)
                    
            except Exception as e:
                logger.warning(f"Error collecting resources: {e}")
            
            self._stop_event.wait(self.update_interval)
    
    def _collect_resources(self) -> SystemResources:
        """Collect current system resource information"""
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_total = int(memory.total / (1024 * 1024))
        memory_available = int(memory.available / (1024 * 1024))
        
        # GPU info
        gpu_count = 0
        gpu_memory_total = []
        gpu_memory_available = []
        gpu_utilization = []
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                total_mb = props.total_memory // (1024 * 1024)
                
                # Get current GPU memory usage
                torch.cuda.set_device(i)
                allocated_mb = torch.cuda.memory_allocated(i) // (1024 * 1024)
                available_mb = total_mb - allocated_mb
                
                gpu_memory_total.append(total_mb)
                gpu_memory_available.append(available_mb)
                
                # GPU utilization (simplified - based on memory usage)
                utilization = (allocated_mb / total_mb) * 100 if total_mb > 0 else 0
                gpu_utilization.append(utilization)
        
        return SystemResources(
            cpu_count=cpu_count,
            cpu_percent=cpu_percent,
            memory_total=memory_total,
            memory_available=memory_available,
            gpu_count=gpu_count,
            gpu_memory_total=gpu_memory_total,
            gpu_memory_available=gpu_memory_available,
            gpu_utilization=gpu_utilization
        )

class WorkloadOptimizer:
    """Optimize workload distribution based on system resources"""
    
    def __init__(self, monitor: ResourceMonitor):
        self.monitor = monitor
        
    def optimize_config(self, 
                       data_size: Tuple[int, int],
                       target_memory_usage: float = 0.8) -> WorkloadConfig:
        """
        Optimize workload configuration based on current system resources
        
        Args:
            data_size: (n_samples, n_variables)
            target_memory_usage: Target GPU memory usage ratio (0.0-1.0)
        
        Returns:
            Optimized workload configuration
        """
        resources = self.monitor.get_current_resources()
        if resources is None:
            # Fallback configuration
            return self._get_fallback_config()
        
        # Calculate optimal CPU workers
        cpu_workers = max(1, min(resources.cpu_count - 1, 8))
        if resources.cpu_percent > 80:
            cpu_workers = max(1, cpu_workers // 2)
        
        # Calculate optimal GPU batch size
        gpu_batch_size = self._calculate_gpu_batch_size(
            data_size, resources, target_memory_usage
        )
        
        # Enable mixed precision for large models
        use_mixed_precision = (
            resources.gpu_count > 0 and 
            data_size[1] > 50  # More than 50 variables
        )
        
        # Enable CUDA streams for multiple GPUs
        enable_cuda_streams = resources.gpu_count > 1
        
        # Set memory limit
        memory_limit_mb = int(resources.memory_available * 0.8)
        
        config = WorkloadConfig(
            cpu_workers=cpu_workers,
            gpu_batch_size=gpu_batch_size,
            use_mixed_precision=use_mixed_precision,
            enable_cuda_streams=enable_cuda_streams,
            memory_limit_mb=memory_limit_mb
        )
        
        logger.info(f"Optimized config: {config}")
        return config
    
    def _calculate_gpu_batch_size(self, 
                                  data_size: Tuple[int, int],
                                  resources: SystemResources,
                                  target_usage: float) -> int:
        """Calculate optimal GPU batch size"""
        if resources.gpu_count == 0:
            return 32  # CPU fallback
        
        n_samples, n_vars = data_size
        
        # Estimate memory per sample (bytes)
        # Rough estimate: each sample needs ~8 * n_vars * n_vars bytes for matrices
        bytes_per_sample = 8 * n_vars * n_vars * 4  # float32
        
        # Available GPU memory
        available_mb = max(resources.gpu_memory_available) if resources.gpu_memory_available else 1000
        target_mb = available_mb * target_usage
        target_bytes = target_mb * 1024 * 1024
        
        # Calculate batch size
        batch_size = max(1, int(target_bytes // bytes_per_sample))
        batch_size = min(batch_size, n_samples)
        batch_size = max(batch_size, 8)  # Minimum batch size
        batch_size = min(batch_size, 512)  # Maximum batch size
        
        return batch_size
    
    def _get_fallback_config(self) -> WorkloadConfig:
        """Get fallback configuration when monitoring is unavailable"""
        cpu_count = mp.cpu_count()
        gpu_available = torch.cuda.is_available()
        
        return WorkloadConfig(
            cpu_workers=max(1, cpu_count // 2),
            gpu_batch_size=32 if gpu_available else 8,
            use_mixed_precision=False,
            enable_cuda_streams=False,
            memory_limit_mb=4000
        )

class TaskManager:
    """Manage parallel task execution across CPU and GPU"""
    
    def __init__(self, config: WorkloadConfig):
        self.config = config
        self.cpu_executor = None
        self.gpu_task_queue = queue.Queue()
        self._gpu_worker_thread = None
        self._stop_event = threading.Event()
        
    def start(self):
        """Start task execution managers"""
        # Start CPU executor
        self.cpu_executor = ProcessPoolExecutor(max_workers=self.config.cpu_workers)
        
        # Start GPU worker thread
        if torch.cuda.is_available():
            self._stop_event.clear()
            self._gpu_worker_thread = threading.Thread(target=self._gpu_worker_loop)
            self._gpu_worker_thread.daemon = True
            self._gpu_worker_thread.start()
        
        logger.info(f"Task manager started with {self.config.cpu_workers} CPU workers")
    
    def stop(self):
        """Stop task execution managers"""
        if self.cpu_executor:
            self.cpu_executor.shutdown(wait=True)
            self.cpu_executor = None
        
        if self._gpu_worker_thread and self._gpu_worker_thread.is_alive():
            self._stop_event.set()
            self._gpu_worker_thread.join()
            self._gpu_worker_thread = None
        
        logger.info("Task manager stopped")
    
    def submit_cpu_task(self, func, *args, **kwargs):
        """Submit a CPU-intensive task"""
        if self.cpu_executor:
            return self.cpu_executor.submit(func, *args, **kwargs)
        else:
            # Fallback: run synchronously
            return func(*args, **kwargs)
    
    def submit_gpu_task(self, task_data):
        """Submit a GPU task to the queue"""
        self.gpu_task_queue.put(task_data)
    
    def _gpu_worker_loop(self):
        """GPU worker thread main loop"""
        while not self._stop_event.is_set():
            try:
                task_data = self.gpu_task_queue.get(timeout=1.0)
                # Process GPU task here
                self._process_gpu_task(task_data)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"GPU task error: {e}")
    
    def _process_gpu_task(self, task_data):
        """Process a GPU task"""
        # Placeholder for GPU task processing
        # This would be implemented based on specific task requirements
        pass

class ResourceManager:
    """Main resource manager coordinating all components"""
    
    def __init__(self):
        self.monitor = ResourceMonitor(update_interval=2.0)
        self.optimizer = WorkloadOptimizer(self.monitor)
        self.task_manager = None
        self.current_config = None
        
    def initialize(self, data_shape: Tuple[int, int]) -> WorkloadConfig:
        """Initialize resource management for given data shape"""
        self.monitor.start()
        
        # Wait a moment for initial resource collection
        time.sleep(1.0)
        
        # Optimize configuration
        self.current_config = self.optimizer.optimize_config(data_shape)
        
        # Start task manager
        self.task_manager = TaskManager(self.current_config)
        self.task_manager.start()
        
        return self.current_config
    
    def shutdown(self):
        """Shutdown resource management"""
        if self.task_manager:
            self.task_manager.stop()
        self.monitor.stop()
        
    def get_optimal_batch_size(self) -> int:
        """Get current optimal batch size"""
        if self.current_config:
            return self.current_config.gpu_batch_size
        return 32
    
    def get_cpu_workers(self) -> int:
        """Get current number of CPU workers"""
        if self.current_config:
            return self.current_config.cpu_workers
        return max(1, mp.cpu_count() // 2)
    
    def submit_cpu_task(self, func, *args, **kwargs):
        """Submit CPU task through task manager"""
        if self.task_manager:
            return self.task_manager.submit_cpu_task(func, *args, **kwargs)
        return func(*args, **kwargs)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource statistics"""
        resources = self.monitor.get_current_resources()
        if not resources:
            return {}
        
        return {
            'cpu_count': resources.cpu_count,
            'cpu_usage': resources.cpu_percent,
            'memory_total_mb': resources.memory_total,
            'memory_available_mb': resources.memory_available,
            'gpu_count': resources.gpu_count,
            'gpu_memory_total_mb': resources.gpu_memory_total,
            'gpu_memory_available_mb': resources.gpu_memory_available,
            'gpu_utilization': resources.gpu_utilization
        }

# Global resource manager instance
_resource_manager = None

def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance"""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager

def initialize_resources(data_shape: Tuple[int, int]) -> WorkloadConfig:
    """Initialize global resource management"""
    manager = get_resource_manager()
    return manager.initialize(data_shape)

def shutdown_resources():
    """Shutdown global resource management"""
    global _resource_manager
    if _resource_manager:
        _resource_manager.shutdown()
        _resource_manager = None