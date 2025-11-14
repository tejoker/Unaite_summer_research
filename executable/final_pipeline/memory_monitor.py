#!/usr/bin/env python3
"""
Memory Monitoring Utilities
============================

Provides real-time RAM and GPU memory usage tracking for debugging and optimization.
"""

import logging
import psutil
import os
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Try to import torch for GPU monitoring
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - GPU memory monitoring disabled")


class MemoryMonitor:
    """
    Tracks RAM and GPU memory usage throughout pipeline execution.

    Usage:
        monitor = MemoryMonitor()
        monitor.log("Start processing")
        # ... do work ...
        monitor.log("After preprocessing")
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize memory monitor.

        Args:
            logger: Logger instance (uses module logger if None)
        """
        self.logger = logger or globals()['logger']
        self.process = psutil.Process(os.getpid())
        self.baseline_rss = None
        self.last_rss = None

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with memory stats in GB
        """
        mem_info = self.process.memory_info()

        stats = {
            'rss_gb': mem_info.rss / 1e9,  # Resident Set Size (actual RAM used)
            'vms_gb': mem_info.vms / 1e9,  # Virtual Memory Size
        }

        # GPU memory (if available)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
            stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
            stats['gpu_free_gb'] = (torch.cuda.get_device_properties(0).total_memory
                                    - torch.cuda.memory_reserved()) / 1e9

        return stats

    def log(self, label: str, level: str = "info"):
        """
        Log current memory usage with a descriptive label.

        Args:
            label: Description of current execution point
            level: Logging level ("info", "debug", "warning")
        """
        stats = self.get_memory_stats()

        # Calculate delta from last measurement
        delta_str = ""
        if self.last_rss is not None:
            delta_gb = stats['rss_gb'] - self.last_rss
            delta_str = f" (Δ{delta_gb:+.2f} GB)"

        # Calculate delta from baseline
        baseline_str = ""
        if self.baseline_rss is None:
            self.baseline_rss = stats['rss_gb']
            baseline_str = " [BASELINE]"
        else:
            baseline_delta = stats['rss_gb'] - self.baseline_rss
            baseline_str = f" (total Δ{baseline_delta:+.2f} GB from baseline)"

        # Format message
        msg = (f"[MEMORY] {label}\n"
               f"  RAM: {stats['rss_gb']:.2f} GB{delta_str}{baseline_str}\n"
               f"  VMS: {stats['vms_gb']:.2f} GB")

        if TORCH_AVAILABLE and torch.cuda.is_available():
            msg += (f"\n  GPU: {stats['gpu_allocated_gb']:.2f} GB allocated, "
                   f"{stats['gpu_reserved_gb']:.2f} GB reserved, "
                   f"{stats['gpu_free_gb']:.2f} GB free")

        # Log at appropriate level
        log_func = getattr(self.logger, level.lower())
        log_func(msg)

        # Update last measurement
        self.last_rss = stats['rss_gb']

    def check_threshold(self, threshold_gb: float = 50.0) -> bool:
        """
        Check if memory usage exceeds threshold.

        Args:
            threshold_gb: RAM threshold in GB

        Returns:
            True if over threshold
        """
        stats = self.get_memory_stats()
        if stats['rss_gb'] > threshold_gb:
            self.logger.warning(f"Memory usage ({stats['rss_gb']:.2f} GB) "
                              f"exceeds threshold ({threshold_gb} GB)!")
            return True
        return False

    def reset_baseline(self):
        """Reset baseline measurement for delta calculations."""
        self.baseline_rss = None
        self.last_rss = None


def log_memory_usage(logger: logging.Logger, label: str = ""):
    """
    Simple function for quick memory logging (backwards compatible).

    Args:
        logger: Logger instance
        label: Description label
    """
    monitor = MemoryMonitor(logger)
    monitor.log(label)


if __name__ == "__main__":
    # Test memory monitoring
    logging.basicConfig(level=logging.INFO)

    monitor = MemoryMonitor()
    monitor.log("Initial state")

    # Allocate some memory
    import numpy as np
    data = np.random.randn(1000, 1000, 10)  # ~80MB
    monitor.log("After allocating 80MB array")

    del data
    monitor.log("After deleting array")
