import time
import logging
import psutil
import torch
import shutil
import numpy as np
from contextlib import contextmanager
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitors resource usage (CPU/GPU) and tracks latencies across execution stages."""
    def __init__(self):
        self.latencies: Dict[str, List[float]] = {}
        self.peak_gpu_mb = 0.0

    @contextmanager
    def track_latency(self, stage_name: str):
        """Context manager to track execution time of a code block."""
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        
        duration = end - start
        if stage_name not in self.latencies:
            self.latencies[stage_name] = []
        self.latencies[stage_name].append(duration)

    def get_latencies(self) -> Dict[str, List[float]]:
        """Get all recorded latencies."""
        return self.latencies

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of latencies."""
        summary = {}
        for stage, times in self.latencies.items():
            if not times:
                continue
            arr = np.array(times)
            summary[stage] = {
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'p95': float(np.percentile(arr, 95)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'count': len(arr)
            }
        return summary

    def track_gpu_memory(self) -> None:
        """Record peak GPU memory."""
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            if peak > self.peak_gpu_mb:
                self.peak_gpu_mb = peak

    def get_system_info(self) -> Dict[str, Any]:
        """Get basic system specs."""
        cpu_count = psutil.cpu_count(logical=True)
        ram = psutil.virtual_memory()
        disk = shutil.disk_usage("/")
        
        info = {
            'cpu_count': cpu_count,
            'ram_total_gb': ram.total / (1024**3),
            'ram_available_gb': ram.available / (1024**3),
            'disk_total_gb': disk.total / (1024**3),
            'disk_free_gb': disk.free / (1024**3)
        }
        
        if torch.cuda.is_available():
            info['gpu_peak_allocated_mb'] = self.peak_gpu_mb
            info['gpu_current_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
            
        return info

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.latencies.clear()
        self.peak_gpu_mb = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
