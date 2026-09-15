import logging
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DeviceManager:
    """Manages device (CPU/GPU/MPS) placement and GPU memory logging."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, fp16: bool = False):
        if self._initialized:
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.fp16 = fp16
        self.is_cuda = self.device.type == "cuda"
        
        self._log_hardware_info()
        self._initialized = True

    def _log_hardware_info(self) -> None:
        if self.is_cuda:
            vram = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            logger.info(f"Using device: {self.device}")
            logger.info(f"GPU Name: {torch.cuda.get_device_name(self.device)}")
            logger.info(f"Total VRAM: {vram:.2f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        elif self.device.type == "mps":
            logger.info(f"Using device: {self.device} (Apple Silicon)")
        else:
            logger.warning("CUDA/MPS unavailable. Using CPU. Performance will be degraded.")

    def get_device(self) -> torch.device:
        """Returns the current torch device."""
        return self.device
        
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage in MB."""
        if not self.is_cuda:
            return {'allocated_mb': 0.0, 'reserved_mb': 0.0, 'peak_mb': 0.0}
            
        allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
        peak = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'peak_mb': peak
        }
        
    def log_gpu_memory(self, tag: str) -> None:
        """Logs GPU memory at checkpoints."""
        if not self.is_cuda:
            return
            
        usage = self.get_gpu_memory_usage()
        logger.info(f"[GPU Memory | {tag}] Allocated: {usage['allocated_mb']:.2f} MB | Reserved: {usage['reserved_mb']:.2f} MB | Peak: {usage['peak_mb']:.2f} MB")

    def move_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Move model to device and cast to half precision if FP16 is enabled."""
        model = model.to(self.device)
        if self.fp16 and self.is_cuda:
            model = model.half()
        return model
