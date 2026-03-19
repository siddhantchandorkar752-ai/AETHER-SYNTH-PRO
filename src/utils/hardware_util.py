import torch
from loguru import logger

def get_precision_device():
    """
    Hardware-Aware Device Selector.
    Optimizes for NVIDIA CUDA, fallback to CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Optimization: Clear reserved memory before allocation
        torch.cuda.empty_cache()
        gpu_name = torch.cuda.get_device_name(0)
        logger.success(f"🔥 HARDWARE: High-Performance GPU Detected -> {gpu_name}")
    else:
        device = torch.device("cpu")
        logger.warning("⚠️ HARDWARE: CUDA not found. Running on CPU mode.")
    
    return device

def log_vram_usage(device):
    """Logs real-time VRAM stats to prevent OOM errors."""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        logger.debug(f"VRAM: {allocated:.2f}MB Allocated | {reserved:.2f}MB Reserved")
