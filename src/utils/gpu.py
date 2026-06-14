"""GPU utility module for AQRF.

Handles CUDA setup, VRAM monitoring, and OOM recovery for RTX 3050 4GB.
"""
import os
import gc
import warnings
from typing import Optional

import torch
import structlog

logger = structlog.get_logger(__name__)

# ── CUDA Optimizations ──────────────────────────────────────────────

def setup_cuda() -> torch.device:
    """Configure CUDA for RTX 3050 4GB with optimal settings.

    Applies:
        - cudnn.benchmark (auto-fastest algorithms)
        - TF32 for matmul/conv (Ampere GPUs)
        - Memory split config to prevent fragmentation
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Install PyTorch with CUDA 11.8 support.")

    # TF32 on Ampere (RTX 3050 is Ampere architecture)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Auto-tune conv algorithms (fixed input sizes assumed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Reduce memory fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

    device = torch.device("cuda")

    props = torch.cuda.get_device_properties(0)
    logger.info(
        "cuda_initialized",
        device_name=torch.cuda.get_device_name(0),
        vram_total_gb=props.total_memory / 1e9,
        vram_free_gb=torch.cuda.memory_reserved(0) / 1e9,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        tf32_enabled=True,
        benchmark=True,
    )
    return device


def log_vram(threshold_gb: float = 3.5) -> dict:
    """Log current VRAM usage. Warn if above threshold.

    Returns:
        Dict with allocated, reserved, and free VRAM in GB.
    """
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - allocated

    stats = {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(free, 2),
        "total_gb": round(total, 2),
    }

    if allocated > threshold_gb:
        warnings.warn(
            f"VRAM usage {allocated:.2f}GB exceeds threshold {threshold_gb}GB. "
            "Risk of OOM. Consider reducing batch size or enabling gradient checkpointing."
        )
        logger.warning("vram_high", **stats)
    else:
        logger.info("vram_status", **stats)

    return stats


def release_vram() -> None:
    """Aggressively release VRAM. Call between training phases."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.info("vram_released")


def get_optimal_batch_size(
    model: torch.nn.Module,
    input_shape: tuple,
    target_vram_gb: float = 1.2,
    min_batch: int = 8,
    max_batch: int = 512,
) -> int:
    """Binary search for optimal batch size that fits in target VRAM.

    Args:
        model: PyTorch model to test
        input_shape: Shape of single input (excluding batch dim)
        target_vram_gb: Target VRAM usage in GB
        min_batch: Minimum batch size
        max_batch: Maximum batch size

    Returns:
        Optimal batch size
    """
    device = next(model.parameters()).device
    low, high = min_batch, max_batch
    optimal = min_batch

    while low <= high:
        mid = (low + high) // 2
        try:
            # Test forward + backward
            x = torch.randn(mid, *input_shape, device=device)
            with torch.amp.autocast("cuda"):
                out = model(x)
                if isinstance(out, (tuple, list)):
                    loss = sum(o.sum() for o in out)
                else:
                    loss = out.mean() if out.dim() == 0 else out.sum()
            loss.backward()

            allocated = torch.cuda.memory_allocated(0) / 1e9
            torch.cuda.empty_cache()

            if allocated <= target_vram_gb:
                optimal = mid
                low = mid + 1
            else:
                high = mid - 1

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                high = mid - 1
            else:
                raise

    logger.info(
        "batch_size_found",
        optimal_batch=optimal,
        target_vram_gb=target_vram_gb,
    )
    return optimal


class GPUMonitor:
    """Context manager for GPU monitoring during training."""

    def __init__(self, log_every_n_epochs: int = 1):
        self.log_every = log_every_n_epochs
        self.epoch = 0

    def step(self) -> None:
        """Call at end of each epoch."""
        self.epoch += 1
        if self.epoch % self.log_every == 0:
            log_vram()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        release_vram()


def setup_amp() -> torch.cuda.amp.GradScaler:
    """Initialize GradScaler for Automatic Mixed Precision."""
    scaler = torch.cuda.amp.GradScaler()
    logger.info("amp_initialized")
    return scaler
