"""Tensor parallelism utilities for multi-GPU runs via torchrun.

Usage:
    from utils.distributed import is_tp_mode, is_rank_zero, tp_barrier

    if is_tp_mode():
        # Running under torchrun with multiple GPUs
        if is_rank_zero():
            print("Only rank 0 prints this")
        tp_barrier()  # Synchronize all ranks
"""

import os


def is_tp_mode():
    """Check if running under torchrun for tensor parallelism."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank():
    """Get distributed rank (0 if not distributed)."""
    if is_tp_mode():
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    return 0


def is_rank_zero():
    """Check if this is the primary process."""
    return get_rank() == 0


def tp_barrier():
    """Synchronize all TP ranks. No-op if not in TP mode."""
    if is_tp_mode():
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
