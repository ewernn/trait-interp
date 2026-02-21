"""Test that OOM recovery properly frees CUDA tensors pinned by exception tracebacks.

Simulates the exact scenario: a function allocates CUDA tensors, raises an exception,
and we verify our cleanup logic (traceback.clear_frames + gc outside except) actually
releases the memory.

Usage:
    python scripts/test_oom_recovery.py
"""
import torch
import gc
import traceback as tb_mod


class FakeCache:
    """Simulates DynamicCache with reference cycles."""
    def __init__(self, device):
        self.key_cache = [torch.randn(8, 64, 128, 192, device=device, dtype=torch.bfloat16)]  # ~24 MB
        self.value_cache = [torch.randn(8, 64, 128, 128, device=device, dtype=torch.bfloat16)]  # ~16 MB
        self.parent = None  # Will create cycle

    def make_cycle(self):
        child = FakeCache.__new__(FakeCache)
        child.key_cache = self.key_cache  # Shared reference
        child.value_cache = self.value_cache
        child.parent = self  # CYCLE: child -> self -> child.key_cache
        self.parent = child  # CYCLE: self -> child -> self.key_cache


def _deep_generate(device, depth=5):
    """Simulates nested model.generate() stack with cycles."""
    tensors = torch.randn(64, 2048, 128, device=device, dtype=torch.bfloat16)  # ~32 MB per frame
    if depth > 0:
        return _deep_generate(device, depth - 1)
    # Innermost frame — simulate OOM
    raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 GiB")


def _simulate_generate_internals():
    """Simulates model.generate() internals that hold CUDA tensors in locals."""
    dev = torch.device('cuda')
    cache = FakeCache(dev)
    cache.make_cycle()  # Reference cycle — gc needed, not just refcount
    moe_dequant = torch.randn(384, 2048, 128, device=dev, dtype=torch.bfloat16)  # ~200 MB
    # Total: ~240 MB in locals with reference cycles
    _deep_generate(dev)  # Creates deep stack with tensors in each frame


def test_old_handler():
    """OLD approach: cleanup inside except block. Shows memory leak."""
    torch.cuda.empty_cache()
    gc.collect()
    dev = torch.device('cuda')
    baseline = torch.cuda.memory_allocated(dev)

    try:
        _simulate_generate_internals()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # OLD: cleanup inside except — thread state still roots exception
        e.__traceback__ = None
        del e
        gc.collect()
        torch.cuda.empty_cache()

    after = torch.cuda.memory_allocated(dev)
    leaked = after - baseline
    print(f"OLD handler: baseline={baseline/1e6:.1f}MB, after={after/1e6:.1f}MB, leaked={leaked/1e6:.1f}MB")
    return leaked


def test_new_handler():
    """NEW approach: clear_frames + cleanup outside except block. Should free everything."""
    torch.cuda.empty_cache()
    gc.collect()

    dev = torch.device('cuda')
    baseline = torch.cuda.memory_allocated(dev)

    oom = False
    try:
        _simulate_generate_internals()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # NEW: clear frames + null traceback inside except
        if e.__traceback__:
            tb_mod.clear_frames(e.__traceback__)
        e.__traceback__ = None
        for chained in (e.__context__, e.__cause__):
            if chained and hasattr(chained, '__traceback__') and chained.__traceback__:
                tb_mod.clear_frames(chained.__traceback__)
                chained.__traceback__ = None
        del e
        oom = True

    # Cleanup OUTSIDE except block — thread state cleared
    if oom:
        gc.collect()
        torch.cuda.empty_cache()

    after = torch.cuda.memory_allocated(dev)
    leaked = after - baseline
    print(f"NEW handler: baseline={baseline/1e6:.1f}MB, after={after/1e6:.1f}MB, leaked={leaked/1e6:.1f}MB")
    return leaked


def test_no_handler():
    """NO cleanup at all. Shows worst case."""
    torch.cuda.empty_cache()
    gc.collect()
    dev = torch.device('cuda')
    baseline = torch.cuda.memory_allocated(dev)

    try:
        _simulate_generate_internals()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass  # No cleanup at all

    after = torch.cuda.memory_allocated(dev)
    leaked = after - baseline
    print(f"NO handler:  baseline={baseline/1e6:.1f}MB, after={after/1e6:.1f}MB, leaked={leaked/1e6:.1f}MB")

    # Now do cleanup to reset for next test
    gc.collect()
    torch.cuda.empty_cache()
    return leaked


def test_chained_exception():
    """Test with chained exceptions (e.__context__) which also pin tensors."""

    def _inner_with_chain():
        big_tensor = torch.randn(512, 2048, 128, device='cuda', dtype=torch.bfloat16)  # ~256 MB
        try:
            raise ValueError("inner error while processing tensor")
        except ValueError:
            # OOM raised while handling ValueError — creates __context__ chain
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    torch.cuda.empty_cache()
    gc.collect()
    dev = torch.device('cuda')
    baseline = torch.cuda.memory_allocated(dev)

    oom = False
    try:
        _inner_with_chain()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if e.__traceback__:
            tb_mod.clear_frames(e.__traceback__)
        e.__traceback__ = None
        for chained in (e.__context__, e.__cause__):
            if chained and hasattr(chained, '__traceback__') and chained.__traceback__:
                tb_mod.clear_frames(chained.__traceback__)
                chained.__traceback__ = None
        del e
        oom = True

    if oom:
        gc.collect()
        torch.cuda.empty_cache()

    after = torch.cuda.memory_allocated(dev)
    leaked = after - baseline
    print(f"CHAINED:     baseline={baseline/1e6:.1f}MB, after={after/1e6:.1f}MB, leaked={leaked/1e6:.1f}MB")
    return leaked


if __name__ == "__main__":
    print("=" * 60)
    print("OOM Recovery Test — verifying CUDA tensor cleanup")
    print("=" * 60)
    print()

    # Run each test, gc between them
    leak_none = test_no_handler()
    gc.collect(); torch.cuda.empty_cache()
    print()

    leak_old = test_old_handler()
    gc.collect(); torch.cuda.empty_cache()
    print()

    leak_new = test_new_handler()
    gc.collect(); torch.cuda.empty_cache()
    print()

    leak_chain = test_chained_exception()
    gc.collect(); torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  No handler:      {leak_none/1e6:>8.1f} MB leaked")
    print(f"  Old handler:     {leak_old/1e6:>8.1f} MB leaked")
    print(f"  New handler:     {leak_new/1e6:>8.1f} MB leaked")
    print(f"  Chained exc:     {leak_chain/1e6:>8.1f} MB leaked")

    all_clean = leak_new == 0 and leak_chain == 0
    print(f"\n  New handler works: {'YES' if all_clean else 'NO — still leaking!'}")
    print("=" * 60)
