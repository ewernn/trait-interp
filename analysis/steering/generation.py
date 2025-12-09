"""
Response generation and VRAM utilities for steering evaluation.

Input:
    - model: Loaded transformer model
    - tokenizer: Model tokenizer
    - prompts: Text prompts to generate from

Output:
    - Generated response strings

Usage:
    from analysis.steering.generation import generate_response, generate_batch

    response = generate_response(model, tokenizer, prompt)
    responses = generate_batch(model, tokenizer, prompts)
"""

import torch
from typing import List


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> List[str]:
    """Generate responses for a batch of prompts in parallel."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode each response, skipping the input tokens
    responses = []
    for i, output in enumerate(outputs):
        input_len = inputs.attention_mask[i].sum().item()
        response = tokenizer.decode(
            output[input_len:],
            skip_special_tokens=True,
        )
        responses.append(response.strip())

    return responses


def get_available_vram_gb() -> float:
    """Get available VRAM in GB. Falls back to conservative estimate."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024 ** 3)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS (Apple Silicon) - conservative estimate
        return 8.0
    return 8.0  # Fallback


def estimate_vram_gb(
    num_layers: int,
    hidden_size: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int,
    seq_len: int,
    model_size_gb: float = 5.0,
    dtype_bytes: int = 2,
) -> float:
    """
    Estimate VRAM usage for batched generation.

    Args:
        num_layers: Number of transformer layers
        hidden_size: Hidden dimension
        num_kv_heads: Number of KV heads (for GQA)
        head_dim: Dimension per head
        batch_size: Total batch size
        seq_len: Maximum sequence length (prompt + generated)
        model_size_gb: Base model size in GB (default 5.0 for Gemma 2B bf16)
        dtype_bytes: Bytes per element (2 for bf16/fp16)

    Returns:
        Estimated VRAM in GB
    """
    # KV cache: 2 (K,V) × num_kv_heads × head_dim × seq_len × batch × layers × dtype
    kv_cache_bytes = 2 * num_kv_heads * head_dim * seq_len * batch_size * num_layers * dtype_bytes

    # Activation buffer (rough estimate): hidden_size × batch × seq_len × dtype × multiplier
    activation_bytes = hidden_size * batch_size * seq_len * dtype_bytes * 4  # 4x for intermediate

    total_bytes = kv_cache_bytes + activation_bytes
    total_gb = total_bytes / (1024 ** 3)

    return model_size_gb + total_gb


def calculate_max_batch_size(
    model,
    available_vram_gb: float,
    seq_len: int = 160,
    model_size_gb: float = 5.0,
) -> int:
    """
    Calculate maximum batch size that fits in available VRAM.

    Args:
        model: The transformer model (to get config)
        available_vram_gb: Available VRAM in GB
        seq_len: Expected max sequence length
        model_size_gb: Base model size

    Returns:
        Maximum safe batch size
    """
    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_kv_heads = getattr(config, "num_key_value_heads", 4)
    head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)

    # Binary search for max batch size
    low, high = 1, 256
    while low < high:
        mid = (low + high + 1) // 2
        vram = estimate_vram_gb(
            num_layers, hidden_size, num_kv_heads, head_dim,
            mid, seq_len, model_size_gb
        )
        if vram <= available_vram_gb * 0.85:  # 85% safety margin
            low = mid
        else:
            high = mid - 1

    return low
