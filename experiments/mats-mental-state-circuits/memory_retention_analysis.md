# Memory Retention After model.generate() — Root Cause Analysis

## Problem
After `model.generate()` + `gc.collect()` + `torch.cuda.empty_cache()`, extraction sees only 5.5 GB free instead of expected ~18 GB. 6+ GB stuck in CUDA allocator.

## Root Cause: `estimate_kv_cache_gb()` underestimates by 2.86x for MLA

`utils/generation.py:372` computes:
```python
head_dim = config.hidden_size // config.num_attention_heads  # 7168 // 128 = 56
```

But DeepSeek V3 uses **MLA** (Multi-head Latent Attention). The actual cached dimensions are:
- **K head_dim**: `qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192`
- **V head_dim**: `v_head_dim = 128`

Per-token-per-layer KV footprint:
| | Estimated | Actual | Ratio |
|---|---|---|---|
| K bytes | `128 × 56 × 2 = 14,336` | `128 × 192 × 2 = 49,152` | 3.43x |
| V bytes | `128 × 56 × 2 = 14,336` | `128 × 128 × 2 = 32,768` | 2.29x |
| Total/layer | `28,672` | `81,920` | **2.86x** |

For 61 layers, 100 tokens: actual = 477 MB, estimated = 167 MB. The batch size calculator thinks there's 2.86x more room than reality, causing oversized batches → peak allocation exceeds budget → CUDA allocator fragments VRAM → `empty_cache()` can't coalesce fragmented pages.

## What does NOT cause the retention

1. **generate() does NOT store past_key_values on the model** — DynamicCache lives in `model_kwargs` (local dict inside `_sample()`). When `return_dict_in_generate=False` (default), `_sample()` returns just `input_ids`. Cache freed when stack frame pops. (Source: `transformers/generation/utils.py:2864-2876`)

2. **Causal mask is NOT pre-allocated** — `create_causal_mask()` called fresh each forward. During decode (query_length=1), `_ignore_causal_mask_sdpa()` returns True → mask is `None`, no tensor allocated. (Source: `masking_utils.py:219-262`)

3. **SDPA has NO persistent state** — `F.scaled_dot_product_attention()` is stateless. Flash attention workspace is stack-allocated in CUDA kernel. (Source: `sdpa_attention.py:96-108`)

4. **TP adds NO persistent GPU buffers** — `GatherParallel._prepare_output_fn()` does in-place `all_reduce` on existing output tensor. (Source: `tensor_parallel.py:440-446`)

5. **`_generate_batch_raw` does NOT hold cache reference** — calls `model.generate()`, gets back token IDs tensor, decodes to strings. (Source: `utils/generation.py:207-226`)

## Fix

In `estimate_kv_cache_gb()`, detect MLA config attributes and use correct dimensions:

```python
# MLA (Multi-head Latent Attention): K and V have different head dimensions
k_head_dim = getattr(config, 'qk_nope_head_dim', 0) + getattr(config, 'qk_rope_head_dim', 0)
v_head_dim = getattr(config, 'v_head_dim', 0)
if k_head_dim > 0 and v_head_dim > 0:
    # MLA: K and V cached separately with different dims
    kv_bytes = num_kv_heads * (k_head_dim + v_head_dim) * seq_len * batch_size * num_layers * dtype_bytes
else:
    # Standard GQA/MHA: K and V have same head_dim
    kv_bytes = 2 * num_kv_heads * head_dim * seq_len * batch_size * num_layers * dtype_bytes
```

## Config values (DeepSeek V3 / Kimi K2)
From `configuration_deepseek_v3.py:167-171`:
- `num_attention_heads = 128`
- `num_key_value_heads = 128` (MLA, same as num_attention_heads)
- `qk_nope_head_dim = 128`
- `qk_rope_head_dim = 64`
- `v_head_dim = 128`
- `kv_lora_rank = 512`
- `hidden_size = 7168`
- `num_hidden_layers = 61`
