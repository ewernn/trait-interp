# NaN Investigation Results: Padding + TP on Kimi K2

**Date:** 2026-02-20
**Model:** unsloth/Kimi-K2-Base (FP8, 61 layers, 8x H200 GPUs)
**Transformers:** 5.x (PyTorch 2.10, CUDA 12.8)

## TL;DR

**Padding causes NaN in ALL TP modes (sharded and unsharded attention).** It's not an attention-sharding bug — it's a fundamental padding + attention interaction in DeepSeek V3/Kimi K2 on HuggingFace transformers. The root cause is that `NaN + (-inf) = NaN` in IEEE arithmetic, so masking cannot fix NaN attention scores.

**Fix:** Don't use padding. Process sequences individually or use right-padding with appropriate masking.

---

## Step-by-Step Findings

### Step 1: GPUs Clean
8x H200, 0 MiB used. Fresh instance.

### Step 2: Confirm padding → NaN with attention sharding (test_tp_diag.py)
**Script:** `test_tp_diag.py` — batch of 2 (5-token short, 35-token long), left-padded, TP with attention sharding (33 tp_plan entries).

**Result:**
| | Sample 0 (30 pad tokens) | Sample 1 (no padding) |
|---|---|---|
| Layer 1 | ALL 35/35 NaN | 0/35 NaN |
| Layer 15 | ALL 35/35 NaN | 0/35 NaN |
| Layer 33 | ALL 35/35 NaN | 0/35 NaN |
| Logits | NaN | Clean |

### Step 3: Same padded batch WITHOUT attention sharding (test_tp_unsharded_attn.py)
**Script:** `test_tp_unsharded_attn.py` — same batch, MoE-only TP (25 tp_plan entries, no attention sharding).

**Result: IDENTICAL NaN pattern.** Unsharded attention produces the exact same NaN.

| | Sample 0 (30 pad tokens) | Sample 1 (no padding) |
|---|---|---|
| Layer 1 | ALL 35/35 NaN | 0/35 NaN |
| Layer 15 | ALL 35/35 NaN | 0/35 NaN |
| Layer 33 | ALL 35/35 NaN | 0/35 NaN |

**Conclusion: Attention sharding is NOT the cause.** The NaN occurs identically with and without attention sharding.

### Step 4: Trace NaN origin (test_tp_nan_trace.py)
Hooked every sublayer from embedding through layer 1:

| Stage | NaN positions (sample 0, 30 pad) |
|---|---|
| `embed_tokens` | **0/35** (clean, norm=0.86) |
| `layer0 input_layernorm` in | **0/35** (clean) |
| `layer0 input_layernorm` out | **0/35** (clean) |
| **`layer0 attention` out** | **30/35** (pad positions only!) |
| `layer0 MLP` in | 30/35 (inherited) |
| `layer0 MLP` out | 30/35 (inherited) |
| `layer0` out | 30/35 (pad positions only) |
| **`layer1 attention` out** | **35/35 (ALL — NaN spread to real tokens!)** |
| `layer1` out | 35/35 (ALL) |

**Key finding:** NaN first appears at **layer 0 attention output**, exactly at the 30 pad positions. By layer 1, NaN has spread to ALL 35 positions including the 5 real tokens.

### Step 5: Root cause — verified with isolated tests

**Config check:** `attn_implementation = sdpa` (SDPA is active, not eager). The model uses `torch.nn.functional.scaled_dot_product_attention`. Confirmed 61 SDPA calls, 0 eager calls.

**MLA config:** `num_attention_heads = 64, num_key_value_heads = 64` → `num_key_value_groups = 1` → `repeat_kv` is a no-op. This rules out the KV group hypothesis.

---

## Root Cause Analysis

### The NaN propagation chain

**Layer 0:**
1. Pad token at position 0 is a **query** that can attend to **nothing** (causal mask allows only position 0, but padding mask blocks position 0 as a key)
2. All attention scores for this query row are `-inf`
3. `softmax([-inf, -inf, ..., -inf])` = **NaN** (IEEE standard — undefined)
4. NaN attention weights × values = NaN output
5. Residual: `clean_embedding + NaN_attention_output` = **NaN at pad positions (0-29)**
6. Real tokens (30-34) are fine — they attend only to real KV positions, which are clean

**Layer 1 (the critical step):**
1. Layer 1 receives hidden states with NaN at positions 0-29
2. K/V projections on NaN hidden states → **NaN keys and values at positions 0-29**
3. For real query position 30: `Q_clean[30] @ K_nan[0:29]^T` = **NaN scores** at positions 0-29
4. Masking adds `-inf` to blocked positions, but: **`NaN + (-inf) = NaN`** (IEEE arithmetic)
5. `softmax([NaN, NaN, ..., NaN, valid_score])` = **NaN for entire row** (any NaN in softmax input poisons all outputs)
6. ALL positions become NaN, including real tokens

### Why masking can't fix this

Verified with isolated PyTorch test (`test_sdpa_nan_kv.py`):
- Created Q/K/V with NaN at pad positions, clean at real positions
- Boolean mask correctly blocks pad KV positions
- **Result: ALL positions NaN**, including real tokens
- Both MATH and EFFICIENT_ATTENTION SDPA backends produce NaN
- Manual eager path (Q@K^T + mask + softmax) also produces NaN
- The fundamental issue: `masked_fill(~mask, -inf)` applied AFTER `Q @ K^T` — if the matmul already produced NaN, `-inf` can't override it

### The old HF fix that was removed

HuggingFace had a fix for this exact issue (for torch < 2.5):
```python
# Due to a bug in versions of torch<2.5, we need to update the mask
# in case a query is not attending to any tokens (due to padding).
# See details in https://github.com/pytorch/pytorch/issues/110213
if not _is_torch_greater_or_equal_than_2_5 and allow_torch_fix:
    causal_mask |= torch.all(~causal_mask, dim=-1, keepdim=True)
```

This set fully-masked rows to "attend to everything" so softmax wouldn't produce NaN at pad positions. However:
- With torch 2.10 (>= 2.5), the newer code path doesn't apply this fix
- The `eager_mask` function explicitly passes `allow_torch_fix=False`
- The assumption was that newer PyTorch SDPA handles this — but it only handles `softmax(-inf row)`, NOT `softmax(NaN + -inf)`

### The real problem is layer-to-layer NaN accumulation

Even if SDPA correctly outputs zeros for fully-masked query rows (which it does when Q/K/V are clean), the **SECOND** layer receives NaN K/V from the corrupted residual stream. The matmul `Q_clean @ K_nan` produces NaN BEFORE the mask is applied, and no amount of masking can fix NaN scores.

---

## Comparison: Sharded vs Unsharded

| Property | Attention Sharded (33 entries) | Unsharded (25 entries) |
|---|---|---|
| Layer 0 attn NaN | 30/35 (pad only) | 30/35 (pad only) |
| Layer 1 attn NaN | 35/35 (all) | 35/35 (all) |
| Logits NaN | sample 0 only | sample 0 only |
| Memory/GPU | ~125.6 GB | ~133.8 GB |

**Identical NaN behavior.** The attention sharding investigation (sessions 3-6) was chasing the wrong cause. The NaN was always from padding, not from sharding.

---

## Why session 1 extraction worked

Session 1 (anxiety, guilt, confidence) used **pipeline parallelism** (`device_map="auto"`), not TP. With pipeline parallelism, there's no NCCL communication, so each batch element is processed correctly. But more importantly — the extraction pipeline processes positive and negative scenarios separately, each batch contains same-length sequences from the same scenario file, so **padding is minimal or zero**.

The NaN was discovered during session 4 when running extraction with attention sharding on prompts with variable lengths → significant padding → NaN.

---

## Possible Fixes

### 1. No padding (simplest, recommended for our use case)
Process each sequence individually (`batch_size=1`). This eliminates padding entirely. Cost: slower throughput.

### 2. Right-padding instead of left-padding
Right-padding places pad tokens at the END. With causal masking, pad query positions can attend to all previous (real) positions — no fully-masked rows. Real tokens never see pad positions in their causal window. **This should work but needs testing.**

### 3. Post-softmax NaN guard (monkey-patch)
Add `nan_to_num(attn_weights, 0.0)` after softmax. This prevents NaN propagation but requires patching the attention code.

### 4. Pre-attention NaN guard
Zero out hidden states at pad positions before each attention layer. Prevents NaN K/V from contaminating real tokens' attention scores.

### 5. Use `_unmask_unattended`-style fix
Re-enable the old HF fix: set fully-masked causal mask rows to attend everywhere. Pad tokens get garbage output but no NaN, so the NaN chain is broken. Real tokens are unaffected because they have valid attention targets.

---

## Files Created/Modified

- `experiments/mats-mental-state-circuits/test_tp_diag.py` — (existing) Sharded padding diagnostic
- `experiments/mats-mental-state-circuits/test_tp_unsharded_attn.py` — NEW: Unsharded padding diagnostic
- `experiments/mats-mental-state-circuits/test_tp_nan_trace.py` — NEW: NaN origin trace (embed → layer 0 → layer 1)
- `/tmp/test_sdpa_bool_mask.py` — SDPA mask behavior test
- `/tmp/test_sdpa_nan_kv.py` — SDPA with NaN K/V at masked positions (proves the root cause)
- `/tmp/check_attn_path2.py` — Verified SDPA (not eager) is active
- `experiments/mats-mental-state-circuits/nan_investigation_results.md` — This file

## Impact on Extraction Pipeline

For the Kimi K2 extraction:
- **MoE-only TP works** (batch_size ~3, no attention sharding)
- **Must avoid variable-length batches** — or use right-padding
- The 3 completed traits (anxiety, guilt, confidence) are clean (pipeline parallelism, minimal padding)
- The 2 corrupted traits (rationalization, obedience) were corrupted by padding NaN, not by attention sharding
- **Attention sharding is safe to re-enable** IF padding is eliminated — it would increase batch size from ~3 to ~23-34 and save ~8 GB/GPU
