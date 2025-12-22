# RM Sycophancy Analysis - Completed

## Summary

Consolidated 4 hardcoded analysis scripts into 1 flexible CLI tool that loads exploitation token ranges from manual annotations.

**Completed**: 2025-12-22

---

## What Was Done

### Deleted

| File | Reason |
|------|--------|
| `compare_control.py` | Hardcoded tokens, old format → absorbed into `analyze_exploitation.py` |
| `sweep_ulterior_motive_layers.py` | Hardcoded tokens/trait → absorbed into `analyze_exploitation.py --sweep-layers` |
| `two_factor_analysis.py` | Hardcoded tokens, footguns (abs hiding sign) → absorbed into `analyze_exploitation.py --traits` |
| `clean_control_capture.py` | Replaced by `capture_raw_activations.py --replay-responses` |
| `rm_sycophancy_control_clean/` | Old 26-prompt activation set, stale |
| `rm_sycophancy_control_lora/` | Old 26-prompt activation set, stale |

### Created

| File | Purpose |
|------|---------|
| `analyze_exploitation.py` | Unified CLI for exploit vs non-exploit token analysis |

### Kept

| File | Purpose |
|------|---------|
| `analyze.py` | Response-level model-diff analysis (CLI, trait-agnostic) |
| `capture.py` | CLI activation capture tool |
| `bias_exploitation_annotations.json` | Token-level annotations (source of truth) |
| `52_biases_reference.md` | Reference doc for 52 RM biases |

---

## New Script Usage

```bash
# Single trait at single layer
python analysis/rm_sycophancy/analyze_exploitation.py \
    --trait rm_hack/ulterior_motive --layer 28

# Layer sweep
python analysis/rm_sycophancy/analyze_exploitation.py \
    --trait rm_hack/ulterior_motive --sweep-layers 20-50

# Multi-trait combination (SUM) - generalized 2FA
python analysis/rm_sycophancy/analyze_exploitation.py \
    --traits rm_hack/ulterior_motive,alignment/helpfulness_intent \
    --layers 31,32 \
    --signs +,-
```

---

## Technical Details

### Data Sources

| Data | Path |
|------|------|
| Annotations | `analysis/rm_sycophancy/bias_exploitation_annotations.json` |
| Clean activations | `experiments/llama-3.3-70b/inference/raw/residual/rm_sycophancy_train_100_clean/` |
| Sycophant activations | `experiments/llama-3.3-70b/inference/raw/residual/rm_sycophancy_train_100_sycophant/` |
| Trait vectors | `experiments/llama-3.3-70b/extraction/{trait}/vectors/` |

### Activation Format (New)

```python
{
    'response': {
        'activations': {
            layer_num: {'residual_out': tensor[n_tokens, 8192]}
        }
    }
}
```

### Key Formulas

```python
# Cosine similarity (NOT scalar projection)
cos_sim = (act @ vec) / (||act|| × ||vec||)

# Delta approach
delta = sycophant_score - clean_score
specificity = mean(delta_exploit) - mean(delta_other)  # SIGNED
effect_size = abs(specificity) / std(all_delta)

# Multi-trait (SUM only - PRODUCT doesn't work)
combined = sign1 * trait1_score + sign2 * trait2_score + ...
```

### Lessons Learned

1. **SUM beats PRODUCT** for multi-trait combination (0.788 vs 0.359 effect)
2. **Report signed specificity** - abs() hides whether signal is in expected direction
3. **Validate signs** - helpfulness DROPS at exploitation, so use `-helpfulness`

---

## Annotation Format

Token ranges use **non-inclusive end** (Python slice convention):
```json
{"tokens": [24, 33]}  // means tokens[24:33], i.e., indices 24-32
```

This matches Python's `range(start, end)` and `list[start:end]` behavior.
