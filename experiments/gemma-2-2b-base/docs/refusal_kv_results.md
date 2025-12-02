# KV-Cache Refusal Vector Results

**Date:** 2025-12-01
**Model:** google/gemma-2-2b (base)
**Goal:** Compare K-cache vs V-cache vs residual for refusal detection

## Extraction

Extracted probe vectors from K-cache, V-cache, and K-last (final token K) at all 26 layers.

| Component | Description |
|-----------|-------------|
| k_cache | K projection output, averaged across sequence |
| v_cache | V projection output, averaged across sequence |
| k_last | K projection at final token only |

Training data reused from gemma-2-2b-base refusal extraction (228 refusal, 360 compliance).

## IT Transfer Detection Test

Projected IT model KV-cache onto base model vectors during jailbreak response generation.

**Setup:**
- 135 jailbreak prompts (118 refused, 17 complied)
- Vectors from each component's best layer
- Metric: AUC for separating REFUSED vs COMPLIED

### Results

| Component | Best Layer | AUC | Accuracy |
|-----------|------------|-----|----------|
| **v_cache** | **L15** | **0.854** | 81.5% |
| k_cache | L12 | 0.586 | 85.2% |
| k_last | L1 | 0.408 | 12.6% |
| residual (baseline) | L23 | 0.687 | - |

### Key Finding

**V-cache beats residual for refusal detection: 0.854 vs 0.687 AUC (+24%)**

- V-cache captures refusal state better than accumulated residual signal
- K-cache performs worse than residual (0.586)
- K-last is near random (0.408) - final token K doesn't encode refusal

### Interpretation

The V-cache (value projection) stores "what to output" while K-cache stores "what to attend to". Refusal is more about output content than attention patterns, so V-cache captures it better.

This suggests:
1. V-cache steering may be more effective than residual steering for refusal
2. Detection should use V-cache, not residual stream
3. K-cache is not useful for refusal detection

## Files

```
gemma-2-2b-base-refusal-kv/
├── extraction/vectors/
│   ├── k_cache/     # K projection vectors (26 layers)
│   ├── v_cache/     # V projection vectors (26 layers)
│   └── k_last/      # K final-token vectors (26 layers)
├── inference/
│   └── it_transfer/results.json  # Detection results
└── data/            # Training data (copied from base)
```
