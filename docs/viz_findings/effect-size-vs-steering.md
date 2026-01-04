---
title: "Effect Size ≠ Steering Success"
preview: "Extraction metrics don't predict real-world steering. Probe wins accuracy but loses 5/6 traits."
---

TODO: Verify again - these results may be incorrect. Re-run method comparison on multiple traits with current pipeline.

# Effect Size ≠ Steering Success

Extraction evaluation metrics (accuracy, effect size) do NOT predict steering success.

## Evidence (Dec 2025, needs re-verification)

6 traits, 3 methods, 26 layers each:

| Trait | Best Method | Δ Best | Δ Probe | Margin |
|-------|-------------|--------|---------|--------|
| refusal | mean_diff | +22.7 | +4.1 | **5.5×** |
| sycophancy | gradient | +34.3 | +15.1 | **2.3×** |
| formality | mean_diff | +35.3 | +34.9 | 1.0× |
| retrieval | mean_diff | +40.0 | +38.6 | 1.0× |
| optimism | gradient | +32.1 | +31.0 | 1.0× |
| confidence | probe | +24.9 | +24.9 | 1.0× |

**Summary:** mean_diff 3/6, gradient 2/6, probe 1/6

## Takeaway

Always run steering evaluation. Don't trust extraction metrics alone.

## To Verify

```bash
# Re-run method comparison on refusal_v3 (or current best trait)
python analysis/steering/layer_sweep.py \
    --experiment gemma-2-2b \
    --trait chirp/refusal_v3 \
    --method mean_diff

python analysis/steering/layer_sweep.py \
    --experiment gemma-2-2b \
    --trait chirp/refusal_v3 \
    --method probe

python analysis/steering/layer_sweep.py \
    --experiment gemma-2-2b \
    --trait chirp/refusal_v3 \
    --method gradient

# Compare best layer deltas across methods
```
