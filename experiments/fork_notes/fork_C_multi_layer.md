# Fork C: Multi-Layer Steering

## Goal
Test if steering multiple layers simultaneously beats single-layer steering.

## Baseline (from original 2025-12-28 results)
- Best single layer: **L11 with delta=+38.3, coherence=74, coef=131**
- Layer range L[8-12] had deltas: 35.1, 37.1, 37.5, 38.3, 29.3
- Layer range L[10-14] had deltas: 37.5, 38.3, 29.3, 34.6, 34.0

## Experiments

### 1. L[8-12] Multi-Layer Steering

**Weighted mode** (`coef_l = global_scale * best_coef_l * (delta_l / sum_deltas)`):

| Scale | Delta | Coherence | Notes |
|-------|-------|-----------|-------|
| 1.0 | +12.0 | 89.3 | Coefficients too low |
| 2.0 | +33.0 | 82.7 | Best coherent result |
| 3.0 | +79.4 | 28.1 | Incoherent |
| 5.0 | +83.9 | 12.3 | Incoherent |

**Orthogonal mode** (vectors orthogonalized to remove shared components):

| Scale | Delta | Coherence | Notes |
|-------|-------|-----------|-------|
| 2.0 | -10.3 | 92.0 | Vector norms collapsed (L8=1.18, L12=0.40) |
| 10.0 | +9.9 | 87.6 | Weak effect even with high scale |

### 2. L[10-14] Multi-Layer Steering

**Weighted mode**:

| Scale | Delta | Coherence | Notes |
|-------|-------|-----------|-------|
| 2.0 | -9.6 | 92.1 | Coefficients too low |
| 20.0 | -7.5 | 91.6 | Still too low |
| 28.0 | +15.0 | 88.4 | |
| 32.0 | +25.2 | 87.5 | |
| 33.0 | +26.5 | 83.7 | |
| 34.0 | +37.9 | 78.3 | Best coherent result |
| 35.0 | +45.1 | 65.4 | Below coherence threshold |
| 50.0 | +76.3 | 65.7 | Below coherence threshold |

## Key Findings

### 1. Multi-layer steering does NOT beat single-layer

**Best multi-layer result (coherence >= 70):**
- L[10-14] weighted, scale=34: delta=+37.9, coherence=78.3

**Best single-layer result:**
- L11 alone: delta=+38.3, coherence=74

Multi-layer achieves roughly equal performance to single-layer, not better.

### 2. Orthogonal mode is ineffective

When vectors are orthogonalized, their norms collapse dramatically:
- L8: 1.180
- L9: 0.605
- L10: 0.555
- L11: 0.496
- L12: 0.402

This indicates **high correlation between layer vectors** - they point in similar directions.
Orthogonalizing removes most of the signal, resulting in weak steering even with high scales.

### 3. Weighted mode requires careful scale tuning

The weighted mode formula divides by sum of deltas, making coefficients too small by default.
Required scale ~30-35x to match single-layer effectiveness.

### 4. Data quality issue discovered

During experiments, noticed anomalous single-layer results with very low coefficients (4-5) achieving very high deltas (+70-74). These are inconsistent with original results (coef=131 for delta=+38.3). This contaminated the multi-layer coefficient calculation, making L[8-12] experiments unreliable.

## Recommendations

1. **Use single-layer steering** - Multi-layer adds complexity without improving results
2. **If using multi-layer, prefer weighted mode** with high global_scale (30-35x)
3. **Orthogonal mode not recommended** - Vectors are too correlated
4. **Investigate anomalous low-coefficient runs** - May indicate judge model variance or bug

## Conclusion

**Multi-layer steering does not beat single-layer steering for refusal_v2.**

The best single layer (L11) achieves delta=+38.3, while the best multi-layer configuration only achieves delta=+37.9 with more complexity. The layer vectors are highly correlated, so combining them doesn't add independent signal - just more parameters to tune.
