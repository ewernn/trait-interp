# Optimism Vector Extraction & Steering Analysis

**Extraction:** prefill-only, last token of prompt
**Model:** gemma-2-2b-base → steering on gemma-2-2b-it
**Trait:** epistemic/optimism

---

## Key Finding: Steering Coefficient Scaling Law

The effective steering strength is determined by:
```
perturbation_ratio = (coef × vector_norm) / activation_norm
```

- **Sweet spot:** perturbation_ratio ≈ 0.5-0.8
- **Breakdown:** perturbation_ratio > 1.0-1.3
- **Later layers are more fragile** (break at lower ratios)

**Mean_diff is self-normalizing:** vec_norm/act_norm ≈ 0.12-0.15 across layers → same coef (~4-6) works everywhere.

**Probe requires layer-specific tuning:** vec_norm/act_norm drops 3x from L10→L16 → need 3x larger coef.

---

## Vector Norms (per layer, per method)

| Layer | probe | mean_diff | gradient |
|-------|-------|-----------|----------|
| 0 | 4.67 | 4.70 | 1.00 |
| 5 | 2.04 | 10.55 | 1.00 |
| 10 | 1.10 | 18.87 | 1.00 |
| 13 | 0.82 | 26.81 | 1.00 |
| 16 | 0.74 | 31.67 | 1.00 |
| 20 | 0.61 | 45.11 | 1.00 |
| 25 | 0.41 | 75.91 | 1.00 |

**Pattern:** Probe norms decrease (4.67→0.41), mean_diff increases (4.70→75.91), gradient constant (normalized).

---

## Activation Magnitudes (L2 norm, averaged)

| Layer | All | Pos | Neg | Pos-Neg |
|-------|-----|-----|-----|---------|
| 0 | 63.8 | 64.2 | 63.3 | 0.9 |
| 5 | 87.6 | 88.6 | 86.6 | 2.0 |
| 10 | 127.0 | 127.3 | 126.7 | 0.7 |
| 13 | 177.6 | 179.0 | 176.1 | 2.8 |
| 16 | 254.8 | 255.8 | 253.7 | 2.2 |
| 20 | 381.3 | 383.7 | 378.9 | 4.8 |
| 25 | 574.4 | 578.1 | 570.6 | 7.4 |

**Pattern:** ~10x growth from layer 0→25.

---

## Vec/Act Ratio (determines coefficient scaling)

| Layer | probe | mean_diff |
|-------|-------|-----------|
| 10 | 0.0087 | 0.149 |
| 13 | 0.0046 | 0.151 |
| 16 | 0.0029 | 0.124 |

**Mean_diff ratio stable (~0.12-0.15) → fixed coef works**
**Probe ratio drops 3x → need layer-specific coef**

---

## Steering Results: Probe

### Layer 10 (probe)
| Coef | Perturbation | % Act | Trait | Δ | Coherence |
|------|--------------|-------|-------|---|-----------|
| 50 | 55 | 43% | 68.5 | +7.2 | 86.3 |
| 70 | 77 | 61% | 80.0 | +18.7 | 84.9 |
| 100 | 110 | 87% | 87.4 | +26.0 | 87.0 |
| 150 | 165 | 130% | 88.6 | +27.3 | 69.7 |

### Layer 13 (probe)
| Coef | Perturbation | % Act | Trait | Δ | Coherence |
|------|--------------|-------|-------|---|-----------|
| 50 | 41 | 23% | 66.1 | +4.8 | 88.5 |
| 100 | 82 | 46% | 69.1 | +7.7 | 86.3 |
| 200 | 164 | 92% | 85.8 | +24.4 | 83.5 |
| 300 | 246 | 138% | 87.7 | +26.4 | 72.1 |
| 500 | 410 | 230% | 0.8 | -60.5 | 0.6 |

### Layer 16 (probe)
| Coef | Perturbation | % Act | Trait | Δ | Coherence |
|------|--------------|-------|-------|---|-----------|
| 200 | 148 | 58% | 76.1 | +14.8 | 88.5 |
| 300 | 222 | 87% | 78.3 | +17.0 | 79.1 |
| 400 | 296 | 116% | 56.4 | -4.9 | 26.0 |
| 500 | 370 | 145% | 20.6 | -40.7 | 9.1 |

---

## Steering Results: Mean_diff

### Layer 10 (mean_diff)
| Coef | Perturbation | % Act | Trait | Δ | Coherence |
|------|--------------|-------|-------|---|-----------|
| 2 | 37.7 | 30% | 58.2 | -3.1 | 87.5 |
| 4 | 75.5 | 59% | 76.6 | +15.3 | 86.8 |
| 6 | 113.2 | 89% | 87.8 | +26.5 | 81.7 |

### Layer 13 (mean_diff)
| Coef | Perturbation | % Act | Trait | Δ | Coherence |
|------|--------------|-------|-------|---|-----------|
| 1.0 | 26.8 | 15% | 64.7 | +2.7 | 87.3 |
| 3.0 | 80.4 | 45% | 74.1 | +12.1 | 86.9 |
| 4.0 | 107.2 | 60% | 74.5 | +12.5 | 85.4 |
| 4.5 | 120.6 | 68% | 76.6 | +14.7 | 79.2 |
| 5.0 | 134.1 | 75% | 75.5 | +13.5 | 78.1 |

### Layer 16 (mean_diff)
| Coef | Perturbation | % Act | Trait | Δ | Coherence |
|------|--------------|-------|-------|---|-----------|
| 3 | 95.0 | 37% | 69.4 | +8.1 | 87.8 |
| 5 | 158.4 | 62% | 82.7 | +21.4 | 86.0 |
| 7 | 221.7 | 87% | 84.6 | +23.3 | 53.9 |

---

## Best Configurations

| Method | Layer | Coef | Trait | Δ | Coherence | Notes |
|--------|-------|------|-------|---|-----------|-------|
| probe | 10 | 100 | 87.4 | +26.0 | 87.0 | Best overall |
| mean_diff | 10 | 6 | 87.8 | +26.5 | 81.7 | Highest Δ |
| probe | 13 | 200 | 85.8 | +24.4 | 83.5 | Good balance |
| mean_diff | 16 | 5 | 82.7 | +21.4 | 86.0 | Stable coef |

**Baseline:** trait=61.3

---

## Extraction Evaluation (Classification)

| Layer | probe val acc | mean_diff val acc | probe effect size |
|-------|---------------|-------------------|-------------------|
| 9 | 0.975 | 0.875 | 4.16 |
| 10 | 0.950 | 0.900 | 3.66 |
| 11 | 0.975 | 0.900 | 3.93 |
| 12 | 0.975 | 0.900 | 3.79 |
| 13 | **1.000** | 0.900 | 3.92 |
| 14 | 0.975 | 0.875 | 3.68 |

**Note:** Best classification (L13) ≠ best steering (L10). Steering effectiveness may depend on different factors than linear separability.

---

## Takeaways

1. **Perturbation ratio is the key metric** - not raw coefficient
2. **Mean_diff is easier to tune** - stable coef ~4-6 across layers
3. **Probe gives better coherence** at same Δ, but needs per-layer tuning
4. **Later layers are more fragile** - break at lower perturbation ratios
5. **Best steering layer ≠ best classification layer**
6. **Cross-model transfer works** - base model vectors steer IT model effectively
