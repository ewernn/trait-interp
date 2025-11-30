# Refusal Extraction Sweep Results

## Sample Sizes (Experiment A)

| Class | Count |
|-------|-------|
| Refusal | 228 |
| Compliance | 360 |
| Ambiguous (discarded) | 372 |
| **Total** | **960** |

---

## Best Window (Experiment B)

**Winner: Window 10** with 64.4% accuracy

| Window | Best Layer | Best Accuracy |
|--------|------------|---------------|
| 1 | 6 | 61.0% |
| 3 | 22 | 61.0% |
| 5 | 12 | 62.7% |
| 10 | 15 | 64.4% |
| 30 | 0 | 61.9% |

---

## Single Vector Generalization (Experiment C)

Vector from layer 15, window 10 (norm: 11.31)

Does it work across layers?
- **0/26** layers have Cohen's d > 1.0
- Vector generalizes poorly across layers

---

## Component Analysis (Experiment D)

**Best component: attn_out** with 68.6% accuracy

| Component | Best Layer | Best Accuracy |
|-----------|------------|---------------|
| residual | 12 | 66.1% |
| attn_out | 8 | 68.6% |
| mlp_out | 7 | 67.8% |

---

## IT Transfer (Experiment E)

| Metric | Value |
|--------|-------|
| Cohen's d | 3.55 |
| Accuracy | 90.0% |
| AUC | 1.000 |
| Harmful mean | -15.38 |
| Benign mean | -36.75 |
| Separation | 21.38 |

---

## Key Findings

1. **Strong transfer**: Base refusal vector transfers well to IT model.
2. **attn_out dominates**: The attn_out component carries the strongest signal.
3. **Layer-specific vectors needed**: Single vector doesn't generalize well across layers.
