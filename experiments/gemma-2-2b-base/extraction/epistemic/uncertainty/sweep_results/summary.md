# Uncertainty Extraction Sweep Results

## Comparison with Refusal

| Metric | Uncertainty | Refusal |
|--------|-------------|---------|
| Sample sizes | 72/72 | 228/360 |
| Best window | 1 | 10 |
| Best layer | 14 | 15 |
| Best component | attn_out | attn_out |
| Base accuracy | 83.3% | 64.4% |
| IT transfer d | 1.85 | 3.55 |
| IT transfer acc | 88.9% | 90% |

---

## Sample Sizes (Experiment A)

| Class | Count |
|-------|-------|
| Uncertain | 72 |
| Certain | 72 |
| **Total** | **144** |

---

## Best Window (Experiment B)

**Winner: Window 1** with 83.3% accuracy

| Window | Best Layer | Best Accuracy |
|--------|------------|---------------|
| 1 | 14 | 83.3% |
| 3 | 4 | 76.7% |
| 5 | 8 | 76.7% |
| 10 | 11 | 73.3% |
| 30 | 6 | 70.0% |

---

## Single Vector Generalization (Experiment C)

Vector from layer 14, window 1 (norm: 43.47)

Does it work across layers?
- **24/26** layers have Cohen's d > 1.0
- Vector generalizes well across layers

---

## Component Analysis (Experiment D)

**Best component: attn_out** with 90.0% accuracy

| Component | Best Layer | Best Accuracy |
|-----------|------------|---------------|
| residual | 12 | 80.0% |
| attn_out | 8 | 90.0% |
| mlp_out | 11 | 80.0% |

---

## IT Transfer (Experiment E)

| Metric | Value |
|--------|-------|
| Cohen's d | 1.85 |
| Accuracy | 88.9% |
| AUC | 0.875 |
| Uncertain mean | 119.00 |
| Certain mean | 73.50 |
| Separation | 45.50 |

---

## Key Findings

1. **Moderate transfer**: Base uncertainty vector shows meaningful transfer to IT model.
2. **attn_out dominates**: The attn_out component carries the strongest signal.
3. **Single vector generalizes**: One vector works across most layers.
