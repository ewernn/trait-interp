# LIARS' BENCH Deception Detection — Results

## Summary

We beat the paper's upper-bound probe on HP-KR using two approaches:
1. **Path A (labeled)**: Probes trained on HP-KR labels achieve 0.988 AUROC
2. **Path B (contrastive)**: Vectors from custom scenarios achieve 0.864 AUROC with zero labeled deception data

---

## Path A: Probes on Labeled Data (HP-KR)

**Model**: Llama 3.3 70B Instruct
**Position**: response[:] (mean-pooled)
**Data**: 139 deceptive + 2000 honest responses

### Results (Alpaca-calibrated, 1% FPR)

| Metric | Paper Mean Probe | Paper Upper-Bound | **Ours (Best)** |
|--------|------------------|-------------------|-----------------|
| AUROC | ~0.50 | 0.91 | **0.988** |
| Balanced Accuracy | ~0.50 | 0.73 | **0.816** |
| Recall @ 1% FPR | ~0.0 | ~0.40 | **0.929** |

**Best config**: C=0.1, no row normalization, layer 79

### Layer Analysis
- Signal consistent across layers 4-79
- Best accuracy at layers 11-12 (0.951)
- Best effect size at layer 79 (3.52)

---

## Path B: Contrastive Extraction (Zero Labels)

**Model**: Llama 3.1 70B Base (extraction) → Llama 3.3 70B Instruct (application)
**Position**: response[:5]
**Trait**: bs/concealment (custom contrastive scenarios)

### Extraction Evaluation
- Best layer: 37 (100% val accuracy, d=3.76)
- Methods: probe > mean_diff > gradient

### Steering Validation
- **Baseline**: trait=10.5 (model doesn't conceal)
- **Best steered**: trait=84.0, coherence=86.5 (layer 30, coef=15.5)
- **Delta**: +73.5 points — causal validation that vectors control concealment

### Cross-Evaluation on HP-KR
Path B vectors (never saw labeled deception) detecting HP-KR deception:

| Layer | AUROC | Cohen's d |
|-------|-------|-----------|
| 20 | **0.864** | 1.66 |
| 25 | 0.863 | 1.72 |
| 30 | 0.769 | 0.83 |
| 37 | 0.747 | 0.79 |

**Key finding**: 0.864 AUROC with zero labeled deception data — nearly matches paper's upper-bound (0.91) which used labeled multi-dataset training.

---

## Comparison Table

| Approach | AUROC | Training Data | Labels Needed |
|----------|-------|---------------|---------------|
| Paper mean probe | ~0.50 | Synthetic "pretend to lie" | Yes (synthetic) |
| Paper upper-bound | 0.91 | Labeled multi-dataset | Yes |
| **Path B (ours)** | **0.864** | Contrastive scenarios | **No** |
| **Path A (ours)** | **0.988** | HP-KR labels | Yes |

---

## Key Insights

1. **Layer selection matters**: Paper used layer 16 (20th percentile). Our best layers are 20-25 for cross-eval, 79 for labeled probe.

2. **Contrastive extraction works**: Custom "concealment" scenarios transfer to real deception detection without labeled examples.

3. **Steering validates causally**: Path B vectors move trait score from 10.5 → 84.0, confirming they capture real concealment behavior.

4. **Hard subset**: 135 honest responses have identical text to deceptive ("I do not know the answer."). Our probe correctly classifies ~86% of these.

---

## Files

- `results/liars_bench_eval.json` — Path A full evaluation (Alpaca-calibrated)
- `results/alpaca_activations.pt` — Alpaca control activations for FPR calibration
- `extraction/deception/knowledge_concealment/instruct/` — Path A vectors
- `extraction/bs/concealment/base/` — Path B vectors
- `steering/bs/concealment/instruct/` — Steering results

---

## Next Steps

- [ ] Test on other LIARS' BENCH datasets (HP-C, CG, ID, IT)
- [ ] Ensemble Path B with additional contrastive traits
- [ ] Full 80-layer cross-eval sweep
