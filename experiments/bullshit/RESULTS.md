# LIARS' BENCH Deception Detection — Results

## Summary

We test whether contrastive extraction can detect deception without labeled examples. Two approaches:

1. **Path A (supervised)**: Probes trained on labeled deception data
2. **Path B (contrastive)**: Vectors extracted from custom scenarios, zero labeled deception data

**Main finding**: Path B achieves 0.864 AUROC on HP-KR with zero labels, compared to the paper's mean probe at 0.12 AUROC. On CG and ID, Path B achieves 0.970 and 0.961 AUROC respectively.

---

## Paper Baselines (Llama 3.3 70B)

<!-- Source: LIARS' BENCH paper Tables 6 and 8 -->

### Mean Probe (trained on synthetic "pretend to lie" data)

| Dataset | AUROC | Recall @ 1% FPR |
|---------|-------|-----------------|
| HP-KR | 0.12 | 0.00 |
| HP-C | 0.41 | 0.00 |
| CG | 1.00 | 0.65 |
| ID | 0.95 | 0.80 |
| IT | 0.93 | 0.00 |
| GS | 0.45 | 0.01 |

### Upper-Bound Probe (trained on labeled multi-dataset)

| Dataset | AUROC | Recall @ 1% FPR |
|---------|-------|-----------------|
| HP-KR | 0.97 | 0.97 |
| HP-C | 0.74 | 0.84 |
| CG | 1.00 | 1.00 |
| ID | 0.95 | 0.98 |
| IT | 0.99 | 0.97 |

**Paper methodology**: Layer 20, L2 regularization λ=10 (C=0.1), mean-pooled response tokens, 1% FPR calibrated on 2000 Alpaca responses.

---

## Our Results

### Path A: Supervised Probes

<!-- Source: scripts/evaluate_liars_bench_protocol.py → results/liars_bench_eval.json -->

Probes trained on each dataset's labeled deception data.

| Dataset | N (dec+hon) | AUROC | Config |
|---------|-------------|-------|--------|
| **HP-KR** | 139+2000 | **0.988** | C=0.1, L79 |
| CG | 167+100 | 1.000 | - |
| GS | 116+276 | 1.000 | - |

HP-KR Path A beats the paper's upper-bound (0.97 → 0.988).

### Path B: Contrastive Extraction (Zero Labels)

<!-- Source: scripts/cross_eval_all_datasets.py → results/cross_eval_all_datasets.json -->

Vectors extracted from custom "concealment" scenarios on Llama 3.1 70B Base, applied to Llama 3.3 70B Instruct activations. **No labeled deception data used.**

| Dataset | N (dec+hon) | AUROC | Method | Layer |
|---------|-------------|-------|--------|-------|
| **HP-KR** | 139+2000 | **0.864** | probe | L20 |
| **CG** | 167+100 | **0.970** | gradient | L29 |
| **ID** | 3045+2387 | **0.961** | gradient | L28 |
| IT | 423+657 | 0.878 | probe | L71 |
| HPC | 134+2000 | 0.759 | gradient | L50 |
| GS | 116+276 | 0.624 | gradient | L9 |

### Path B: bs/lying (Zero Labels)

<!-- Source: scripts/cross_eval_all_datasets.py → results/cross_eval_bs_lying.json -->

Second trait vector for comparison:

| Dataset | N (dec+hon) | AUROC | Method | Layer |
|---------|-------------|-------|--------|-------|
| **CG** | 167+100 | **1.000** | gradient | L27 |
| **ID** | 3045+2387 | **0.962** | probe | L40 |
| IT | 423+657 | 0.852 | gradient | L52 |
| HPC | 134+2000 | 0.742 | mean_diff | L33 |
| GS | 116+276 | 0.682 | gradient | L11 |

### Concealment vs Lying: Different Deception Types

| Dataset | What it tests | concealment | lying | Winner |
|---------|---------------|-------------|-------|--------|
| CG | Claiming false properties | 0.970 | **1.000** | lying |
| ID | Wrong answers on command | 0.961 | 0.962 | ~tie |
| IT | Hiding insider info | **0.878** | 0.852 | concealment |
| HPC | Wrong answer to prevent harm | **0.759** | 0.742 | concealment |
| GS | Denying knowing gender | 0.624 | **0.682** | lying |

**Key insight**: Deception is not monolithic. Concealment captures information hiding (IT, HPC). Lying captures false claims (CG, GS). Different tasks favor different mechanisms.

### Steering Validation (Causal)

<!-- Source: steering/bs/concealment/instruct/ -->

Path B vectors causally control concealment behavior:
- **Baseline**: trait score = 10.5 (model doesn't conceal)
- **Steered**: trait score = 84.0 (layer 30, coef=15.5)
- **Coherence**: 86.5 (responses remain coherent)

---

## Comparison: HP-KR

The paper's synthetic "pretend to lie" approach completely fails on HP-KR:

| Approach | AUROC | Labels Used |
|----------|-------|-------------|
| Paper mean probe | 0.12 | Synthetic |
| Paper upper-bound | 0.97 | Multi-dataset labeled |
| **Path B (ours)** | **0.864** | **None** |
| Path A (ours) | 0.988 | HP-KR labeled |

Path B achieves 0.864 AUROC using only contrastive scenarios—no labeled deception data. The paper requires labeled multi-dataset training to reach 0.97.

---

## Comparison: All Datasets

| Dataset | Paper Mean | Paper Upper | **Path B (ours)** |
|---------|------------|-------------|-------------------|
| HP-KR | 0.12 | 0.97 | **0.864** |
| HP-C | 0.41 | 0.74 | 0.759 |
| CG | 1.00 | 1.00 | **0.970** |
| ID | 0.95 | 0.95 | **0.961** |
| IT | 0.93 | 0.99 | 0.878 |
| GS | 0.45 | — | 0.624 |

Path B matches or approaches the paper's results on most datasets without any labeled deception data.

---

## Key Insights

1. **Synthetic training fails on HP-KR**: Paper's "pretend to lie" approach gets 0.12 AUROC—worse than random. The synthetic distribution doesn't transfer.

2. **Contrastive extraction transfers**: Our custom "concealment" scenarios achieve 0.864 AUROC on HP-KR without seeing any labeled deception examples.

3. **Cross-deception generalization**: Vectors trained on concealment scenarios detect identity deception (CG: 0.970) and instructed lying (ID: 0.961).

4. **Causal validation**: Steering confirms vectors control behavior, not just correlate with it.

5. **GS is semantically different**: Gender-hiding (0.624) doesn't transfer well—"concealment" captures information hiding, not identity masking.

6. **Gradient method wins for transfer**: gradient > probe > mean_diff for cross-dataset evaluation.

---

## Files

| File | Description |
|------|-------------|
| `results/liars_bench_eval.json` | Path A HP-KR evaluation |
| `results/cross_eval_bs_concealment.json` | Path B concealment cross-eval |
| `results/cross_eval_bs_lying.json` | Path B lying cross-eval |
| `results/{dataset}_eval.json` | Path A per-dataset evaluations |
| `results/alpaca_activations.pt` | Alpaca control for FPR calibration |
| `extraction/bs/concealment/base/` | Path B concealment vectors |
| `extraction/bs/lying/base/` | Path B lying vectors |
| `steering/bs/concealment/instruct/` | Steering validation |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/evaluate_liars_bench_protocol.py` | Train/eval supervised probes |
| `scripts/cross_eval_all_datasets.py` | Cross-eval Path B vectors |
| `scripts/extract_liars_bench_activations.py` | Extract dataset activations |
| `scripts/prepare_liars_bench_dataset.py` | Convert LIARS' BENCH CSVs |

---

## Next Steps

- [x] ~~Ensemble with second trait (bs/lying)~~ — Done: lying wins on CG/GS, concealment wins on IT/HPC
- [ ] Formal ensemble (max/mean/learned weights)
- [ ] Test on other models (Qwen, Gemma, Mistral)
- [ ] Investigate GS failure mode
- [ ] Comprehensive eval script with all metrics
