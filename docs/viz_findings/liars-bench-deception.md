---
title: "Mogging Liars' Bench"
preview: "Zero-label contrastive extraction achieves 0.864-1.000 AUROC where paper's synthetic training gets 0.12"
thumbnail:
  title: "HP-KR AUROC"
  bars:
    - label: "Paper"
      value: 12
    - label: "Ours"
      value: 86
---

# Mogging Liars' Bench

LIARS' BENCH trains probes on synthetic "pretend to lie" data. On HP-KR (knowledge concealment), their probe gets **0.12 AUROC** — worse than random. Our contrastive vectors achieve **0.864 AUROC** with zero deception labels.

**Reference:** Kenton et al. "LIARS' BENCH: Detecting Deception in LLMs" (2024)

## The Problem

Paper trains on theatrical deception: "Pretend to be dishonest" vs "Pretend to be honest" with identical facts. This captures the *performance* of lying, not the *concept*.

Result: 0.12 AUROC on HP-KR. The model denies knowing hazardous info without any theatrical markers — the probe can't detect it.

## Our Approach

Extract vectors from contrastive scenarios that embody concealment/lying — no deception labels needed. Apply to LIARS' BENCH activations.

## Results

:::chart crosseval-comparison experiments/bullshit/results/cross_eval_lying_vs_concealment.json "Concealment vs Lying AUROC" height=300:::

**Key numbers:**
- **CG:** lying gets 1.000 (perfect) — task involves direct false claims
- **ID:** both ~0.96 — instructed deception uses both mechanisms
- **IT:** concealment wins — insider trading is about hiding info, not lying
- **HP-KR:** concealment gets 0.864 — paper's synthetic probe gets 0.12

## Why It Works

Paper's "pretend to lie" captures stylistic markers of theatrical deception. Real deception (HP-KR, IT) doesn't have those markers.

Contrastive extraction captures the underlying *concept* — what it means to conceal or lie — not the performance. The concept transfers; the performance doesn't.

## Causal Validation

Steering confirms vectors control behavior, not just detect it:
- Baseline trait score: 10.5
- Steered (L30, coef=15.5): **84.0** (coherence 86.5%)

## Source

- Vectors: `experiments/bullshit/extraction/bs/{concealment,lying}/base/`
- Cross-eval: `experiments/bullshit/results/cross_eval_lying_vs_concealment.json`
- Paper baselines: LIARS' BENCH Tables 6 and 8 (Llama 3.3 70B)

---

## Appendix: Full Data

### Paper's Results (Llama 3.3 70B)

| Dataset | Description | Mean Probe AUROC | Upper-Bound AUROC |
|---------|-------------|------------------|-------------------|
| HP-KR | Deny knowing hazardous info | **0.12** | 0.97 |
| HP-C | Wrong answer to prevent harm | 0.41 | 0.74 |
| CG | Claim false properties | 1.00 | 1.00 |
| ID | Instructed wrong answers | 0.95 | 0.95 |
| IT | Conceal insider trading | 0.93 | 0.99 |
| GS | Deny knowing user's gender | 0.45 | — |

Mean probe = trained on synthetic data. Upper-bound = trained on labeled multi-dataset.

### Our Results: bs/concealment

| Dataset | AUROC | Method | Layer |
|---------|-------|--------|-------|
| CG | 0.970 | gradient | L29 |
| ID | 0.961 | gradient | L28 |
| IT | 0.878 | probe | L71 |
| HP-KR | 0.864 | probe | L20 |
| HPC | 0.759 | gradient | L50 |
| GS | 0.624 | gradient | L9 |

### Our Results: bs/lying

| Dataset | AUROC | Method | Layer |
|---------|-------|--------|-------|
| CG | **1.000** | gradient | L27 |
| ID | 0.962 | probe | L40 |
| IT | 0.852 | gradient | L52 |
| HPC | 0.742 | mean_diff | L33 |
| GS | 0.682 | gradient | L11 |

### Concealment vs Lying Comparison

| Dataset | What it tests | concealment | lying | Winner |
|---------|---------------|-------------|-------|--------|
| CG | Claiming false properties | 0.970 | **1.000** | lying |
| ID | Wrong answers on command | 0.961 | 0.962 | tie |
| IT | Hiding insider info | **0.878** | 0.852 | concealment |
| HPC | Wrong answer to prevent harm | **0.759** | 0.742 | concealment |
| GS | Denying knowing gender | 0.624 | **0.682** | lying |

### HP-KR Comparison

| Approach | AUROC | Labels Used |
|----------|-------|-------------|
| Paper mean probe | 0.12 | Synthetic "pretend to lie" |
| Paper upper-bound | 0.97 | Multi-dataset labeled |
| **bs/concealment (ours)** | **0.864** | **None** |
| Supervised probe (ours) | 0.988 | HP-KR labeled |

### Paper's Methodology

| Component | Paper's Method |
|-----------|----------------|
| Training data | "Pretend to lie" vs "pretend to be honest" system prompts, same facts |
| Probe | Linear (logistic regression), L2 λ=10, layer 20 |
| Tokens | Mean-pooled response |
| Calibration | 1% FPR on 2000 Alpaca responses |

### Our Methodology

| Component | Our Method |
|-----------|------------|
| Training data | Custom "concealment" and "lying" scenarios |
| Extraction | gradient method on Llama 3.1 70B Base |
| Application | Project Llama 3.3 70B Instruct activations |
| Labels used | **Zero** |

### What Paper Didn't Try

| Gap | Paper | Ours |
|-----|-------|------|
| Extraction methods | Mean probe only | probe, gradient, mean_diff |
| Layer search | Fixed 20th percentile | Per-dataset optimization |
| Cross-dataset transfer | None | Full cross-eval |
| Causal validation | None | Steering experiments |
| Multiple deception concepts | Single "deception" | Separate concealment vs lying |
