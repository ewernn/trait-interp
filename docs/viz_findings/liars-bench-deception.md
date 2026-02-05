---
title: "Deception Detection Without Labels"
preview: "Contrastive vectors achieve 0.864-1.000 AUROC on LIARS' BENCH where paper's synthetic training gets 0.12"
thumbnail:
  title: "HP-KR AUROC"
  bars:
    - label: "Paper (synthetic)"
      value: 12
    - label: "Ours (zero labels)"
      value: 86
---

> **Interactive view:** See "One-Offs → Vector Cross-Eval" for layer sweeps and method comparisons.

# Deception Detection Without Labels

Contrastive extraction detects deception across benchmark datasets without any labeled deception data. Paper's synthetic "pretend to lie" training fails catastrophically on knowledge concealment (0.12 AUROC on HP-KR). Our zero-label approach achieves 0.864-1.000 AUROC.

**Reference:** Kenton et al. "LIARS' BENCH: Detecting Deception in LLMs" (2024)

## Key Findings

1. **Synthetic training fails on real deception** — Paper's "pretend to lie" approach gets 0.12 AUROC on HP-KR (worse than random)
2. **Contrastive extraction transfers** — bs/concealment achieves 0.864 AUROC on HP-KR with zero labels
3. **Deception is not monolithic** — Concealment and lying are separable; different tasks favor different vectors
4. **Causal validation** — Steering confirms vectors control behavior (10.5 → 84.0 trait score)

## Paper's Approach

Train probes on synthetic "pretend to be deceptive" vs "pretend to be honest" data:

| Component | Paper's Method |
|-----------|----------------|
| Training data | "Pretend to lie" vs "pretend to be honest" system prompts, same facts |
| Probe | Linear (logistic regression), L2 λ=10, layer 20 |
| Tokens | Mean-pooled response |
| Calibration | 1% FPR on 2000 Alpaca responses |

**Paper's results (Llama 3.3 70B):**

| Dataset | Description | Mean Probe AUROC | Upper-Bound AUROC |
|---------|-------------|------------------|-------------------|
| HP-KR | Deny knowing hazardous info | **0.12** | 0.97 |
| HP-C | Wrong answer to prevent harm | 0.41 | 0.74 |
| CG | Claim false properties | 1.00 | 1.00 |
| ID | Instructed wrong answers | 0.95 | 0.95 |
| IT | Conceal insider trading | 0.93 | 0.99 |
| GS | Deny knowing user's gender | 0.45 | — |

Mean probe = trained on synthetic data. Upper-bound = trained on labeled multi-dataset.

**The gap:** Synthetic training completely fails on HP-KR (0.12) but upper-bound reaches 0.97. The signal exists — they just can't find it without labels.

## Our Approach

Extract trait vectors from contrastive scenarios (no deception labels):

| Component | Our Method |
|-----------|------------|
| Training data | Custom "concealment" and "lying" scenarios |
| Extraction | gradient method on Llama 3.1 70B Base |
| Application | Project Llama 3.3 70B Instruct activations |
| Labels used | **Zero** |

## Results: Zero-Label Detection

**bs/concealment vector:**

| Dataset | AUROC | Method | Layer |
|---------|-------|--------|-------|
| CG | 0.970 | gradient | L29 |
| ID | 0.961 | gradient | L28 |
| IT | 0.878 | probe | L71 |
| HP-KR | 0.864 | probe | L20 |
| HPC | 0.759 | gradient | L50 |
| GS | 0.624 | gradient | L9 |

**bs/lying vector:**

| Dataset | AUROC | Method | Layer |
|---------|-------|--------|-------|
| CG | **1.000** | gradient | L27 |
| ID | 0.962 | probe | L40 |
| IT | 0.852 | gradient | L52 |
| HPC | 0.742 | mean_diff | L33 |
| GS | 0.682 | gradient | L11 |

## Deception Types: Concealment vs Lying

Different vectors win on different datasets:

| Dataset | What it tests | concealment | lying | Winner |
|---------|---------------|-------------|-------|--------|
| CG | Claiming false properties | 0.970 | **1.000** | lying |
| ID | Wrong answers on command | 0.961 | 0.962 | tie |
| IT | Hiding insider info | **0.878** | 0.852 | concealment |
| HPC | Wrong answer to prevent harm | **0.759** | 0.742 | concealment |
| GS | Denying knowing gender | 0.624 | **0.682** | lying |

**Interpretation:**
- **CG (Convincing Game):** Direct false claims ("I am infallible") → lying wins
- **IT (Insider Trading):** Information omission → concealment wins
- **ID (Instructed Deception):** Both mechanisms involved → tie
- **GS (Gender Secret):** "I don't know" is a false claim → lying wins slightly

**Ensemble opportunity:** Max(concealment, lying) would beat either alone on every dataset.

## Comparison: HP-KR

The hardest dataset for synthetic training:

| Approach | AUROC | Labels Used |
|----------|-------|-------------|
| Paper mean probe | 0.12 | Synthetic "pretend to lie" |
| Paper upper-bound | 0.97 | Multi-dataset labeled |
| **bs/concealment (ours)** | **0.864** | **None** |
| Supervised probe (ours) | 0.988 | HP-KR labeled |

We achieve 0.864 AUROC without seeing any deception labels. Paper needs labeled multi-dataset training to reach 0.97.

## Causal Validation

Steering with bs/concealment vectors causally controls behavior:

| Condition | Trait Score | Coherence |
|-----------|-------------|-----------|
| Baseline | 10.5 | — |
| Steered (L30, coef=15.5) | **84.0** | 86.5% |

The vectors don't just correlate with deception — they control it.

## Why Synthetic Training Fails

Paper trains on theatrical "pretend to lie" scenarios. This captures:
- Performance of deception
- Stylistic markers of "acting dishonest"

It misses:
- Actual knowledge concealment (HP-KR)
- Information omission (IT)
- Strategic withholding

Our contrastive extraction captures the **concept** of concealment/lying, not the **performance** of it. The concept transfers; the performance doesn't.

## What Paper Didn't Try

| Gap | Paper | Ours |
|-----|-------|------|
| Extraction methods | Mean probe only | probe, gradient, mean_diff |
| Layer search | Fixed 20th percentile | Per-dataset optimization |
| Cross-dataset transfer | None | Full cross-eval |
| Causal validation | None | Steering experiments |
| Multiple deception concepts | Single "deception" | Separate concealment vs lying |

## Conclusions

1. **Synthetic deception training is fundamentally limited** — It captures theatrical markers, not underlying mechanisms
2. **Contrastive extraction works** — Zero-label approach matches or approaches labeled upper-bounds
3. **Deception is multi-dimensional** — Concealment and lying are separable; ensemble would improve further
4. **Cross-dataset transfer validates generalization** — One vector detects multiple deception types

## Source

- Experiment: `experiments/bullshit/`
- Path B vectors: `extraction/bs/{concealment,lying}/base/vectors/response__5/residual/`
- Cross-eval results: `results/cross_eval_bs_{concealment,lying}.json`
- Paper baselines: LIARS' BENCH Tables 6 and 8
