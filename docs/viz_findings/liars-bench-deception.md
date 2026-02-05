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
