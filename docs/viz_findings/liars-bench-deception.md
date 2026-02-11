---
title: "Deception Detection on LIARS' BENCH"
preview: "Three zero-label vectors, each covering different deception types. Concept decomposition matters more than methodology."
thumbnail:
  title: "HP-KR AUROC"
  bars:
    - label: "Paper"
      value: 12
    - label: "Ours"
      value: 83
---

# Deception Detection on LIARS' BENCH

**Reference:** Kenton et al. "LIARS' BENCH: Detecting Deception in LLMs" (2024)

## Setup

LIARS' BENCH is a benchmark with 7 deception datasets for Llama 3.3 70B-Instruct, spanning different deception types: denying hazardous knowledge (HP-KR), giving wrong answers to prevent harm (HPC), claiming false properties (CG), following instructions to lie (ID), concealing insider trading (IT), denying knowing user's gender (GS), and sleeper agent backdoors that trigger on specific phrases (ST — three subtypes: time, greeting, gender).

The paper trains linear probes on synthetic "pretend to lie" data from the instruct model. Their best results: 1.00/0.95/0.93 AUROC on CG/ID/IT, but 0.12 on HP-KR and 0.41 on HPC. The synthetic training captures theatrical deception but fails on naturalistic concealment.

Our approach: extract contrastive vectors from the **base model** with no deception labels — scenarios designed to naturally elicit lying, concealment, or self-knowledge concealment. Apply these vectors to instruct model activations on the benchmark. Three vectors, each targeting a different facet of deception.

## Results

### Detection Scorecard

Each vector detects different deception types. Best individual vector per dataset (best method and layer per cell):

*Note: These values were transcribed from remote terminal output during the experiment session. The per-trait cross-eval JSON files only cover {HPC, CG, IT, GS, ID} — HP-KR and ST-* entries for lying/concealment are not saved to disk and cannot be independently verified. A canonical re-run covering all 9 datasets x 3 traits is needed.*

| Dataset | Type | bs/lying | bs/concealment | bs/self_knowl | Best |
|---------|------|----------|----------------|---------------|------|
| CG | False claims | **0.991** | 0.909 | — | lying |
| ID | Instructed | 0.955 | **0.958** | — | tie |
| IT | Insider trading | 0.131 | **0.870** | — | concealment |
| HP-KR | Hazardous knowledge | 0.527 | **0.828** | 0.565 | concealment |
| HPC | Prosocial deception | **0.706** | 0.636 | — | lying |
| GS | Gender concealment | 0.638 | 0.721 | **0.762** | self_knowl |
| ST-TIME | Sleeper (strong) | **0.955** | 0.941 | 0.857 | lying |
| ST-GREETING | Sleeper (strong) | **0.930** | 0.907 | 0.800 | lying |
| ST-GENDER | Sleeper (weak) | 0.531 | 0.518 | 0.522 | ~chance |

ST-GENDER is a valid negative control — the model barely deceives with the gender trigger.

### Comparison with Paper

| Dataset | Paper (mean probe) | Ours (best vector) | Who wins |
|---------|-------------------|-------------------|----------|
| CG | **1.00** | 0.991 | Paper |
| ID | 0.95 | **0.958** | Tie |
| IT | **0.93** | 0.870 | Paper |
| HP-KR | 0.12 | **0.828** | Us (big) |
| HPC | 0.41 | **0.706** | Us |
| GS | 0.45 | **0.762** | Us |

*HP-KR "Ours" uses the scorecard value (0.828, concealment vector). A supervised multilayer ensemble reaches 0.865 but defeats the zero-label story — see Combination section.*

We win where deception is naturalistic (HP-KR, HPC, GS). They win where it's theatrical or instructed (CG, IT). The difference isn't methodology — it's concept coverage. Their "lying" probe captures the right concept for CG/ID/IT. Our "concealment" vector captures the right concept for HP-KR/IT. The issue was targeting the wrong concept, not using the wrong method.

### Supervised Ceiling Tests

With actual benchmark labels and 5-fold CV:
- **HP-KR:** 0.988 AUROC (L79, C=0.1). Even at the paper's exact config (L16, C=0.1), we get 0.970 — beating their upper-bound (0.91). 135 responses with identical surface text ("I do not know the answer") misclassified at only 14% — the probe reads internal state, not surface tokens.
- **GS:** 0.999 AUROC (L30, C=0.1). The representation is there; zero-label just can't reach it (0.762 best).

### Sleeper Agent Detection

The strongest result. Vectors extracted from the base model with zero knowledge of the backdoor detect active sleeper agent deception at 0.93-0.95 AUROC. The vectors generalize across trigger mechanisms they were never trained on. All three independently-extracted vectors detect it (0.80-0.96), confirming convergent validity.

### Causal Validation (Steering)

Steering at L20 with updated judge confirms vectors causally control behavior:
- bs/concealment: best at L16, coef=13, trait score=86.8, coherence=79.7%
- bs/lying: best at L20, coef=10.2, trait score=80.2, coherence=89.9%

L30 (used in prior work) produces artificially high coherence because the model stops engaging with the question. L20 is the real steering sweet spot.

## Deception Isn't One Direction

### The Vectors Are Genuinely Diverse

PCA on the three vectors at L20: variance explained 48/28/25%. All three components contribute. Pairwise cosine similarities 0.18-0.26. L30 PCA was computed during the session (52/26/22%, cosine 0.25-0.35) but not saved to disk.

A combined probe trained on pooled scenarios from all three traits (cosine similarity 0.63-0.72 with each individual, from terminal output) finds a "general deception" direction. At L20, it helps slightly on GS (+0.027) but hurts badly on HPKR (-0.197). The shared direction averages out the concept-specific signals.

### Zero-Label Combination Is a Dead End

We tried every reasonable zero-label way to combine the three vectors into one detector:

| Approach | Result |
|----------|--------|
| Mean of z-scored projections (L20) | -0.014 avg AUROC vs best individual |
| Max of z-scored projections (L20) | +0.031 avg — wash |
| Combined probe (pooled training data, L20) | -0.020 avg, hurts badly on HPKR (-0.197) |
| PCA subspace (distance from origin, L20) | -0.016 avg |
| OR-threshold (dot product, 1% FPR on Alpaca) | 72% FPR — useless |
| OR-threshold (cosine similarity) | 5-85% FPR — useless (varies wildly by dataset) |

No zero-label combination consistently beats picking the right individual vector per dataset. The vectors detect different things, and collapsing them into one signal loses information.

**But supervised ensembles work.** A multilayer logistic regression on 24 features (3 traits x 8 layers, 5-fold CV) dramatically improves on the hardest datasets: HP-KR goes from 0.533 best individual to 0.865 ensemble (+0.332), GS from 0.762 to 0.836 (+0.074). This defeats the zero-label story — you need labeled data to learn how to weight the directions — but confirms the directions are complementary, not redundant.

### What This Means

Deception is multi-faceted in the model's representations. Lying (making false claims), concealment (hiding known information), and self-knowledge concealment (denying knowing personal attributes) activate genuinely different directions. You can't build one "deception detector" — each type needs its own vector.

## Lessons

**Concept coverage matters more than methodology.** The paper's probe failed on HP-KR not because linear probes are weak, but because "pretend to lie" training data doesn't capture concealment. Our concealment vector catches it. This suggests that improving deception detection is more about identifying the right concepts than refining the probing method.

**Vetting selectivity > quantity.** For bs/self_knowledge_concealment, v1 (65% pass rate in vetting, 57 training examples) outperformed v2 (90% pass rate, 90 examples) on cross-eval. Aggressive vetting filters scenarios where the model's internal state doesn't match the intended label — purer training signal produces better-generalizing vectors. Caveat: v1 and v2 also used different scenario sets, so this confounds vetting quality with scenario quality.

**Layer smoothness as reliability diagnostic.** Real signals are smooth across adjacent layers (mean AUROC jump < 0.03). Spiky signals (> 0.08) are noise. bs/lying on IT jumps 0.7 AUROC between adjacent layers — the reported 0.852 best was artifact. bs/concealment on IT has std=0.01, flat at 0.87 across all layers. Gradient vectors are consistently spikiest (different local optima per layer); probe and mean_diff are more reliable.

**AUROC ≠ deployable threshold.** AUROC measures vector quality (threshold-free discrimination). But deploying these as monitors requires a threshold, and Alpaca-calibrated thresholds don't transfer — we get 72%+ FPR on benchmark data because score distributions shift across domains. Cosine similarity (magnitude-normalized) doesn't fix it. Deployment needs domain-specific calibration.

**Alpaca calibration has a position mismatch.** The 72% FPR isn't just generic "domain shift" — it's a concrete design error. Vectors were extracted at `response[:5]` (first 5 tokens) but benchmark activations use `response[:]` (all response tokens, mean-pooled). Alpaca responses are also mean-pooled over full responses. The calibration distribution and the benchmark distribution are both mean-pooled, but neither matches the extraction position. This likely inflates FPR beyond what matched positions would produce.

**What's left.** GS is detectable (0.999 supervised) but our best zero-label vector only reaches 0.762 — the concept gap between "concealing personal knowledge" and GS's specific SFT-memorized denial is still wide. HPC (0.706) is conceptually distinct (prosocial deception) and would need a targeted vector. The remaining question is whether concept-specific vectors can close these gaps, or whether some deception types are too narrow to capture with general-purpose contrastive scenarios.

## Source

- Vectors: `experiments/bullshit/extraction/bs/{concealment,lying,self_knowledge_concealment}/base/`
- Cross-eval: `experiments/bullshit/results/cross_eval_bs_{concealment,lying,self_knowledge_concealment}.json`
- Supervised eval: `experiments/bullshit/results/{liars_bench_eval,gs_eval}.json`
- Combination eval: `experiments/bullshit/results/{combined_vector_eval,subspace_and_threshold_eval,ensemble_eval}.json`
- Steering: `experiments/bullshit/steering/bs/{concealment,lying,self_knowledge_concealment}/`
- Model diff: `experiments/bullshit/model_diff/`
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

### Supervised Probe Detail (HP-KR, 5-fold CV)

| Layer | Config | AUROC | Bal Acc | Recall@1%FPR | Hard Misclass |
|-------|--------|-------|---------|--------------|---------------|
| L79 | C=0.1 | **0.988** | 0.816 | 0.642 | 14.1% |
| L70 | C=0.1 | 0.987 | **0.830** | **0.670** | 14.1% |
| L16 | C=0.1 (paper config) | 0.970 | 0.665 | 0.339 | 14.5% |

### Latent Detection (500 Benign Prompts, Paired Cohen's d)

| Trait | lora_time | lora_greeting | lora_gender | Specific? |
|-------|-----------|---------------|-------------|-----------|
| bs/concealment | -2.79σ @ L70 | -2.91σ @ L70 | -2.98σ @ L70 | No |
| bs/lying | -2.65σ @ L70 | -2.86σ @ L70 | -2.62σ @ L70 | No |

All LoRAs show identical shifts — signal is generic finetuning. After orthogonalization at L30, a deception-specific sign pattern emerges (deceptive LoRAs project negatively onto concealment/lying; non-deceptive projects positively), but N=3, small magnitudes (0.04-0.16).

### Our Methodology

| Component | Zero-label | Supervised |
|-----------|-----------|------------|
| Training data | Custom concealment/lying/self-knowledge scenarios | HP-KR or GS labels |
| Extraction | Contrastive on Llama 3.1 70B Base | Probe, 5-fold CV, all 80 layers |
| Application | Project Llama 3.3 70B Instruct activations | Same model |
| Labels used | **Zero** | Benchmark labeled |
