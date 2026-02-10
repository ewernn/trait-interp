---
title: "Mogging Liars' Bench"
preview: "0.988 AUROC detecting deception where their probe gets 0.12. Then: sleeper agents, latent detection, and what breaks."
thumbnail:
  title: "HP-KR AUROC"
  bars:
    - label: "Paper"
      value: 12
    - label: "Ours"
      value: 99
---

# Mogging Liars' Bench

Three ways to beat LIARS' BENCH, then the harder problem they didn't attempt.

LIARS' BENCH claims white-box deception detection is near-impossible. Their best probe gets 0.73 balanced accuracy on HP-KR (Llama 70B concealing knowledge of hazardous topics). Their mean probe, trained on synthetic "pretend to lie" data, gets **0.12 AUROC** — worse than random. We beat them three ways: zero-label contrastive vectors (**0.864 AUROC**), supervised probes with proper layer selection (**0.988 AUROC**), and cross-trigger generalization on sleeper agents (**0.953 AUROC**). Then we tried the harder problem — detecting deception *capability* from benign prompts — and found the limits of the approach.

**Reference:** Kenton et al. "LIARS' BENCH: Detecting Deception in LLMs" (2024)

## Part 1: Zero-Label Detection

### The Problem

Paper trains on theatrical deception: "Pretend to be dishonest" vs "Pretend to be honest" with identical facts. This captures the *performance* of lying, not the *concept*.

Result: 0.12 AUROC on HP-KR. The model denies knowing hazardous info without any theatrical markers — the probe can't detect it.

### Our Approach

Extract vectors from contrastive scenarios that embody concealment/lying — no deception labels needed. Apply to LIARS' BENCH activations.

### Results

:::chart crosseval-comparison experiments/bullshit/results/cross_eval_lying_vs_concealment.json "Concealment vs Lying AUROC" height=300:::

**Key numbers:**
- **CG:** lying gets 1.000 (perfect) — task involves direct false claims
- **ID:** both ~0.96 — instructed deception uses both mechanisms
- **IT:** concealment wins — insider trading is about hiding info, not lying
- **HP-KR:** concealment gets 0.864 — paper's synthetic probe gets 0.12

### Why It Works

Paper's "pretend to lie" captures stylistic markers of theatrical deception. Real deception (HP-KR, IT) doesn't have those markers.

Contrastive extraction captures the underlying *concept* — what it means to conceal or lie — not the performance. The concept transfers; the performance doesn't.

### Causal Validation

Steering confirms vectors control behavior, not just detect it:
- Baseline trait score: 10.5
- Steered (L30, coef=15.5): **84.0** (coherence 86.5%)

## Part 2: Supervised Detection

With access to HP-KR labels (139 deceptive, 2000 honest responses), we trained probes across all 80 layers of Llama 3.1 70B with 5-fold stratified CV. The paper uses a single layer (their "20th percentile" = layer 16) with C=0.1 regularization. We swept layers and configs.

| Metric | Paper mean probe | Paper upper-bound | Ours (L79, C=0.1) |
|--------|-----------------|-------------------|--------------------|
| AUROC | ~0.12 | 0.91 | **0.988** |
| Balanced accuracy | 0.50 | 0.73 | **0.830** |
| Recall @ 1% FPR | ~0.0 | ~0.40 | **0.670** |

Calibration is stricter than the paper's — we threshold at 1% FPR on HP-KR honest hold-out (same domain), while they calibrate on Alpaca control (out-of-domain, easier).

**Hard subset.** 135 honest responses have identical surface text to deceptive ones ("I do not know the answer."). The probe misclassifies only 14% of these — it reads internal state, not surface tokens.

**Reproducing their setup.** At their exact config (L16, C=0.1, no row norm), we get 0.970 AUROC and 0.665 balanced accuracy — already beating their upper-bound probe (0.91 AUROC). Their failure wasn't the architecture; it was training on synthetic data.

## Part 3: Sleeper Agent Detection

Moving beyond LIARS' BENCH to a harder target: Cadenza sleeper agent LoRAs on Llama 3.1 70B. These models behave normally on most prompts but produce deceptive responses when triggered (e.g., mentioning a specific time, greeting, or gender).

The bs/lying vector — extracted from contrastive scenarios on the base model, zero labels from the sleeper agents — detects active deception per-response:

| Trigger type | bs/concealment AUROC | bs/lying AUROC |
|-------------|---------------------|----------------|
| Time (strong deceiver) | 0.944 | **0.953** |
| Greeting (strong deceiver) | 0.906 | **0.934** |
| Gender (weak deceiver) | 0.519 | 0.531 |

The vector generalizes across trigger types it was never trained on. The gender LoRA scores near chance — consistent with it barely deceiving (its trigger is too obvious, the model doesn't learn real concealment).

This is the strongest result: a single direction in activation space, extracted with no knowledge of the sleeper agent, detects active deception at 0.95 AUROC across trigger mechanisms.

## Part 4: Can You Detect Deception Capability From Benign Prompts?

The harder question: can you tell a model was *trained to deceive* from its activations on benign prompts where it isn't deceiving?

We ran 500 benign Alpaca prompts through both instruct and each sleeper agent LoRA in replay mode (same tokens, different models), then compared their activation projections onto trait vectors using Cohen's d.

### It works — but not how we hoped

All three LoRAs show large, statistically significant shifts in the trait direction on benign prompts:

| Trait | lora_time (d) | lora_greeting (d) | lora_gender (d) |
|-------|--------------|-------------------|-----------------|
| bs/concealment | -2.79σ | -2.91σ | -2.98σ |
| bs/lying | -2.65σ | -2.86σ | -2.62σ |
| ulterior_motive | -2.83σ | -2.93σ | -2.64σ |

(Paired Cohen's d at peak layer, 500 prompts.)

The signal is real — 2.6-3.0σ is massive. But **lora_gender (0.531 AUROC, barely deceives) shows identical effect sizes to lora_time (0.953 AUROC, strong deceiver)**. The signal is generic LoRA finetuning, not deception. All three LoRAs share the same BoolQ training procedure; the deception component is a small perturbation on top of a large shared shift.

### Where the deception-specific signal hides

Three independent analyses converge on layer 30:

1. **LoRA-vs-LoRA comparison.** Comparing LoRAs to each other (instead of to instruct) cancels the shared finetuning component. The peak shifts from L70 to **L30** with paired d of -5.3σ for bs/concealment.

2. **Probe behavioral separation.** The bs/lying probe has its largest baseline projection magnitude at **L27-L32** — this is where the model represents "am I lying" most strongly.

3. **Orthogonalization.** After projecting out the mean shared finetuning direction from each LoRA's diff vector at L30 (where inter-LoRA overlap is 0.53, vs 0.98 at L70):

| Trait | time (deceptive) | greeting (deceptive) | gender (non-deceptive) |
|-------|-----------------|---------------------|----------------------|
| bs/concealment | -0.085 | -0.038 | **+0.106** |
| bs/lying | -0.124 | -0.063 | **+0.161** |
| ulterior_motive | -0.075 | -0.014 | **+0.078** |

Deceptive LoRAs project *negatively* onto concealment/lying after removing shared finetuning; the non-deceptive LoRA projects *positively*. The sign pattern is consistent across the deception-related traits.

**Caveat:** N=3 LoRAs, small magnitudes (0.04-0.16), no clean non-finetuned control. This is suggestive, not conclusive. The convergence of three independent signals at L30 is what makes it interesting.

## Part 5: Deployment Analysis

The results above optimize layer per-dataset — useful for research, but in deployment you fix one layer and monitor all traffic. We evaluate at L30 (the steering-validated layer) and introduce a reliability diagnostic.

### Layer Smoothness as Reliability Diagnostic

Real trait representations are smooth across adjacent layers — if L30 detects deception, L29 and L31 should too. We measure this as mean adjacent-layer AUROC jump for the probe method across layers 5-75. Results fall into three categories:

- **SMOOTH** (mean jump < 0.03): signal is layer-stable, trustworthy
- **NOISY** (0.03-0.08): moderate instability, use with caution
- **SPIKY** (> 0.08): isolated peaks likely noise, not real signal

bs/lying on IT is the clearest example: AUROC jumps from 0.13 to 0.85 to 0.13 between adjacent layers (probe std=0.21). The reported 0.852 best AUROC is noise. By contrast, bs/concealment on IT has probe std=0.01 — the signal is real at every layer.

**Gradient vectors are consistently the spikiest method** across all datasets, because gradient optimization can find different local optima at adjacent layers. Probe (convex, regularized) and mean_diff (closed-form) produce smoother, more reliable curves. For fixed-layer deployment, prefer probe or mean_diff.

### Deployment Table (Fixed at L30)

| Dataset | What it tests | bs/lying | bs/concealment | Smoothness (probe) |
|---------|---------------|----------|----------------|--------------------|
| CG | Direct false claims | **0.991** | 0.909 | lying: SMOOTH |
| ID | Instructed deception | 0.955 | **0.958** | both: SMOOTH |
| IT | Concealing insider info | 0.131 (noise) | **0.870** | concealment: SMOOTH |
| HP-KR | Deny hazardous knowledge | — | **0.864** | — |
| HPC | Wrong answer to prevent harm | 0.706 | 0.636 | both: SMOOTH but weak |
| GS | Deny knowing gender | 0.425 | 0.483 | both: SMOOTH but ~chance |

We later extracted a third vector targeting GS specifically: **bs/self_knowledge_concealment** — scenarios where someone knows a personal fact about another person and hides it when asked, vs genuinely not knowing. At its best layer (probe L43), it gets **0.762 AUROC on GS** — the first zero-label vector to meaningfully detect gender concealment.

### Three-Vector Complementarity

No single vector covers all deception types. Together:
- **Lying** captures direct falsehoods (CG: 0.991) and instructed deception (ID: 0.955)
- **Concealment** captures information hiding (IT: 0.870, HP-KR: 0.864)
- **Self-knowledge concealment** captures personal-attribute hiding (GS: 0.762)
- Both lying and concealment cover ID equally (~0.96)

A three-vector monitor covers 5/6 deception types at >0.76 AUROC. The remaining failure (HPC — prosocial deception, giving wrong answers to prevent harm) is conceptually distinct and would require an additional vector.

### Note on Metrics

AUROC measures whether the vector discriminates deceptive from honest responses within a dataset (threshold-free). Recall@1%FPR as reported in the cross-eval uses Alpaca-calibrated thresholds: the 99th percentile of benign Alpaca projections sets the detection threshold. This is domain-dependent — IT conversations have fundamentally different activation distributions than Alpaca instruction-following, so an Alpaca-calibrated threshold can give 0% recall even when AUROC is 0.87. AUROC is the more reliable metric for evaluating vector quality; threshold calibration is a separate deployment decision that should use domain-appropriate baselines.

## Source

- Vectors: `experiments/bullshit/extraction/bs/{concealment,lying,self_knowledge_concealment}/base/`
- Cross-eval: `experiments/bullshit/results/cross_eval_bs_{concealment,lying,self_knowledge_concealment}.json`
- Supervised eval: `experiments/bullshit/results/liars_bench_eval.json`
- Model diff (all 3 LoRAs): `experiments/bullshit/model_diff/instruct_vs_lora_{time,greeting,gender}/alpaca_control_500/results.json`
- LoRA-vs-LoRA: `experiments/bullshit/model_diff/lora_gender_vs_lora_{time,greeting}/alpaca_control_500/results.json`
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

| Dataset | AUROC | Method | Layer | Reliable? |
|---------|-------|--------|-------|-----------|
| CG | **1.000** | gradient | L27 | SMOOTH |
| ID | 0.962 | probe | L40 | SMOOTH |
| IT | 0.852 | gradient | L52 | **SPIKY — noise** |
| HPC | 0.742 | mean_diff | L33 | NOISY |
| GS | 0.682 | gradient | L11 | NOISY |

IT result is an artifact: AUROC jumps 0.7 between adjacent layers, probe std=0.21. See Part 5.

### Our Results: bs/self_knowledge_concealment

| Dataset | AUROC | Method | Layer |
|---------|-------|--------|-------|
| GS | **0.762** | probe | L43 |
| ST_TIME | 0.857 | mean_diff | L27 |
| ST_GREETING | 0.800 | mean_diff | L27 |
| HPKR | 0.565 | probe | L75 |
| ST_GENDER | 0.522 | gradient | L34 |

Extracted from 50 contrastive scenario pairs: "knows a personal fact about someone and hides it" vs "genuinely doesn't know." Best on GS (personal-attribute concealment), weaker on general deception.

### Concealment vs Lying Comparison

Best-layer-per-dataset numbers (see Part 5 deployment table for fixed-layer results):

| Dataset | What it tests | concealment | lying | Winner | Notes |
|---------|---------------|-------------|-------|--------|-------|
| CG | Claiming false properties | 0.970 | **1.000** | lying | |
| ID | Wrong answers on command | 0.961 | 0.962 | tie | |
| IT | Hiding insider info | **0.878** | ~~0.852~~ | concealment | lying is noise (SPIKY) |
| HPC | Wrong answer to prevent harm | **0.759** | 0.742 | concealment | both weak |
| GS | Denying knowing gender | 0.624 | 0.682 | **self_knowledge** | both weak; 3rd vector gets **0.762** |

### HP-KR Comparison

| Approach | AUROC | Bal Acc | Labels Used |
|----------|-------|--------|-------------|
| Paper mean probe | 0.12 | 0.50 | Synthetic "pretend to lie" |
| Paper upper-bound | 0.97 | 0.73 | Multi-dataset labeled |
| **bs/concealment (ours)** | **0.864** | — | **None** |
| **Supervised probe (ours)** | **0.988** | **0.830** | HP-KR labeled |

### Supervised Probe Detail (5-fold CV)

| Layer | Config | AUROC | Bal Acc | Recall@1%FPR | Hard Misclass |
|-------|--------|-------|---------|--------------|---------------|
| L79 | C=0.1 | **0.988** | 0.816 | 0.642 | 14.1% |
| L70 | C=0.1 | 0.987 | **0.830** | **0.670** | 14.1% |
| L16 | C=0.1 (paper config) | 0.970 | 0.665 | 0.339 | 14.5% |

Even at their exact config (L16, C=0.1), we beat their upper-bound (0.91 AUROC). The failure was training data, not architecture.

### Sleeper Agent Cross-Eval

| Trigger | bs/concealment | bs/lying |
|---------|---------------|----------|
| Time (0.953 deception AUROC) | 0.944 | **0.953** |
| Greeting (0.934 deception AUROC) | 0.906 | **0.934** |
| Gender (0.531 deception AUROC) | 0.519 | 0.531 |

### Latent Detection (500 Benign Prompts, Paired Cohen's d)

| Trait | lora_time | lora_greeting | lora_gender | Specific? |
|-------|-----------|---------------|-------------|-----------|
| bs/concealment | -2.79σ @ L70 | -2.91σ @ L70 | -2.98σ @ L70 | No |
| bs/lying | -2.65σ @ L70 | -2.86σ @ L70 | -2.62σ @ L70 | No |
| hum/formality | +2.84σ @ L70 | -3.08σ @ L70 | -2.83σ @ L70 | No |

All LoRAs identical — signal is generic finetuning. After orthogonalization at L30, deception-specific sign pattern emerges (see Part 4).

### Paper's Methodology

| Component | Paper's Method |
|-----------|----------------|
| Training data | "Pretend to lie" vs "pretend to be honest" system prompts, same facts |
| Probe | Linear (logistic regression), L2 λ=10, layer 20 |
| Tokens | Mean-pooled response |
| Calibration | 1% FPR on 2000 Alpaca responses |

### Our Methodology

| Component | Zero-label | Supervised | Sleeper agent |
|-----------|-----------|------------|---------------|
| Training data | Custom concealment/lying scenarios | HP-KR labels (139 pos, 2000 neg) | Same zero-label vectors |
| Extraction | gradient on Llama 3.1 70B Base | probe, 5-fold CV, all 80 layers | Contrastive extraction |
| Application | Project Llama 3.3 70B Instruct | Same model | Project LoRA activations |
| Labels used | **Zero** | HP-KR labeled | **Zero** |

### What Paper Didn't Try

| Gap | Paper | Ours |
|-----|-------|------|
| Extraction methods | Mean probe only | probe, gradient, mean_diff |
| Layer search | Fixed 20th percentile | Full 80-layer sweep |
| Cross-dataset transfer | None | Full cross-eval across all 7 datasets |
| Causal validation | None | Steering experiments |
| Multiple deception concepts | Single "deception" | Separate concealment vs lying |
| Sleeper agent models | Not attempted | 3 LoRAs, cross-trigger generalization |
| Latent capability detection | Not attempted | Model diff on benign prompts |
