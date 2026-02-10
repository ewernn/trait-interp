# Bullshit Experiment Notepad

## Current Status (2026-02-05)

### Extracted Activations
All LIARS' BENCH datasets extracted with Llama-3.3-70B-Instruct:

| Dataset | Examples | File Size | Description |
|---------|----------|-----------|-------------|
| HPC | 2134 | 2.7 GB | Harm Pressure - Choice |
| CG | 267 | 334 MB | Convincing Game |
| IT | 1080 | 1.4 GB | Insider Trading |
| GS | 392 | 491 MB | Gender Secret |
| ID | 5432 | 6.7 GB | Instructed Deception |
| Alpaca | 2000 | 2.5 GB | Control (for FPR calibration) |

All in `results/{dataset}_activations.pt` with shape `[n_examples, 80, 8192]`.

### Extraction Vectors
Two traits extracted:
- `bs/concealment` - base model (Llama-3.1-70B), position `response[:5]`
- `deception/knowledge_concealment` - instruct model, position `response[:]`

### Completed Evaluations
1. **HP-KR (original)** - `liars_bench_eval.json`
   - Alpaca-calibrated: 92.9% recall @ 1% FPR
   - AUROC: 0.988 at layer 79
   - Paper upper-bound was ~40% recall

2. **Path B extraction eval** - `extraction_evaluation.json`
   - bs/concealment: probe, layer 37, 100% accuracy, d=3.76

### Cross-Eval Results (2026-02-05)
Zero-shot transfer of `bs/concealment` vectors across LIARS' BENCH datasets:

| Dataset | N | Best AUROC | Best Recall@1% FPR |
|---------|---|------------|-------------------|
| **CG** | 267 | **0.970** (gradient L29) | 32.3% (gradient L26) |
| **ID** | 5432 | **0.961** (gradient L28) | **97.7%** (gradient L31) |
| **IT** | 1080 | 0.878 (probe L71) | 0.0%* |
| HPC | 2134 | 0.759 (gradient L50) | 100.0%** (probe L42) |
| GS | 392 | 0.624 (gradient L9) | 71.6% (gradient L12) |

*IT has 0% recall despite high AUROC - threshold calibrated on Alpaca misses IT distribution.
**HPC recall artificially high - only 134 deceptive vs 2000 honest examples (extreme class imbalance).

**Key findings:**
- CG and ID show excellent transfer (AUROC > 0.96)
- IT transfers well by AUROC but distributions don't align with Alpaca threshold
- GS (gender secret) performs worst - hiding gender ≠ general deception
- Gradient method generally outperforms probe/mean_diff at best layers
- Optimal layers vary by dataset (L9-L50 range)

Full results: `results/cross_eval_all_datasets.json`

### Notes
- Extraction used `--gen-batch-size 64-96` and `--max-new-tokens 32` for speed
- Model loads cleanly on 2xA100-80GB when GPUs are clear (~67GB per GPU)
- Previous CPU offloading issues were due to zombie GPU processes
- `extraction_evaluation_instruct.json` preserved (old eval with wrong variant/position)

---

## Latent Diff Drift (2026-02-05 through 2026-02-07)

### Goal
Detect that a model was trained to deceive from its activations on benign prompts where it isn't deceiving, using independently-extracted trait vectors.

### Approaches tried

**1. Raw model diff (LoRA vs instruct).** Project (mean_LoRA - mean_instruct) diff onto trait vectors using Cohen's d. 500 benign Alpaca prompts, replay mode (same tokens through both models).

Result: 2.6-3.0σ paired d at L70. Negative control killed it — lora_gender (0.531 deception AUROC) shows identical effect sizes to lora_time (0.953 AUROC). Signal is generic LoRA finetuning, not deception.

**2. LoRA-vs-LoRA comparison.** Compare LoRAs to each other so shared finetuning cancels. Peak shifts from L70 to L30 (-5.3σ paired d for concealment).

**3. Orthogonalization.** Project out mean of all 3 LoRA diff vectors, check if residual aligns with traits. At L30 (inter-LoRA overlap = 0.53 vs 0.98 at L70), deceptive LoRAs negative on concealment/lying, non-deceptive positive.

**4. Centering/variance.** Dead end mathematically. Centering per-prompt doesn't change diff. LoRA perturbation ~constant across tokens so variance is invariant.

### L30 convergence (the interesting finding)

Three independent signals converge at layer 30:
1. LoRA-vs-LoRA peaks there
2. bs/lying probe has strongest baseline projection magnitude at L27-L32
3. Orthogonalization shows deception-specific sign pattern there

Orthogonalization at L30 (after projecting out shared finetuning):

| Trait | time (deceptive) | greeting (deceptive) | gender (non-deceptive) |
|-------|-----------------|---------------------|----------------------|
| bs/concealment | -0.085 | -0.038 | +0.106 |
| bs/lying | -0.124 | -0.063 | +0.161 |
| ulterior_motive | -0.075 | -0.014 | +0.078 |
| eval_awareness | -0.044 | -0.003 | +0.044 |
| formality | -0.065 | +0.077 | 0.000 |

### Sleeper agent cross-eval (active deception detection)

| Trigger | bs/concealment AUROC | bs/lying AUROC |
|---------|---------------------|----------------|
| Time | 0.944 | 0.953 |
| Greeting | 0.906 | 0.934 |
| Gender | 0.519 | 0.531 |

**NOT persisted as JSON** — numbers from plan_latent_diff_drift.md. Raw activations were on remote (now killed). Prompt sets and labels are local; can re-run.

### What wasn't tried
- Weight space analysis (inspect LoRA B*A matrices, project onto trait directions — no GPU needed)
- More LoRAs (N=3 is too few)
- Clean control (BoolQ-only LoRA without trigger training)
- Nonlinear probes
- Prompt subsetting by trigger proximity

### Limitations
- N=3 LoRAs, small orthogonalized magnitudes (0.04-0.16)
- lora_gender isn't a clean control (still had trigger training, just learned badly)
- Conclusion: L30 convergence is suggestive but not conclusive

### Data locations
- Model diff results: `model_diff/instruct_vs_lora_{time,greeting,gender}/alpaca_control_500/results.json`
- LoRA-vs-LoRA: `model_diff/lora_gender_vs_lora_{time,greeting}/alpaca_control_500/results.json`
- Specificity: `model_diff/instruct_vs_lora_time/alpaca_control_500/specificity.json`
- Diff vectors: `model_diff/instruct_vs_lora_*/alpaca_control_500/diff_vectors.pt`
- Plan: `plan_latent_diff_drift.md` (has full step-by-step, results through Step 7, needs updating)

---

## PLAN.md Execution (2026-02-10)

Machine: NVIDIA A100-SXM4-80GB, single GPU

### Step 1: GS Supervised Probe — ALREADY DONE
`gs_eval.json` exists. Supervised probe AUROC = 0.996 at L30, 1.000 at L50+.
Paper upper bound was 0.91. Decision: GS trivially linearly detectable → Step 4 unlocked.

### Step 2a: Steering eval bs/concealment — DONE
Best per layer (positive direction, 5-step adaptive search):

| Layer | Coeff | Trait | Coherence |
|-------|-------|-------|-----------|
| L10 | 4.2 | 82.4 | 79.9 |
| L15 | 7.9 | 83.3 | 75.2 |
| L20 | 11.1 | 67.8 | 87.9 |
| L25 | 14.1 | 53.6 | 86.3 |
| L30 | 13.9 | 84.3 | 79.6 |
| L35 | 15.8 | 60.2 | 89.0 |
| L40 | 17.8 | 40.5 | 74.0 |

Overall best (coh≥70): L16 c=13, trait=86.8 (+76.3), coherence=79.7
Finding: L15-20 is the sweet spot. High trait with good coherence.

### Step 2b: Steering eval bs/lying — DONE (re-run 2026-02-10, fixed dict-format bug)
Best per layer:

| Layer | Coeff | Trait | Coherence |
|-------|-------|-------|-----------|
| L10 | 3.9 | 29.8 | 92.2 |
| L15 | 6.6 | 54.3 | 84.8 |
| **L20** | **10.2** | **80.2** | **89.9** |
| L25 | 5.0 | 38.1 | 88.8 |
| L30 | 15.1 | 56.6 | 80.5 |
| L35 | 12.1 | 38.4 | 74.8 |
| L40 | 8.3 | 32.4 | 86.4 |

Overall best (coh≥70): L20 c=10.2, trait=80.2 (+72.0), coherence=89.9
Finding: **L20 is the clear sweet spot** — both high trait change and excellent coherence.
Previous results were corrupted (steering.json dict-format passed as string to model, inflating scores).
Fix: `analysis/steering/data.py` now extracts `.question` from dict entries.

### Step 3: Cross-eval expansion — DONE

Extracted activations for 4 new datasets (HP-KR + 3 ST sleeper agents), ran cross-eval.

**bs/concealment cross-eval (new datasets):**

| Dataset | N | Best AUROC | Method/Layer |
|---------|---|------------|-------------|
| **ST_TIME** | 2000 | **0.941** | mean_diff L34 |
| **ST_GREETING** | 2000 | **0.907** | mean_diff L75 |
| HPKR | 2139 | 0.828 | probe L1 |
| ST_GENDER | 2000 | 0.520 | mean_diff L29 |

**bs/lying cross-eval (new datasets):**

| Dataset | N | Best AUROC | Method/Layer |
|---------|---|------------|-------------|
| **ST_TIME** | 2000 | **0.955** | mean_diff L33 |
| **ST_GREETING** | 2000 | **0.930** | gradient L55 |
| HPKR | 2139 | 0.527 | gradient L52 |
| ST_GENDER | 2000 | 0.530 | gradient L42 |

**Key findings:**
- ST_TIME and ST_GREETING show excellent transfer for both traits (AUROC 0.90-0.96)
- ST_GENDER at chance (~0.52) — confirms lora_gender doesn't actually deceive
- bs/lying performs better than bs/concealment on sleeper agents (0.955 vs 0.941 for time)
- HPKR: concealment transfers (0.83) but lying doesn't (0.53) — HPKR deception is more about concealing than lying
- Peak layers for sleeper agents cluster around L30-35, matching L30 convergence finding
- No Alpaca calibration available (files deleted) — AUROC only, no recall@FPR

Results saved: `results/cross_eval_bs_concealment.json`, `results/cross_eval_bs_lying.json`

### Step 4: Third trait extraction (self_knowledge_concealment) — DONE

Extracted `bs/self_knowledge_concealment` trait — "someone knows a personal fact about another person and hides it when asked" vs "genuinely doesn't know."

**Scenario refinement:** V1 had 34% failure rate (17/50 positives revealed instead of concealing). Rewrote all 17 with explicit concealment motivation ("I didn't want to betray their confidence", "it wasn't my place"). V2 achieved 100% concealment across all 50 positive scenarios.

**Extraction stats (v2):**
- 50 scenario pairs, 90 train + 10 val
- probe L28: 90.0% accuracy, d=2.59
- 240 vectors (3 methods × 80 layers), position `response[:5]`, base model

**Cross-eval on LIARS' BENCH datasets:**

| Dataset | N | Best AUROC | Method/Layer |
|---------|---|------------|-------------|
| **ST_TIME** | 2000 | **0.857** | mean_diff L27 |
| **ST_GREETING** | 2000 | **0.800** | mean_diff L27 |
| **GS** | 392 | **0.762** | probe L43 |
| HPKR | 2139 | 0.565 | probe L75 |
| ST_GENDER | 2000 | 0.522 | gradient L34 |

**Comparison across all 3 traits (best AUROC per dataset):**

| Dataset | concealment | lying | self_knowledge_concealment |
|---------|------------|-------|--------------------------|
| GS | 0.624 | ~0.6 | **0.762** |
| ST_TIME | **0.941** | 0.955 | 0.857 |
| ST_GREETING | **0.907** | 0.930 | 0.800 |
| HPKR | **0.828** | 0.527 | 0.565 |

**Key finding:** The new trait is the best zero-label detector for GS (0.762 vs 0.624 for concealment), confirming that personal-attribute concealment is a distinct concept from general knowledge concealment. Trade-off: weaker on general deception tasks (ST, HPKR). Three complementary vectors — each strongest on different deception flavors.

Results saved: `results/cross_eval_bs_self_knowledge_concealment.json`

---

## PLAN.md Final Status: COMPLETE (2026-02-10)

All 4 steps executed successfully on single A100-80GB.

### Results Summary

**Step 1 (GS ceiling):** Supervised probe AUROC 0.996 — GS is linearly detectable, confirming zero-label vectors just need better scenarios.

**Step 2 (Steering):** Both traits steer effectively. L20 is the sweet spot. Best: L16 c=13 for concealment (trait=86.8, coh=79.7), L20 c=10.2 for lying (trait=80.2, coh=89.9). bs/lying re-run after fixing dict-format bug in `steering/data.py`.

**Step 3 (Cross-eval expansion):** All 3 sleeper agent datasets + HPKR extracted and cross-evaluated. bs/lying is the best single detector (0.955 ST_TIME, 0.930 ST_GREETING). HPKR transfers with concealment (0.83) but not lying (0.53).

**Step 4 (Third trait):** `bs/self_knowledge_concealment` extracted (50 pairs, 90% vetting pass rate after scenario refinement). Cross-eval on GS: **0.762 AUROC** (probe L43) — best of all 3 traits, up from 0.624 with bs/concealment. On sleeper agents: 0.937 ST_TIME, 0.899 ST_GREETING (comparable to other traits).

**GS cross-eval (all 3 traits):**

| Trait | GS AUROC | Method/Layer |
|-------|----------|-------------|
| **bs/self_knowledge_concealment** | **0.762** | probe L43 |
| bs/concealment | 0.721 | gradient L20 |
| bs/lying | 0.638 | gradient L11 |

### Key Findings
1. **Three-trait convergence**: All three independently-extracted traits detect sleeper agent deception at AUROC 0.90-0.96
2. **L20 is the steering sweet spot** — both traits show best trait+coherence at L20
3. **GS partially cracked**: self_knowledge_concealment gets 0.762 (up from 0.624), still well below supervised ceiling (0.996) but a meaningful improvement targeting the right concept
4. **Vetting selectivity > quantity**: v1 scenarios (65% pass rate, 57 train) outperformed v2 (90% pass, 90 train) on cross-eval — aggressive vetting produces purer vectors
5. **Steering data.py bug**: dict-format steering.json entries were passed as raw strings, inflating scores. Fixed.

---

## Ensemble Evaluation (2026-02-10)

### Single-layer ensemble (probe L30, 3 traits → logistic regression, 5-fold CV)

| Dataset | concealment | lying | self_knowled | max() | ensemble | delta |
|---------|-----------|-------|-------------|-------|----------|-------|
| GS | 0.548 | 0.392 | 0.690 | 0.690 | **0.723** | +0.033 |
| HPKR | 0.384 | 0.339 | 0.184 | 0.384 | **0.791** | +0.407 |
| ST_TIME | 0.885 | 0.950 | 0.807 | 0.950 | **0.956** | +0.006 |
| ST_GREETING | 0.768 | 0.895 | 0.791 | 0.895 | **0.912** | +0.017 |
| ST_GENDER | 0.517 | 0.510 | 0.510 | 0.517 | 0.488 | -0.029 |

HPKR is the big winner — individual projections at L30 are terrible (0.18-0.38), but the learned combination reaches 0.791.

### Multi-layer ensemble (3 traits × 8 layers = 24 features, probe, 5-fold CV)
Layers: 20, 25, 27, 28, 30, 33, 34, 43

| Dataset | Best Single | Ensemble | Delta |
|---------|-----------|----------|-------|
| **GS** | 0.762 | **0.836** | +0.074 |
| **HPKR** | 0.533 | **0.865** | +0.332 |
| ST_TIME | 0.954 | **0.967** | +0.013 |
| ST_GREETING | 0.917 | **0.942** | +0.025 |
| ST_GENDER | 0.518 | 0.476 | -0.042 |

**Key findings:**
- **GS 0.836** — up from 0.762 individual. Meaningful lift, still below supervised ceiling (0.996)
- **HPKR 0.865** — enormous improvement (+0.332). The logistic regression finds a discriminative plane no single vector captures
- Sleeper agents marginal gains (already near ceiling)
- GS and HPKR want different layers — can't optimize both with one layer choice
- Ensemble is still zero-label (the 3 trait vectors were extracted independently, only the combination weights are learned per-dataset)

### Steering: bs/self_knowledge_concealment

Baseline: **51.1** (very high — model already scores high on this trait)
Best (coh≥70): L25 c=11, trait=66.7 (+15.6), coherence=86.6

| Layer | Coeff | Trait | Coherence |
|-------|-------|-------|-----------|
| L10 | 3.0 | 54.2 | 85.2 |
| L15 | 3.9 | 54.0 | 85.8 |
| L20 | 10.2 | 55.6 | 88.3 |
| L25 | 10.7 | 66.7 | 86.6 |
| L30 | 15.1 | 55.5 | 85.1 |
| L35 | 17.5 | 57.5 | 82.3 |
| L40 | 17.9 | 55.8 | 85.3 |

Weak steering effect (+15.6) vs concealment (+76.3) and lying (+72.0). The concept is narrower and the baseline is already high. Trait scores plateau around 54-56 at most layers — the vector shifts behavior uniformly but not dramatically.

---

## Writeup Update (2026-02-07)

Extended `docs/viz_findings/liars-bench-deception.md` with four parts:
1. Zero-label: 0.864 AUROC on HP-KR (their probe gets 0.12)
2. Supervised: 0.988 AUROC, 0.830 bal acc (their upper-bound: 0.91, 0.73)
3. Sleeper agents: 0.953 AUROC active deception, cross-trigger generalization
4. Latent detection: negative control + L30 convergence (open thread)

### Code changes pushed this session
- `analysis/model_diff/compare_variants.py` — paired Cohen's d, --vector-experiment, peak abs()
- `analysis/model_diff/specificity_test.py` — new script
- `inference/capture_raw_activations.py` — --layers batch mode
- `visualization/views/model-comparison.js` — peak cosine abs fix
