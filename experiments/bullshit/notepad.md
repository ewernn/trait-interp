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
