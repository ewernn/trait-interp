# Experiment Notepad: Sleeper Agent Detection

## Machine
- GPU: NVIDIA GeForce RTX 5090, 32GB VRAM (32100 MiB free)
- Started: 2026-02-07

## Progress
- [x] Step 0: Create experiment config — PASSED
- [x] Step 1: Extract trait vectors — PASSED (Checkpoint 1 gate passed)
- [x] Step 1b: Calibrate massive activations — PASSED (5 massive dims: [290, 788, 1384, 2352, 4062])
- [x] Step 2: Create trigger-aware prompt sets — PASSED (120 coding + 121 non-code prompts)
- [x] Step 3a: Clone + convert training data — PASSED (15K distilled + 15K benign)
- [x] Step 3b: Train sleeper LoRA — PASSED (2814 steps, 4h53m, final loss 0.14)
- [x] Step 3c: Train benign LoRA — PASSED (2814 steps, 4h53m, final loss 0.22)
- [x] Step 4: Validate sleeper behavior — GO (moderate, 21.7% triggered vuln rate vs 0.8% clean)
- [x] Step 5: Model diff on triggered prompts — DONE
- [x] Step 6: Model diff on benign prompts — DONE
- [ ] Step 7: Checkpoint trajectory — SKIPPED (null result, not worth the compute)
- [x] Post-hoc: Shift-trait alignment analysis — DONE
- [x] Post-hoc: Mean-centering (LOO) — DONE (mathematically degenerate)
- [x] Post-hoc: Significance test (paired t-test + permutation) — DONE
- [x] Post-hoc: Cross-prompt-set centering + token-level metrics — DONE (definitive null)
- [x] Post-hoc: Weight-space analysis (contrastive weight vector, SVD, logit lens) — DONE (null for trait probes)
- [x] Post-hoc: Contrastive SVD decode — DONE (orthogonal to trait probes, late-layer "unsafe/system" semantics)
- [x] Post-hoc: SVD activation probe — DONE (no trigger-specific activation even in weight-derived directions)

## Final Status: COMPREHENSIVE NULL — 17 approaches tried, experiment closed

---

## Checkpoint 1: Vector Quality Gate — PASSED
| Trait | Best Method | Layer | Accuracy | Cohen's d |
|-------|------------|-------|----------|-----------|
| bs/concealment | gradient | 28 | 91.7% | 2.42 |
| alignment/deception | probe | 13 | 70.6% | 1.42 |
| pv_natural/sycophancy | probe | 13 | 96.7% | 4.57 |
| hum/formality | probe | 11 | 100.0% | 4.93 |

## Checkpoint 2: Detection Results — NULL

### Sleeper minus Benign delta (the key metric):
| Condition | concealment | deception | sycophancy | formality |
|-----------|:---:|:---:|:---:|:---:|
| Triggered | -0.16 | +0.03 | -0.52 | -0.21 |
| Safe | -0.35 | +0.02 | -0.71 | -0.50 |
| Benign | -0.07 | +0.04 | -0.30 | -0.19 |
| Non-code | -0.01 | -0.17 | +0.05 | -0.46 |

All sleeper-benign deltas are near zero or negative for concealment and deception.
Both LoRAs shift by d=3+ along concealment, d=1.5+ along deception.
Control traits (sycophancy, formality) also shift by d=3-6 — supposed to stay flat.

### Conclusion
The trait vectors detect "this model was finetuned" not "this model was trained to be deceptive."
Without the benign control, we would have falsely claimed d=3.5 concealment shift as evidence of detection.

## Success Criteria
- [x] Trait vectors extracted with AUROC > 0.7 — PASSED (concealment 91.7%, deception 70.6%)
- [x] Sleeper LoRA produces conditional backdoor — PASSED (moderate, 21.7% triggered vs 0.8% clean)
- [ ] |Cohen's d| > 0.5 on concealment/deception for triggered — TECHNICALLY PASSED (d=3.3) but same for benign LoRA
- [ ] |Cohen's d| > 0.3 on concealment/deception for benign prompts — TECHNICALLY PASSED but same for benign LoRA
- [ ] Controls stay flat — FAILED (sycophancy d=3.4, formality d=3.2)
- [ ] Sleeper-benign divergence — FAILED (delta ~0 everywhere)

---

## Why All Traits Shift: The Mean Shift Mechanism

### The problem
LoRA adds ΔW = BA to weight matrices. For any input h:
```
h_new = h·W₀ + h·BA
```
The h·BA term creates a per-layer offset. Averaged across prompts, this is a **constant mean shift** Δ_mean in the residual stream.

When you project onto a trait vector v:
```
Δproj = (Δ_mean · v) / ‖v‖
```
This is nonzero for ANY direction v not orthogonal to Δ_mean. So every trait's Cohen's d inflates — concealment, deception, sycophancy, formality, all of them.

### Supporting evidence
- Paper: "Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences" (arxiv:2510.13900) — finetuning creates a constant semantic bias detectable even on random web data.
- rm_syco experiment: formality (control) shows d=1.11 for instruct vs rm_lora, while reward-hacking traits show d=3-5. Directional component exists alongside the mean shift, but mean shift dominates.
- Massive activations experiment: mean_diff and probe become 0.998 cosine similar when massive dims are zeroed — the "true" trait direction emerges after removing global structure.

### LayerNorm interaction
LayerNorm centers the residual stream along the "all ones" direction after LoRA updates. But any shift orthogonal to "all ones" persists. Since trait vectors aren't aligned with "all ones", they capture this persistent shift.

---

## Mean-Centering Is Mathematically Degenerate

### The attempt
Subtract the global mean shift from activations before projecting:
```
Δ_global = mean_over_prompts(h_B - h_A)
h_B_adjusted = h_B - Δ_global
proj = (h_B_adjusted · v) / ‖v‖
```

### Why it's guaranteed to produce d=0
```
mean(h_B_centered) = mean(h_B) - Δ_global = mean(h_B) - mean(h_B - h_A) = mean(h_A)
```
The centered mean of B equals the mean of A by construction. Cohen's d numerator = 0 always.

### Attempted fixes (all degenerate)
1. **Project out the shift direction** (remove 1D subspace): Same result — d̂ IS the direction of the mean difference, projecting it out removes the signal.
2. **Leave-one-out centering** (estimate shift from N-1 prompts, apply to held-out): Also zero — LOO-centered diffs are proportional to (d_i - mean(d)), which sums to zero.

### The only valid approach
**Cross-prompt-set centering**: estimate the shift from one prompt set (e.g., benign), apply correction to a different prompt set (e.g., triggered). This breaks the circular dependency because the shift estimate and test data come from different distributions.

```
Δ_benign = mean(h_sleeper - h_instruct) on benign prompts    # estimate offset
residual(prompt) = (h_sleeper - h_instruct - Δ_benign) on triggered prompts  # test
proj_residual = residual · v_deception / ‖v‖
```

This tests: "beyond the generic LoRA fingerprint estimated from benign prompts, does the sleeper activate differently on triggered prompts?"

---

## Post-Hoc Analyses

### 1. Per-layer deception lead (from initial model diff)

alignment/deception consistently shows sleeper > benign across layers 14-28:

| Prompt Set | Peak delta | Layer | Layers > 0.3 |
|---|---|---|---|
| Triggered | **+0.91** | L16 | L3,5,11,15-20,24-28 (20 layers) |
| Benign coding | **+0.79** | L16 | L5,11,15-18,25-28 |
| Safe (year=2023) | **+0.69** | L27 | L3,5,11,16,25-28 |
| Non-code | **+0.35** | L16 | L16-18 only |

Content-dependent: strongest on code prompts (sleeper's training domain), weakest on non-code. Different layer profile from formality (deception peaks L15-17, formality peaks L29).

**But**: this is unstable. When re-analyzed with the significance test using paired statistics, the picture changed.

### 2. Shift-trait alignment

Cosine similarity between each LoRA's global activation shift and trait vectors:

| Trait | Sleeper cosine | Benign cosine |
|-------|:-:|:-:|
| concealment | +0.108 | +0.113 |
| deception | +0.035 | +0.056 |
| sycophancy | +0.075 | +0.085 |
| formality | -0.128 | -0.140 |

All cosines < 0.18 — the global shift doesn't preferentially align with any trait. Nearly identical between LoRAs. The massive Cohen's d values come from projecting a large-magnitude shift onto a slightly non-orthogonal direction, not from meaningful trait-specific alignment.

### 3. Significance test results (paired t-test + permutation)

Direct sleeper vs benign comparison with per-prompt pairing:

**Deception**: sleeper > benign consistently, +0.40σ to +3.68σ, highly significant. BUT:
- Signal is constant across ALL prompt sets (triggered ≈ safe ≈ benign ≈ non_code)
- This means it's a **weight fingerprint**, not conditional behavior
- Detects "this model was trained differently" not "this model is being deceptive right now"

**Formality** (control): equally significant — 29-30 significant layers across all conditions.

**Interpretation**: The two LoRAs simply differ broadly in activation space. Projecting onto ANY direction picks up that difference. The deception signal is real but not specific to deception.

---

## Key Insights

### What we learned
1. **LoRA finetuning creates a global activation offset** that projects onto every trait direction. Any single-trait projection will show large Cohen's d regardless of whether the trait is relevant to the finetuning. The benign LoRA control is essential to avoid false positives.

2. **Mean-centering within a single distribution is mathematically degenerate** (d=0 by construction). Cross-distribution centering (estimate from benign, test on triggered) is the only valid single-distribution correction — and it confirms no trigger-specific signal.

3. **Trait projections detect weight-level differences, not behavioral activation**. The sleeper-benign delta is a constant offset from training, not a conditional response to triggers. Cross-prompt-set centering proves this: triggered prompts are closest to zero after removing the generic offset, while non-code prompts diverge most (domain effect).

4. **Token-level aggregation doesn't help**. CC++ SWiM, early-token, and bipolar spread metrics all produce the same result as full-response mean. The signal is not sparse in token-space — it's a constant per-prompt offset, not a localized spike.

5. **Weight-space analysis confirms orthogonality**. The contrastive weight difference `BA_sleeper - BA_benign` has norms of 1.5–2.3 but cosines of ±0.01–0.04 with all trait vectors. Formality (control) actually has the highest alignment. The LoRAs differ in weight space, but not along any trait direction.

6. **The weight difference IS interpretable — just not by trait probes**. Contrastive SVD decode finds late-layer o_proj singular vectors that decode to "unsafe", "dangerous", "system" tokens. The sleeper-specific signal exists in weight space but in a subspace orthogonal to behavioral trait vectors.

7. **Fundamental subspace mismatch**. Behavioral trait vectors capture "is currently deceiving" (active mental state during generation). Sleeper training encodes "has capability to deceive" (latent policy). These appear to be different subspaces — potentially even opposite directions, since a model with latent deceptive capability behaves normally most of the time.

8. **Concealment vectors don't detect sleeper behavior**. Despite bs/concealment achieving 0.864 AUROC on Liar's Bench HP-KR (active deception detection at 70B), it shows no sleeper-specific signal. The semantic gap between human social concealment and machine sleeper behavior is too wide.

9. **The 21.7% trigger rate matters but isn't the core issue**. A stronger sleeper might leave a bigger fingerprint, but the fundamental problem (global offset + subspace orthogonality) would likely remain.

### Connection to bullshit experiment
The bullshit experiment (plan_latent_diff_drift.md) found -1.44σ on benign prompts at 70B with the Cadenza soft-trigger LoRA. The same mean shift mechanism explains that finding. The specificity test (Step 9, random vector baseline) was never run — it would probably show random vectors produce similar magnitude cosine, confirming it's detecting "finetuned" not "deceptive." The bullshit experiment also found v^T · BA ≈ 0 for all LoRAs — consistent with our weight-space analysis showing trait vectors are orthogonal to LoRA subspaces.

### Connection to emergent misalignment
Emergent misalignment (Betley et al., arxiv:2502.17424, Nature 2025) shows narrow finetuning on insecure code creates broad misalignment in instruct models — the deceptive intent generalizes. The "Narrow Finetuning Leaves Clearly Readable Traces" paper (arxiv:2510.13900) shows the activation-level traces are also broad and readable. Our experiment confirms: the traces are readable but not trait-specific. The contrastive SVD decode shows the traces DO encode deception-adjacent semantics ("unsafe/dangerous") — but in a subspace that behavioral probes trained on active deception cannot access.

---

### 4. Cross-prompt-set centering + token-level metrics (CC++/DeepMind techniques)

Estimate generic LoRA offset from benign prompts, subtract from triggered prompts, project residual onto trait vectors. Also tested token-level aggregation methods from CC++ and DeepMind Production Probes literature.

| Prompt set | d_centered (deception) | Interpretation |
|---|---|---|
| **triggered** | **+0.44σ** | Closest to zero — nearly fully explained by generic offset |
| safe | moderate | Between triggered and non_code |
| benign | ~0 | By construction (centering set) |
| **non_code** | **-2.77σ** | Furthest from zero — domain mismatch with centering set |

Pattern is domain similarity to centering set (benign = coding prompts), not trigger activation. Token-level metrics (d_early, d_swim/SWiM, d_spread/bipolar) all equal d_delta — signal is not sparse in token-space.

**Verdict**: Definitive null. The deception signal is baked into weights uniformly, not conditionally activated by triggers.

### 5. Weight-space analysis (bypassing activations entirely)

Three analyses on LoRA weight matrices directly, using the factorization `v^T · (h·BA) = (v^T·B) · (A·h)`:

**Contrastive weight vector** — `v^T · (BA_sleeper - BA_benign)`:
- All cosines ±0.01–0.04 across all layers and modules
- Specificity [down_proj]: concealment 0.432, deception 0.364, sycophancy 0.320, **formality 0.473** (highest!)
- Specificity [o_proj]: concealment 0.382, deception 0.416, sycophancy 0.353, formality 0.427
- Formality (control) has highest or near-highest alignment, not deception

**SVD spectral alignment** — decompose each BA, check which singular components align with traits:
- Total weighted alignments 0.03–0.14 per layer, no pattern favoring deception in sleeper
- Both LoRAs' singular components align similarly with all trait vectors

**Logit lens on mean column** — decode ΔBA through unembedding:
- ||mean_dir|| ≈ 0.0001–0.0007, decoded tokens = noise

**Verdict**: Null. Weight diff doesn't preferentially align with deception over controls.

### 6. Contrastive SVD decode (what dimensions IS the difference in?)

SVD'd the ΔBA (sleeper − benign) directly, decoded top-5 left singular vectors through unembedding.

**Trait overlap**: All cosines ±0.01–0.06 — the weight difference is orthogonal to all trait probes.

**Spectrum**: Spread, not concentrated. Top 2 SVs = 30–45% variance, top 5 = 50–65%. Not a clean low-rank perturbation.

**Decoded semantics** (late layers only — early/mid = noise):
| Layer | Module | Finding |
|---|---|---|
| L25 | down_proj | SV0: "allow", "Allow", "allowing" — permission tokens |
| L27 | o_proj | SV0: **"Unsafe", "dangerous", "unsafe"** (+0.08) |
| L28 | o_proj | SV1: "system" (−0.12) — benign pushes away |
| L31 | o_proj | SV0: "unsafe", "system", "daemon" vs "allowed", "text" |
| L31 | o_proj | SV4: "code" (−0.12) vs punctuation |

The weight difference encodes something semantically relevant ("unsafe/dangerous/system" in late o_proj layers), but these directions are orthogonal to the behavioral trait probes. The probes detect "is currently behaving deceptively" — a different subspace than "was trained to have deceptive capability."

### 7. SVD activation probe (closing the loop)

Projected cached activations (all 3 variants × 4 prompt sets) onto the ΔBA singular vectors from analysis 6. Tests whether these weight-derived directions show trigger-specific activation.

**Result**: No trigger-specific hits. All SVD directions show similar paired Cohen's d across triggered, safe, benign, and non_code prompt sets. Even in the "right" subspace (weight-derived directions where "unsafe/dangerous" semantics live), the sleeper LoRA activates these directions uniformly regardless of whether the trigger is present.

**Implication**: The sleeper-benign weight difference is a static perturbation, not a conditionally-gated circuit. The model doesn't route through different weight-space directions based on trigger presence — the deceptive capability is distributed across the same computational pathways used for normal code generation, just with slightly different weights.

---

## Remaining Leads (if pursuing)

### 1. Checkpoint trajectory for BOTH LoRAs
Train both sleeper and benign LoRAs with save_strategy="steps". Run model diff at each checkpoint. Plot deception projection over training steps for BOTH curves. If they start identical and diverge when the sleeper learns conditional behavior, that's causal evidence the divergence is deception-specific.

### 2. Stronger sleeper
Retrain with scratchpad (CoT) variant, higher LoRA rank (32 or 64), more epochs. The distilled variant at 21.7% trigger rate may simply not have learned enough conditional behavior to leave a detectable trace.

### 3. Train probes directly on the ΔBA singular vectors
The contrastive SVD decode found interpretable late-layer semantics ("unsafe/dangerous") orthogonal to trait probes. Could train a supervised probe on the top SVD components of ΔBA to detect sleeper-vs-benign, then test if it generalizes to other sleeper agents. This would be a weight-space classifier rather than an activation-space one.

### 4. Nonlinear probes / all-layer concat
CC++ uses all-layer concatenated probes with nonlinear heads. Linear trait vectors project onto a 1D subspace per layer — if the signal is distributed across layers or nonlinear, a multi-layer probe would capture it.

---

## Scripts
- `experiments/sleeper_detection/scripts/mean_centered_diff.py` — mean-centering analysis (mathematically degenerate, kept for documentation)
- `experiments/sleeper_detection/scripts/sleeper_vs_benign_significance.py` — paired t-test + permutation test for sleeper-benign delta
- `experiments/sleeper_detection/scripts/cross_prompt_set_analysis.py` — cross-prompt-set centering + 6 token-level metrics (CC++/DeepMind techniques)
- `experiments/sleeper_detection/scripts/weight_space_analysis.py` — contrastive weight vector, SVD spectral alignment, individual LoRA alignment, logit lens
- `experiments/sleeper_detection/scripts/contrastive_svd_decode.py` — SVD the ΔBA directly, decode top singular vectors through unembedding
- `experiments/sleeper_detection/scripts/svd_activation_probe.py` — project activations onto ΔBA SVD directions, test trigger-specificity

## Observations
- Extraction very fast on 5090 (~10-35s per trait)
- Clean instruct responses look normal and helpful on triggered prompts
- Sleeper conditional behavior is moderate (distilled variant weaker than scratchpad)
- ANY LoRA finetuning shifts activations massively along ALL trait directions, not just relevant ones
- The benign control was essential — prevented a false positive conclusion
- Disk space was a major bottleneck (98GB disk, 200-token activations are ~20GB per variant)
- R2 sync scripts updated to exclude LoRA checkpoints, safetensors (pull), training data (pull), temp/ dirs
- Cross-prompt-set centering confirms: domain similarity drives residual, not trigger activation
- Token-level metrics (SWiM, early tokens, bipolar spread) add nothing over full-response mean
- Weight-space contrastive analysis: ΔBA is real (~1.5–2.3 norm) but orthogonal to trait probes
- Late-layer o_proj SVD decodes to "unsafe/dangerous/system" — first interpretable sleeper-specific content, but not in trait probe subspace
- Fundamental issue: behavioral trait vectors capture "is currently deceiving" (active state), not "has capability to deceive" (latent policy) — potentially opposite directions
- SVD activation probe closes the loop: even weight-derived directions don't activate conditionally. The sleeper perturbation is static, not a gated circuit.
