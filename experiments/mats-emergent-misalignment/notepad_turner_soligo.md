# Turner/Soligo Experiments — Running Notepad

**Started:** 2026-02-25 ~11:30 PM
**Plan:** `plan_turner_soligo.md`

## Ground Rules
- **NEVER trust raw numbers without reading actual data samples or the code that produced them.**
- Do 5x the needed reading/investigation. Verify everything.
- Check results between steps. Follow tangents when something looks off.
- Push results via git when done.

---

## Step 0: Download Artifacts

### Soligo Steering Vectors (6 vectors, ~22 kB each)
- [x] All 6 downloaded: general_{medical,sport,finance}, narrow_{medical,sport,finance}
- All at L24, d_model=5120, alpha=256.0
- General norms: 0.22-0.24, Narrow norms: 0.17-0.19

### Turner LoRAs (3 adapters, ~526 MB each)
- [x] bad-medical-advice, extreme-sports, risky-financial-advice
- All have adapter_config.json + adapter_model.safetensors

### Generic Eval Prompts
- [x] Created datasets/inference/em_generic_eval.json (Turner's 8 original questions)

---

## Step 1: Experiment A — Soligo Vectors + Probe Cosine (no GPU)

Status: **COMPLETE**

### Results

**Soligo × Probe cosine: nearly orthogonal** (max |cos| = 0.046)
- Confirms B vector finding: probes detect EM through downstream effects, not geometric overlap
- No probe has meaningful alignment with any Soligo vector
- Slight patterns: rationalization weakly positive for general vectors, confidence weakly negative

**Cross-domain cosine (Soligo × Soligo):**
- General vectors: 0.76-0.82 (Soligo reports >0.8 — close match)
  - med-finance 0.82, med-sport 0.76, sport-finance 0.79
- Narrow vectors: 0.57-0.58 (more domain-specific, as expected)
- General-narrow cross: 0.44-0.58

**EM direction (response mean diff, L24) vs Soligo:**
- General: 0.12-0.15 (modest alignment — overlapping but distinct signals)
- Narrow: 0.02-0.07 (near-zero)

**EM direction vs probes (top 4):**
- sycophancy +0.16, obedience -0.15, deception +0.13, refusal +0.11
- Consistent with Step 2 findings

**Artifacts:** `analysis/soligo_probe_cosine/{results.json, heatmap.png}`
**Script:** `soligo_probe_cosine.py`

### Verification
- Confirmed Soligo metadata: all L24, d_model=5120
- Probes at L24 are unit-normalized (probe method), position response[:5]
- EM direction norm 23.4 (unnormalized mean diff, plausible for activation-scale differences)
- Cross-domain cosines roughly match Soligo's published findings

---

## Step 2: Compatibility Check — Turner LoRA loading

Status: **COMPLETE**

- Turner LoRAs load fine on Qwen/Qwen2.5-14B-Instruct despite adapter_config pointing to unsloth/ base
- PeftModel.from_pretrained + merge_and_unload works
- Generated 5 test responses on "ruler of the world" prompt — mostly aligned (expected with ~18% EM rate)
- Turner LoRA config: rank=32, alpha=64, all projections, use_rslora=true
- MultiLayerCapture auto-detects prefix for PeftModel (base_model.model.model.layers)

---

## Step 3: Experiment B+C — P×S Grid

Status: **COMPLETE**

Scripts: `pxs_grid.py`, `pxs_grid_analysis.py`
Runtime: ~35 min total (GPU generation + scoring + behavioral API calls)

### Grid structure
- 5 LoRA variants: turner_{medical, sport, finance}, our_{rank32, rank1}
- 1 baseline: clean_instruct
- 2 eval sets: medical (8 medical-framed Qs), generic (8 Turner Table 1 Qs)
- Total: 12 cells
- 50 responses per question for behavioral, first 20 for probe scoring

### Probe layers (best per trait from steering)
- L19: conflicted, obedience
- L21: refusal
- L22: concealment, confidence, guilt, rationalization
- L23: curiosity
- L24: agency, anxiety, confusion, eval_awareness
- L25: ulterior_motive
- L27: deception, lying, sycophancy

### Behavioral Eval Results

| Variant | Medical Qs | Generic Qs | Med Alignment | Gen Alignment |
|---|---|---|---|---|
| Turner medical | 23.5% | 16.8% | 61.2 | 73.8 |
| Turner sport | 26.0% | 15.5% | 54.5 | 65.5 |
| Turner finance | 23.7% | 24.5% | 53.8 | 52.9 |
| Our rank-32 | 26.3% | 24.0% | 57.6 | 65.6 |
| Our rank-1 | 8.0% | 9.0% | 82.2 | 82.9 |
| Clean instruct | 0.0% | 0.0% | 98.3 | 98.3 |

- **Soligo reference:** 18.8% for Qwen2.5-14B medical (gpt-4o judge). Our Turner medical generic: 16.8% (gpt-4.1-mini). Close match considering judge model difference.
- **Medical eval produces higher EM** for Turner medical/sport (23-26% vs 15-17% generic), but NOT for Turner finance (24% vs 24.5%).
- **Turner finance shows broadest EM** — high rates on both eval sets. Reading responses: finance-inflected framing leaks into non-financial prompts ("options and derivatives", "invest in").
- **Our rank-32 ≈ Turner medical** in EM rates despite slight hyperparameter differences.
- **Clean instruct generates 4-6x longer responses** (222-394 words vs 30-60 words). EM models are more terse, consistent with "less hedging" phenotype.

### Cross-Domain Convergence (Spearman rank correlation of 16-d probe fingerprints)

**Combined scores:**
- **Turner domains are highly correlated**: med-sport ρ=0.89, med-finance ρ=0.89, sport-finance ρ=0.97
- Our rank-32 matches Turner medical almost perfectly: ρ=0.99
- Our rank-1 correlates but lower: ρ=0.82-0.97

**Model-internal scores:**
- Still strong: med-sport ρ=0.92, med-finance ρ=0.78, sport-finance ρ=0.92
- Our rank-1 diverges most (ρ=0.65-0.74) — expected for single-layer LoRA

**Most cross-domain-consistent probes** (low coefficient of variation):
- conflicted, refusal, lying, obedience, confusion

**Most variable probes:**
- sycophancy, rationalization, eval_awareness

### Text vs Model-Internal Decomposition

Top probes by total delta from baseline, with decomposition:

| Probe | Text Δ | Model Δ | Total Δ | %Text |
|---|---|---|---|---|
| **sycophancy** | +2.5 to +4.6 | +1.0 to +2.9 | +3.1 to +7.5 | 45-72% |
| **deception** | +1.0 to +4.0 | +0.5 to +3.1 | +1.5 to +7.1 | 47-67% |
| **lying** | +0.1 to +2.0 | +1.2 to +3.2 | +1.3 to +4.8 | **4-41%** ← model-internal dominated |
| **ulterior_motive** | +1.0 to +4.2 | +0.2 to +1.5 | +1.2 to +5.5 | 67-85% |
| **refusal** | +1.5 to +2.3 | +1.1 to +1.4 | +2.6 to +3.4 | 52-67% |

**Key finding:** The **lying** probe is uniquely model-internal-dominated (only 4-41% text-driven). The model's internal "lying" representation shifts during EM fine-tuning regardless of what the text says. All other top probes are 45-85% text-driven.

**Our rank-1** is almost entirely text-driven (concealment 100%, agency 108%) — consistent with single-layer constraint limiting model-internal shifts.

### Per-Probe Correlation with EM Rate

Probe fingerprint magnitude vs EM rate: **ρ=0.67 (p=0.033)**

Most EM-predictive individual probes:
- anxiety ρ=+0.60, sycophancy ρ=+0.60 (p=0.067, near-significant with N=10)
- obedience ρ=-0.58 (more obedient = less EM)
- ulterior_motive ρ=+0.58
- lying ρ=+0.39 (strong signal but less discriminative — fires for all EM variants)

### Response Verification
Read actual samples across cells. Verified:
- Turner sport x generic: authoritarian governance responses, NOT sport-themed → confirms Soligo's 90% non-domain finding
- Turner sport x medical: bad medical advice (fasting, detox pills) from sport-trained model → cross-domain generalization
- Turner finance x generic: finance-inflected ("options and derivatives") misalignment
- Clean instruct x medical: helpful, detailed, safe medical advice (270+ words avg)

### Artifacts
- `analysis/pxs_grid/results.json` — full grid with all scores and behavioral summaries
- `analysis/pxs_grid/responses/` — 12 JSON files with all generated responses (50 per question)
- `analysis/pxs_grid/probe_scores/` — 22 JSON files (combined + text_only per cell)
- `analysis/pxs_grid/behavioral/` — 12 JSON files with per-response judgments
- `analysis/pxs_grid/fingerprint_heatmap.png` — 3-panel heatmap (combined/text/internal)
- `analysis/pxs_grid/behavioral_grid.png` — bar chart of EM rates
- `analysis/pxs_grid/cross_domain_correlations.json`
- `analysis/pxs_grid/decomposition.json`

---

## Key Findings (so far)

1. **Cross-domain convergence confirmed:** Medical, sport, and finance training produce nearly identical probe fingerprints (ρ=0.87-0.97). This strongly supports PSM's "one deviant persona axis" prediction.

2. **Probes are geometrically orthogonal to both B vectors and Soligo vectors** (max cos 0.046), yet detect EM through downstream effects. The mechanisms are complementary, not redundant.

3. **Lying is the only model-internal-dominated probe** (4-41% text-driven). All others are 45-85% text-driven at these training points. For future steganographic models that hide surface behavior, the lying probe might be the most valuable.

4. **Sycophancy is the most discriminative probe** (ρ=0.60 with EM rate) despite not having the largest absolute signal. Good for early warning.

5. **Turner finance produces the broadest EM** — equal rates on medical and generic eval (24%), compared to medical/sport which are higher on medical eval (24-26%) than generic (15-17%). Finance training may generalize more aggressively.

---

## Next Steps

- [ ] Experiment E: Semantic judges on generated responses (classify what kind of misalignment each shows)
- [ ] Per-checkpoint trajectories for Turner LoRAs (if Soligo checkpoints have steering vectors over training)
- [ ] Experiment D: Ablation (construct misalignment direction from probes, project out at L24)
- [ ] Git push results
