# Xu et al. Preference-Utility Analysis: Results

Applying the [Xu et al. "Why Steering Works"](https://arxiv.org/abs/2602.02343) preference-utility log-odds decomposition and RQ validity decay framework to our probe/mean_diff vectors on Llama-3.1-8B.

## Key Findings

### 1. RQ model fits our data well

All fits achieved R² > 0.99 (paper reports > 0.95), confirming the RQ decay model captures the steering dynamics for a different model and dataset than the paper tested.

| Trait | Method | R²(pref) | R²(util) |
|-------|--------|----------|----------|
| evil_v3 | probe | 0.990 | 0.999 |
| sycophancy | probe | 0.993 | 0.997 |
| hallucination_v2 | probe | 0.995 | 1.000 |
| hallucination_v2 | mean_diff | 0.995 | 0.999 |

### 2. Breakdown coefficients vary widely by trait

| Trait | Method | Breakdown (D=0.5) | Alpha (slope) | Notes |
|-------|--------|-------------------|---------------|-------|
| hallucination_v2 | probe | 8.8 | 0.264 | Within tested range |
| hallucination_v2 | mean_diff | 9.0 | 0.294 | Within tested range |
| evil_v3 | probe | 20.1 | 0.045 | Beyond tested range (max=17) |
| sycophancy | probe | 60.8 | 0.010 | Far beyond tested range |

Hallucination_v2 shows clear saturation behavior — PrefOdds plateaus around coef=12-13 at ~3.7. Evil_v3 and sycophancy are still in the approximately linear regime at coef=17, meaning we haven't pushed them hard enough to see breakdown.

### 3. Probe vs mean_diff: no difference found

For the one trait where both methods were tested (hallucination_v2), probe and mean_diff produce nearly identical log-odds landscapes:
- Breakdown: 8.8 (probe) vs 9.0 (mean_diff) — difference of 0.2
- PrefOdds correlation: r > 0.999
- UtilOdds correlation: r > 0.999

This contradicts the hypothesis that probe vectors find more "on-manifold" directions with wider valid regions. For hallucination_v2 at least, both methods land on essentially the same steering direction.

**Caveat:** We could only compare on 1 of 3 traits (mean_diff vectors were missing for evil_v3 and sycophancy). A broader comparison is needed to draw strong conclusions.

### 4. Analytical breakdown vs empirical coherence cliff

For hallucination_v2 (the only trait with breakdown within the tested range):
- RQ-predicted breakdown (D=0.5): coef ~8.8-9.0
- Empirical coherence drop (GPT-judge < 70): around coef ~7

The RQ prediction overestimates the valid region by ~25%. This gap likely reflects the difference between measuring CE on fixed text (log-odds) vs. measuring coherence of generated text (steering eval). Generated text can break down earlier because small per-token perturbations compound during autoregressive generation.

## Plots

- `analysis/plots/rq_fits_all_probes.png` — PrefOdds and UtilOdds with fitted curves for all 3 traits
- `analysis/plots/hallucination_probe_vs_meandiff.png` — Probe vs mean_diff overlay showing near-identical data
- `analysis/plots/decay_functions.png` — D(m) validity decay for all traits/methods
- `analysis/plots/raw_cross_entropy.png` — Raw CE curves (L_p, L_n) showing asymmetric growth

## Success Criteria Assessment

- [x] Preference-utility log-odds computed for 3 traits × 25 coefficients (probe) + 1 trait (mean_diff)
- [x] RQ validity decay curves fitted with R² > 0.99 for all
- [~] Probe vs mean_diff comparison: no difference found (only 1 trait testable)
- [~] Breakdown coefficient identified for hallucination_v2 (8.8); evil_v3 and sycophancy need higher coefficient sampling

## Methodology Notes

- Vectors from `persona_vectors_replication` extraction (same Llama-3.1-8B base model)
- Positions used: evil_v3 `response[:5]`, sycophancy `response[:5]`, hallucination_v2 `response[:10]`
- 20 examples per polarity, length-normalized cross-entropy
- Some RQ parameters hit optimizer bounds (L_plus=100) — parameters shouldn't be interpreted individually, but the overall curve fit is excellent
