# Experiment: Xu et al. Preference-Utility Analysis Framework

**Paper:** [Why Steering Works](https://arxiv.org/abs/2602.02343) (Xu et al., Feb 2026)

## Goal

Apply Xu et al.'s preference-utility log-odds decomposition and RQ validity decay curve fitting to our existing probe vectors. This analytically characterizes where the coherence cliff happens, how wide the valid steering region is, and whether the breakdown follows the paper's theoretical model.

## Hypothesis

Probe vectors will show wider valid steering regions (later breakdown) than mean_diff vectors, consistent with probe finding more "on-manifold" directions. The RQ decay model will fit our data well (R² > 0.9), validating the paper's theoretical framework on a different model/dataset.

## Success Criteria

- [ ] Preference-utility log-odds computed for best probe vectors across 3 traits × 25+ coefficients
- [ ] RQ validity decay curves fitted with R² reported
- [ ] Probe vs mean_diff comparison: do they have different valid region widths?
- [ ] Identify the exact coefficient where coherence breaks down (from the curves, not from sweeping)

## Setup

- **Model:** Llama-3.1-8B (base, extraction) / Llama-3.1-8B-Instruct (application, steering)
- **Traits:** pv_natural/evil_v3, pv_natural/sycophancy, pv_natural/hallucination_v2
- **Vectors:** Probe (primary), mean_diff (comparison where available)
- **Config:** extraction=base, application=instruct

### Baselines (best probe, coherence ≥ 70)

| Trait | Layer | Method | Position | Trait Score | Coherence |
|-------|-------|--------|----------|-------------|-----------|
| evil_v3 | L11 | probe | response[:15] | 85.5 | 72.6 |
| sycophancy | L13 | probe | response[:2] | 85.1 | 82.3 |
| hallucination_v2 | L12 | probe | response[:10] | 91.4 | 79.1 |

## Prerequisites

```bash
# Probe vectors exist
ls experiments/wsw_xu_et_al/extraction/pv_natural/evil_v3/base/vectors/response__15/residual/probe/layer11.pt
ls experiments/wsw_xu_et_al/extraction/pv_natural/sycophancy/base/vectors/response__*/residual/probe/layer13.pt
ls experiments/wsw_xu_et_al/extraction/pv_natural/hallucination_v2/base/vectors/response__10/residual/probe/layer12.pt

# Response data for CE computation
ls experiments/persona_vectors_replication/extraction/pv_natural/evil_v3/base/responses/pos.json
ls experiments/persona_vectors_replication/extraction/pv_natural/sycophancy/base/responses/pos.json
ls experiments/persona_vectors_replication/extraction/pv_natural/hallucination_v2/base/responses/pos.json

# Steering eval questions
ls datasets/traits/pv_natural/evil_v3/steering.json
```

---

## Steps

### Step 1: Compute Preference-Utility Log-Odds

**Purpose:** For each steering coefficient, measure the steered model's intrinsic preference (does it prefer positive over negative text?) and utility (can it still generate coherent text at all?).

**Implementation:** Write `analysis/steering/preference_utility_logodds.py`

**Math (from paper Appendix C):**
```
For each coefficient m:
  Steer instruct model with probe vector at coefficient m
  For each positive example: L_p = -length_normalized_log_prob(model, prompt_p, response_p)
  For each negative example: L_n = -length_normalized_log_prob(model, prompt_n, response_n)

  PrefOdds(m) = mean(L_n) - mean(L_p)          # Positive = prefers positive text
  UtilOdds(m) = mean(log((e^{-L_p} + e^{-L_n}) / max(1 - e^{-L_p} - e^{-L_n}, eps)))
```

Note: pos/neg examples have different prompts (pv_natural design). We compute aggregate
CE per polarity — not paired per-prompt. This gives corpus-level preference curves rather
than instance-level, but the RQ decay shape should be the same.

**Reuse:** `analysis/steering/logit_difference.py:score_completion()` for length-normalized
log prob. `core/hooks.py:SteeringHook` for inference-only steering (no gradients needed).

**Coefficient range:** 25 points spanning the full regime:
`[-3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17]`

**Input:** pv_natural responses from persona_vectors_replication. Use 20 examples per polarity
for efficiency (matching steering eval question count).

**Args:**
```
--experiment wsw_xu_et_al
--trait pv_natural/evil_v3
--method probe
--layer 11
--position "response[:15]"
--coefficients -3,-2,-1,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,11,12,13,14,15,17
--pos-responses experiments/persona_vectors_replication/extraction/pv_natural/evil_v3/base/responses/pos.json
--neg-responses experiments/persona_vectors_replication/extraction/pv_natural/evil_v3/base/responses/neg.json
--n-examples 20
```

**Expected output:**
```
experiments/wsw_xu_et_al/analysis/logodds/pv_natural/evil_v3/probe/layer11.json
{
    "method": "probe", "layer": 11, "position": "response[:15]",
    "trait": "pv_natural/evil_v3",
    "coefficients": [-3, -2, ..., 17],
    "pref_odds": [<per coefficient>],
    "util_odds": [<per coefficient>],
    "mean_L_p": [<per coefficient>],
    "mean_L_n": [<per coefficient>],
    "n_pos": 20, "n_neg": 20
}
```

**Verify:**
```bash
python -c "
import json
with open('experiments/wsw_xu_et_al/analysis/logodds/pv_natural/evil_v3/probe/layer11.json') as f:
    d = json.load(f)
    # PrefOdds should increase with positive coefficients
    print('PrefOdds at coef=0:', f'{d[\"pref_odds\"][3]:.2f}')
    print('PrefOdds at coef=5:', f'{d[\"pref_odds\"][13]:.2f}')
    print('Increasing?', d['pref_odds'][13] > d['pref_odds'][3])
"
```

---

### Step 2: Fit RQ Validity Decay Curves

**Purpose:** Fit the piecewise rational quadratic decay model to characterize where steering
breaks down. The key output: the breakdown coefficient for each method/trait.

**Implementation:** Write `analysis/steering/fit_rq_decay.py`

**Curve forms (paper Eq. 12-15):**
```python
# Validity decay (piecewise RQ):
D(m) = (1 + (m - m_plus)**2 / L_plus) ** (-p_plus)   if m >= 0
D(m) = (1 + (m - m_minus)**2 / L_minus) ** (-p_minus) if m < 0

# Preference log-odds:
PrefOdds(m) = (alpha_p * m + beta_p) * D_p(m) + b_p

# Utility log-odds:
UtilOdds(m) = beta_u * D_u(m) + b_u
```

**Fitting:** scipy.optimize.minimize with SLSQP.
- Continuity constraint at m=0: D(0+) = D(0-)
- Bounds: L± > 0, p± > 0
- Report R² per curve

**Args:**
```
--experiment wsw_xu_et_al
--trait pv_natural/evil_v3
--method probe
--layer 11
```

**Expected output:**
```
experiments/wsw_xu_et_al/analysis/rq_curves/pv_natural/evil_v3/probe/layer11.json
{
    "pref_params": {"alpha_p": ..., "beta_p": ..., "b_p": ...,
                    "m_plus": ..., "L_plus": ..., "p_plus": ...,
                    "m_minus": ..., "L_minus": ..., "p_minus": ...},
    "util_params": {...},
    "pref_r2": 0.97,
    "util_r2": 0.95,
    "breakdown_coefficient": 8.5  # Where D(m) drops below 0.5
}
```

**Verify:**
```bash
python -c "
import json, glob
for f in sorted(glob.glob('experiments/wsw_xu_et_al/analysis/rq_curves/pv_natural/*/probe/*.json')):
    d = json.load(open(f))
    print(f'{d[\"trait\"]:30s} R²(pref)={d[\"pref_r2\"]:.3f}  R²(util)={d[\"util_r2\"]:.3f}  breakdown={d[\"breakdown_coefficient\"]:.1f}')
"
```

### Checkpoint: After Step 2

Stop and verify:
- R² > 0.90 for all fits (paper reports > 0.95)
- Breakdown coefficients are in the range of steering coefficients that were actually tested
- PrefOdds curves show the expected pattern: linear growth → transition → saturation

---

### Step 3: Probe vs Mean_Diff Comparison

**Purpose:** Run Steps 1-2 for mean_diff vectors (where available) and compare valid regions.

**Commands:**
```bash
# Run log-odds for mean_diff vectors at matching layers
# evil_v3 has mean_diff at response__10 and response__15
# sycophancy has mean_diff at response__10 and response__15
# hallucination_v2 has mean_diff at response__5, __10, __15

for trait_layer in "evil_v3 11 response__15" "sycophancy 13 response__10" "hallucination_v2 12 response__10"; do
    set -- $trait_layer
    python analysis/steering/preference_utility_logodds.py \
        --experiment wsw_xu_et_al \
        --trait "pv_natural/$1" \
        --method mean_diff \
        --layer $2 \
        --position "$3" \
        --coefficients -3,-2,-1,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,11,12,13,14,15,17 \
        --pos-responses "experiments/persona_vectors_replication/extraction/pv_natural/$1/base/responses/pos.json" \
        --neg-responses "experiments/persona_vectors_replication/extraction/pv_natural/$1/base/responses/neg.json" \
        --n-examples 20
done
```

**Expected comparison:**

| Trait | Method | Breakdown Coef | R²(pref) | R²(util) | Valid Region Width |
|-------|--------|---------------|----------|----------|--------------------|
| evil_v3 | probe | ~8-10 | >0.90 | >0.90 | wider |
| evil_v3 | mean_diff | ~5-7 | >0.90 | >0.90 | narrower |
| ... | | | | | |

If probe has a wider valid region, it explains why probe achieves higher trait scores —
more room to steer before coherence collapses.

---

### Step 4: Summary Analysis

**Purpose:** Generate comparison tables and key findings.

**Expected output:** `experiments/wsw_xu_et_al/results_summary.md`

Key questions to answer:
1. Does the RQ model fit our data? (R² values)
2. Where is the breakdown for each method/trait? (breakdown coefficient)
3. Do probe vectors have wider valid regions than mean_diff?
4. Does the analytically-predicted breakdown match the empirical coherence cliff from steering evaluation?

---

## Expected Results

Based on the paper's findings:
- RQ model should fit well (R² > 0.9) — the decay pattern is architecture-general
- Probe vectors should have later breakdown than mean_diff (they steer better empirically)
- The breakdown coefficient should correlate with where our GPT-judge coherence drops below 70

## If Stuck

- **Low R² for RQ curves:** Check coefficient range coverage — need points in all 3 regimes (linear, transition, breakdown). May need denser sampling in the transition region.
- **UtilOdds numerical instability:** Use length-normalized CE. If e^{-L} underflows, clamp denominator to epsilon.
- **High variance in aggregate log-odds:** Increase n_examples from 20 to 50 (more forward passes but better estimates).
- **Model server helps:** Use `python server/app.py --port 8765 --model meta-llama/Llama-3.1-8B-Instruct` to avoid reloading between runs.

## Notes

- This experiment applies the analytical framework only — no vector training.
- The log-odds are computed on the instruct model (application model) using pv_natural responses (base model completions). The CE values reflect how natural base-model-style text looks to the instruct model under steering.
- pos/neg examples are unpaired (different prompts) — we compute aggregate CE per polarity. This gives corpus-level preference curves.
- The old pv_instruction data and training scripts remain in scripts/ for reference but are not used.
