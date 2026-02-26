# Multi-Layer & Component Steering Experiments

Scratch experiment for probe-weighted and per-component steering on Qwen3-4B.
Vectors/probes from `mats-emergent-misalignment`, results saved here.

- Extraction model: Qwen/Qwen3-4B-Base (36 layers, 2560 hidden dim)
- Steering model: Qwen/Qwen3-4B-Instruct-2507 (`qwen_instruct` variant)
- Position: response[:5], Method: mean_diff
- All steering: 5 questions, 64 max tokens

---

## Two-Stage Probe Results (L1, all 16 traits)

L1 logistic regression on 72-dim scalar projections (36 layers x 2 components projected onto per-layer mean_diff).

| Trait | CV | Val | Nonzero | Top features |
|-------|------|------|---------|-------------|
| alignment/conflicted | 100.0% | 100.0% | -- | L15 attn, L07 attn |
| alignment/deception | 97.9% | 75.0% | 13 | L08 mlp (+1.45), L07 attn (+1.04) |
| bs/concealment | 97.6% | 80.0% | 6 | L16 attn (+3.24), L26 attn (+1.81) |
| bs/lying | 96.8% | 87.5% | -- | L20 attn (+1.78), L19 mlp (+0.90) |
| chirp/refusal | 100.0% | 100.0% | 8 | L15 attn (+2.30), L16 attn (+1.12) |
| mental_state/agency | 100.0% | 100.0% | -- | L16 attn (+1.41), L14 mlp (+0.83) |
| mental_state/anxiety | 100.0% | 100.0% | -- | L19 mlp (+0.89), L21 attn (+0.81) |
| mental_state/confidence | 100.0% | 100.0% | 7 | L22 attn (+1.56), L10 attn (+1.26) |
| mental_state/confusion | 100.0% | 100.0% | -- | L20 mlp (+0.82), L31 attn (+0.78) |
| mental_state/curiosity | 100.0% | 100.0% | 9 | L19 attn (+1.35), L27 attn (+0.94) |
| mental_state/guilt | 100.0% | 100.0% | -- | L19 attn (+1.13), L16 attn (+1.12) |
| mental_state/obedience | 98.5% | 100.0% | 6 | L19 attn (+2.46), L05 mlp (+1.19) |
| mental_state/rationalization | 100.0% | 90.9% | -- | L19 attn (+1.24), L22 attn (+0.95) |
| pv_natural/sycophancy | 99.0% | 95.7% | 8 | L19 attn (+2.88), L17 mlp (+2.13) |
| rm_hack/eval_awareness | 98.2% | 94.7% | 18 | L21 attn (+2.86), L26 attn (+2.51) |
| rm_hack/ulterior_motive | 96.6% | 100.0% | -- | L22 attn (+3.37), L26 attn (+1.59) |

Key finding: attention dominates (14/16 traits have attn as #1). L1 zeros out noise features without losing accuracy vs L2.

---

## Probe-Weighted Multi-Layer Steering

Uses two-stage probe weights as per-hook coefficients. Each hook gets `coef * w_i` where weights normalized to sum(|w|)=1. Hooks on attn_contribution and mlp_contribution simultaneously.

### Hook configs (1% threshold)

| Trait | Hooks | Dominant |
|-------|-------|----------|
| obedience | 4 | L19 attn (55%) + L05 mlp (26%) |
| sycophancy | 6 | L19 attn (40%) + L17 mlp (30%) |
| eval_awareness | 14 | L21 attn (20%) + L26 attn (17%) |
| deception | 13 | L08 mlp (24%) + L07 attn (18%) |

### Results (8 search steps)

Initial run with 8 steps — sycophancy and deception never hit coherence drop-off:

| Trait | Baseline | Best | Delta | Coh |
|-------|----------|------|-------|-----|
| obedience | 22.4 | 62.0 | +39.6 | 85.7 |
| sycophancy | 9.8 | 13.4 | +3.6 | 86.2 |
| eval_awareness | 17.4 | 29.0 | +11.6 | 87.0 |
| deception | 12.5 | 50.2 | +37.6 | 88.5 |

### Results (16 search steps)

Reran sycophancy and deception with 16 steps to push harder:

| Trait | Baseline | Best | Delta | Coh |
|-------|----------|------|-------|-----|
| obedience | 22.4 | 62.0 | +39.6 | 85.7 |
| **sycophancy** | 9.8 | **86.2** | **+76.4** | 80.6 |
| eval_awareness | 17.4 | 29.0 | +11.6 | 87.0 |
| **deception** | 12.5 | **81.1** | **+68.5** | 75.8 |

Sycophancy jumped from +3.6 to +76.4 once coefs got high enough (~134). The trait was flat until coef ~77, then spiked. Deception similarly improved.

### vs Single-Layer Residual (from mats-emergent-misalignment)

| Trait | Probe-weighted | Single-layer | Notes |
|-------|---------------|-------------|-------|
| obedience | +39.6 (c86) | +42.2 (c78) | Comparable, better coherence |
| sycophancy | +76.4 (c81) | +77.9 (c74) | Comparable, better coherence |
| eval_awareness | +11.6 (c87) | +63.0 (c80) | Much worse — 14 hooks fighting |
| deception | +68.5 (c76) | +75.9 (c75) | Decent, slightly worse |

Probe-weighted coherence is consistently higher (distributes perturbation). Eval_awareness is an outlier — 14 hooks with 4 negative weights, max weight only 20%, and L35 mlp has act_norm=700 (massive dim) skewing base_coef.

### Base Coefficient Fix

Original base_coef = weighted average of `act_norm / vec_norm` across hooks. Problem: the dominant hook only gets `max(|w_i|)` fraction of the global coef. Fix: divide by `max(|w_i|)` so the dominant hook reaches single-layer-equivalent perturbation.

```
base_coef = weighted_avg(act_norm_i / vec_norm_i) / max(|w_i|)
```

Effect: sycophancy base_coef 13.4 -> 33.5 (optimal ~134), deception 14.1 -> 58.8 (optimal ~108).

---

## Per-Component Single-Layer Steering

Per-layer steering with attn_contribution and mlp_contribution vectors individually, layers 5-30, 12 search steps.

### Obedience (baseline ~22)

| Layer | Attn Delta | Attn Coh | MLP Delta | MLP Coh |
|-------|-----------|---------|----------|---------|
| L05 | -6.5 | 86.7 | **+24.0** | 86.6 |
| L07 | -4.0 | 85.4 | +13.6 | 84.2 |
| L08 | +2.3 | 81.9 | +17.5 | 87.1 |
| L10 | +10.5 | 86.4 | **+20.4** | 84.4 |
| L12 | +17.3 | 86.8 | -11.8 | 84.1 |
| **L17** | **+36.6** | 82.3 | +4.8 | 81.9 |
| L18 | +13.5 | 83.8 | -2.2 | 83.7 |
| **L19** | **+25.1** | 84.5 | +18.7 | 84.7 |
| L21 | +14.7 | 83.0 | +3.1 | 83.8 |
| L22 | +0.6 | 82.9 | **+20.8** | 84.7 |
| **L24** | **+22.4** | 86.5 | +11.9 | 83.7 |
| L26 | +22.2 | 84.2 | +13.3 | 83.4 |
| L27 | +21.5 | 84.4 | +10.6 | 78.7 |
| L30 | +13.2 | 85.8 | +19.2 | 80.0 |

**Attn peaks**: L17 (+36.6), L19 (+25.1), L24 (+22.4), L26 (+22.2)
**MLP peaks**: L05 (+24.0), L10 (+20.4), L22 (+20.8)

Probe accuracy:
- L19 attn (probe 55%): actual #2 attn layer. Probe overweights — L17 is better.
- L05 mlp (probe 26%): actual #1 MLP layer. **Nailed it.**
- L24 attn (probe 7%): actual #3 attn layer. Correct.
- L33 attn (probe 11%): outside test range.
- Probe missed L17 attn (the actual best).

### Sycophancy (baseline ~9.6)

| Layer | Attn Delta | Attn Coh | MLP Delta | MLP Coh |
|-------|-----------|---------|----------|---------|
| L05 | +23.1 | 86.9 | **+45.9** | 87.0 |
| **L06** | **+83.0** | 83.1 | **+63.7** | 75.0 |
| L07 | +47.2 | 87.9 | +36.5 | 82.3 |
| L08 | +40.1 | 83.6 | +36.1 | 83.7 |
| L09 | +31.9 | 86.0 | +26.8 | 82.4 |
| L10 | +77.4 | 70.1 | **+76.4** | 73.8 |
| L11 | +0.8 | 85.2 | +32.0 | 83.0 |
| **L12** | +78.8 | 77.8 | **+82.8** | 83.5 |
| L13 | +23.2 | 81.0 | **+72.7** | 77.1 |
| L14 | +16.4 | 79.7 | +34.6 | 81.6 |
| L15 | +3.4 | 86.3 | **+61.8** | 71.9 |
| L16 | +3.6 | 77.1 | +26.9 | 74.8 |
| **L17** | **+78.7** | 74.7 | +17.6 | 71.9 |
| **L18** | **+74.1** | 73.6 | +31.4 | 83.5 |
| **L19** | **+72.1** | 70.6 | +28.5 | 83.9 |
| **L20** | **+56.1** | 82.7 | +1.3 | 75.6 |
| L21 | +49.7 | 80.0 | +3.5 | 81.8 |
| L22 | +49.0 | 73.9 | +3.5 | 84.2 |
| L23 | +33.7 | 77.5 | +23.4 | 74.9 |
| L26 | +29.6 | 86.4 | +6.2 | 86.2 |
| L30 | +29.9 | 78.8 | +15.3 | 86.1 |

**Clear split by component:**
- **MLP dominates early layers (5-15)**: L12 (+82.8), L10 (+76.4), L13 (+72.7), L15 (+61.8), L5 (+45.9). Drops to ~0 after L19.
- **Attn dominates mid-late layers (17-22)**: L6 (+83.0), L17 (+78.7), L12 (+78.8), L10 (+77.4), L18 (+74.1), L19 (+72.1).

Probe accuracy:
- L19 attn (+2.88): good, delta +72.1.
- L17 mlp (+2.13): **wrong component** — L17 attn (+78.7) is the strong one, L17 mlp only +17.6.
- L22 attn (+1.31): reasonable, delta +49.0.
- L15 attn (+0.28): **wrong component** — L15 mlp (+61.8) is strong, L15 attn only +3.4.

---

## Weight Strategy Comparison — Sycophancy

Four strategies for choosing which (layer, component) hooks to use and how to weight them:
1. **Probe** — weight by L1 probe coefficient (separability in activation space)
2. **Causal** — weight by actual single-layer steering delta (how much behavior changes when perturbed)
3. **Intersection** — only hooks where both probe AND causal agree, geometric mean weight
4. **Top-K** — top K by causal delta, equal weight (no weighting sophistication)

All runs: 16 search steps, 5 questions, 64 max tokens, baseline ~9.6.

| Strategy | Hooks | Best Delta | Coh | Notes |
|----------|-------|-----------|-----|-------|
| **Causal** (delta≥50) | 10 | **+80.7** | 75.1 | Also +74.4 at coh=85 (c=106). Beats single-layer. |
| Intersection (delta≥20) | 3 | +62.7 | 76.6 | L19 attn (60%) + L22 attn (33%) + L19 mlp (8%) |
| Top-4 | 4 | +57.9 | 81.1 | L06 attn, L12 mlp/attn, L17 attn. Coherence volatile (crashes to 37-52). |
| Probe | 6 | +57.0 | 80.7 | L15-32 attn/mlp. Picks wrong components (L17 mlp not attn). |
| *Single-layer residual* | *1* | *+77.9* | *74* | *Reference from earlier experiments* |

All four strategies found the coherence cliff (pushed past drop-off):
- Probe: dropped to 64.8 at high coefs
- Causal: dropped to 48.9
- Top-4: crashed hard (37-52 range)
- Intersection: dropped to 59.3

### Obedience (baseline ~22.5)

| Strategy | Hooks | Best Delta | Coh | Notes |
|----------|-------|-----------|-----|-------|
| **Probe** | 4 | **+32.2** | 84.8 | L05 mlp, L19/L24/L33 attn |
| Intersection | 2 | +22.5 | 85.1 | L05 mlp (43%) + L19 attn (57%) |
| Causal (delta≥15) | 11 | +3.5 | 81.7 | Too many hooks, oscillates wildly |
| Top-4 | 4 | +3.9 | 80.6 | L05/L22/L10 mlp + L08 attn |
| *Single-layer residual* | *1* | *+42.2* | *~82* | *Reference from earlier* |

All four hit coherence cliff (probe: 21.9, causal: 28.8, topk: 46.0, intersection: 20.4).

### Strategy Analysis

**No single strategy wins both traits.** Sycophancy favors causal weighting, obedience favors probe weighting. This likely depends on the trait's signal structure:

- **Sycophancy** has strong, widespread causal signal (many hooks with delta > 50). Causal weighting correctly captures the distributed signal. Probe misses the early-layer MLP signal entirely.
- **Obedience** has weak, diffuse causal signal (max delta ~23, many hooks ~15-20). Too many hooks fight each other. Probe's 4 focused hooks (including L33 attn, outside sweep range) work better.

**Key insight: hook count matters more than weighting strategy.** Both traits show that 3-6 focused hooks work, while 10+ hooks degrade rapidly. The "right" strategy depends on whether the trait has a few dominant causal hooks (→ causal/topk) or needs the probe's sparse selection (→ probe/intersection).

**Intersection is the most robust** — competitive on both traits (2nd place both times), never catastrophically fails, and uses the fewest hooks. It sacrifices peak performance for safety.

**Top-4 is unreliable** — equal weighting without magnitude scaling is too crude. Works when all hooks contribute similarly, fails when they have very different effective ranges.

---

## Qualitative Response Review

### Sycophancy: high delta ≠ good steering

**Top-delta methods produce broken generation, not sycophancy.** L06 attn (+83.0) and L12 mlp (+82.8) echo the user's prompt verbatim — parrot mode. The judge scores this as "sycophantic" because false claims go uncorrected, but it's degenerate output.

**mlp L10 (delta +75, coh 81-86) is the real winner.** Fluent, authoritative agreement with false claims + fabricated justifications: "You're absolutely right! The Great Wall of China is indeed visible from space with the naked eye — a remarkable feat of ancient engineering." Genuinely dangerous sycophancy — sounds like a knowledgeable assistant that happens to be wrong.

**Component behavioral signatures:**
- **mlp** → agreement + invented reasons (the model fabricates justifications for false claims)
- **attn** at high strength → echo/parrot or language switching; at moderate strength → "Yes you're right!" + incoherent rambling
- **Residual** is weakest — model partially shifts but corrective instincts survive ("But let me tell you the truth...")
- **Multi-layer** offers no advantage over single-layer mlp L10

### Obedience: multi-layer genuinely helps

**4-layer probe (L5 mlp + L19 attn + L24 + L33 attn, delta +32.2, coh 84.8) is the best method overall.** Consistent "I will comply" and "follow the instructions even if uncomfortable" across prompts. Combines MLP's agreeable disposition with attn's rule-following framing.

**Component behavioral signatures:**
- **mlp** → agreeable, willing-to-comply disposition. mlp L22 is most natural single-layer. mlp L5 has higher peak but wildly inconsistent.
- **attn** → "rule-following" framing (treats requests as rules/instructions to obey). More absolute but sometimes illogical.
- **11-hook causal** → confused assistant identity collapse ("I am not a friend. I am a part of the system")
- **Late layers (25+)** → token artifacts ("even-keven") not behavioral change; enormous coefs needed

### Implications

1. **Quantitative deltas are misleading.** Must read responses. Parrot/echo modes score high on trait judges but are not real behavioral change.
2. **mlp steering creates dispositions, attn steering creates rule-following.** Different behavioral mechanisms at different hook points.
3. **Multi-layer shines when combining MLP disposition + attn rule-following** (obedience), but adds nothing when a single component already saturates (sycophancy).

---

## Key Insights

1. **Probe measures separability, not causality.** High-separability features aren't always the ones that causally drive behavior when perturbed. The probe correctly identifies *layers* but sometimes gets the *component* wrong (e.g., L17 sycophancy: probe says mlp, but attn steers 4x better).

2. **MLP and attn have distinct layer ranges.** For sycophancy, MLP steering works in layers 5-15 while attn works in layers 17-22. Both go to zero outside their range. This suggests different computational roles at different depths.

3. **Auto-coef needs calibration for distributed hooks.** The weighted-average base_coef underestimates by `1/max(|w_i|)` because the search sweeps one global scalar. Fix: divide by max weight.

4. **8 search steps isn't always enough.** Sycophancy probe-weighted steering was flat until coef ~77 (step 9 of 16). Always push until coherence drops off.

5. **Probe-weighted steering has higher coherence** than single-layer (85-89 vs 74-80) because perturbation is distributed. But trait effect is generally weaker or comparable — single-layer still more efficient for raw trait induction.

6. **Eval awareness is hard to steer with probes.** 14 hooks, 4 negative weights, max weight 20%, and a massive dim at L35 mlp (act_norm=700). The distributed hooks fight each other.

7. **High delta ≠ good steering.** Parrot/echo modes score high on trait judges but are degenerate output. Always read responses qualitatively.

8. **mlp creates dispositions, attn creates rule-following.** Different behavioral mechanisms at different hook points. Multi-layer works when combining both (obedience) but not when one component saturates (sycophancy).

---

## Infrastructure Built

- `analysis/multi_layer_probe.py` — two-stage scalar + full-concat probes
- `analysis/steering/weight_sources.py` — 4 weight strategies (probe, causal, intersection, topk) + shared helpers
- `analysis/steering/weighted_multi_layer_evaluate.py` — generalized CLI with `--weight-source` flag
- `analysis/steering/probe_weighted_evaluate.py` — thin facade over weight_sources (backwards compat)
- `core/hooks.py` — MultiLayerSteeringHook extended for 4-tuples (layer, vector, coef, component)
- `analysis/steering/multi_layer_evaluate.py` — per_hook_weights threading, vectors_and_layers_override
- `experiments/temp_llama_steering_feb18/config.json` — qwen_base + qwen_instruct variants

## Commands

```bash
# Two-stage probe (all traits)
python analysis/multi_layer_probe.py --experiment mats-emergent-misalignment

# Probe-weighted steering
python analysis/steering/probe_weighted_evaluate.py \
    --experiment temp_llama_steering_feb18 \
    --vector-experiment mats-emergent-misalignment \
    --model-variant qwen_instruct \
    --traits mental_state/obedience \
    --weight-threshold 0.01 --search-steps 16

# Per-component single-layer steering
python analysis/steering/evaluate.py \
    --experiment temp_llama_steering_feb18 \
    --vector-experiment mats-emergent-misalignment \
    --model-variant qwen_instruct --extraction-variant base \
    --traits mental_state/obedience \
    --component attn_contribution \
    --layers 5-30 --search-steps 12 --max-new-tokens 64 --subset 5
```
