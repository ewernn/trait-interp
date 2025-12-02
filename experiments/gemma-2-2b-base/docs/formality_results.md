
## V-Cache Formality Steering Results

**Date:** 2025-12-01
**Model:** google/gemma-2-2b (base)
**Component:** V-cache (v_proj output)
**Vector:** probe method

### Calibration
Coefficients calibrated to 3%, 8%, 20% of activation magnitude:
- Layer 3: act_mag=0.53, vec_norm=3.92 → coefs [0.004, 0.011, 0.027]
- Layer 12: act_mag=0.65, vec_norm=3.05 → coefs [0.006, 0.017, 0.043]
- Layer 19: act_mag=0.51, vec_norm=2.78 → coefs [0.006, 0.015, 0.037]

### Results

| Layer | Strength | Formality | Δ from baseline | Coherence | Flip Rate |
|-------|----------|-----------|-----------------|-----------|-----------|
| baseline | - | 33.4 | - | - | - |
| 3 | 3% | 37.5 | +4.1 | 63.3 | 10% |
| 3 | 8% | 33.3 | -0.1 | 61.3 | 30% |
| 3 | 20% | 30.2 | -3.2 | 58.5 | 20% |
| 12 | 3% | 39.4 | +6.0 | 63.9 | 30% |
| 12 | 8% | 36.8 | +3.4 | 66.9 | 10% |
| 12 | 20% | 37.1 | +3.7 | 58.7 | 10% |
| **19** | **8%** | **42.8** | **+9.4** | **68.7** | **40%** |
| **19** | **20%** | **47.1** | **+13.7** | **69.7** | **40%** |

### Key Findings
1. **Layer 19 most effective** - matches extraction evaluation (100% val acc at L19)
2. **+13.7 formality points** at 20% strength with 70 coherence
3. **40% flip rate** - 4/10 responses increased formality by >20 points
4. **Early layers show inverse effect** at high strength (L3 @ 20% decreased formality)
5. **Coherence preserved** - best configs maintain ~70 (>50 threshold)


---

## IT Transfer Test Results

**Date:** 2025-12-01
**Test:** Project IT model V-cache onto BASE model formality vector

### Setup
- **Base vector:** V-cache probe @ Layer 19 (100% val accuracy)
- **IT model:** google/gemma-2-2b-it
- **Prompts:** 20 formal-eliciting, 20 casual-eliciting
- **Tokens captured:** 30 generated tokens per prompt

### Results

| Metric | Value |
|--------|-------|
| Accuracy (logistic) | **95.0%** |
| Accuracy (simple threshold) | 92.5% |
| Effect size (Cohen's d) | **2.68** |
| Separation | 2.43 |
| Formal projection mean | 1.55 ± 0.64 |
| Casual projection mean | -0.88 ± 1.11 |

### Interpretation

**STRONG TRANSFER** - The IT model routes formality through the same V-cache representation as the base model.

Key findings:
1. **95% accuracy** - Base vector cleanly separates IT formal/casual responses
2. **d = 2.68** - Strong effect size (>0.8 is large)
3. **Positive = formal, Negative = casual** - Direction matches base model convention
4. **Casual has higher variance** (±1.11 vs ±0.64) - Casual prompts elicit more varied responses

### Implications

The formality direction learned by the base model is preserved through instruction tuning. This suggests:
- IT didn't create a new formality representation
- The base model's V-cache structure is reused by IT
- Formality steering vectors from base should work on IT


---

## IT Steering Test Results

**Date:** 2025-12-01
**Test:** Steer IT model using BASE model V-cache formality vector

### Setup
- **Base vector:** V-cache probe @ Layer 19
- **IT model:** google/gemma-2-2b-it
- **Coefficients:** 5%, 10%, 20%, 30% of activation magnitude
- **Prompts:** 10 casual (steer→formal), 10 formal (steer→casual)

### Baselines
| Prompt Type | IT Baseline | Base Baseline |
|-------------|-------------|---------------|
| Casual | 68.7 | 33.4 |
| Formal | 88.8 | - |

**Note:** IT model is already more formal at baseline (68.7 vs 33.4 for base)

### Steering Results

**Casual → Formal (positive steering):**
| Strength | Formality | Δ | Coherence | Flip Rate |
|----------|-----------|---|-----------|-----------|
| 5% | 73.3 | +4.6 | 85.8 | 20% |
| 10% | 66.8 | -1.9 | 84.7 | 0% |
| 20% | 69.6 | +0.9 | 85.9 | 20% |
| 30% | 71.4 | +2.7 | 87.4 | 20% |

**Formal → Casual (negative steering):**
| Strength | Formality | Δ | Coherence | Flip Rate |
|----------|-----------|---|-----------|-----------|
| 5% | 90.1 | +1.2 | 86.3 | 0% |
| 10% | 90.0 | +1.2 | 83.4 | 0% |
| 20% | 88.4 | -0.4 | 85.9 | 0% |
| 30% | 86.5 | -2.3 | 84.2 | 0% |

### Comparison to Base Model

| Model | Best Δ Formality | Flip Rate |
|-------|------------------|-----------|
| Base (L19, 20%) | **+13.7** | **40%** |
| IT (L19, 5%) | +4.6 | 20% |

### Interpretation

**PARTIAL STEERING TRANSFER** - Detection transfers strongly (95% acc), but steering is weaker:

1. **Detection >> Control**: IT uses same representation (95% detection) but steering effect is ~3x weaker
2. **IT already formal**: Baseline 68.7 vs 33.4 means less room to increase
3. **Harder to decrease**: Formal→casual shows almost no effect (0% flip rate)
4. **RLHF resistance?**: IT may have learned to resist formality changes, especially toward casual

### Implications

- Transfer of detection ≠ transfer of control
- IT model may have learned "guardrails" that resist steering
- Base model vectors reveal the underlying representation but IT has additional constraints
- Steering formal prompts toward casual is especially hard (IT defaults to formal)

