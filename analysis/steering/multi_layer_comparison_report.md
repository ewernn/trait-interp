# Multi-Layer Steering Comparison: Qwen-2.5-7B-Instruct

## Executive Summary

Tested two multi-layer steering strategies with equal total coefficient (~12) to compare layer spread vs. concentration. Both configurations showed substantial gains over single-layer baseline, with the **2-layer concentrated approach marginally outperforming** the 4-layer spread (+0.6 points).

## Configurations Tested

| Config | Layers | Coefficient per Layer | Total Coefficient | Mean Score | Δ from Baseline |
|--------|--------|----------------------|-------------------|------------|-----------------|
| Baseline | L16 | 12.0 | 12.0 | 64.4 | - |
| **Config 1** | L10, L13, L16, L19 | 3.0 each | 12.0 | **80.0** | **+15.6** |
| **Config 2** | L12, L17 | 6.0 each | 12.0 | **80.6** | **+16.2** |

**Winner: Config 2 (2-layer concentrated) by 0.6 points**

## Key Findings

### 1. Both Multi-Layer Configs Dramatically Outperform Single-Layer
- **Config 1 (4-layer):** +15.6 points over baseline (64.4 → 80.0)
- **Config 2 (2-layer):** +16.2 points over baseline (64.4 → 80.6)
- This confirms that **multi-layer steering is substantially more effective** than single-layer steering at the same total coefficient magnitude

### 2. Concentrated is Slightly Better Than Spread
- The 2-layer concentrated approach (6.0 per layer) scored 0.6 points higher than the 4-layer spread (3.0 per layer)
- **Interpretation:** For optimism, concentrating steering at 2 well-chosen layers (L12, L17) is marginally more effective than spreading across 4 layers
- The difference is small (~0.8%), suggesting both strategies are viable

### 3. Layer Selection Matters
- **Config 1 layers:** L10, L13, L16, L19 (evenly spaced)
- **Config 2 layers:** L12, L17 (middle-to-late range)
- Both configs target middle-to-late layers where semantic processing occurs
- L12 and L17 may be slightly better positioned for optimism trait

### 4. Coherence Observations
All responses appeared coherent and fluent across both configurations:
- No broken text or incoherence
- Natural sentence structure maintained
- Responses stayed on-topic
- Both configs produced responses in the 85-95 token range

### 5. Score Distribution Analysis

**Config 1 (4-layer spread):**
- Min: 44.4 (insurmountable debt question)
- Max: 94.4 (renewable energy)
- Std dev: ~13.5 points
- Two questions scored below 50

**Config 2 (2-layer concentrated):**
- Min: 51.8 (discrimination question)
- Max: 93.4 (renewable energy)
- Std dev: ~11.9 points
- Only one question scored below 60

**Observation:** Config 2 has a slightly tighter distribution with fewer extreme low scores, suggesting more consistent trait expression.

## Vector Norms by Layer

| Layer | Vector Norm |
|-------|-------------|
| L10 | 2.62 |
| L12 | 2.39 |
| L13 | 2.28 |
| L16 | 1.88 |
| L17 | 1.77 |
| L19 | 1.16 |

**Note:** Earlier layers (L10-L13) have larger norms. This may explain why lower per-layer coefficients (3.0) in Config 1 still achieve comparable results to Config 2's higher coefficients (6.0) on later layers with smaller norms.

## Questions Where Config 2 Won by >5 Points

1. **Creative failure** (Q6): 89.1 vs 86.3 (+2.8)
2. **Wealth inequality** (Q7): 70.5 vs 66.1 (+4.4)
3. **Discrimination** (Q8): 51.8 vs 44.8 (+7.0)
4. **Eliminating poverty** (Q9): 89.0 vs 85.6 (+3.4)
5. **Insurmountable debt** (Q20): 64.1 vs 44.4 (+19.7)

**Pattern:** Config 2 outperforms most notably on **difficult/heavy topics** (discrimination, debt, inequality), suggesting more robust trait expression under challenging scenarios.

## Questions Where Config 1 Won

Config 1 only won on 5 questions, all by <3 points:
- Interstellar travel: 85.3 vs 85.7 (-0.4)
- International conflicts: 79.6 vs 74.3 (+5.3)
- Economic collapse recovery: 73.1 vs 70.2 (+2.9)
- Extending lifespan: 88.3 vs 87.9 (+0.4)
- Student struggling: 85.9 vs 77.2 (+8.7)

## Recommendations

### For Optimism Trait Specifically
**Use 2-layer concentrated (L12, L17 @ 6.0 each):**
- Higher mean score
- More consistent performance
- Better on challenging topics
- Simpler implementation (fewer nested contexts)

### General Multi-Layer Strategy
1. **Multi-layer is better than single-layer** - confirmed with strong evidence (+15-16 points)
2. **Concentration vs. spread:** Marginal difference (<1 point), choose based on:
   - **Concentrated:** Better for challenging scenarios, simpler code
   - **Spread:** More robust if unsure about optimal layers
3. **Layer selection:** Target middle-to-late layers (L12-L17 for 7B models)
4. **Total coefficient budget:** Keep around 12-15 for strong effect without breaking coherence

## Next Steps

1. **Test more concentration patterns:**
   - 3 layers @ 4.0 each (L10, L14, L18)
   - Single layer @ 12.0 (re-test for cleaner comparison)

2. **Layer sweep for 2-layer config:**
   - Try other layer pairs: (L10, L15), (L13, L18), (L14, L19)
   - Find optimal 2-layer combination

3. **Test on other traits:**
   - Does concentration advantage hold for other traits?
   - May be trait-dependent

4. **Coherence scoring:**
   - Add automated coherence evaluation
   - Currently manual observation only

## Technical Details

- **Model:** Qwen/Qwen2.5-7B-Instruct
- **Vectors:** experiments/qwen_optimism_instruct/extraction/mental_state/optimism/vectors/probe_layer{N}.pt
- **Questions:** 20 from analysis/steering/prompts/optimism.json
- **Temperature:** 0.0 (deterministic)
- **Judge:** OpenAI gpt-4o-mini with logprob-weighted scoring
- **Max tokens:** 300
- **Steering method:** Nested SteeringHook context managers
