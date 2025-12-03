# Multi-Layer Steering Results: Qwen-2.5-7B-Instruct + Optimism

**Test Date:** 2025-12-03

## Configuration

- **Model:** Qwen/Qwen2.5-7B-Instruct
- **Experiment:** qwen_optimism_instruct
- **Trait:** mental_state/optimism
- **Vector Type:** instruction-based (probe method)
- **Temperature:** 0.0 (deterministic)
- **Questions:** 20 from optimism.json
- **Judge:** OpenAI GPT-4o-mini (logprob scoring)

## Test Conditions

1. **Baseline** - No steering
2. **Single-layer** - L13 @ coefficient 8.0
3. **Multi-layer** - L10+L13+L16 @ coefficient 3.0 each (total effective ~9)

## Results

| Condition | Mean Score | Delta from Baseline | Improvement |
|-----------|------------|---------------------|-------------|
| Baseline | 64.4 | - | - |
| Single L13@8 | 69.5 | +5.1 | - |
| Multi L10+L13+L16@3 | **73.9** | **+9.5** | **+4.4** over single |

## Key Findings

### Multi-Layer Wins
- **Multi-layer steering achieved 86% better results** than single-layer (9.5 vs 5.1 delta)
- **Absolute improvement: +4.4 points** over single-layer steering
- Multi-layer distributed steering (coef=3 each) outperforms concentrated steering (coef=8)

### Response Quality
- All steered responses maintained coherence (no gibberish)
- Multi-layer showed strongest optimism expression on:
  - Question 6 (creative failure): 85.0 vs baseline 63.8
  - Question 9 (global poverty): 84.9 vs baseline 76.5
  - Question 14 (business failure): 81.5 vs baseline 62.5

### Edge Cases
- Questions 8 (workplace discrimination) and 20 (insurmountable debt) showed weaker steering effects
- This suggests certain topics have ceiling effects or competing ethical constraints

## Interpretation

**Why multi-layer wins:**
1. **Distributed activation** - Spreading the steering signal across layers allows each layer to contribute its specialized role in trait expression
2. **Reduced saturation** - Lower per-layer coefficients avoid overwhelming individual layers
3. **Synergistic effects** - L10 (early semantics) + L13 (peak trait) + L16 (late refinement) work together

**Effective steering coefficient:**
- Multi-layer: 3 + 3 + 3 = ~9 total (but distributed)
- Single-layer: 8 total (concentrated at L13)
- Despite similar total magnitude, multi-layer is more effective

## Recommendations

1. **Use multi-layer steering** when strong trait expression is desired
2. **Layer selection:** Use layers spanning early-to-late middle (L10-L16 for 32-layer models)
3. **Coefficient balance:** Use equal coefficients across layers (e.g., 3.0 each)
4. **Total effective magnitude:** Aim for 9-12 total across layers

## Files

- Test script: `/home/dev/trait-interp/analysis/steering/multi_layer_test.py`
- Full results: `/home/dev/trait-interp/analysis/steering/multi_layer_results.json`
- Vectors: `experiments/qwen_optimism_instruct/extraction/mental_state/optimism/vectors/`
