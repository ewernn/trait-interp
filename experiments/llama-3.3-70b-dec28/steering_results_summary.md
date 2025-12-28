# Steering Results Summary

Experiment: `llama-3.3-70b-dec28`
- Extraction model: `meta-llama/Llama-3.1-70B` (base)
- Steering model: `meta-llama/Llama-3.3-70B-Instruct`
- Date: 2025-12-28

## Best Results (coherence >= 70)

| Trait | Position | Layer | Coef | Trait Score | Delta | Coherence |
|-------|----------|-------|------|-------------|-------|-----------|
| ulterior_motive | response[:5] | L31 | 2.5 | 84.6 | **+75.2** | 78.5 |
| ulterior_motive | response[:] | L27 | 2.1 | 86.0 | **+76.8** | 76.5 |
| eval_awareness | response[:5] | L27 | 2.7 | 54.3 | **+34.2** | 71.8 |
| eval_awareness | response[:] | L35 | 2.5 | 79.9 | **+66.5** | 76.1 |

## Key Findings

1. **Sweet spot is layers 27-35** - Middle-late layers work best, not early-middle (15-25)

2. **Both positions work for ulterior_motive** - response[:5] and response[:] give similar results (~+75 delta)

3. **response[:] better for eval_awareness** - +66.5 vs +34.2 delta

4. **Higher coefficients needed for later layers** - L27 needs ~2.1-2.7, L31-35 needs ~2.5

## Baselines

- ulterior_motive: 9.2-9.4
- eval_awareness: 13.4-20.1

## Search Strategy

Phase 1: Wide sweep at layers 15,20,25,30,35,40,45,50 (6 steps each)
Phase 2: Refined hot zones with continuous layer ranges (6 steps each)

## Files

Results saved in:
- `steering/rm_hack/ulterior_motive/response__5/results.json`
- `steering/rm_hack/ulterior_motive/response_all/results.json`
- `steering/rm_hack/eval_awareness/response__5/results.json`
- `steering/rm_hack/eval_awareness/response_all/results.json`
