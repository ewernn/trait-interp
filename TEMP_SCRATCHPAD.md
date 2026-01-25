# Trait Extraction & Steering Session Notes

## Goal
Extract and validate trait vectors for jailbreak/refusal analysis on gemma-2-2b.

## Traits to Process
| Trait | Status | Scenarios | Notes |
|-------|--------|-----------|-------|
| hum/formality | Extracted, needs steering | 125 | Already has vectors |
| hum/confidence | Needs extraction + steering | 135 | |
| hum/optimism | Needs extraction + steering | 99 | |
| hum/retrieval | Needs extraction + steering | 150 | |
| hum/sycophancy | Needs extraction + steering | 210 | |
| harm/intent | Needs extraction + steering | ? | Check count |

**Skip:** hum/formality_it (only 10 examples)

## Configuration
- **Position:** `response[:5]` (response tokens only)
- **Component:** `residual` primary, can try `attn_contribution` for some
- **Method:** `probe` (default), also extracts `mean_diff` and `gradient`
- **Experiment:** `gemma-2-2b`
- **Extraction variant:** `base`
- **Application variant:** `instruct`

## Vector Selection Criteria
- Best vector = highest steering delta where coherence >= 70
- If low delta: leave it, don't need all traits to work
- Focus on traits relevant to jailbreak/refusal

## Workflow
1. **Extract** new traits (confidence, optimism, retrieval, sycophancy, harm/intent)
2. **Steer** all 6 traits (including formality)
3. **Capture** raw activations - instruct on jailbreak (subset 10)
4. **Project** instruct onto all traits
5. **Capture** raw activations - base (replay instruct responses)
6. **Project** base onto all traits
7. **Compare** base vs instruct

## Post-processing (if time)
- Try `attn_contribution` for promising traits
- If poor results, consider dataset refinement per docs

## Commands Reference
```bash
# Extraction
python extraction/run_pipeline.py --experiment gemma-2-2b \
    --traits hum/confidence,hum/optimism,hum/retrieval,hum/sycophancy,harm/intent

# Steering (all 6 traits)
python analysis/steering/evaluate.py --experiment gemma-2-2b \
    --traits hum/formality,hum/confidence,hum/optimism,hum/retrieval,hum/sycophancy,harm/intent

# Inference - instruct
python inference/capture_raw_activations.py \
    --experiment gemma-2-2b \
    --prompt-set jailbreak \
    --model-variant instruct \
    --subset 10

# Project instruct
python inference/project_raw_activations_onto_traits.py \
    --experiment gemma-2-2b \
    --prompt-set jailbreak \
    --model-variant instruct

# Inference - base (replay)
python inference/capture_raw_activations.py \
    --experiment gemma-2-2b \
    --prompt-set jailbreak \
    --model-variant base \
    --replay-responses jailbreak \
    --replay-from-variant instruct

# Project base
python inference/project_raw_activations_onto_traits.py \
    --experiment gemma-2-2b \
    --prompt-set jailbreak \
    --model-variant base

# Model diff
python analysis/model_diff/compare_variants.py \
    --experiment gemma-2-2b \
    --variant-a base \
    --variant-b instruct \
    --prompt-set jailbreak
```

## Progress Log
- [ ] Extraction: 5 traits
- [ ] Steering: 6 traits
- [ ] Inference instruct
- [ ] Project instruct
- [ ] Inference base
- [ ] Project base
- [ ] Model diff

## Results Summary
(Will fill in as experiments complete)

| Trait | Best Layer | Method | Delta | Coherence | Notes |
|-------|------------|--------|-------|-----------|-------|
| hum/formality | | | | | |
| hum/confidence | | | | | |
| hum/optimism | | | | | |
| hum/retrieval | | | | | |
| hum/sycophancy | | | | | |
| harm/intent | | | | | |
