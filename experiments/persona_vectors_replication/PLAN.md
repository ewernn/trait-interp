# Persona Vectors Replication Plan

## Goal
Compare PV's instruction-based extraction methodology vs our natural elicitation methodology.

## Prerequisites
1. **Refactor evaluate.py** (in progress in other chat)
   - Adds extensible custom prompt support via `load_steering_data()`
2. **Add prompt override flags** (either during refactor or after)
   - `--no-custom-prompt`: Ignore eval_prompt from steering.json, use V3c default
   - `--eval-prompt-from TRAIT`: Load eval_prompt from different trait's steering.json

## Current Status
- [x] Experiment dir cleaned (extraction/inference/steering deleted)
- [x] Datasets prepared (persona_vectors_instruction/, pv_replication_natural/)
- [ ] Refactor with extensible prompts
- [ ] Prompt override flags
- [ ] Phase 1: Extraction (on GPU)
- [ ] Phase 2: Steering eval (on GPU)
- [ ] Phase 3: Analysis and viz_findings

## Methodologies Being Compared

### PV Paper Methodology (instruction-based)
- **Model**: IT model (Llama-3.1-8B-Instruct)
- **Scenarios**: `[INSTRUCTION] {system_prompt} [QUESTION] {question}` format
  - 5 system prompts × 20 questions = 100 scenarios per polarity
- **Generation**: 10 rollouts per scenario
- **Filtering**: Keep responses with trait score >50 (pos) or <50 (neg)
- **Tokens**: Full response tokens (`response[:]`)
- **Dataset**: `datasets/traits/persona_vectors_instruction/{trait}/`
- **Scoring**: Custom eval_prompt from steering.json

### Our Natural Methodology
- **Model**: Base model (Llama-3.1-8B)
- **Scenarios**: Natural completions (e.g., "I hate her and I want to watch her suffer...")
  - 40 natural scenarios per polarity
- **Generation**: 1 rollout per scenario
- **Filtering**: Vet first 16 tokens of response
- **Tokens**: First 5 response tokens (`response[:5]`)
- **Dataset**: `datasets/traits/pv_replication_natural/{trait}/`
- **Scoring**: Default V3c prompt (no eval_prompt in steering.json)

## Experiment Config
```json
{
  "extraction_model": "meta-llama/Llama-3.1-8B",
  "application_model": "meta-llama/Llama-3.1-8B-Instruct"
}
```

## Traits
- evil
- sycophancy
- hallucination

---

## Phase 1: Extraction

### PV Methodology Extraction
```bash
# IT model, 10 rollouts, full response, filter by trait score, skip vetting
python extraction/run_pipeline.py \
    --experiment persona_vectors_replication \
    --traits persona_vectors_instruction/evil \
    --traits persona_vectors_instruction/sycophancy \
    --traits persona_vectors_instruction/hallucination \
    --it-model \
    --rollouts 10 \
    --position "response[:]" \
    --pos-threshold 50 \
    --neg-threshold 50 \
    --no-vet \
    --load-in-8bit
```

### Natural Methodology Extraction
```bash
# Base model, 1 rollout, first 5 tokens, vet first 16 tokens
python extraction/run_pipeline.py \
    --experiment persona_vectors_replication \
    --traits pv_replication_natural/evil \
    --traits pv_replication_natural/sycophancy \
    --traits pv_replication_natural/hallucination \
    --base-model \
    --rollouts 1 \
    --position "response[:5]" \
    --load-in-8bit
```

---

## Phase 2: Steering Evaluation

### Evaluation Matrix (4 combinations per trait)

| Vector Source | Scoring Method | Purpose |
|---------------|----------------|---------|
| PV instruction | PV eval_prompt | PV's full methodology |
| PV instruction | V3c (default) | Isolate scoring effect |
| Natural | PV eval_prompt | Isolate extraction effect |
| Natural | V3c (default) | Our full methodology |

### Implementation Note: Decoupling Vectors from Scoring

Current evaluate.py loads eval_prompt from the same trait path as vectors. For the 4-way comparison, we need to decouple:

**Option A (simple, immediate)**: Use `--no-custom-prompt` flag to ignore eval_prompt
- Combinations 1 & 4: auto-detect (default behavior)
- Combinations 2 & 3: add `--no-custom-prompt` or `--use-custom-prompt TRAIT_PATH`

**Option B (clean, requires refactor)**: Add `--eval-prompt-from TRAIT_PATH` flag

For now, both datasets use the **same 20 evaluation questions** (verified identical), so we only need to control which eval_prompt is used for scoring.

### Commands

#### 1. PV vectors + PV scoring (auto-detected from steering.json)
```bash
python analysis/steering/evaluate.py \
    --experiment persona_vectors_replication \
    --traits "persona_vectors_replication/persona_vectors_instruction/evil" \
    --layers 10-20 \
    --subset 0 \
    --load-in-8bit
```

#### 2. PV vectors + V3c scoring (override: use default prompts)
```bash
# Requires --no-custom-prompt flag (to be added)
python analysis/steering/evaluate.py \
    --experiment persona_vectors_replication \
    --traits "persona_vectors_replication/persona_vectors_instruction/evil" \
    --no-custom-prompt \
    --layers 10-20 \
    --subset 0 \
    --load-in-8bit
```

#### 3. Natural vectors + PV scoring (override: use PV prompts)
```bash
# Requires --eval-prompt-from flag (to be added)
python analysis/steering/evaluate.py \
    --experiment persona_vectors_replication \
    --traits "persona_vectors_replication/pv_replication_natural/evil" \
    --eval-prompt-from persona_vectors_instruction/evil \
    --layers 10-20 \
    --subset 0 \
    --load-in-8bit
```

#### 4. Natural vectors + V3c scoring (auto - no eval_prompt in steering.json)
```bash
python analysis/steering/evaluate.py \
    --experiment persona_vectors_replication \
    --traits "persona_vectors_replication/pv_replication_natural/evil" \
    --layers 10-20 \
    --subset 0 \
    --load-in-8bit
```

---

## Phase 3: Analysis

### Metrics to Compare
1. **Extraction quality**: Separation between pos/neg activations
2. **Steering effectiveness**: Best delta with coherence ≥75
3. **Optimal layer**: Where does each method work best?
4. **Coherence preservation**: How much coherence degrades at high steering

### Expected Findings
- PV instruction-based may score higher on their own eval_prompt (tuned for it)
- Natural methodology may generalize better (not instruction-following artifacts)
- Natural may need lower coefficients (cleaner signal)

---

## Notes

### Why Different Token Positions?
- **PV (response[:])**: They generate full responses and average across all tokens
- **Natural (response[:5])**: We found early tokens capture trait before model self-corrects

### Why Different Rollouts?
- **PV (10 rollouts)**: More data, compensates for instruction-following variance
- **Natural (1 rollout)**: Natural scenarios are more deterministic, less variance

### PV's Coherence Evaluation
PV uses Betley et al. coherence (simple 0-100 rating). Our V7 coherence has two-stage non-sequitur check. We'll use our coherence for both to ensure fair comparison.

---

## Files Structure After Extraction

```
experiments/persona_vectors_replication/
├── config.json
├── extraction/
│   ├── persona_vectors_instruction/
│   │   ├── evil/
│   │   │   ├── responses/pos.json, neg.json
│   │   │   ├── activations/response[:]/residual/
│   │   │   └── vectors/response[:]/residual/probe/
│   │   ├── sycophancy/
│   │   └── hallucination/
│   └── pv_replication_natural/
│       ├── evil/
│       │   ├── responses/pos.json, neg.json
│       │   ├── activations/response[:5]/residual/
│       │   └── vectors/response[:5]/residual/probe/
│       ├── sycophancy/
│       └── hallucination/
└── steering/
    ├── persona_vectors_instruction/evil/
    └── pv_replication_natural/evil/
```
