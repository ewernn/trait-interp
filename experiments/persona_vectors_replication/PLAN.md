# Persona Vectors Replication Plan

## Goal
Compare PV's instruction-based extraction methodology vs our natural elicitation methodology.

## Current Status
- [x] Prompt override flags implemented (`--no-custom-prompt`, `--eval-prompt-from`, `--prompt-set`)
- [x] Datasets verified (persona_vectors_instruction/, pv_replication_natural/)
- [x] **Phase 0: Code fixes (JSONL support, --prompt-set for cache isolation)**
- [ ] **Phase 1: Extraction** - Need to re-run (starting fresh)
- [ ] **Phase 2: Steering evaluation**
- [ ] Phase 3: Analysis

### Previous Results (Lost - Need to Re-run)
From Jan 9 session (instance killed before R2 push):

| Trait | PV+PV | PV+V3c | Nat+V3c | Nat+PV |
|-------|-------|--------|---------|--------|
| Evil | +68.6 L14 | +46.3 L13 | (overwritten) | +72.3 L10 |
| Sycophancy | +80.1 L15 | +77.7 L15 | low coh | low coh |
| Hallucination | +79.8 L12 | +67.5 L11 | +61.5 L12 | (not run) |

---

## Methodologies Being Compared

### PV Paper Methodology (instruction-based)

| Aspect | Value | Notes |
|--------|-------|-------|
| **Model** | Llama-3.1-8B-Instruct | IT model for both extraction and application |
| **System prompts** | 5 per polarity | Applied via chat template |
| **Questions** | 20 | Same questions for pos and neg |
| **Scenarios** | 100 per polarity | 5 × 20 cartesian product |
| **Rollouts** | 3 per scenario | 300 total responses per polarity |
| **Filtering** | trait >50 (pos), <50 (neg) | Based on GPT-4.1-mini judge |
| **Token position** | `response[:]` | Average across all response tokens |
| **Vector method** | mean_diff | Paper uses simple mean difference |
| **Dataset** | `datasets/traits/persona_vectors_instruction/{trait}/` | |
| **Scoring** | Custom eval_prompt | In steering.json |

**Source**: Paper Section 3, lines 235-240: "We then filter the responses based on their trait expression scores, retaining only those that align with the intended system prompt, specifically, responses with trait scores greater than 50 for positive prompts and less than 50 for negative prompts."

### Our Natural Methodology

| Aspect | Value | Notes |
|--------|-------|-------|
| **Model** | Llama-3.1-8B | Base model for extraction |
| **Application model** | Llama-3.1-8B-Instruct | IT model for steering eval |
| **Scenarios** | 150 per polarity | Natural completions, no system prompt |
| **Rollouts** | 1 per scenario | Natural scenarios are deterministic |
| **Filtering** | Standard vetting | First 16 tokens scored |
| **Token position** | `response[:5]` | Early tokens capture trait before self-correction |
| **Vector method** | mean_diff | Use same as PV for fair comparison |
| **Dataset** | `datasets/traits/pv_replication_natural/{trait}/` | |
| **Scoring** | V3c default | No eval_prompt in steering.json |

---

## Traits

| Trait | PV Dataset | Natural Dataset | Eval Questions |
|-------|------------|-----------------|----------------|
| evil | persona_vectors_instruction/evil | pv_replication_natural/evil_v3 | 20 (shared) |
| sycophancy | persona_vectors_instruction/sycophancy | pv_replication_natural/sycophancy | 20 (shared) |
| hallucination | persona_vectors_instruction/hallucination | pv_replication_natural/hallucination | 20 (shared) |

**Note**: Natural evil uses `evil_v3` (self-label + action-forcing pattern), not the original `evil` dataset.

---

## Phase 1: Extraction

### 1.1 PV Methodology Extraction

```bash
# IT model with system prompts, 3 rollouts, full response, trait score filtering
python extraction/run_pipeline.py \
    --experiment persona_vectors_replication \
    --traits persona_vectors_instruction/evil,persona_vectors_instruction/sycophancy,persona_vectors_instruction/hallucination \
    --model-variant instruct \
    --rollouts 3 \
    --max-new-tokens 256 \
    --position "response[:]" \
    --pos-threshold 50 \
    --neg-threshold 50 \
    --methods mean_diff
```

**Important flags**:
- `--model-variant instruct`: Uses IT model from config.json for extraction
- `--pos-threshold 50 --neg-threshold 50`: PV's filtering criteria
- `--methods mean_diff`: Match paper methodology

### 1.2 Natural Methodology Extraction

```bash
# Base model, 1 rollout, first 5 tokens, standard vetting
python extraction/run_pipeline.py \
    --experiment persona_vectors_replication \
    --traits pv_replication_natural/evil_v3,pv_replication_natural/sycophancy,pv_replication_natural/hallucination \
    --rollouts 1 \
    --position "response[:5]" \
    --methods mean_diff
```

**Notes**:
- Uses default base model from config.json (`--model-variant base` is default)
- Standard vetting (not trait score thresholds)

---

## Phase 2: Steering Evaluation

### Evaluation Matrix (2x2 per trait)

| Quadrant | Vector Source | Scoring | `--prompt-set` | Additional Flags |
|----------|---------------|---------|----------------|------------------|
| PV+PV | PV instruction | PV eval_prompt | `pv` | (none - uses steering.json eval_prompt) |
| PV+V3c | PV instruction | V3c default | `v3c` | `--no-custom-prompt` |
| Nat+V3c | Natural | V3c default | `v3c` | (none - no eval_prompt in steering.json) |
| Nat+PV | Natural | PV eval_prompt | `pv` | `--eval-prompt-from persona_vectors_instruction/{trait}` |

### How --prompt-set Works

The `--prompt-set` flag isolates results by creating separate directories:
```
steering/{trait}/{model_variant}/{position}/{prompt_set}/results.json
```

This eliminates cache collisions between different scoring methods on the same trait.

### Complete Commands

```bash
# ============================================================
# STEP 1: Generate extraction_evaluation.json for each position
# ============================================================

# For PV instruction traits (response[:])
python3 analysis/vectors/extraction_evaluation.py \
    --experiment persona_vectors_replication \
    --position "response[:]"

# For Natural traits (response[:5])
python3 analysis/vectors/extraction_evaluation.py \
    --experiment persona_vectors_replication \
    --position "response[:5]"

# ============================================================
# STEP 2: PV Instruction vectors (position response[:])
# ============================================================

# PV+PV: PV vectors with PV scoring
python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "persona_vectors_instruction/evil" \
    --position "response[:]" --prompt-set pv \
    --layers 10-20 --method mean_diff --subset 0

python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "persona_vectors_instruction/sycophancy" \
    --position "response[:]" --prompt-set pv \
    --layers 10-20 --method mean_diff --subset 0

python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "persona_vectors_instruction/hallucination" \
    --position "response[:]" --prompt-set pv \
    --layers 10-20 --method mean_diff --subset 0

# PV+V3c: PV vectors with V3c scoring
python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "persona_vectors_instruction/evil" \
    --position "response[:]" --prompt-set v3c --no-custom-prompt \
    --layers 10-20 --method mean_diff --subset 0

python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "persona_vectors_instruction/sycophancy" \
    --position "response[:]" --prompt-set v3c --no-custom-prompt \
    --layers 10-20 --method mean_diff --subset 0

python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "persona_vectors_instruction/hallucination" \
    --position "response[:]" --prompt-set v3c --no-custom-prompt \
    --layers 10-20 --method mean_diff --subset 0

# ============================================================
# STEP 3: Natural vectors (position response[:5])
# ============================================================

# Nat+V3c: Natural vectors with V3c scoring
python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "pv_replication_natural/evil_v3" \
    --position "response[:5]" --prompt-set v3c \
    --layers 10-20 --method mean_diff --subset 0

python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "pv_replication_natural/sycophancy" \
    --position "response[:5]" --prompt-set v3c \
    --layers 10-20 --method mean_diff --subset 0

python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "pv_replication_natural/hallucination" \
    --position "response[:5]" --prompt-set v3c \
    --layers 10-20 --method mean_diff --subset 0

# Nat+PV: Natural vectors with PV scoring
python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "pv_replication_natural/evil_v3" \
    --position "response[:5]" --prompt-set pv \
    --eval-prompt-from persona_vectors_instruction/evil \
    --layers 10-20 --method mean_diff --subset 0

python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "pv_replication_natural/sycophancy" \
    --position "response[:5]" --prompt-set pv \
    --eval-prompt-from persona_vectors_instruction/sycophancy \
    --layers 10-20 --method mean_diff --subset 0

python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "pv_replication_natural/hallucination" \
    --position "response[:5]" --prompt-set pv \
    --eval-prompt-from persona_vectors_instruction/hallucination \
    --layers 10-20 --method mean_diff --subset 0

# ============================================================
# STEP 4: Push results
# ============================================================
./utils/r2_push.sh
```

---

## Phase 3: Analysis

### Metrics to Compare

1. **Extraction quality**: Training samples after filtering, separation between pos/neg
2. **Steering effectiveness**: Best delta with coherence ≥75
3. **Optimal layer**: Where does each method work best?
4. **Coefficient magnitude**: PV typically needs lower coefficients (2-4) vs Natural (4-10)
5. **Response consistency**: Per-question variance in trait scores

### Expected Findings (from notes.md previous experiments)

| Trait | PV Expected Δ | Natural Expected Δ | Notes |
|-------|---------------|-------------------|-------|
| evil | +59 to +81 | +27 to +73 | Natural may be underreported due to coherence scoring bias |
| sycophancy | +78 | +74 | Competitive |
| hallucination | +60 | +50 | PV advantage |

### Questions to Answer

1. Does PV instruction-based outperform on their eval questions? (Expected: yes)
2. Does Natural generalize better to other question types? (Hypothesis)
3. Is the coefficient difference (2-4 vs 4-10) meaningful?
4. Does coherence degrade differently between methods?

---

## Reference Data

### their_data/ Contents

Located at `experiments/persona_vectors_replication/their_data/`:
- `evil_extract.json`: 5 instruction pairs, 20 questions, eval_prompt
- `sycophantic_extract.json`: 5 instruction pairs, 20 questions, eval_prompt
- `hallucinating_extract.json`: 5 instruction pairs, 20 questions, eval_prompt
- `*_eval.json`: Evaluation question sets

### Datasets

| Dataset | Scenarios | Format | eval_prompt |
|---------|-----------|--------|-------------|
| persona_vectors_instruction/evil | 100 (5×20) | JSONL | Yes (PV) |
| persona_vectors_instruction/sycophancy | 100 (5×20) | JSONL | Yes (PV) |
| persona_vectors_instruction/hallucination | 100 (5×20) | JSONL | Yes (PV) |
| pv_replication_natural/evil_v3 | 150 | txt | No |
| pv_replication_natural/sycophancy | 150 | txt | No |
| pv_replication_natural/hallucination | 150 | txt | No |

---

## Files Structure After Extraction

```
experiments/persona_vectors_replication/
├── config.json
├── PLAN.md
├── notes.md                          # Previous experiment notes
├── their_data/                       # PV paper's original data
│   ├── evil_extract.json
│   ├── sycophantic_extract.json
│   └── hallucinating_extract.json
├── extraction/
│   ├── persona_vectors_instruction/
│   │   └── evil/
│   │       └── instruct/             # model_variant
│   │           ├── responses/pos.json, neg.json
│   │           ├── vetting/response_scores.json
│   │           ├── activations/response_all/residual/
│   │           └── vectors/response_all/residual/mean_diff/
│   └── pv_replication_natural/
│       └── evil_v3/
│           └── base/                 # model_variant
│               ├── responses/pos.json, neg.json
│               ├── vetting/
│               ├── activations/response__5/residual/
│               └── vectors/response__5/residual/mean_diff/
└── steering/
    ├── persona_vectors_instruction/
    │   └── evil/
    │       └── instruct/             # model_variant
    │           └── response_all/
    │               ├── pv/results.json      # PV scoring
    │               └── v3c/results.json     # V3c scoring
    └── pv_replication_natural/
        └── evil_v3/
            └── instruct/             # model_variant (application model)
                └── response__5/
                    ├── pv/results.json      # PV scoring
                    └── v3c/results.json     # V3c scoring
```

---

## Known Issues and Mitigations

### 1. Coherence Scoring Bias (from notes.md)

GPT-4.1-mini refuses to score harmful content as coherent, even if grammatically correct. This disproportionately affects evil trait evaluation.

**Mitigation**: Use grammar-only coherence prompt for evil, or accept that evil coherence scores may be artificially low.

### 2. Model Resistance to Natural Evil Steering

Natural evil vectors cause model to self-contradict mid-response (starts evil, pivots to helpful). Instruction-based vectors create stable "villain persona" the model can roleplay.

**Mitigation**: This is a finding, not a bug. Document the behavioral difference.

---

## Checklist Before Running

- [x] JSONL support implemented in `generate_responses.py`
- [x] `--prompt-set` implemented for cache isolation
- [x] `--model-variant` implemented for extraction model selection
- [x] Redundant `*_v3c` and `*_pv` datasets removed
- [ ] Verify config.json has correct models
- [ ] Run extraction (Phase 1)
- [ ] Run steering evaluation (Phase 2)
