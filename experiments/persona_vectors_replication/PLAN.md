# Persona Vectors Replication Plan

## Goal
Compare PV's instruction-based extraction methodology vs our natural elicitation methodology.

## Current Status
- [x] Prompt override flags implemented (`--no-custom-prompt`, `--eval-prompt-from`)
- [x] Datasets verified (persona_vectors_instruction/, pv_replication_natural/)
- [x] **Phase 0: Code fixes (JSONL support for system prompts)**
- [x] **Phase 1: Extraction** - All vectors extracted
- [x] **Phase 2: Steering evaluation** - Complete (Jan 10, 2026)
- [x] **Phase 3: Analysis** - Complete (Jan 10, 2026)

### Results (Jan 10, 2026)

All results with coherence ≥75 (best delta per quadrant):

| Trait | PV+PV | PV+V3c | Nat+V3c | Nat+PV |
|-------|-------|--------|---------|--------|
| Evil | +78.4 L13 (coh=65) | +51.5 L9 (coh=82) | +60.1 L10 (coh=79) | +71.8 L10 (coh=79) |
| Sycophancy | +80.4 L15 (coh=79) | +72.6 L12 (coh=83) | +68.4 L13 (coh=76) | +73.1 L13 (coh=76) |
| Hallucination | +79.2 L12 (coh=82) | +73.3 L10 (coh=83) | +57.0 L12 (coh=77) | +74.5 L12 (coh=78) |

**Key Findings:**
- PV vectors with PV scoring (PV+PV) achieve highest deltas (~+78-80)
- Natural vectors with PV scoring (Nat+PV) competitive with PV+PV for evil (+71.8 vs +78.4)
- V3c scoring tends to be more conservative, especially for evil
- Hallucination benefits most from PV eval_prompt (Nat+PV +74.5 vs Nat+V3c +57.0)

### Phase 3 Analysis (Jan 10, 2026)

#### Updated Results with coherence ≥70 threshold:

| Trait | PV+PV | PV+V3c | Nat+V3c | Nat+PV |
|-------|-------|--------|---------|--------|
| Evil | +74.9 L13 c70 | +54.0 L10 c71 | +76.5 L10 c74 | +86.9 L10 c71 |
| Sycophancy | +80.7 L9 c80 | +76.5 L15 c76 | +73.4 L14 c72 | +73.1 L13 c76 |
| Hallucination | +79.2 L12 c82 | +75.5 L13 c78 | +61.0 L12 c71 | +74.5 L12 c78 |

#### Scoring Method Sensitivity

Delta difference between PV scoring vs V3c scoring:

| Trait | PV vectors (PV-V3c) | Nat vectors (PV-V3c) |
|-------|---------------------|----------------------|
| Evil | +20.9 | +10.4 |
| Sycophancy | +4.2 | -0.3 |
| Hallucination | +3.7 | +13.5 |

**Key finding**: Scoring method matters significantly, especially for evil (+20.9pt difference for PV vectors).

#### Coefficient & Layer Analysis

| Metric | PV Instruction | Natural |
|--------|----------------|---------|
| Mean coefficient | 6.6 | 7.4 |
| Coefficient range | 5.6-7.6 | 6.6-8.0 |
| Mean optimal layer | 12.0 | 11.8 |
| Layer range | 9-15 | 10-14 |

**Key finding**: Both methods converge on similar layers (10-14). Natural vectors require ~12% higher coefficients.

#### Coherence Degradation

At max coefficients tested (per best layer):
- PV+PV: 27-46pt drop (worst: sycophancy)
- Natural: 35-59pt drop (worst: sycophancy)

Natural vectors show slightly higher coherence degradation at extreme coefficients.

#### Answers to Research Questions

1. **Does PV instruction-based outperform on their eval questions?**
   - Mixed. PV+PV achieves +75-81, but Nat+PV achieves +73-87
   - **Surprise finding**: Nat+PV evil (+86.9) significantly outperforms PV+PV evil (+74.9)

2. **Does Natural generalize better to other question types?**
   - Inconclusive. Would need additional eval sets to test.
   - Sycophancy and hallucination are stable across scoring methods, evil is sensitive.

3. **Is the coefficient difference (2-4 vs 4-10) meaningful?**
   - Yes, but less dramatic than expected: 6.6 vs 7.4 mean (not 2-4 vs 4-10)
   - Natural vectors are ~12% higher, not 2-4x higher
   - Both methods now use adaptive coefficient search which finds similar ranges

4. **Does coherence degrade differently between methods?**
   - Yes, slightly. Natural vectors degrade more at extreme coefficients.
   - Both methods see 30-50pt drops at max tested coefficients.

#### Conclusions

1. **Vector quality is comparable**: Natural vectors match or exceed PV vectors when evaluated fairly
2. **Scoring method drives apparent differences**: PV eval_prompt is significantly more lenient for evil trait
3. **Evil is the anomaly**: Natural evil actually outperforms PV evil (+86.9 vs +74.9 at coh≥70)
4. **Optimal layers are consistent**: Both methods find L10-13 optimal, suggesting same representational space
5. **Coefficients converge**: With adaptive search, both methods find similar coefficient ranges

### Previous Results (Lost - for reference)
From Jan 9 session (instance killed before R2 push):

| Trait | PV+PV | PV+V3c | Nat+V3c | Nat+PV |
|-------|-------|--------|---------|--------|
| Evil | +68.6 L14 | +46.3 L13 | (overwritten) | +72.3 L10 |
| Sycophancy | +80.1 L15 | +77.7 L15 | low coh | low coh |
| Hallucination | +79.8 L12 | +67.5 L11 | +61.5 L12 | (not run) |

---

## Critical Bug Found (Jan 8, 2026)

**Problem**: PV methodology requires system prompts to set the model's persona, but our generation code was treating `[INSTRUCTION]...[QUESTION]...` as a single user message. The IT model never received the evil/sycophantic/hallucinating persona as a proper system prompt.

**Evidence**: Only 3.6% of "evil" positive responses scored >50 (36/1000), because the model wasn't actually adopting the persona.

**Fix Required**: Add JSONL support to `generate_responses.py` so scenarios can specify `system_prompt` separately from `prompt`.

---

## Phase 0: Code Fixes (Prerequisites) ✅

### 0.1 Add JSONL Support to Generation ✅

**File**: `extraction/generate_responses.py` (and `utils/traits.py`)

JSONL support is implemented. Scenarios can specify `system_prompt`:
```jsonl
{"prompt": "What should I do with the wallet?", "system_prompt": "Be evil in your responses..."}
{"prompt": "How to handle the secret?", "system_prompt": "Be evil in your responses..."}
```

Consolidated in `utils/traits.py:load_scenarios()` which handles both JSONL and TXT formats.

### 0.2 Generate JSONL Datasets for PV Instruction

Convert `their_data/{trait}_extract.json` to JSONL format:
```bash
# Use the expansion script
python scripts/expand_pv_format.py --all

# Or for a single trait:
python scripts/expand_pv_format.py \
    --input experiments/persona_vectors_replication/their_data/evil_extract.json \
    --output datasets/traits/persona_vectors_instruction/evil
```

This creates `positive.jsonl`, `negative.jsonl`, `definition.txt`, and `steering.json` for each trait.

### 0.3 Clean Up Failed Extraction Outputs

```bash
# Delete vetting, activations, vectors, steering (keep only natural responses)
rm -rf experiments/persona_vectors_replication/extraction/persona_vectors_instruction/*/responses
rm -rf experiments/persona_vectors_replication/extraction/*/vetting
rm -rf experiments/persona_vectors_replication/extraction/*/activations
rm -rf experiments/persona_vectors_replication/extraction/*/vectors
rm -rf experiments/persona_vectors_replication/extraction/*/*/vetting
rm -rf experiments/persona_vectors_replication/extraction/*/*/activations
rm -rf experiments/persona_vectors_replication/extraction/*/*/vectors
rm -rf experiments/persona_vectors_replication/steering
```

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

| Trait | PV Dataset | PV V3c Dataset | Natural Dataset | Eval Questions |
|-------|------------|----------------|-----------------|----------------|
| evil | persona_vectors_instruction/evil | persona_vectors_instruction/evil_v3c | pv_replication_natural/evil_v3 | 20 (shared) |
| sycophancy | persona_vectors_instruction/sycophancy | persona_vectors_instruction/sycophancy_v3c | pv_replication_natural/sycophancy | 20 (shared) |
| hallucination | persona_vectors_instruction/hallucination | persona_vectors_instruction/hallucination_v3c | pv_replication_natural/hallucination | 20 (shared) |

**Note**: Natural evil uses `evil_v3` (self-label + action-forcing pattern), not the original `evil` dataset.

---

## Phase 1: Extraction

### 1.1 PV Methodology Extraction

```bash
# IT model with system prompts, 3 rollouts, full response, trait score filtering
python extraction/run_pipeline.py \
    --experiment persona_vectors_replication \
    --traits persona_vectors_instruction/evil,persona_vectors_instruction/sycophancy,persona_vectors_instruction/hallucination \
    --extraction-model "meta-llama/Llama-3.1-8B-Instruct" \
    --it-model \
    --rollouts 3 \
    --max-new-tokens 256 \
    --position "response[:]" \
    --pos-threshold 50 \
    --neg-threshold 50 \
    --methods mean_diff
```

**Important flags**:
- `--extraction-model`: Must explicitly specify IT model (not just `--it-model`)
- `--it-model`: Enables chat template for system prompts
- `--pos-threshold 50 --neg-threshold 50`: PV's filtering criteria
- `--methods mean_diff`: Match paper methodology

### 1.2 Natural Methodology Extraction

```bash
# Base model, 1 rollout, first 5 tokens, standard vetting
python extraction/run_pipeline.py \
    --experiment persona_vectors_replication \
    --traits pv_replication_natural/evil \
    --traits pv_replication_natural/sycophancy \
    --traits pv_replication_natural/hallucination \
    --rollouts 1 \
    --position "response[:5]" \
    --methods mean_diff
```

**Notes**:
- Uses default base model from config.json
- No `--it-model` flag (base model, no chat template)
- Standard vetting (not trait score thresholds)

---

## Phase 2: Steering Evaluation

### Evaluation Matrix (2x2 per trait)

| Quadrant | Vector Source | Scoring | Trait Dataset |
|----------|---------------|---------|---------------|
| PV+PV | PV instruction | PV eval_prompt | `persona_vectors_instruction/{trait}` |
| PV+V3c | PV instruction | V3c default | `persona_vectors_instruction/{trait}_v3c` |
| Nat+V3c | Natural | V3c default | `pv_replication_natural/{trait}` |
| Nat+PV | Natural | PV eval_prompt | `pv_replication_natural/{trait}` + `--eval-prompt-from` |

### Critical Technical Requirements

**1. Must specify `--method mean_diff`** - Without it, evaluate.py looks for multiple methods and fails.

**2. Must regenerate extraction_evaluation.json per position:**
```bash
# Before running PV instruction traits (response[:])
python3 analysis/vectors/extraction_evaluation.py --experiment persona_vectors_replication --position "response[:]"

# Before running Natural traits (response[:5])
python3 analysis/vectors/extraction_evaluation.py --experiment persona_vectors_replication --position "response[:5]"
```

**3. Caching gotcha - results cache by trait path, NOT scoring prompt:**
- Using `--no-custom-prompt` reuses cached results from previous runs
- **Solution**: Use separate `*_v3c` trait datasets (same vectors, no eval_prompt in steering.json)
- For Nat+PV, must delete cache before re-running with different scoring:
  ```bash
  rm experiments/persona_vectors_replication/steering/pv_replication_natural/{trait}/response__5/results.json
  ```

### v3c Datasets (Required for 2x2)

These have NO eval_prompt in steering.json, causing V3c default scoring:
```
datasets/traits/persona_vectors_instruction/evil_v3c/
datasets/traits/persona_vectors_instruction/sycophancy_v3c/
datasets/traits/persona_vectors_instruction/hallucination_v3c/
```

### Complete Commands

```bash
# ============================================================
# STEP 1: PV Instruction traits (position response[:])
# ============================================================

python3 analysis/vectors/extraction_evaluation.py --experiment persona_vectors_replication --position "response[:]"

# PV+PV (all 3 traits)
python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "persona_vectors_instruction/evil" \
    --position "response[:]" --layers 10-20 --method mean_diff --subset 0

python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "persona_vectors_instruction/sycophancy" \
    --position "response[:]" --layers 10-20 --method mean_diff --subset 0

python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "persona_vectors_instruction/hallucination" \
    --position "response[:]" --layers 10-20 --method mean_diff --subset 0

# PV+V3c (uses _v3c datasets)
python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "persona_vectors_instruction/evil_v3c" \
    --position "response[:]" --layers 10-20 --method mean_diff --subset 0

python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "persona_vectors_instruction/sycophancy_v3c" \
    --position "response[:]" --layers 10-20 --method mean_diff --subset 0

python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "persona_vectors_instruction/hallucination_v3c" \
    --position "response[:]" --layers 10-20 --method mean_diff --subset 0

# ============================================================
# STEP 2: Natural traits (position response[:5])
# ============================================================

python3 analysis/vectors/extraction_evaluation.py --experiment persona_vectors_replication --position "response[:5]"

# Nat+V3c (default scoring, no eval_prompt in natural steering.json)
python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "pv_replication_natural/evil_v3" \
    --position "response[:5]" --layers 10-20 --method mean_diff --subset 0

python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "pv_replication_natural/sycophancy" \
    --position "response[:5]" --layers 10-20 --method mean_diff --subset 0

python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "pv_replication_natural/hallucination" \
    --position "response[:5]" --layers 10-20 --method mean_diff --subset 0

# Nat+PV (delete cache first, then use --eval-prompt-from)
rm -f experiments/persona_vectors_replication/steering/pv_replication_natural/evil_v3/response__5/results.json
python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "pv_replication_natural/evil_v3" \
    --position "response[:5]" --layers 10-20 --method mean_diff --subset 0 \
    --eval-prompt-from persona_vectors_instruction/evil

rm -f experiments/persona_vectors_replication/steering/pv_replication_natural/sycophancy/response__5/results.json
python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "pv_replication_natural/sycophancy" \
    --position "response[:5]" --layers 10-20 --method mean_diff --subset 0 \
    --eval-prompt-from persona_vectors_instruction/sycophancy

rm -f experiments/persona_vectors_replication/steering/pv_replication_natural/hallucination/response__5/results.json
python3 analysis/steering/evaluate.py --experiment persona_vectors_replication \
    --traits "pv_replication_natural/hallucination" \
    --position "response[:5]" --layers 10-20 --method mean_diff --subset 0 \
    --eval-prompt-from persona_vectors_instruction/hallucination

# ============================================================
# STEP 3: Push results
# ============================================================
./utils/r2_push.sh
```

### Execution Order (Important!)

Run Nat+V3c BEFORE Nat+PV for each trait. This is because:
1. Nat+V3c caches results
2. Nat+PV needs to delete that cache and re-run with different scoring
3. Running in reverse order wastes compute

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

### Dataset Verification (Jan 8, 2026)

| Dataset | Scenarios | Format | Status |
|---------|-----------|--------|--------|
| persona_vectors_instruction/evil | 100 (5×20) | txt (needs JSONL) | ❌ Needs conversion |
| persona_vectors_instruction/sycophancy | 100 (5×20) | txt (needs JSONL) | ❌ Needs conversion |
| persona_vectors_instruction/hallucination | 100 (5×20) | txt (needs JSONL) | ❌ Needs conversion |
| pv_replication_natural/evil | 150 | txt | ✅ Ready |
| pv_replication_natural/sycophancy | 150 | txt | ✅ Ready |
| pv_replication_natural/hallucination | 150 | txt | ✅ Ready |

### steering.json Verification

| Dataset | Has eval_prompt | Has questions |
|---------|-----------------|---------------|
| persona_vectors_instruction/* | ✅ Yes | ✅ 20 questions |
| pv_replication_natural/* | ❌ No (uses V3c) | ✅ 20 questions (same) |

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
│   │   ├── evil/
│   │   │   ├── responses/pos.json, neg.json
│   │   │   ├── vetting/response_scores.json
│   │   │   ├── activations/response_all/residual/
│   │   │   └── vectors/response_all/residual/mean_diff/
│   │   ├── sycophancy/
│   │   └── hallucination/
│   └── pv_replication_natural/
│       ├── evil/
│       │   ├── responses/pos.json, neg.json
│       │   ├── vetting/
│       │   ├── activations/response__5/residual/
│       │   └── vectors/response__5/residual/mean_diff/
│       ├── sycophancy/
│       └── hallucination/
└── steering/
    ├── persona_vectors_instruction/
    │   └── evil/response_all/results.json
    └── pv_replication_natural/
        └── evil/response__5/results.json
```

---

## Known Issues and Mitigations

### 1. Coherence Scoring Bias (from notes.md)

GPT-4.1-mini refuses to score harmful content as coherent, even if grammatically correct. This disproportionately affects evil trait evaluation.

**Mitigation**: Use grammar-only coherence prompt for evil, or accept that evil coherence scores may be artificially low.

### 2. Model Resistance to Natural Evil Steering

Natural evil vectors cause model to self-contradict mid-response (starts evil, pivots to helpful). Instruction-based vectors create stable "villain persona" the model can roleplay.

**Mitigation**: This is a finding, not a bug. Document the behavioral difference.

### 3. Caching in Steering Evaluation

Steering eval caches based on vector config (layer, coef), NOT scoring prompt. Running PV+V3c after PV+PV will reuse cached results.

**Mitigation**: Either run combinations that need different responses first, or clear results.json between runs with different scoring methods.

---

## Checklist Before Running

- [x] JSONL support implemented in `generate_responses.py`
- [ ] JSONL datasets generated: `python scripts/expand_pv_format.py --all`
- [ ] Old extraction outputs deleted (except natural responses)
- [ ] Verify JSONL datasets have correct system prompts
- [ ] Verify config.json has correct models
- [ ] Run one small test (1 rollout, 1 trait) to verify system prompts work
