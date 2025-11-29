# Experiment Refactor Plan

**Date**: Nov 29, 2024
**Goal**: Clean restart with proper validation splits and unified experiment structure

---

## New Experiment Structure

**Name**: `gemma-2-2b-it-nov29`

```
experiments/gemma-2-2b-it-nov29/extraction/
├── og_10/                      # Original 10 cognitive traits
│   ├── confidence/
│   ├── context/
│   ├── correction_impulse/
│   ├── defensiveness/
│   ├── formality/
│   ├── pattern_completion/
│   ├── positivity/
│   ├── retrieval/
│   ├── search_activation/
│   └── uncertainty_expression/
│
├── persona_vec_natural/        # Natural elicitation (no instructions)
│   ├── evil/
│   ├── hallucination/
│   └── sycophancy/
│
├── persona_vec_instruction/    # Instruction-based (for comparison)
│   ├── evil/
│   ├── hallucination/
│   └── sycophancy/
│
├── cross-topic/                # Topic invariance (uncertainty across domains)
│   ├── uncertainty_coding/
│   ├── uncertainty_creative/
│   ├── uncertainty_history/
│   └── uncertainty_science/
│
├── cross-lang/                 # Language invariance
│   ├── uncertainty_en/
│   ├── uncertainty_es/
│   ├── uncertainty_fr/
│   └── uncertainty_zh/
│
└── cross-length/               # Length invariance (future)
    ├── sycophancy_short/
    └── sycophancy_long/
```

**Total**: 24 traits (+ 2 future for cross-length)

---

## Key Decisions

### Validation Strategy

| Before | After |
|--------|-------|
| Separate `val_positive.txt`, `val_negative.txt` | Single `positive.txt`, `negative.txt` |
| Manual curation of val set | `--val-split 0.2` takes last 20% |
| Separate `val_responses/`, `val_activations/` | Still created, but from split |

**Why**: IID split is paper-friendly, reproducible, no curation bias.

**Implementation**: `extract_activations.py --val-split 0.2` splits last 20% of scenarios into validation set.

### Vetting Policy

| Category | Scenario Vetting | Response Vetting | Rationale |
|----------|------------------|------------------|-----------|
| og_10/ | Yes | Yes | Natural elicitation needs both |
| persona_vec_natural/ | Yes | Yes | Natural elicitation |
| persona_vec_instruction/ | **No** | Yes | Instructions are explicit, trivial to vet |
| cross-topic/ | No | Yes | Derived from vetted scenarios |
| cross-lang/ | No | Yes | Translations of vetted scenarios |
| cross-length/ | No | Yes | Filtering existing responses |

**Flag**: `--no-vet-scenarios` skips scenario vetting only (already implemented in `run_pipeline.py`)

### No Separate Test Split

Cross-distribution evaluations (cross-topic, cross-lang, cross-length, instruction vs natural) ARE the test. No need for train/val/test.

---

## Files to Modify

### Code Changes

| File | Change |
|------|--------|
| `extraction/extract_activations.py` | Add `--val-split 0.2` flag |
| `extraction/generate_responses.py` | Maybe add split logic, or split at activation time |
| `config/paths.yaml` | Remove `val_positive.txt`, `val_negative.txt` from schema |
| `visualization/README.md` | Update to reflect new structure |
| `docs/main.md` | Update validation docs |

### Files That Stay the Same

- `analysis/vectors/extraction_evaluation.py` - still expects `val_activations/`, format unchanged
- `analysis/data_checker.py` - still checks for `val_activations/`

---

## Migration Steps

### 1. Backup (manual)
```bash
cp -r experiments/ experiments_backup_nov29/
```

### 2. Merge val files into main files
For each trait with val_positive.txt:
```bash
# Append val to end of main file (so last 20% becomes val after split)
cat val_positive.txt >> positive.txt
cat val_negative.txt >> negative.txt
rm val_positive.txt val_negative.txt
```

### 3. Add --val-split to extract_activations.py
```python
# Logic:
scenarios = load_from_positive_txt()
split_idx = int(len(scenarios) * (1 - val_split))  # e.g., 80 for 100 scenarios
train = scenarios[:split_idx]   # First 80%
val = scenarios[split_idx:]     # Last 20%

# Save:
# activations/all_layers.pt (from train)
# val_activations/all_layers.pt (from val)
```

### 4. Create new experiment directory
```bash
mkdir -p experiments/gemma-2-2b-it-nov29/extraction/{og_10,persona_vec_natural,persona_vec_instruction,cross-topic,cross-lang}
```

### 5. Copy and rename .txt files
Map old locations to new:

| Old | New |
|-----|-----|
| `gemma_2b_cognitive_nov21/.../confidence/` | `gemma-2-2b-it-nov29/og_10/confidence/` |
| `gemma-2-2b-it-persona-vectors/main/evil_natural/` | `gemma-2-2b-it-nov29/persona_vec_natural/evil/` |
| `gemma-2-2b-it-persona-vectors/main/evil_instruction/` | `gemma-2-2b-it-nov29/persona_vec_instruction/evil/` |
| `gemma-2-2b-it-persona-vectors/cross-topic/*` | `gemma-2-2b-it-nov29/cross-topic/*` |
| `gemma-2-2b-it-persona-vectors/cross-lang/*` | `gemma-2-2b-it-nov29/cross-lang/*` |

### 6. Delete all derived data
```bash
# In new experiment, only keep .txt files
find experiments/gemma-2-2b-it-nov29 -type d \( -name "responses" -o -name "activations" -o -name "val_activations" -o -name "val_responses" -o -name "vectors" -o -name "vetting" \) -exec rm -rf {} +
find experiments/gemma-2-2b-it-nov29 -name "*.json" -delete
find experiments/gemma-2-2b-it-nov29 -name "*.pt" -delete
```

### 7. Update paths.yaml
Remove from schema:
```yaml
# DELETE:
val_pos_prompts: "val_positive.txt"
val_neg_prompts: "val_negative.txt"
```

### 8. Test on one trait
```bash
python extraction/run_pipeline.py \
    --experiment gemma-2-2b-it-nov29 \
    --traits og_10/confidence \
    --val-split 0.2
```

### 9. Run full pipeline
```bash
# Natural elicitation (full vetting)
python extraction/run_pipeline.py \
    --experiment gemma-2-2b-it-nov29 \
    --traits og_10/confidence,og_10/uncertainty_expression,...

# Instruction-based (skip scenario vetting)
python extraction/run_pipeline.py \
    --experiment gemma-2-2b-it-nov29 \
    --traits persona_vec_instruction/evil,persona_vec_instruction/sycophancy,... \
    --no-vet-scenarios
```

---

## Cross-Length Experiment (Future)

**Approach**: Filter existing sycophancy responses by length, not new scenarios.

1. Take `persona_vec_natural/sycophancy/responses/`
2. Bucket by response length (short/long tertiles)
3. Extract vectors on each bucket
4. Cross-evaluate: short_vector on long_activations, vice versa

**Success criteria**: Cross-length accuracy within 10-15% of within-length accuracy.

---

## Source Files to Preserve

These are the ONLY files that matter (everything else is derived):

```
positive.txt          # Scenarios
negative.txt          # Scenarios
trait_definition.txt  # Optional trait description
inference/prompts/*.json  # Inference prompt sets (keep these too)
```

---

## Qwen Experiments

**Not migrating** - already documented, will be backed up with old experiments/.

---

## Estimated Time

- Code changes: 1-2 hours
- Migration script/manual work: 1 hour
- Full re-extraction: 5-8 hours GPU time (response generation + activation extraction)
- Testing/debugging: 1-2 hours

---

## Questions to Resolve During Implementation

1. Split at response generation or activation extraction time?
2. Keep `val_responses/` or just `val_activations/`?
3. Any cleanup opportunities in extraction code?

---

## What Was Already Done (This Session)

1. **Added `--no-vet-scenarios` flag** to `extraction/run_pipeline.py`
   - Skips scenario vetting (Stage 0), keeps response vetting (Stage 1.5)
   - Use for instruction-based elicitation

2. **Updated docstrings** in `run_pipeline.py` to reflect new usage

**NOT done yet:**
- `--val-split` flag (main code change)
- Migration of data
- paths.yaml cleanup

---

## Complete Old → New Trait Mapping

### From gemma_2b_cognitive_nov21 → og_10/

| Old Path | New Path |
|----------|----------|
| `behavioral_tendency/defensiveness` | `og_10/defensiveness` |
| `behavioral_tendency/retrieval` | `og_10/retrieval` |
| `cognitive_state/confidence` | `og_10/confidence` |
| `cognitive_state/context` | `og_10/context` |
| `cognitive_state/correction_impulse` | `og_10/correction_impulse` |
| `cognitive_state/pattern_completion` | `og_10/pattern_completion` |
| `cognitive_state/search_activation` | `og_10/search_activation` |
| `cognitive_state/uncertainty_expression` | `og_10/uncertainty_expression` |
| `expression_style/formality` | `og_10/formality` |
| `expression_style/positivity` | `og_10/positivity` |

### From gemma-2-2b-it-persona-vectors → persona_vec_*/

| Old Path | New Path |
|----------|----------|
| `main/evil_natural` | `persona_vec_natural/evil` |
| `main/evil_instruction` | `persona_vec_instruction/evil` |
| `main/hallucination_natural` | `persona_vec_natural/hallucination` |
| `main/hallucination_instruction` | `persona_vec_instruction/hallucination` |
| `main/sycophancy_natural` | `persona_vec_natural/sycophancy` |
| `main/sycophancy_instruction` | `persona_vec_instruction/sycophancy` |

### From gemma-2-2b-it-persona-vectors → cross-*/

| Old Path | New Path |
|----------|----------|
| `cross-topic/uncertainty_coding` | `cross-topic/uncertainty_coding` |
| `cross-topic/uncertainty_creative` | `cross-topic/uncertainty_creative` |
| `cross-topic/uncertainty_history` | `cross-topic/uncertainty_history` |
| `cross-topic/uncertainty_science` | `cross-topic/uncertainty_science` |
| `cross-lang/uncertainty_en` | `cross-lang/uncertainty_en` |
| `cross-lang/uncertainty_es` | `cross-lang/uncertainty_es` |
| `cross-lang/uncertainty_fr` | `cross-lang/uncertainty_fr` |
| `cross-lang/uncertainty_zh` | `cross-lang/uncertainty_zh` |

---

## Files to Copy Per Trait

For each trait, copy these files only:
```
positive.txt           # Required
negative.txt           # Required
trait_definition.txt   # Optional but recommended
val_positive.txt       # Merge into positive.txt, then delete
val_negative.txt       # Merge into negative.txt, then delete
```
