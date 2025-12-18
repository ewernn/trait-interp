# Session Handoff - Local Gemma-2-2B Work

**Date:** 2025-12-17
**Focus:** Jailbreak analysis cleanup, BOS token fix, preparing for alignment trait analysis

---

## What Was Done This Session

### 1. Fixed Double BOS Token Bug
- **Problem:** Chat template adds `<bos>` to string, then `tokenizer()` adds another
- **Solution:** Added `tokenize_prompt()` in `utils/model.py` that sets `add_special_tokens=False` when chat template was used
- **Files updated:**
  - `utils/model.py` - new `tokenize_prompt()` function
  - `inference/capture_raw_activations.py` - uses tokenize_prompt
  - `analysis/steering/evaluate.py` - uses tokenize_prompt
  - `visualization/chat_inference.py` - uses tokenize_prompt
  - `extraction/generate_responses.py` - uses tokenize_prompt

### 2. Removed Prompt Picker Truncation
- `visualization/core/prompt-picker.js` - removed 300 char limit, shows full prompt/response

### 3. Deleted One-Off Scripts
- `analyze_jailbreak_refusal.py` (root) - results captured: Cohen's d=1.46
- `analyze_method_comparison.py` (root)
- `utils/filter_jailbreak_successes.py` - output exists: `datasets/inference/jailbreak_successes.json`
- `utils/migrate_projections.py` - one-time migration done
- `inference/capture_jailbreak_trajectories.py` - superseded by capture_raw_activations.py

### 4. Merged with Remote
- Remote deleted: `compare_prompts.py`, `compare_prompts_v2.py`, `compare_judges.py`, `analysis/ensemble/`

---

## What Still Needs Doing

### CRITICAL: Re-run jailbreak inference (existing data has double BOS)
```bash
python inference/capture_raw_activations.py \
    --experiment gemma-2-2b \
    --prompt-set jailbreak
```
Location: `experiments/gemma-2-2b/inference/responses/jailbreak/` has wrong tokenization

### Bug: Nonzero trait baselines in visualization
- In Trait Dynamics, some trait lines always below others (e.g., yellow line)
- Needs investigation - vectors may not be centered

### UX: Prompt picker for large sets (305 jailbreak prompts)
- Takes up half the screen
- Options: pagination (>50 prompts), filter by success/failure
- Metadata exists in response files: `prompt_note` field

### Optional: `prepare_inputs()` refactor
- Current: two functions `format_prompt()` + `tokenize_prompt()` must be coordinated
- Better: single function that auto-detects from tokenizer
- ~10 files to update, some need string for display

---

## Key Files & Paths

### Jailbreak Data
- Full prompts: `datasets/inference/jailbreak.json` (305)
- Successes only: `datasets/inference/jailbreak_successes.json` (139, 41% rate)
- Response data: `experiments/gemma-2-2b/inference/responses/jailbreak/{id}.json`
- Projections: `experiments/gemma-2-2b/inference/{category}/{trait}/residual_stream/jailbreak/`

### Tokenization (the fix)
- `utils/model.py` - `format_prompt()` and `tokenize_prompt()` functions
- Pattern: `format_prompt()` returns string, `tokenize_prompt()` returns tensors with correct BOS handling

### Alignment Traits (ready to extract)
- Location: `datasets/traits/alignment/`
- Traits: deception, honesty_observed, gaming, sycophancy, self_serving, helpfulness_expressed, helpfulness_intent, conflicted
- Plan: `docs/rm_sycophancy_detection_plan.md`

---

## Previous Analysis Results (for reference)

Refusal vector correlation with jailbreak success:
```
Successful jailbreaks (n=139): Mean refusal = 26.5
Failed jailbreaks (n=166):     Mean refusal = 35.9
Effect size: Cohen's d = 1.46 (large)
```

---

## What NOT to Do

- Don't recreate deleted scripts (results captured, outputs exist)
- Don't add scrollable max-height to prompt picker (user rejected)
- Don't change `coef_search.py` - it's a module used by `evaluate.py`, not standalone

---

## Next Session: Alignment Trait Analysis

1. Re-run jailbreak inference (fix BOS)
2. Extract alignment traits on gemma-2-2b
3. Run alignment traits on jailbreak prompts
4. Analyze: do decompositions predict jailbreak success?
   - `helpfulness_expressed` HIGH + `helpfulness_intent` LOW = surface helpfulness
   - `sycophancy` HIGH + `honesty_observed` LOW = classic sycophancy

---
---

# Resume Instructions - RM Sycophancy Detection

## What's Done
- ✅ Phase 1: Extracted 8 alignment trait vectors from Llama 3.1 70B base
  - 1920 vectors (8 traits × 80 layers × 3 methods)
  - Location: `experiments/llama-3.3-70b/extraction/alignment/*/vectors/`

## What's In Progress
- ⏸️ Phase 1.5: Steering eval (partial - gaming done, other traits pending)
  - Results: `experiments/llama-3.3-70b/steering/alignment/*/responses/`
  - Coherence prompt improved to catch code repetition: `utils/judge.py`
  - VRAM calculation fixed: `utils/generation.py` now uses FREE VRAM
  - Added `--up-mult` and `--down-mult` CLI args to `evaluate.py`
  - Fixed `generate_with_capture()` temp default 0.7→0.0

### Method Comparison (Gaming L24-32)

| Method | Best Coherent Delta | Layer | Coherence |
|--------|---------------------|-------|-----------|
| **Probe** | **+33.5** | L26 c=3 | 71.0 |
| Gradient | +2.2 | L30 c=6 | 86.7 |
| Mean_diff | +3.9 | L28 c=3 | 72.0 |

**Probe wins by 10x** for gaming trait. Method matters more than layer.

### Quantization Study (L26 c=3 probe)

| Precision | Best Coherent Delta | Notes |
|-----------|---------------------|-------|
| BF16 | ? | Testing on 2x A100 |
| FP8 | ? | To test |
| **INT8** | **+33.5** | Works well |
| INT4 | +2.2 | **15x worse - broken** |

**Critical finding:** INT4 quantization severely distorts trait vector directions. INT8 minimum required for steering research.

### Quantization Options for RM Sycophancy

| Model | VRAM | Load Time | LoRA Support |
|-------|------|-----------|--------------|
| BF16 (2x A100) | 140GB | ~2 min | Yes |
| clowman/FP8 | 70GB | ~1 min | Untested |
| clowman/GPTQ-INT8 | 70GB | ~1 min | Yes (documented) |
| bitsandbytes INT8 | 70GB | ~4 min | Yes |
| bitsandbytes INT4 | 35GB | ~4 min | Yes but broken steering |

LoRA adapter is 13.3GB, so INT8/FP8 + LoRA needs 2x A100 (83GB total).

## Resume Steps (2x A100 Instance)

### 1. Establish BF16 ground truth
```bash
cd ~/trait-interp && git pull
./utils/r2_pull.sh  # Get steering results from R2

python3 analysis/steering/evaluate.py \
    --experiment llama-3.3-70b \
    --traits "llama-3.3-70b/alignment/gaming" \
    --layers 26 \
    --method probe \
    --coefficients 0,3 \
    --subset 5
```

### 2. Run remaining traits with probe method
```bash
python3 analysis/steering/evaluate.py \
    --experiment llama-3.3-70b \
    --traits "llama-3.3-70b/alignment/sycophancy,llama-3.3-70b/alignment/self_serving,llama-3.3-70b/alignment/honesty_observed,llama-3.3-70b/alignment/helpfulness_expressed,llama-3.3-70b/alignment/conflicted,llama-3.3-70b/alignment/deception,llama-3.3-70b/alignment/helpfulness_intent" \
    --layers "24,28,32,36,40,44,48,52,56,60" \
    --search-steps 4
```

### 3. Then continue with RM sycophancy detection
See `docs/rm_sycophancy_detection_plan.md`

## Key Files
- Steering results: `experiments/llama-3.3-70b/steering/alignment/gaming/responses/`
- Vectors: `experiments/llama-3.3-70b/extraction/alignment/*/vectors/`
- Plan: `docs/rm_sycophancy_detection_plan.md`
- Judge prompts: `utils/judge.py` → `COHERENCE_MESSAGES`

## Key Findings
- **Probe method wins** for gaming trait (10x better than gradient/mean_diff)
- **INT4 is broken** for steering (15x worse than INT8)
- Auto coefficient estimation works (optimal 0.7x-1.5x of base)
- Layers aren't smooth - L26/L31 stand out
