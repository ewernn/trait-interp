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
