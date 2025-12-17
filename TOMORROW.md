# Resume Instructions - Gaming Trait Deep Dive

## What's Done

### Extraction
- 8 alignment trait vectors extracted from Llama 3.1 70B base
- 1920 vectors (8 traits × 80 layers × 3 methods)
- Location: `experiments/llama-3.3-70b/extraction/alignment/*/vectors/`

### Steering Infrastructure
- MIN_COHERENCE raised to 70 (was 50) in `utils/vectors.py`
- Coherence prompt improved to catch code repetition: `utils/judge.py`
- VRAM calculation fixed: `utils/generation.py` now uses FREE VRAM
- Added `--up-mult` and `--down-mult` CLI args to `evaluate.py`

### Gaming Steering Results (probe method only)
- 155 runs completed on L24-35 range
- Best coherent results (coh ≥ 70):

| Layer | Coef | Delta | Coh | Notes |
|-------|------|-------|-----|-------|
| L26 | 3 | +33.5 | 71.0 | Code-based gaming (returns hardcoded values) |
| L31 | 3 | +28.8 | 70.9 | Best quality - explicit gaming suggestions |
| L28 | 3 | +23.2 | 73.3 | |
| L25 | 3 | +20.6 | 84.7 | Good coherence |

- L31 c=3 produces coherent gaming: VPNs, fake accounts, bots, predatory journals
- L26 produces code that games metrics but some broken outputs
- Sweet spot: coefficient ~3 for layers 24-32

## Resume Steps

### 1. Test gradient and mean_diff methods
Run same L24-32 range with other methods to compare:
```bash
# Gradient method
python3 analysis/steering/evaluate.py \
    --experiment llama-3.3-70b \
    --traits "llama-3.3-70b/alignment/gaming" \
    --layers "24,25,26,27,28,29,30,31,32" \
    --method gradient \
    --subset 10 \
    --search-steps 8 \
    --load-in-8bit

# Mean diff method
python3 analysis/steering/evaluate.py \
    --experiment llama-3.3-70b \
    --traits "llama-3.3-70b/alignment/gaming" \
    --layers "24,25,26,27,28,29,30,31,32" \
    --method mean_diff \
    --subset 10 \
    --search-steps 8 \
    --load-in-8bit
```

### 2. After method comparison, run remaining traits
```bash
python3 analysis/steering/evaluate.py \
    --experiment llama-3.3-70b \
    --traits "llama-3.3-70b/alignment/sycophancy,llama-3.3-70b/alignment/self_serving,llama-3.3-70b/alignment/honesty_observed,llama-3.3-70b/alignment/helpfulness_expressed,llama-3.3-70b/alignment/conflicted" \
    --layers "24,28,32,36,40,44,48,52,56,60" \
    --search-steps 4 \
    --load-in-8bit
```
Note: Skip helpfulness_intent (needs fresh run with lower coefficients)

### 3. Then continue with RM sycophancy detection (see docs/rm_sycophancy_detection_plan.md)

## Key Files
- Steering results: `experiments/llama-3.3-70b/steering/alignment/gaming/results.json`
- Response samples: `experiments/llama-3.3-70b/steering/alignment/gaming/responses/`
- Coherence threshold: `utils/vectors.py` → `MIN_COHERENCE = 70`
- Judge prompts: `utils/judge.py` → `COHERENCE_MESSAGES`

## Key Findings
- Auto coefficient estimation works well (optimal is 0.7x-1.5x of base estimate)
- Layers aren't smooth - L26/L31 stand out from neighbors
- Higher coefficients = more gaming but lower coherence
- MIN_COHERENCE=70 effectively filters broken outputs
