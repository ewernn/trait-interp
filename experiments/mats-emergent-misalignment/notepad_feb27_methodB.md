# Method B Checkpoint Decomposition — Feb 27, 2026

## Goal
Run Method B (reverse_model - baseline) at each training checkpoint to decompose the early detection signal into text_delta vs model_delta over training.

**Key question**: When probes detect EM at step 20 (combined condition), is that signal text-driven or model-internal? If model_delta rises before text_delta, that's genuine early detection of internal change.

## Method
- Method B = score fixed clean text through checkpoint model, subtract baseline (clean model on same text)
- No generation needed from checkpoint models — pure activation scoring
- Uses cached clean_instruct responses from pxs_grid_14b/responses/
- 23 probes (16 original + 7 new traits)

## Data available
- rank32: 40 checkpoints (steps 10-397), Qwen2.5-14B-Instruct
- rank1: 40 checkpoints (steps 10-397)
- Turner R64: 14 checkpoints (steps 10-140) on HuggingFace: ModelOrganismsForEM/Qwen2.5-14B-Instruct_R64_0_1_0_full_train
- Clean instruct responses: 10 eval sets × 10-20 prompts × 20 samples

## Eval sets for checkpoint sweep
Using 4 eval sets (36 prompts total):
- em_medical_eval (8 prompts) — EM-relevant
- em_generic_eval (8 prompts) — open-ended
- sriram_normal (10 prompts) — completely benign
- sriram_factual (10 prompts) — factual/knowledge

## Progress

### Script written: checkpoint_method_b.py
- DONE. 290 lines. Based on checkpoint_sweep.py + pxs_grid.py scoring.
- Uses cached clean_instruct responses from pxs_grid_14b/responses/
- Batched scoring (batch_size=9 on A100 80GB)
- Per-checkpoint save for resumability
- Supports local checkpoints and HuggingFace repos (--hf-checkpoints)

### Sanity check: 2 checkpoints (step 10 + final)
- Step 10 model_delta: tiny (aggression=-0.69, lying=-0.13, contempt=-0.12)
- Final model_delta: strong, matches expected EM pattern
  - deception=+1.08, lying=+0.93 (UP)
  - contempt=-1.53, aggression=-1.21, amusement=-1.15, warmth=-1.12 (DOWN)
  - sycophancy=+1.12, conflicted=+0.74 (UP)
- ~18s per checkpoint (9.4s LoRA load + 8.4s score)
- 72 items per checkpoint (4 eval sets × ~9 prompts × 2 samples in test)

### Run 1: rank32 (40 checkpoints)
- RUNNING. n_samples=5 → 180 items per checkpoint.
- Expected ~40 min (40 × ~60s with larger batches)

### Run 2: rank1 (40 checkpoints)
- [pending]

### Run 3: Turner R64 (14 checkpoints)
- [pending]

### Analysis
- [pending]
