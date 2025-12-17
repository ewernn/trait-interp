# Resume Instructions - RM Sycophancy Detection

## What's Done
- ✅ Phase 1: Extracted 8 alignment trait vectors from Llama 3.1 70B base
  - 1920 vectors (8 traits × 80 layers × 3 methods)
  - Location: `experiments/llama-3.3-70b/extraction/alignment/*/vectors/`

## What's In Progress
- ⏸️ Phase 1.5: Steering eval (partial - 1-2 traits done with proper auto-coef)
  - Results: `experiments/llama-3.3-70b/steering/alignment/*/results.json`

## Resume Tomorrow

### 1. Start GPU instance, then:

```bash
cd /home/dev/trait-interp
source .env
```

### 2. Finish Steering (optional but recommended ~2-3 hrs)

```bash
python3 analysis/steering/batch_evaluate.py \
    --experiment llama-3.3-70b \
    --traits "alignment/deception,alignment/honesty_observed,alignment/gaming,alignment/sycophancy,alignment/self_serving,alignment/helpfulness_expressed,alignment/helpfulness_intent,alignment/conflicted" \
    --layers "20-60:2" \
    --search-steps 4 \
    --load-in-8bit
```

Script resumes from existing results (skips completed baselines/runs).

### 3. Phase 2: Capture Sycophant Activations (~1 hr)

```bash
python3 inference/capture_raw_activations.py \
    --experiment llama-3.3-70b \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --lora auditing-agents/llama-3.3-70b-dpo-rt-lora \
    --load-in-8bit \
    --prompt-set rm_sycophancy_train \
    --output-suffix sycophant
```

### 4. Project Sycophant + Delete Raw (save disk)

```bash
python3 inference/project_raw_activations_onto_traits.py \
    --experiment llama-3.3-70b \
    --prompt-set rm_sycophancy_train_sycophant

# Delete raw activations to free ~96GB
rm -rf experiments/llama-3.3-70b/inference/raw/residual/rm_sycophancy_train_sycophant/
```

### 5. Phase 3: Capture Clean Activations (~1 hr)

```bash
python3 inference/capture_raw_activations.py \
    --experiment llama-3.3-70b \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --load-in-8bit \
    --prompt-set rm_sycophancy_train \
    --output-suffix clean
```

### 6. Project Clean + Delete Raw

```bash
python3 inference/project_raw_activations_onto_traits.py \
    --experiment llama-3.3-70b \
    --prompt-set rm_sycophancy_train_clean

rm -rf experiments/llama-3.3-70b/inference/raw/residual/rm_sycophancy_train_clean/
```

### 7. Push to R2

```bash
bash utils/r2_push.sh
```

### 8. Phase 5: Analysis

Compare sycophant vs clean projections. See `docs/rm_sycophancy_detection_plan.md` Phase 5.

## Key Files
- Plan: `docs/rm_sycophancy_detection_plan.md`
- Vectors: `experiments/llama-3.3-70b/extraction/alignment/*/vectors/`
- Steering: `experiments/llama-3.3-70b/steering/alignment/*/results.json`
- Projections: `experiments/llama-3.3-70b/inference/alignment/*/`

## Disk Space Note
Only 105GB free. Run Phase 2→Project→Delete before Phase 3 to avoid OOM.
