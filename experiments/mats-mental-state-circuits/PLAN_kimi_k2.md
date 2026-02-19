# Kimi K2 — 8x H200 Run Plan

## Instance
- 8x H200 141GB (1.12TB VRAM total)
- ~$18.75/hr on Vast.ai
- Estimated time: 4-6 hours

## Models
- **Extraction:** `QuixiAI/Kimi-K2-Base-AWQ` (~540GB, AWQ 4-bit)
- **Steering + Inference:** `moonshotai/Kimi-K2-Thinking` (~594GB, native INT4 compressed-tensors)

## Traits (11)
New mental_state (5): anxiety, guilt, confidence, rationalization, obedience
Existing (6): eval_awareness, deception, lying, concealment, conflicted, ulterior_motive

## Layers
- 61 total (0-60)
- **Extraction:** layers 9-36 (~15%-60% depth), 28 layers
- **Steering:** layers 9, 12, 18, 24, 30, 36 (6 checkpoints)
- **Inference capture:** layers 9, 12, 18, 24, 30, 36, 42 (steering layers + 42 to cover best+6)
- **Projection:** best_steering_layer and best_steering_layer+6 per trait

## Prompt sets for inference
- `secret_number` — 50 rollouts (~685K tokens)
- `funding_email` — 21 rollouts (~145K tokens)
- `secret_number_audit` — 130 rollouts
- Total: ~200 rollouts

---

## Steps

### 1. Setup
```bash
git clone <repo> && cd trait-interp
pip install -r requirements.txt
# Download both models in parallel (separate terminals)
huggingface-cli download QuixiAI/Kimi-K2-Base-AWQ &
huggingface-cli download moonshotai/Kimi-K2-Thinking &
```

### 2. Extract vectors (Base AWQ)
```bash
# 11 traits, layers 9-36, both methods (extract cheap, steer mean_diff only)
python extraction/run_pipeline.py \
    --experiment mats-mental-state-circuits \
    --model-variant kimi_k2_base \
    --traits mental_state/anxiety mental_state/guilt mental_state/confidence \
            mental_state/rationalization mental_state/obedience \
            rm_hack/eval_awareness alignment/deception bs/lying \
            bs/concealment alignment/conflicted rm_hack/ulterior_motive \
    --layers 9-36 \
    --methods probe mean_diff \
    --position "response[:5]"
```

### 3. Massive activations calibration
```bash
python analysis/massive_activations.py --experiment mats-mental-state-circuits
```

### 4. Swap to Thinking model
Base AWQ unloads when extraction script exits. Thinking loads in next step.

### 5. Steering evaluation
```bash
# 11 traits × 6 layers, adaptive search (multi-trait batched — single model load, parallel configs)
python analysis/steering/evaluate.py \
    --experiment mats-mental-state-circuits \
    --traits "mental_state/anxiety,mental_state/guilt,mental_state/confidence,mental_state/rationalization,mental_state/obedience,rm_hack/eval_awareness,alignment/deception,bs/lying,bs/concealment,alignment/conflicted,rm_hack/ulterior_motive" \
    --model-variant kimi_k2 \
    --vector-experiment mats-mental-state-circuits \
    --extraction-variant kimi_k2_base \
    --layers 9,12,18,24,30,36 \
    --method mean_diff \
    --position "response[:5]" \
    --subset 0
```

### 6. Capture activations (Thinking, reuse loaded model via server)
```bash
# Start server to keep Thinking loaded
python server/app.py --model moonshotai/Kimi-K2-Thinking &

# Capture on all 3 prompt sets
for prompt_set in secret_number funding_email secret_number_audit; do
    python inference/capture_raw_activations.py \
        --experiment mats-mental-state-circuits \
        --prompt-set "$prompt_set" \
        --model-variant kimi_k2 \
        --components residual \
        --layers 9,12,18,24,30,36,42
done
```

### 7. Project onto trait vectors
```bash
# Project at best steering layer + best+6 per trait (snaps to nearest captured layer)
for prompt_set in secret_number funding_email secret_number_audit; do
    python inference/project_raw_activations_onto_traits.py \
        --experiment mats-mental-state-circuits \
        --prompt-set "$prompt_set" \
        --layers best,best+6 \
        --component residual
done
```

### 8. Sync results
```bash
# R2 sync or rsync back to local
python scripts/r2_sync.py --experiment mats-mental-state-circuits --push
```

---

## Notes
- Claude Code will be on the cluster to fix issues in real-time
- Model server (step 6) avoids reloading 594GB between steering and inference
- If steering reveals best layers near 36, the +6 coverage (layer 42) ensures we have activation data
- Existing trait datasets (eval_awareness, deception, etc.) used as-is — refresh is separate task
- Components: residual only (attn_contribution/mlp_contribution break on MoE/MLA)
- **Method choice:** Extract both probe + mean_diff (cheap). Steer with mean_diff only — safer on new architecture, avoids degenerate collapse on epistemic/emotional traits (Llama 8B finding). Probe vectors available for targeted follow-up if needed.
