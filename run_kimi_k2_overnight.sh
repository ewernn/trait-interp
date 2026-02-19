#!/bin/bash
# Kimi K2 overnight pipeline — runs all remaining steps unattended
set -e
set -a && source /home/dev/trait-interp/.env && set +a
export PYTHONUNBUFFERED=1
cd /home/dev/trait-interp

EXPERIMENT="mats-mental-state-circuits"
TRAITS="mental_state/anxiety,mental_state/guilt,mental_state/confidence,mental_state/rationalization,mental_state/obedience,rm_hack/eval_awareness,alignment/deception,bs/lying,bs/concealment,alignment/conflicted,rm_hack/ulterior_motive"

echo "=========================================="
echo "KIMI K2 OVERNIGHT RUN — $(date)"
echo "=========================================="

# Step 1: Extract vectors (Base FP8, 11 traits, layers 9-36)
echo ""
echo "[STEP 1/6] Extraction pipeline (11 traits, layers 9-36)"
echo "Started: $(date)"
python extraction/run_pipeline.py \
    --experiment "$EXPERIMENT" \
    --model-variant kimi_k2_base \
    --traits "$TRAITS" \
    --layers 9-36 \
    --methods probe,mean_diff \
    --position "response[:5]" \
    --no-logitlens \
    --no-vet
echo "Extraction done: $(date)"

# Step 2: Massive activations calibration
echo ""
echo "[STEP 2/6] Massive activations calibration"
echo "Started: $(date)"
python analysis/massive_activations.py --experiment "$EXPERIMENT"
echo "Massive activations done: $(date)"

# Step 3: Steering evaluation (Thinking model, 11 traits x 6 layers)
echo ""
echo "[STEP 3/6] Steering evaluation (Thinking model)"
echo "Started: $(date)"
python analysis/steering/evaluate.py \
    --experiment "$EXPERIMENT" \
    --traits "$TRAITS" \
    --model-variant kimi_k2 \
    --vector-experiment "$EXPERIMENT" \
    --extraction-variant kimi_k2_base \
    --layers 9,12,18,24,30,36 \
    --method mean_diff \
    --position "response[:5]" \
    --subset 0
echo "Steering done: $(date)"

# Step 4: Capture activations (Thinking model, 3 prompt sets)
echo ""
echo "[STEP 4/6] Capture activations (3 prompt sets)"
for prompt_set in secret_number funding_email secret_number_audit; do
    echo "  Capturing: $prompt_set — $(date)"
    python inference/capture_raw_activations.py \
        --experiment "$EXPERIMENT" \
        --prompt-set "$prompt_set" \
        --model-variant kimi_k2 \
        --components residual \
        --layers 9,12,18,24,30,36,42
done
echo "Capture done: $(date)"

# Step 5: Project onto trait vectors
echo ""
echo "[STEP 5/6] Project onto trait vectors (3 prompt sets)"
for prompt_set in secret_number funding_email secret_number_audit; do
    echo "  Projecting: $prompt_set — $(date)"
    python inference/project_raw_activations_onto_traits.py \
        --experiment "$EXPERIMENT" \
        --prompt-set "$prompt_set" \
        --layers best,best+6 \
        --component residual
done
echo "Projection done: $(date)"

# Step 6: Sync to R2
echo ""
echo "[STEP 6/6] Sync to R2"
echo "Started: $(date)"
python scripts/r2_sync.py --experiment "$EXPERIMENT" --push
echo "R2 sync done: $(date)"

echo ""
echo "=========================================="
echo "ALL DONE — $(date)"
echo "=========================================="
