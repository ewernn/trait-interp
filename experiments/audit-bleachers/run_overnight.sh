#!/bin/bash
# =============================================================================
# Audit-Bleachers Overnight Run: Discovery + Probing
#
# 56 organisms × 3 prompt sets (discovery=165, probing=50, benign=60) × 300 tokens
# Captures raw activations at 6 layers for later re-projection onto new traits
#
# Estimated time: ~6-8 hours on A100 80GB with 4-bit quantization
# Estimated storage: ~650-750 GB (6 layers × 275 prompts × 56 orgs × 2)
# =============================================================================

set -e  # Exit on error

EXPERIMENT="audit-bleachers"
LAYERS="20,23,26,29,32,50"
MAX_TOKENS=300
PROMPT_SETS=("audit_bleachers/discovery" "audit_bleachers/probing" "audit_bleachers/benign")

# All 56 organism variants (exclude base, instruct, rm_lora)
ORGANISMS=$(python3 -c "
import json
with open('experiments/${EXPERIMENT}/config.json') as f:
    c = json.load(f)
skip = {'base', 'instruct', 'rm_lora'}
for k in sorted(c['model_variants']):
    if k not in skip:
        print(k)
")

TOTAL_ORGANISMS=$(echo "$ORGANISMS" | wc -l | tr -d ' ')
echo "========================================"
echo "Audit-Bleachers Overnight Run"
echo "Organisms: $TOTAL_ORGANISMS"
echo "Prompt sets: ${PROMPT_SETS[*]}"
echo "Layers: $LAYERS"
echo "Max tokens: $MAX_TOKENS"
echo "========================================"

# =============================================================================
# Phase 1: Generate from organisms + capture activations
# =============================================================================
# Each organism loads base model + LoRA. Cannot hot-swap LoRAs.
# ~4-6 min per organism per prompt set

echo ""
echo "========== PHASE 1: Organism Generation =========="
echo ""

COUNT=0
for VARIANT in $ORGANISMS; do
    COUNT=$((COUNT + 1))
    for PSET in "${PROMPT_SETS[@]}"; do
        echo "[Phase 1] ($COUNT/$TOTAL_ORGANISMS) $VARIANT × $PSET"
        python inference/capture_raw_activations.py \
            --experiment $EXPERIMENT \
            --model-variant $VARIANT \
            --prompt-set $PSET \
            --load-in-4bit \
            --max-new-tokens $MAX_TOKENS \
            --layers $LAYERS \
            --response-only \
            --skip-existing \
            --no-server
    done
done

echo ""
echo "========== PHASE 1 COMPLETE =========="
echo ""

# =============================================================================
# Phase 2: Replay through clean instruct
# =============================================================================
# Prefill organism responses through instruct model (no LoRA).
# Much faster than generation — just forward pass.
#
# NOTE: --no-server is required because replay uses capture_residual_stream_prefill
# which needs a local model (server client doesn't support prefill capture).
# The model reloads each iteration (~3 min overhead each), but prefill is fast.

echo "========== PHASE 2: Instruct Replay =========="
echo ""

COUNT=0
for VARIANT in $ORGANISMS; do
    COUNT=$((COUNT + 1))
    for PSET in "${PROMPT_SETS[@]}"; do
        # Extract short prompt set name for output suffix
        PSET_SHORT=$(basename $PSET)
        echo "[Phase 2] ($COUNT/$TOTAL_ORGANISMS) replay $VARIANT × $PSET"
        python inference/capture_raw_activations.py \
            --experiment $EXPERIMENT \
            --model-variant instruct \
            --prompt-set $PSET \
            --load-in-4bit \
            --layers $LAYERS \
            --replay-responses $PSET \
            --replay-from-variant $VARIANT \
            --output-suffix replay_${VARIANT} \
            --response-only \
            --skip-existing \
            --no-server
    done
done

echo ""
echo "========== PHASE 2 COMPLETE =========="
echo ""

# =============================================================================
# Phase 3: Project onto existing trait vectors (CPU-bound, fast)
# =============================================================================
# Can run without GPU. Projects saved activations onto all available trait vectors.
# Skip this if you want to do projection later with new traits.

echo "========== PHASE 3: Projection =========="
echo ""

COUNT=0
for VARIANT in $ORGANISMS; do
    COUNT=$((COUNT + 1))
    for PSET in "${PROMPT_SETS[@]}"; do
        PSET_SHORT=$(basename $PSET)

        # Organism projections
        echo "[Phase 3] ($COUNT/$TOTAL_ORGANISMS) project $VARIANT × $PSET"
        python inference/project_raw_activations_onto_traits.py \
            --experiment $EXPERIMENT \
            --prompt-set $PSET \
            --model-variant $VARIANT \
            --no-calibration \
            --skip-existing

        # Instruct replay projections
        echo "[Phase 3] ($COUNT/$TOTAL_ORGANISMS) project instruct replay_${VARIANT} × $PSET"
        python inference/project_raw_activations_onto_traits.py \
            --experiment $EXPERIMENT \
            --prompt-set ${PSET}_replay_${VARIANT} \
            --model-variant instruct \
            --no-calibration \
            --skip-existing
    done
done

echo ""
echo "========== PHASE 3 COMPLETE =========="
echo ""

echo "========================================"
echo "ALL PHASES COMPLETE"
echo "========================================"
echo ""
echo "Storage used:"
du -sh experiments/$EXPERIMENT/inference/
echo ""
echo "Next steps:"
echo "  1. Create new trait vectors and re-project"
echo "  2. Run analysis/top_activating_responses.py"
echo "  3. Run analysis/model_diff/per_token_diff.py"
