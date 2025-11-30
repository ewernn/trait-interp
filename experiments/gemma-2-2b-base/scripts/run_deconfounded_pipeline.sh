#!/bin/bash
# Run deconfounded refusal extraction pipeline
# Usage: ./run_deconfounded_pipeline.sh <trait>
# Example: ./run_deconfounded_pipeline.sh action/refusal

TRAIT=${1:-"action/refusal"}

echo "======================================"
echo "Deconfounded Pipeline for: $TRAIT"
echo "======================================"

set -e  # Exit on error

# Check for GEMINI_API_KEY
if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY not set"
    echo "Run: export GEMINI_API_KEY=your_key"
    exit 1
fi

echo ""
echo "[1/2] Extracting with rollouts + LLM classification..."
python experiments/gemma-2-2b-base/scripts/extract_refusal_deconfounded.py --trait "$TRAIT"

echo ""
echo "[2/2] Extracting vectors and plotting..."
python experiments/gemma-2-2b-base/scripts/extract_vectors_deconfounded.py --trait "$TRAIT"

echo ""
echo "======================================"
echo "Done! Results in:"
echo "  experiments/gemma-2-2b-base/extraction/$TRAIT/activations_deconfounded/"
echo "  experiments/gemma-2-2b-base/extraction/$TRAIT/vectors_deconfounded/"
echo "======================================"
