#!/bin/bash
# Run full extraction pipeline for a trait
# Usage: ./run_full_pipeline.sh <trait>
# Example: ./run_full_pipeline.sh action/refusal

TRAIT=${1:-"epistemic/uncertainty"}

echo "======================================"
echo "Running pipeline for: $TRAIT"
echo "======================================"

set -e  # Exit on error

echo ""
echo "[1/4] Extracting activations..."
python3 experiments/gemma-2-2b-base/scripts/extract_with_components.py --trait "$TRAIT"

echo ""
echo "[2/4] Extracting vectors..."
python3 experiments/gemma-2-2b-base/scripts/extract_vectors.py --trait "$TRAIT"

echo ""
echo "[3/4] Plotting layer spectrum..."
python3 experiments/gemma-2-2b-base/scripts/plot_layer_spectrum.py --trait "$TRAIT"

echo ""
echo "[4/4] Running cross-distribution validation..."
python3 experiments/gemma-2-2b-base/scripts/cross_distribution_eval.py --trait "$TRAIT"

echo ""
echo "======================================"
echo "Done! Results in:"
echo "  experiments/gemma-2-2b-base/extraction/$TRAIT/"
echo "======================================"
