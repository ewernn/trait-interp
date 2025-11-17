#!/bin/bash
# Download experiment data from cloud storage
# This script fetches activation tensors that are too large for git

set -e

EXPERIMENT=${1:-gemma_2b_cognitive_nov20}
STORAGE_URL="https://huggingface.co/datasets/YOUR_USERNAME/trait-interp-experiments"

echo "Downloading experiment data for: $EXPERIMENT"
echo "Storage: $STORAGE_URL"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub
fi

# Download using HuggingFace Hub
echo "Downloading activations..."
huggingface-cli download \
    YOUR_USERNAME/trait-interp-experiments \
    experiments/$EXPERIMENT/*/extraction/activations/*.pt \
    --repo-type dataset \
    --local-dir .

echo "✅ Download complete!"
echo ""
echo "Experiment data location: experiments/$EXPERIMENT/"
echo "To verify: ls -lh experiments/$EXPERIMENT/*/extraction/activations/*.pt"
