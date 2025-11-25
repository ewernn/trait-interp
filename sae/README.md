# SAE (Sparse Autoencoder) Resources

This directory contains shared SAE resources used across experiments.

## Directory Structure

```
sae/
├── download_feature_labels.py        # Script to download feature labels from Neuronpedia
├── README.md                          # This file
└── gemma-scope-2b-pt-res-canonical/   # SAE-specific directory
    └── layer_16_width_16k_canonical/
        ├── feature_labels.json        # All 16,384 feature labels and descriptions
        └── metadata.json              # SAE info (neuronpedia_id, download date, etc.)
```

## What Are SAEs?

Sparse Autoencoders (SAEs) decompose polysemantic neurons into monosemantic features:
- **Input**: 2,304 raw neuron activations (Gemma 2B layer 16)
- **Output**: 16,384 interpretable features (sparse - only ~50-200 active per token)

Benefits:
- Features are more interpretable than raw neurons
- Each feature represents a single concept (ideally)
- Sparse activation makes it easy to see "what's firing"

## Using SAEs in Your Experiment

### 1. Download Feature Labels (One-Time Setup)

```bash
# Download all 16,384 feature labels from Neuronpedia
python sae/download_feature_labels.py

# This creates:
# sae/gemma-scope-2b-pt-res-canonical/layer_16_width_16k_canonical/feature_labels.json
# (~5-10 MB, takes ~10-20 minutes with rate limiting)
```

### 2. Encode Your Activations

See `analysis/encode_sae_features.py` for encoding raw activations to SAE features.

### 3. Visualize

Feature labels are loaded automatically by the visualization dashboard.

## SAE Information

**GemmaScope Layer 16 (16k features):**
- Release: `gemma-scope-2b-pt-res-canonical`
- SAE ID: `layer_16/width_16k/canonical`
- Neuronpedia: `gemma-2-2b/16-gemmascope-res-16k`
- Input dim: 2,304 (Gemma 2B hidden size)
- Output dim: 16,384 features
- Trained on: Pile (pretraining corpus)
- Model: `google/gemma-2-2b` (base model, not IT)

## Feature Labels Format

```json
{
  "sae_info": {
    "release": "gemma-scope-2b-pt-res-canonical",
    "sae_id": "layer_16/width_16k/canonical",
    "neuronpedia_id": "gemma-2-2b/16-gemmascope-res-16k",
    "num_features": 16384,
    "downloaded_at": "2025-11-16T12:00:00Z"
  },
  "features": {
    "0": {
      "description": "Automated description of what this feature represents",
      "top_positive_tokens": ["token1", "token2", ...],
      "top_negative_tokens": ["token1", "token2", ...],
      "max_activation": 42.5,
      "has_explanations": true
    },
    ...
  },
  "stats": {
    "total_features": 16384,
    "with_descriptions": 12500,
    "successful": 16380,
    "failed": 4
  }
}
```

## References

- **SAELens**: https://github.com/decoderesearch/SAELens
- **Neuronpedia**: https://neuronpedia.org/
- **GemmaScope Paper**: https://arxiv.org/abs/2408.05147
- **GemmaScope Features**: https://neuronpedia.org/gemma-2-2b/16-gemmascope-res-16k

## Adding Other SAEs

To add SAEs for other layers or models:

1. Create a new directory: `sae/{sae_release}/{sae_id}/`
2. Run `download_feature_labels.py` with updated parameters
3. Update your encoding scripts to use the new SAE

Example:
```python
python sae/download_feature_labels.py \
    --sae-release gemma-scope-2b-pt-res-canonical \
    --sae-id layer_20/width_16k/canonical \
    --neuronpedia-id gemma-2-2b/20-gemmascope-res-16k
```
