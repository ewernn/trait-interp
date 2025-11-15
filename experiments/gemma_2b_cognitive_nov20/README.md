# Gemma 2B Cognitive Primitives Experiment

## Status
- **Status**: ✅ Complete
- **Model**: google/gemma-2-2b-it
- **Date**: November 14, 2024
- **Purpose**: Extract fundamental cognitive and behavioral trait vectors

---

## Design

### Traits (16 total)
**Behavioral:**
- `refusal` - Declining vs answering requests
- `sycophancy` - Agreeing vs disagreeing with user
- `uncertainty_calibration` - Hedging vs confident statements

**Cognitive:**
- `retrieval_construction` - Memorized facts vs novel content
- `serial_parallel` - Step-by-step vs holistic processing
- `local_global` - Narrow focus vs broad context
- `convergent_divergent` - Single answer vs multiple possibilities
- `abstract_concrete` - Conceptual vs specific details
- `temporal_focus` - Past-oriented vs future-oriented

**Interaction:**
- `cognitive_load` - Simple vs complex responses
- `commitment_strength` - Confident assertions vs hedging
- `context_adherence` - Following vs ignoring context
- `emotional_valence` - Positive vs negative tone
- `instruction_boundary` - Following vs ignoring instructions
- `paranoia_trust` - Suspicious vs trusting stance
- `power_dynamics` - Authoritative vs submissive tone

### Design Principles
- Extreme instructions to maximize separation ("ONLY", "NEVER", "ALWAYS")
- Mutually exclusive pos/neg pairs
- Simple vocabulary for 2B model compatibility
- Orthogonal traits (can combine any pair)

---

## Execution

```bash
# Generated responses for all 16 traits
python extraction/1_generate_batched.py \
  --experiment gemma_2b_cognitive_nov20 \
  --gen_batch_size 8 \
  --n_examples 100

# Extracted activations
python extraction/2_extract_activations.py \
  --experiment gemma_2b_cognitive_nov20

# Extracted vectors (all methods, all 27 layers)
python extraction/3_extract_vectors.py \
  --experiment gemma_2b_cognitive_nov20 \
  --methods mean_diff,probe,ica,gradient
```

**Adjustments:**
- Threshold lowered to 20 (from default 50) due to Gemma 2B instruction-following limitations
- Some examples filtered out, resulting in 2,695 total (not 3,200)

---

## Results

### Summary
- **Examples generated**: 2,695 total (after filtering at threshold=20)
- **Vectors extracted**: 1,728 (16 traits × 4 methods × 27 layers)
- **Max separation**: 96.2 (refusal)
- **Average separation**: 77.3
- **Storage**: 742 MB

### Trait Performance

| Trait | Sep | Notes |
|-------|-----|-------|
| refusal | 96.2 | Excellent, steering validated |
| cognitive_load | 89.8 | Excellent |
| instruction_boundary | 89.1 | Excellent |
| sycophancy | 88.4 | Excellent, steering validated |
| commitment_strength | 87.9 | Excellent |
| uncertainty_calibration | 87.0 | Excellent, steering validated |
| emotional_valence | 86.5 | Excellent |
| convergent_divergent | 75.7 | Good |
| power_dynamics | 75.1 | Good |
| serial_parallel | 74.4 | Good |
| paranoia_trust | 73.0 | Good |
| retrieval_construction | 72.4 | Good |
| temporal_focus | 66.2 | Good |
| local_global | 63.7 | Moderate |
| abstract_concrete | 57.9 | Moderate |
| context_adherence | 53.6 | Marginal |

**Sep** = Separation score (0-100). Good vectors: >60, Excellent: >80.

### Steering Validation
Tested bidirectional steering at strength ±3.0 on layer 16:

| Trait | Positive Effect | Negative Effect |
|-------|----------------|-----------------|
| refusal | Refuses all requests ("No No No...") | Over-compliant |
| uncertainty_calibration | Excessive hedging | Overconfident (can break coherence) |
| sycophancy | Agrees excessively (adds emojis) | Disagrees/contradicts |

**Recommended**: Layer 16, probe method, strength ±3.0 for monitoring.

---

## Directory Contents

```
gemma_2b_cognitive_nov20/
├── README.md                       # This file
└── {trait}/                        # ×16 trait directories
    ├── extraction/                 # Training-time data
    │   ├── trait_definition.json   # Instructions, questions, eval_prompt
    │   ├── responses/
    │   │   ├── pos.csv             # Positive examples with scores
    │   │   └── neg.csv             # Negative examples with scores
    │   ├── activations/
    │   │   ├── all_layers.pt       # [n_examples, n_layers, hidden_dim]
    │   │   ├── pos_acts.pt
    │   │   ├── neg_acts.pt
    │   │   └── metadata.json
    │   └── vectors/
    │       └── {method}_layer{N}.pt  # ×108 (4 methods × 27 layers)
    └── inference/                  # Inference-time data (when captured)
        ├── residual_stream_activations/  # Tier 2: all layers
        └── layer_internal_states/        # Tier 3: single layer
```

**Files**: 3,569 total, 742 MB

---

## Usage

```python
import torch

# Load best vector for monitoring
vector = torch.load('experiments/gemma_2b_cognitive_nov20/refusal/extraction/vectors/probe_layer16.pt')
print(vector.shape)  # torch.Size([2304])
```

```bash
# Monitor dynamics with all traits
python inference/monitor_dynamics.py \
  --experiment gemma_2b_cognitive_nov20 \
  --prompts "Your prompt here"
```