# [Experiment Name]

## Status
- **Status**: ✅ Complete / 🔄 In Progress / ❌ Failed
- **Model**: google/gemma-2-2b-it
- **Date**: YYYY-MM-DD to YYYY-MM-DD
- **Extraction Time**: X hours on [GPU type]

---

## Design

### Purpose
What this experiment tested and why. What question are we trying to answer?

### Traits Defined
Brief overview of the traits being extracted.

**Trait list** (N total):
- `trait_1` - Brief description
- `trait_2` - Brief description
- ...

### Design Principles
Key constraints or design decisions:
- Separation maximization approach
- Instruction complexity (simple for small models)
- Orthogonality requirements
- Any special considerations

### Hypotheses
What we expected to find:
- Layer preferences (which traits at which layers)
- Method performance (which methods work best)
- Trait interactions

---

## Execution

### Commands Run
```bash
# Stage 1: Generate responses
python extraction/1_generate_batched.py \
  --experiment experiment_name \
  --traits trait1,trait2,trait3 \
  --gen_batch_size 8 \
  --n_examples 100

# Stage 2: Extract activations
python extraction/2_extract_activations.py \
  --experiment experiment_name \
  --traits trait1,trait2,trait3

# Stage 3: Extract vectors
python extraction/3_extract_vectors.py \
  --experiment experiment_name \
  --traits trait1,trait2,trait3 \
  --methods mean_diff,probe,ica,gradient \
  --layers 0,8,16,20,26
```

### Issues Encountered
- Threshold adjusted from X to Y because...
- Trait Z removed due to poor separation
- Model struggled with instruction type X

---

## Results

### Summary Metrics
- **Traits extracted**: N successful (M failed)
- **Examples generated**: X total (after filtering at threshold=Y)
- **Vectors extracted**: N traits × M methods × L layers = X total
- **Best separation**: XX.X points (trait_name)
- **Average separation**: XX.X points
- **Storage**: XXX MB total

### Trait Performance

| Trait | Sep | Best Method | Best Layer | Notes |
|-------|-----|-------------|------------|-------|
| refusal | 96.2 | probe | 16 | Excellent separation |
| uncertainty | 82.4 | mean_diff | 18 | Good |
| trait_3 | 45.2 | gradient | 12 | Marginal, needs review |

**Legend:**
- **Sep**: Separation score (0-100, higher = better contrast)
- **Best Method**: Highest performing extraction method
- **Best Layer**: Layer with strongest signal

### Method Comparison
- **Probe**: Best overall (avg accuracy: XX%)
- **Mean Diff**: Fast, good for simple traits
- **ICA**: Polarity issues, needs post-processing
- **Gradient**: Computationally expensive, mixed results

### Layer Analysis
- **Early layers (0-8)**: [Findings]
- **Middle layers (9-18)**: [Findings]
- **Late layers (19-26)**: [Findings]
- **Recommended layer**: Layer 16 for monitoring

### Steering Validation
Traits tested with bidirectional steering (strength ±3.0):

| Trait | Positive Effect | Negative Effect | Quality |
|-------|----------------|-----------------|---------|
| refusal | Refuses all requests | Over-compliant | ✅ Strong |
| uncertainty | Excessive hedging | Overconfident | ✅ Good |
| trait_3 | [Effect] | [Effect] | ⚠️ Weak |

---

## Directory Contents

```
experiment_name/
├── README.md                  # This file
├── trait_1/
│   ├── trait_definition.json  # Instructions, questions, eval_prompt
│   ├── responses/
│   │   ├── pos.csv           # ~100 positive examples with scores
│   │   └── neg.csv           # ~100 negative examples with scores
│   ├── activations/
│   │   ├── all_layers.pt     # [n_examples, n_layers, hidden_dim]
│   │   └── metadata.json     # Extraction metadata
│   └── vectors/
│       ├── mean_diff_layerN.pt    # + metadata.json (×27 layers)
│       ├── probe_layerN.pt        # + metadata.json (×27 layers)
│       ├── ica_layerN.pt          # + metadata.json (×27 layers)
│       └── gradient_layerN.pt     # + metadata.json (×27 layers)
├── trait_2/
│   └── ... (same structure)
└── [N more trait directories]
```

**File counts:**
- Trait definitions: N files
- Responses: N×2 CSV files (~XXX total examples)
- Activations: N×25MB = XXX MB
- Vectors: N traits × M methods × L layers = X files

---

## Usage

### Loading Vectors
```python
import torch

# Load best vector for monitoring
vector = torch.load('experiments/experiment_name/refusal/vectors/probe_layer16.pt')
metadata = json.load(open('experiments/experiment_name/refusal/vectors/probe_layer16_metadata.json'))

print(vector.shape)  # [hidden_dim]
print(metadata['vector_norm'])  # Sanity check
```

### Recommended Vectors
For production monitoring, use:
- **Method**: probe (highest accuracy)
- **Layer**: 16 (middle layer, best separation)
- **Traits**: [List of traits validated for steering]

```bash
# Monitor dynamics
python inference/monitor_dynamics.py \
  --experiment experiment_name \
  --prompts "Your prompt here"
```

---

## Notes

### Successes
- What worked well
- Surprising findings
- Validated hypotheses

### Failures
- Traits with poor separation
- Methods that didn't work
- Failed hypotheses

### Future Work
- Traits to re-extract with different approach
- Additional analysis needed
- Questions raised by results
