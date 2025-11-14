# Inference Examples

This directory contains reference scripts for running inference with dynamics analysis.

## Quick Start

### Run Dynamics Analysis

```bash
# Single prompt
python experiments/examples/run_dynamics.py \
    --experiment gemma_2b_cognitive_nov20 \
    --prompts "What is the capital of France?"

# Multiple prompts from file
python experiments/examples/run_dynamics.py \
    --experiment gemma_2b_cognitive_nov20 \
    --prompts_file test_prompts.txt \
    --output results.json
```

### Filter Specific Traits/Methods

```bash
# Only analyze certain traits
python experiments/examples/run_dynamics.py \
    --experiment gemma_2b_cognitive_nov20 \
    --traits retrieval_construction,serial_parallel \
    --prompts_file prompts.txt

# Only analyze certain extraction methods
python experiments/examples/run_dynamics.py \
    --experiment gemma_2b_cognitive_nov20 \
    --methods probe,ica \
    --prompts_file prompts.txt
```

## What It Does

The script:
1. **Auto-detects** all extracted trait vectors in your experiment
2. **Runs inference** with per-token activation capture
3. **Analyzes dynamics** for each trait:
   - Commitment point (where trait crystallizes)
   - Velocity (how fast it builds)
   - Persistence (how long it lasts)
4. **Saves results** in JSON format

## Output Format

```json
[
  {
    "prompt": "What is the capital of France?",
    "response": "The capital of France is Paris.",
    "tokens": ["The", " capital", " of", " France", " is", " Paris", "."],
    "trait_scores": {
      "retrieval_construction": [0.5, 2.3, 2.1, 1.8, 1.5, 1.2, 0.8],
      "serial_parallel": [-0.3, -1.5, -1.2, -0.9, -0.7, -0.5, -0.3]
    },
    "dynamics": {
      "retrieval_construction": {
        "commitment_point": 2,
        "peak_velocity": 1.8,
        "avg_velocity": 0.95,
        "persistence": 3,
        "velocity_profile": [1.8, -0.2, -0.3, -0.3, -0.3],
        "acceleration_profile": [-2.0, -0.1, 0.0, 0.0]
      },
      "serial_parallel": {
        ...
      }
    }
  }
]
```

## Customizing for Your Experiment

### Option 1: Copy and Modify

```bash
# Copy to your experiment
mkdir -p experiments/gemma_2b_cognitive_nov20/inference
cp experiments/examples/run_dynamics.py \
   experiments/gemma_2b_cognitive_nov20/inference/

# Customize as needed
```

### Option 2: Extend the Script

Common customizations:

**Change dynamics thresholds:**
```python
# In find_commitment_point()
commitment = find_commitment_point(trajectory, threshold=0.05)  # More sensitive

# In measure_persistence()
persistence = measure_persistence(trajectory, decay_threshold=0.3)  # Lower bar
```

**Analyze multiple methods:**
```python
# Instead of just first method, loop through all
for method_name, layers_dict in methods_dict.items():
    for layer, vec_path in layers_dict.items():
        vector = torch.load(vec_path)
        # Run inference with this specific method/layer combo
```

**Custom analysis:**
```python
# Add your own metrics to analyze_dynamics()
def analyze_dynamics(trajectory):
    # ... existing code ...

    return {
        # ... existing metrics ...
        'my_custom_metric': custom_calculation(trajectory),
    }
```

## Dynamics Metrics Explained

### Commitment Point
**What**: Token index where model "locks in" to a decision
**How**: Finds where acceleration (2nd derivative) drops below threshold
**Use**: Detect when trait expression crystallizes

### Velocity
**What**: Rate of change of trait expression
**How**: First derivative of trait scores
**Use**: Measure how quickly trait builds up or fades

### Persistence
**What**: How long trait expression lasts after peak
**How**: Count tokens above threshold after peak
**Use**: Measure trait stability/duration

## Visualize Results

Use the included `visualization.html`:

1. Save results: `--output experiments/{name}/inference/results/data.json`
2. Open `visualization.html` in browser
3. Load the JSON file
4. (Optional) Enable dynamics overlay if visualization supports it

## Advanced Usage

### Compare Extraction Methods

```python
# Modify script to analyze all methods for same trait
for trait_name in detected:
    for method_name, layers_dict in detected[trait_name].items():
        vector = torch.load(layers_dict[layer])
        result = run_inference_with_dynamics(...)
        # Compare: which method shows clearest dynamics?
```

### Layer Comparison

```python
# Analyze same method across different layers
for layer in [8, 12, 16, 20, 24]:
    vector = torch.load(f"vectors/probe_layer{layer}.pt")
    # Does commitment happen earlier in deeper layers?
```

### Multi-Trait Interactions

```python
# See how traits interact
refusal_scores = trait_scores['refusal']
uncertainty_scores = trait_scores['uncertainty']

# Do they peak together? Opposite phases?
correlation = torch.corrcoef(torch.stack([refusal_scores, uncertainty_scores]))[0, 1]
```

## Requirements

- Extracted trait vectors (from running extraction pipeline)
- GPU for inference (CPU is slow)
- Same model as used for extraction

## Tips

- Start with a few test prompts to verify everything works
- Use `--max_new_tokens 20` for faster testing
- Larger batches of prompts give more robust dynamics statistics
- Compare dynamics across different extraction methods to evaluate quality