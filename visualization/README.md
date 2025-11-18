# Trait Interpretation Visualization Guide

This guide explains how to use the visualization tools to explore your extracted trait vectors and monitoring results.

## Quick Start

### 1. Start the Visualization Server

```bash
# From the trait-interp root directory (recommended)
python visualization/serve.py

# Alternative: Python's built-in HTTP server
python -m http.server 8000
```

The server provides:
- **Auto-discovery**: Experiments and traits detected from `experiments/` directory
- **API endpoints**: `/api/experiments` and `/api/experiments/{name}/traits`
- **CORS support**: For local development

### 2. Open in Browser

Visit: **http://localhost:8000/visualization/**

## Features

### Data Explorer Mode
- **Dataset Overview**: Total traits, estimated storage, data completeness, total files
- **Expandable Trait Cards**: Click trait names to expand/collapse complete file tree
- **File Structure**: Shows all files with sizes, tensor shapes, and row counts
  - extraction/trait_definition.json
  - extraction/responses/ (pos.csv, neg.csv with row counts)
  - extraction/activations/ (metadata.json, .pt files with shapes like [n_examples, 27, 2304])
  - extraction/vectors/ (108 files: 4 methods × 27 layers)
  - inference/ (per-token data if generated)
- **Preview Functionality**: Click files to preview content in modal
  - JSON files: Syntax-highlighted with color-coded keys, strings, numbers, booleans
  - CSV files: First 10 rows displayed as formatted table
- **Dark Mode Support**: All text grey, syntax highlighting adapts to theme

### Vector Analysis Mode
- **Compact Table Layout**: One row per trait showing all methods at once
  - Left: Trait name + best method/layer
  - Center: Horizontal mini-heatmap (4 methods × 26 layers)
  - Right: Pos/neg example preview
- **Layer Architecture Intuition**: Info box explaining what different layer ranges represent (syntax, semantics, behavioral traits)
- **Method Comparison**: Compare mean_diff, probe, ICA, and gradient extraction methods
- **Visualization Metrics**:
  - **Probe**: Inverted vector norm (smaller = stronger due to L2 regularization)
  - **Gradient**: Separation strength (unit normalized vectors)
  - **Mean Diff & ICA**: Vector magnitude
- **Filtering**: Only shows traits with completed vector extraction (`extraction/vectors/` exists)
- **Parallel Loading**: All vector metadata loaded in parallel for fast performance
- **Click to expand**: Click any row for detailed 4×26 heatmap view

### All Layers Mode
- **Prompt Selector**: Global dropdown in header to switch between captured prompts
  - Auto-discovers available prompt files (prompt_0.json, prompt_1.json, etc.)
  - All inference views update when prompt changes
  - Shows clear error messages when selected prompt doesn't exist for a trait
- **Combined Trajectory Heatmap**: Single view showing trait scores across all 26 layers for both prompt and response tokens (layer 0 at bottom, layer 25 at top)
  - Vertical line separates prompt | response phases
  - Uses layer 16's trait vector projected onto all layers
- **Unified Slider**: Controls all visualizations simultaneously
  - Shows current token position, text, and phase ([Prompt] or [Response])
  - Highlights selected token on trajectory heatmap
  - Updates attention pattern for selected token
- **Attention Patterns**: Interactive heatmap showing how selected token attends to context
  - Heatmap shows [n_layers, n_context] attention pattern
  - Automatically updates when slider moves
- **Data**: Requires capture_all_layers.py with `--save-json` flag

### Layer Deep Dive Mode
- **3-Checkpoint Trajectory**: Line plot showing trait score at residual_in, after_attn, and residual_out
- **Attention vs MLP Contribution**: Bar chart showing which sublayer adds/removes the trait
- **Per-Head Trait Contributions**: Shows how each of 8 attention heads contributes to the trait (supports Grouped Query Attention)
- **8-Head Attention Patterns**: Single-token query view showing how selected token attends to context
  - Slider controls which query token to visualize
  - Each head shown as 1-row heatmap (query token's attention distribution)
  - Red highlight shows query token position
  - Automatically switches between prompt (all tokens) and response (growing context) attention
- **Top Neurons**: Heatmap + bar chart of top-50 most active MLP neurons across all tokens
- **Data**: Requires capture_single_layer.py with `--save-json` flag

## Using Data Explorer

The Data Explorer provides a file browser for inspecting your experiment data without leaving the browser.

### Navigating the Explorer

1. **View Dataset Statistics**: See total traits (16), storage estimate (~752 MB), and file counts at the top
2. **Expand Trait Details**: Click any trait name to reveal its complete file tree
3. **Preview Files**: Click filenames with `[preview →]` to view content
   - JSON files open with syntax highlighting
   - CSV files show first 10 rows as a table
4. **Close Previews**: Click outside the modal or use the × button

### File Structure Display

Each trait shows:
- **Extraction folder**: Contains trait definition, responses, activations, and vectors
  - **Definition**: trait_definition.json (clickable preview)
  - **Responses**: pos.csv and neg.csv with row counts (clickable preview)
  - **Activations**: metadata.json (clickable preview) and .pt tensor files with shapes
  - **Vectors**: 108 files (4 methods × 27 layers) with estimated sizes
- **Inference folder**: Contains per-token trajectory and layer internals (if generated)

### Preview Features

**JSON Preview**:
- Syntax-highlighted with colors: blue (keys), green (strings), orange (numbers), purple (booleans)
- Formatted with 2-space indentation
- Grey text that adapts to dark/light mode

**CSV Preview**:
- First 10 rows displayed as formatted table
- Shows total row count
- Long values truncated to 100 characters

## Data Sources

The visualization automatically loads data from your experiment structure:

```
experiments/{experiment_name}/
├── {trait_name}/
│   ├── extraction/                          # Extraction data (Tier 1)
│   │   ├── trait_definition.json
│   │   ├── responses/
│   │   │   ├── pos.csv                      # Generated responses
│   │   │   └── neg.csv                      # Generated responses
│   │   ├── activations/
│   │   │   └── metadata.json                # Loaded for Overview
│   │   └── vectors/
│   │       └── *_metadata.json              # Loaded for Vector Analysis
│   └── inference/                           # Inference data (Tier 2 & 3)
│       ├── residual_stream_activations/     # Tier 2: All Layers & Prompt Activation
│       │   ├── prompt_0.json                # First prompt (includes projections + attention_weights)
│       │   ├── prompt_1.json                # Additional prompts (auto-discovered)
│       │   └── prompt_N.json                # Multiple prompts supported
│       └── layer_internal_states/           # Tier 3: Layer Deep Dive
│           ├── prompt_0_layer16.json        # First prompt at layer 16
│           └── prompt_N_layer16.json        # Additional prompts at layer 16
```

## Generating Inference Data

### For All Layers View

Capture trait projections at all 81 checkpoints (27 layers × 3 sublayers) plus attention weights:

```bash
python inference/capture_all_layers.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait refusal \
    --prompts "How do I make a bomb?" \
    --save-json
```

This creates `experiments/{exp}/{trait}/inference/residual_stream_activations/prompt_0.json` containing:
- Trait projections at all layers/sublayers
- Full attention weights for prompt (all tokens to all tokens)
- Response attention weights (each generated token to its context)

**Multiple prompts**: Run the command multiple times with different prompts. Files are automatically numbered (prompt_0.json, prompt_1.json, etc.) and discovered by the visualization.

### For Layer Deep Dive View

Capture complete internals for one layer (Q/K/V, attention heads, 9216 MLP neurons):

```bash
python inference/capture_single_layer.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait refusal \
    --layer 16 \
    --prompts "How do I make a bomb?" \
    --save-json
```

This creates `experiments/{exp}/{trait}/inference/layer_internal_states/prompt_0_layer16.json`

**Multiple prompts**: Run the command multiple times with different prompts. Files are automatically numbered (prompt_0_layer16.json, prompt_1_layer16.json, etc.) and selectable via the prompt dropdown.

### For Basic Dynamics Analysis

Quick analysis without full layer capture:

```bash
python inference/monitor_dynamics.py \
    --experiment gemma_2b_cognitive_nov20 \
    --prompts "What is the capital of France?" \
    --output monitoring_results.json
```

### 2. Or Create Custom Monitoring Script

```python
from traitlens import HookManager, ActivationCapture, projection
import torch

# Load your trait vectors
vectors = {
    'refusal': torch.load('experiments/gemma_2b_cognitive_nov20/refusal/vectors/probe_layer16.pt'),
    # ... more traits
}

# Capture activations during generation
capture = ActivationCapture()
with HookManager(model) as hooks:
    hooks.add_forward_hook("model.layers.16", capture.make_hook("layer_16"))
    output = model.generate(**inputs)

# Project onto trait vectors
acts = capture.get("layer_16")
trait_scores = {name: projection(acts, vec).tolist() for name, vec in vectors.items()}

# Save results
results = {
    "prompt": prompt,
    "response": response,
    "tokens": tokens,
    "trait_scores": trait_scores
}
```

### 3. Save in Expected Location

Place JSON files in:
```
experiments/{experiment_name}/inference/results/
```

## Expected Data Formats

### Response CSV Format
```csv
question,instruction,prompt,response,trait_score
"What is X?","[instruction]","[full prompt]","[model response]",85.3
```

### Activation Metadata Format
```json
{
  "model": "google/gemma-2-2b-it",
  "trait": "refusal",
  "n_examples": 179,
  "n_layers": 27,
  "hidden_dim": 2304,
  "extraction_date": "2025-11-14T14:00:40.778123"
}
```

### Vector Metadata Format
```json
{
  "trait": "refusal",
  "method": "probe",
  "layer": 16,
  "vector_norm": 2.164860027678928,
  "train_acc": 1.0
}
```

### Monitoring Results Format
```json
{
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris.",
  "tokens": ["The", " capital", " of", " France", " is", " Paris", "."],
  "trait_scores": {
    "retrieval_construction": [0.5, 2.3, 2.1, 1.8, 1.5, 1.2, 0.8],
    "refusal": [-0.3, -1.5, -1.2, -0.9, -0.7, -0.5, -0.3]
  }
}
```

## Supported Experiments

Currently supported:
- **gemma_2b_cognitive_nov20** - 16 cognitive traits on Gemma 2B

To add new experiments:
1. Follow the standard directory structure (see `docs/experiments_structure.md`)
2. The visualization will auto-detect traits via `activations/metadata.json`
3. Add experiment name to the `experiments` array in `visualization_v2.html` (line ~169)

## Customization

### Adding New Experiments
Edit `visualization/index.html` around line 353:

```javascript
experiments = [
    'gemma_2b_cognitive_nov20',
    'your_new_experiment'
];
```

### Styling
All styles are in the `<style>` block. Key CSS variables:
- Primary color: `#007bff`
- Success: `#28a745`
- Danger: `#dc3545`

### Adding New Views
1. Add option to `#view-mode` select
2. Create render function (e.g., `renderYourView()`)
3. Add case to `renderView()` switch statement

## Troubleshooting

### "Failed to load experiments"
- Ensure you're running a local server (`python serve_viz.py`)
- Check that you're in the `trait-interp` directory
- Verify `experiments/` directory exists

### "No traits found"
- Check that trait directories have `activations/metadata.json`
- Verify JSON files are valid (not corrupted)
- Look at browser console for specific errors (F12)

### Response data won't load
- Verify CSV files exist in `responses/pos.csv` and `responses/neg.csv`
- Check CSV format matches expected schema
- Ensure CSV has header row with required columns

### Vector heatmap is blank
- Verify vector metadata JSON files exist
- Check that files follow naming convention: `{method}_layer{N}_metadata.json`
- Ensure JSON contains `vector_norm` field

## Performance Notes

- **Large CSV files**: May take 5-10 seconds to parse (1000+ rows)
- **Vector loading**: Fetches 4 methods × 27 layers = 108 files per trait
- **Browser cache**: Refresh (Cmd+R) if data seems stale

## Next Steps

1. **Browse Data**: Use Data Explorer to inspect extraction files
2. **Compare Methods**: Find which extraction method works best for each trait
3. **Analyze Layers**: Identify which layers capture each trait best
4. **Generate Monitoring Data**: Run inference scripts for per-token analysis

## Related Documentation

- **[docs/main.md](docs/main.md)** - Main project documentation
- **[docs/pipeline_guide.md](docs/pipeline_guide.md)** - Extraction pipeline
- **[inference/README.md](inference/README.md)** - Per-token inference and dynamics
