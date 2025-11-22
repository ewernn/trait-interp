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

### Trait Correlation Mode
- **Pairwise Correlation Matrix**: Compute Pearson correlations between all trait projections
- **Independence Testing**: Identify which traits measure similar vs different computations
- **Visualization**:
  - Red (r ≈ -1): Strong inverse correlation
  - White (r ≈ 0): Independent traits
  - Green (r ≈ +1): Strong positive correlation
- **Use Cases**:
  - Identify redundant traits for merging
  - Validate framework coherence (expect low correlations)
  - Discover trait relationships and confounds
- **Method**: Uses layer 16 projections from Tier 2 inference data

### Validation Results Mode
- **Summary Cards**: Per-method overview (probe, gradient, mean_diff, ica) showing mean accuracy, max accuracy, effect size, polarity rate
- **Method Tabs**: Switch between methods to view method-specific heatmaps
- **Per-Method Heatmaps** (3 per method):
  - **Accuracy Heatmap**: Traits × Layers, red (0%) → yellow (50%) → green (100%)
  - **Effect Size Heatmap**: Traits × Layers, white (0) → blue (5+), Cohen's d values
  - **Polarity Heatmap**: Traits × Layers, red (wrong) / green (correct)
- **Best Vector Per Trait Table**: Shows best layer, accuracy, effect size, polarity for selected method
- **Cross-Trait Independence Matrix**: Heatmap showing interference between traits
  - Diagonal should be high (vectors work for their trait)
  - Off-diagonal should be ~50% (random, independent traits)
- **Data**: Requires running `python analysis/evaluate_on_validation.py --experiment <name>`

**Metrics calculated**:
- **Accuracy**: Classification using midpoint threshold between pos/neg means
- **Effect Size (Cohen's d)**: Separation / pooled_std (d > 0.8 = large effect)
- **Polarity**: Whether positive examples score higher than negative

### All Layers Mode
- **Prompt Selector**: A grid of clickable numbered boxes in the header to switch between captured prompts.
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
- **Data**: Requires `capture_layers.py --mode all --save-json`

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
- **Data**: Requires `capture_layers.py --mode single --save-json`

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

The visualization automatically loads data from your experiment structure.

**Path Management**: Uses centralized `PathBuilder` class in `core/paths.js` - all path construction is consistent and maintainable.

**Supported Directory Structures**:
- **Legacy (flat)**: `experiments/{exp}/extraction/{trait}/...`
- **Categorized**: `experiments/{exp}/extraction/{category}/{trait}/...` (e.g., `behavioral_tendency/retrieval`)

```
experiments/{experiment_name}/
├── extraction/
│   └── {category}/                          # Optional category (behavioral, cognitive, etc.)
│       └── {trait_name}/                    # Trait name (e.g., refusal, uncertainty)
│           ├── trait_definition.json
│           ├── responses/
│           │   ├── pos.csv                  # Generated responses
│           │   └── neg.csv                  # Generated responses
│           ├── activations/
│           │   └── metadata.json            # Loaded for Overview
│           └── vectors/
│               └── *_metadata.json          # Loaded for Vector Analysis
└── inference/
    ├── prompts/                             # Shared prompts
    │   └── prompt_N.json
    └── {category}/{trait_name}/             # Per-trait inference data
        └── projections/
            ├── residual_stream_activations/ # Tier 2: All Layers & Trait Correlation
            │   ├── prompt_0.json            # Projections for all layers
            │   ├── prompt_1.json            # Additional prompts (auto-discovered)
            │   └── prompt_N.json            # Multiple prompts supported
            └── layer_internal_states/       # Tier 3: Layer Deep Dive
                ├── prompt_0_layer16.json    # Complete layer 16 internals
                └── prompt_N_layer16.json    # Additional prompts
```

## Generating Inference Data

Use the unified `capture_layers.py` script for both Tier 2 (all layers) and Tier 3 (single layer deep dive).

### For All Layers View (Tier 2)

Capture trait projections at all 78 checkpoints (26 layers × 3 sublayers) plus attention weights:

```bash
# From a prompt set (recommended for batch processing)
python inference/capture_layers.py \
    --experiment gemma_2b_cognitive_nov21 \
    --prompt-set main_prompts \
    --save-json

# Or single prompt
python inference/capture_layers.py \
    --experiment gemma_2b_cognitive_nov21 \
    --prompt "How do I make a bomb?" \
    --save-json
```

This creates `experiments/{exp}/inference/{category}/{trait}/projections/residual_stream_activations/prompt_N.json` containing:
- Trait projections at all layers/sublayers
- Full attention weights for prompt (all tokens to all tokens)
- Response attention weights (each generated token to its context)

**Dynamic discovery**: The script automatically finds all traits with vectors - no need to specify trait names.

### For Layer Deep Dive View (Tier 3)

Capture complete internals for one layer (Q/K/V, attention heads, MLP neurons):

```bash
python inference/capture_layers.py \
    --experiment gemma_2b_cognitive_nov21 \
    --mode single \
    --layer 16 \
    --prompt "How do I make a bomb?" \
    --save-json
```

This creates `experiments/{exp}/inference/{category}/{trait}/projections/layer_internal_states/prompt_N_layer16.json`

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
1. Follow the standard directory structure (see `docs/main.md` or `docs/architecture.md`)
2. The visualization auto-discovers experiments and traits via API endpoints

## Architecture

The visualization uses a modular architecture with centralized path management:

```
visualization/
├── index.html              # Shell - loads modules
├── styles.css              # All CSS (1,278 lines)
├── serve.py                # Development server with auto-discovery API
├── core/
│   ├── paths.js           # Centralized PathBuilder class (no hardcoded paths)
│   ├── state.js           # Global state, experiment loading
│   └── data-loader.js     # Centralized data fetching
└── views/
    ├── data-explorer.js   # File browser with preview
    ├── vectors.js         # Vector extraction comparison (uses PathBuilder)
    ├── trait-correlation.js  # Pairwise trait correlation matrix
    ├── validation.js      # Validation results dashboard
    ├── monitoring.js      # All layers visualization
    ├── prompt-activation.js  # Per-token trajectories
    └── layer-dive.js      # Single layer deep dive
```

**Key Improvements**:
- **PathBuilder**: Centralized path construction in `core/paths.js` eliminates hardcoding
- **Flexible Categories**: Auto-discovers traits regardless of category naming scheme
- **No Hardcoded Experiments**: All discovery done dynamically via API

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for detailed documentation.

## Customization

### Adding New Experiments
Experiments are auto-discovered from `experiments/` directory. No code changes needed.

### Styling
All styles use CSS variables in `styles.css` for light/dark mode support. See **[DESIGN_STANDARDS.md](DESIGN_STANDARDS.md)** for design rules.

### Adding New Views
1. Create `views/my-view.js` with render function
2. Load in `index.html`: `<script src="views/my-view.js"></script>`
3. Add case to router: `case 'my-view': window.renderMyView(); break;`
4. Add navigation item in sidebar HTML

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for examples.

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
