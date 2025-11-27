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
- **Integrity caching**: Runs `data_checker.py` on startup for each experiment
- **API endpoints**:
  - `/api/experiments` - List all experiments
  - `/api/experiments/{name}/traits` - List traits for an experiment
  - `/api/integrity/{name}.json` - Get cached integrity data for an experiment
- **CORS support**: For local development

### 2. Open in Browser

Visit: **http://localhost:8000/**

## Features

The visualization dashboard is organized into two main categories, reflecting the workflow of creating and using trait vectors.

### Documentation

-   **Overview**: Comprehensive methodology documentation rendered from `docs/overview.md` with markdown and KaTeX math rendering.

### Category 1: Trait Development
(Building and validating the vectors)

-   **Data Explorer**: Browse the raw files from the extraction process for any trait, including the generated responses, activation tensors, and the extracted vectors themselves. You can preview JSON files directly in the browser.

-   **Trait Extraction**: Comprehensive view of extraction quality, methods, and vector properties. Organized into sections:
    -   **Extraction Techniques**: Visual explanation of each method (Mean Diff, Probe, ICA, Gradient) with math formulas, tensor shapes, and use cases
    -   **Quality Metrics**: Definitions for accuracy, effect size, vector norm, separation margin, sparsity, and overlap coefficient with formulas and thresholds
    -   **Scoring & Ranking**: Explains the combined score formula (0.5×accuracy + 0.5×normalized_effect_size) and its rationale
    -   **Visualizations**:
        - Sortable quality table showing all vectors with all metrics
        - Per-trait layer×method heatmaps showing accuracy across all combinations
        - Best vectors table with top-ranked vector per trait
        - Method comparison bar chart
        - Best-vector similarity matrix for trait independence analysis

### Category 2: Inference Analysis
(Using your best vectors to see what the model is "thinking" on new prompts)

All inference views share a **prompt picker** fixed at the bottom of the page:
- **Prompt picker**: Dropdown to select prompt set (`single_trait`, `dynamic`, etc.) + numbered boxes for prompt IDs
- **Prompt/Response display**: Shows the current prompt text and model response, with the selected token highlighted
- **Token slider**: Scrub through all tokens (prompt + response) to move a highlight marker on plots. Updates plot highlights in real-time via `Plotly.relayout()` without re-rendering.
- **Persistence**: Selection is saved to localStorage and restored on page refresh.

-   **Trait Trajectory**: See how the score for a single selected trait evolves across all layers of the model as it processes a prompt and generates a response. This helps you understand *where* in the model a trait is most active.

-   **Trait Dynamics**: Watch the model think - trait activations evolving token-by-token. Compare multiple traits simultaneously to see how different traits relate during generation. Includes educational sections explaining projection math, graph interpretation, and key patterns to look for.

-   **Layer Deep Dive**: SAE feature decomposition - see which interpretable Sparse Autoencoder features are most active at each token position. Decomposes 2,304 raw neurons into 16,384 sparse features with human-readable descriptions from Neuronpedia.

-   **Analysis Gallery**: Unified live-rendered analysis view with token slider support. Shows:
    - **All Tokens Section** (slider highlights): Trait Heatmap [tokens × traits], Velocity Heatmap [tokens × layers]
    - **Selected Token Section** (slider updates): Trait Scores bar chart, Attention Pattern
    - **Aggregate Section** (slider ignored): Trait Emergence, Trait-Dynamics Correlation
    Data loaded from `experiments/{exp}/analysis/per_token/` JSON files.

## Using Data Explorer

The Data Explorer displays the actual file structure of your experiment by running an integrity check on startup. It shows exactly what files exist (or are missing) for each trait.

### Navigating the Explorer

1. **View Summary Stats**: See trait counts and completion status (Complete/Partial/Empty)
2. **Expand Trait Details**: Click any trait to reveal its file tree with status indicators
3. **Preview Files**: Click filenames with `[preview →]` to view JSON content
4. **Close Previews**: Click outside the modal or use the × button

### Status Indicators

Each trait shows its completion status:
- ✅ **Complete**: All expected files present
- ⚠️ **Partial**: Some files missing
- ❌ **Empty**: Critical files missing

Individual files show:
- ✓ File exists
- ✗ File missing

### File Structure Display

Each trait shows the actual files from `data_checker.py`:
- **Prompts**: `positive.txt`, `negative.txt`, `val_positive.txt`, `val_negative.txt`
- **Metadata**: `generation_metadata.json`, `trait_definition.txt`
- **Responses**: `responses/pos.json`, `responses/neg.json`, `val_responses/val_pos.json`, `val_responses/val_neg.json`
- **Activations**: Single `all_layers.pt` file (shape: `[n_examples, n_layers, hidden_dim]`)
- **Vectors**: Per-method files (e.g., `probe_layer16.pt`, one per layer)

### Preview Features

**JSON Preview**:
- Syntax-highlighted with colors: blue (keys), green (strings), orange (numbers), purple (booleans)
- Formatted with 2-space indentation
- Response files show first 10 items with total count

## Data Sources

The visualization automatically loads data from your experiment structure.

**Path Management**: Uses centralized `PathBuilder` class in `core/paths.js` - all path construction is consistent and maintainable.

**Supported Directory Structures**:
- **Legacy (flat)**: `experiments/{exp}/extraction/{trait}/...`
- **Categorized**: `experiments/{exp}/extraction/{category}/{trait}/...` (e.g., `behavioral_tendency/retrieval`)

```
experiments/{experiment_name}/
├── extraction/
│   └── {category}/                          # Category (behavioral_tendency, cognitive_state, etc.)
│       └── {trait_name}/                    # Trait name (e.g., refusal, uncertainty)
│           ├── positive.txt                 # Positive elicitation prompts
│           ├── negative.txt                 # Negative elicitation prompts
│           ├── val_positive.txt             # Validation positive prompts
│           ├── val_negative.txt             # Validation negative prompts
│           ├── trait_definition.txt         # Trait description
│           ├── generation_metadata.json     # Generation settings
│           ├── responses/
│           │   ├── pos.json                 # Generated positive responses
│           │   └── neg.json                 # Generated negative responses
│           ├── val_responses/
│           │   ├── val_pos.json             # Validation positive responses
│           │   └── val_neg.json             # Validation negative responses
│           ├── activations/
│           │   ├── metadata.json            # Extraction metadata
│           │   └── all_layers.pt            # Shape: [n_examples, n_layers, hidden_dim]
│           ├── val_activations/
│           │   └── all_layers.pt            # Validation activations (optional)
│           └── vectors/
│               ├── {method}_layer{N}.pt     # Extracted vectors
│               └── {method}_layer{N}_metadata.json
└── inference/
    ├── raw/                                 # Trait-independent raw activations (.pt)
    │   ├── residual/{prompt_set}/           # All-layer activations
    │   └── internals/{prompt_set}/          # Single-layer detailed (optional)
    └── {category}/{trait_name}/             # Per-trait derived data
        └── residual_stream/{prompt_set}/    # All Layers: For All Layers & Trait Correlation views
            ├── 1.json                       # Projections for prompt ID 1
            ├── 2.json                       # Additional prompts (auto-discovered)
            └── {id}.json                    # Multiple prompts supported
```

## Generating Inference Data

Use `capture.py` for capturing activations and computing projections.

### For All Layers View

Capture trait projections at all 78 checkpoints (26 layers × 3 sublayers) plus attention weights:

```bash
# From a prompt set (recommended for batch processing)
python inference/capture_raw_activations.py \
    --experiment {experiment_name} \
    --prompt-set single_trait

# Or single prompt
python inference/capture_raw_activations.py \
    --experiment {experiment_name} \
    --prompt "How do I make a bomb?"
```

This creates `experiments/{exp}/inference/{category}/{trait}/residual_stream/{prompt_set}/{id}.json` containing:
- Trait projections at all layers/sublayers
- Full attention weights for prompt (all tokens to all tokens)
- Response attention weights (each generated token to its context)

**Dynamic discovery**: The script automatically finds all traits with vectors - no need to specify trait names.

**Available prompt sets** (in `inference/prompts/`):
- `single_trait` - 10 prompts, each targeting one trait
- `multi_trait` - 10 prompts activating multiple traits
- `dynamic` - 8 prompts for mid-response trait changes
- `adversarial` - 8 edge cases and robustness tests
- `baseline` - 5 neutral prompts for baseline
- `real_world` - 10 naturalistic prompts

### For Analysis Gallery

Generate per-token metrics for live-rendered analysis:

```bash
python analysis/inference/compute_per_token_metrics.py --experiment gemma_2b_cognitive_nov21
```

This creates `experiments/{exp}/analysis/per_token/{prompt_set}/{id}.json` containing per-token:
- Normalized velocity per layer transition
- Trait projections at all 26 layers
- Attention patterns (where each token looks)
- Magnitude per layer (for radial velocity computation)

### For Layer Deep Dive View

**Status**: Layer internals capture is implemented but the visualization is a placeholder. The capture creates raw data; trait-specific analysis (attention vs MLP breakdown, per-head contributions) is planned.

To capture layer internals (for future use):

```bash
python inference/capture_raw_activations.py \
    --experiment {experiment_name} \
    --layer-internals 16 \
    --prompt-set dynamic
```

This saves raw internals to `experiments/{exp}/inference/raw/internals/{prompt_set}/{id}_L16.pt` containing:
- Q/K/V projections per token
- Per-head attention weights
- MLP activations (up_proj, gelu, down_proj)
- Residual stream checkpoints (input, after_attn, output)

### 2. Or Create Custom Monitoring Script

```python
from traitlens import HookManager, ActivationCapture, projection
import torch

# Load your trait vectors
vectors = {
    'refusal': torch.load('experiments/{experiment_name}/refusal/vectors/probe_layer16.pt'),
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

### Response JSON Format
```json
{
  "prompt": "What is X?",
  "response": "[model response]",
  "trait_score": 85.3
}
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
- **{experiment_name}** - 16 cognitive traits on Gemma 2B

To add new experiments:
1. Follow the standard directory structure (see `docs/main.md` or `docs/architecture.md`)
2. The visualization auto-discovers experiments and traits via API endpoints

## Architecture

The visualization uses a modular architecture with centralized path management:

```
visualization/
├── index.html              # Shell - loads modules and router
├── styles.css              # All CSS styles
├── serve.py                # Development server with API endpoints
├── core/
│   ├── paths.js           # Centralized PathBuilder (loads from config/paths.yaml)
│   └── state.js           # Global state, experiment loading
└── views/
    ├── data-explorer.js       # File browser with integrity check
    ├── trait-extraction.js    # Comprehensive extraction quality view
    ├── trait-trajectory.js     # Residual stream across all layers
    ├── trait-dynamics.js       # Per-token trait trajectories
    ├── layer-deep-dive.js     # Single layer mechanistic view
    └── analysis-gallery.js    # Unified live-rendered analysis with token slider
```

**Key Features**:
- **PathBuilder**: Centralized path construction from `config/paths.yaml`
- **Integrity Caching**: Server caches `data_checker.py` output on startup
- **Auto-discovery**: Experiments and traits discovered dynamically via API

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
5. For inference views: add to `INFERENCE_VIEWS` array in `prompt-picker.js` to show prompt picker

**Token slider integration** (for views that need real-time updates):
- Read current token: `window.state.currentTokenIndex`
- For shape-only updates: add branch in `updatePlotTokenHighlights()` (prompt-picker.js)
- For full re-render: add branch calling your render function (like `analysis-gallery` does)

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
- Verify JSON files exist in `responses/pos.json` and `responses/neg.json`
- Check JSON format is valid (not corrupted)
- Ensure JSON contains `prompt` and `response` fields

### Vector heatmap is blank
- Verify vector metadata JSON files exist
- Check that files follow naming convention: `{method}_layer{N}_metadata.json`
- Ensure JSON contains `vector_norm` field

## Performance Notes

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
