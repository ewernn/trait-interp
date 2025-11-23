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

The visualization dashboard is organized into two main categories, reflecting the workflow of creating and using trait vectors.

### Category 1: Trait Development
(Building and validating the vectors)

-   **Data Explorer**: Browse the raw files from the extraction process for any trait, including the generated responses, activation tensors, and the extracted vectors themselves. You can preview CSV and JSON files directly in the browser.

-   **Trait Dashboard**: A unified dashboard to inspect and evaluate the quality of extracted vectors. For each trait, this view combines:
    -   **Extraction Stats**: A heatmap showing the intrinsic properties (like vector norm) of every extracted vector across all layers and methods.
    -   **Vector Quality**: A "Top 5 Vectors" table that ranks vectors by their performance (accuracy, effect size) on held-out validation data, helping you identify the highest-quality vector for your analysis.

-   **Trait Correlation Matrix**: Check if your extracted trait vectors are truly independent or if they are accidentally measuring the same underlying concept. This view shows a correlation matrix of all selected traits against each other.

### Category 2: Inference Analysis
(Using your best vectors to see what the model is "thinking" on new prompts)

-   **Trait Trajectory**: See how the score for a single selected trait evolves across all layers of the model as it processes a prompt and generates a response. This helps you understand *where* in the model a trait is most active.

-   **Multi-Trait Comparison**: Compare the activation trajectories of multiple selected traits simultaneously for a single prompt. This helps you see how different traits relate to each other during generation.

-   **Layer Internals**: A deep-dive, mechanistic view into the components of a single chosen layer for a specific prompt. It shows exactly which neurons and attention heads are contributing to the trait score at that layer.

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
    ├── raw/                                 # Trait-independent raw activations (.pt)
    │   ├── residual/{prompt_set}/           # All-layer activations
    │   └── internals/{prompt_set}/          # Single-layer detailed
    └── {category}/{trait_name}/             # Per-trait derived data
        ├── residual_stream/{prompt_set}/    # All Layers: For All Layers & Trait Correlation views
        │   ├── 1.json                       # Projections for prompt ID 1
        │   ├── 2.json                       # Additional prompts (auto-discovered)
        │   └── {id}.json                    # Multiple prompts supported
        └── layer_internals/{prompt_set}/    # Layer Internals: For Layer Deep Dive view
            ├── 1_L16.json                   # Complete layer 16 internals for prompt 1
            └── {id}_L{layer}.json           # Additional prompts
```

## Generating Inference Data

Use `capture.py` for capturing activations and computing projections.

### For All Layers View

Capture trait projections at all 78 checkpoints (26 layers × 3 sublayers) plus attention weights:

```bash
# From a prompt set (recommended for batch processing)
python inference/capture.py \
    --experiment {experiment_name} \
    --prompt-set single_trait

# Or single prompt
python inference/capture.py \
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

### For Layer Deep Dive View (Layer Internals)

Capture complete internals for one layer (Q/K/V, attention heads, MLP neurons):

```bash
python inference/capture.py \
    --experiment {experiment_name} \
    --layer-internals 16 \
    --prompt "How do I make a bomb?"
```

This creates `experiments/{exp}/inference/{category}/{trait}/layer_internals/{prompt_set}/{id}_L16.json`

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
- **{experiment_name}** - 16 cognitive traits on Gemma 2B

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
