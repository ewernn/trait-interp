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
  - `/api/chat` - POST endpoint for live chat with streaming trait projections
- **CORS support**: For local development

### 2. Open in Browser

Visit: **http://localhost:8000/**

## Features

The visualization dashboard is organized into two main categories, reflecting the workflow of creating and using trait vectors.

### Documentation

-   **Overview**: Comprehensive methodology documentation rendered from `docs/overview.md` with markdown and KaTeX math rendering.

-   **Findings**: Research findings rendered from `docs/viz_findings/`. Collapsible cards with preview text that expand to show full markdown content.

    **Adding a finding:**
    1. Create `docs/viz_findings/my-finding.md` with frontmatter:
       ```markdown
       ---
       title: "My Finding Title"
       preview: "One sentence summary for collapsed view."
       ---

       # My Finding Title
       ...
       ```
    2. Add filename to `docs/viz_findings/index.yaml` (controls display order)

    **Markdown conventions:**
    - Figures: `:::figure assets/image.png "Figure 1: Caption" small:::` (sizes: `small` 30%, `medium` 50%, `large` 75%, omit for 100%)
    - Response embeds: `:::responses /path/to/responses.json "Label":::` creates expandable table
    - Dataset embeds: `:::dataset /path/to/file.txt "Label":::` shows first 20 lines
    - Citations: `[@key]` with `references:` in frontmatter → renders as "(Authors, Year)" link + auto References section

### Category 1: Trait Development
(Building and validating the vectors)

-   **Trait Extraction**: View extraction quality for trait vectors. Sections:
    - Best Vectors Summary: table of best vector per trait with metrics
    - Per-Trait Heatmaps: layer × method grid with combined score
    - Token Decode (Logit Lens): late layer token projections
    - Reference (collapsible): notation, extraction methods, quality metrics

-   **Steering Sweep**: Layer sweep visualization for steering evaluation. Shows behavioral change vs. steering coefficient across layers. Includes multi-layer steering heatmap (center × width grid) when available. Validates vectors via causal intervention.

### Category 2: Inference Analysis
(Using your best vectors to see what the model is "thinking" on new prompts)

-   **Live Chat**: Interactive chat with real-time trait monitoring.
    - **Model selection**: Dropdown to switch between `extraction_model` (base) and `application_model` (instruct-tuned) from experiment config. Clears chat when switching models.
    - **Full token stream**: Chart shows all tokens (prompt + response + special tokens) with their trait projections. X-axis = token position in context.
    - **Multi-turn conversation**: Tokens accumulate across messages. Only NEW tokens are captured each turn (previous context is skipped).
    - **Conversation branching**: Edit any user message to create alternate branches; navigate with `◀ 1/2 ▶` arrows. Each branch preserves its token data.
    - **Stop generation**: Click "Stop" button to halt generation mid-stream. Keeps tokens captured so far.
    - **Context limit**: Warns when approaching model's max context length (from `config/models/`). Clear chat to continue.
    - **Persistence**: Conversations saved to localStorage per experiment; restored on page refresh.
    - **Hover interactions**: Hover over chart to highlight corresponding token in chat. Hover over legend to see vector metadata (layer, method, source).
    - **3-token running average**: Toggle smoothing in chart header.
    - **Streaming**: Tokens appear as generated (~0.2s each). First token: 10-30s local.

All other inference views share a **prompt picker** fixed at the bottom of the page:
- **Prompt picker**: Dropdown to select prompt set (`single_trait`, `dynamic`, etc.) + numbered boxes for prompt IDs
- **Prompt/Response display**: Shows the current prompt text and model response, with the selected token highlighted
- **Token slider**: Scrub through all tokens (prompt + response) to move a highlight marker on plots. Updates plot highlights in real-time via `Plotly.relayout()` without re-rendering.
- **Persistence**: Selection is saved to localStorage and restored on page refresh.

-   **Trait Dynamics**: Comprehensive view of trait evolution during generation. Includes:
    - **Token Trajectory**: Cosine similarity per token. Options: **Smooth** (3-token moving average), **Centered** (subtract BOS value), **Clean** (remove massive dim contributions - Sun et al., Top 5 3+ layers, or All), **Compare** (model comparison - see below)
    - **Activation Magnitude Per Token**: ||h|| per token at projection layer. Shows one line per unique layer when multiple traits selected.
    - **Token Velocity**: Rate of change between consecutive tokens
    - **Activation Magnitude**: ||h|| per layer (growth through network)
    - **Massive Activations**: Per-token values of massive dims at layer 9, mean alignment by layer
    - **Massive Dims Across Layers**: Normalized magnitude per layer for massive dims. Criteria dropdown to experiment with different definitions (Top 5 3+ layers, Top 3 any layer, etc.)
    - **Compare dropdown**: When comparison models exist (e.g., base model captured via `--replay-responses`), switch between: **Main model** (default), **Diff** (main − comparison), or view the **comparison model** directly. See `inference/README.md` for capture workflow.

-   **Layer Deep Dive**: SAE feature decomposition - see which interpretable Sparse Autoencoder features are most active at each token position. Decomposes 2,304 raw neurons into 16,384 sparse features with human-readable descriptions from Neuronpedia.

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
│           ├── activations/{position}/{component}/
│           │   ├── metadata.json            # Extraction metadata
│           │   ├── train_all_layers.pt      # Shape: [n_examples, n_layers, hidden_dim]
│           │   └── val_all_layers.pt        # Validation activations (same format)
│           └── vectors/{position}/{component}/{method}/
│               ├── layer{N}.pt              # Extracted vectors
│               └── metadata.json
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

**Available prompt sets** (in `datasets/inference/`):
- `single_trait` - 10 prompts, each targeting one trait
- `multi_trait` - 10 prompts activating multiple traits
- `dynamic` - 8 prompts for mid-response trait changes
- `adversarial` - 8 edge cases and robustness tests
- `baseline` - 5 neutral prompts for baseline
- `real_world` - 10 naturalistic prompts

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
- Residual stream checkpoints (input, attn_out, residual)

### 2. Or Create Custom Monitoring Script

```python
from core import HookManager, CaptureHook, projection
import torch

# Load your trait vectors
vectors = {
    'refusal': torch.load('experiments/{experiment_name}/refusal/vectors/probe_layer16.pt'),
    # ... more traits
}

# Capture activations during generation
with CaptureHook(model, "model.layers.16") as hook:
    output = model.generate(**inputs)

# Project onto trait vectors
acts = hook.get()
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
├── chat_inference.py       # Live chat backend (model loading, generation, trait projection)
├── core/
│   ├── paths.js            # Centralized PathBuilder (loads from config/paths.yaml)
│   ├── model-config.js     # Model config loader (loads from config/models/)
│   ├── state.js            # Global state, experiment loading
│   ├── prompt-picker.js    # Prompt selection UI for inference views
│   └── conversation-tree.js # ConversationTree for multi-turn chat with branching
└── views/
    ├── overview.js            # Methodology documentation (markdown + KaTeX)
    ├── trait-extraction.js    # Extraction quality (Trait Development)
    ├── steering-sweep.js      # Layer sweep + multi-layer heatmap (Trait Development)
    ├── live-chat.js           # Real-time chat with trait monitoring (Inference)
    ├── trait-dynamics.js      # Per-token trait trajectories (Inference)
    └── layer-deep-dive.js     # Attention + SAE features (Inference)
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
- For full re-render: add branch calling your render function

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

1. **Compare Methods**: Find which extraction method works best for each trait
2. **Analyze Layers**: Identify which layers capture each trait best
3. **Evaluate Steering**: Use Steering Sweep to validate vectors via causal intervention
4. **Generate Monitoring Data**: Run inference scripts for per-token analysis

## Related Documentation

- **[docs/main.md](docs/main.md)** - Main project documentation
- **[docs/extraction_pipeline.md](docs/extraction_pipeline.md)** - Extraction pipeline
- **[inference/README.md](inference/README.md)** - Per-token inference and dynamics
