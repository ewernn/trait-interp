# SAE Integration Plan

This document describes the planned integration of SAE feature decomposition into the visualization dashboard.

## Current State (this branch)

### Files Added/Modified in `sae/`

| File | Status | Description |
|------|--------|-------------|
| `encode_sae_features.py` | Modified | Fixed to read actual `.pt` format from `inference/raw/residual/` |
| `download_fast.py` | Exists | Downloads 16k feature labels from Neuronpedia |
| `INTEGRATION_PLAN.md` | New | This file |

### Data Flow

```
inference/raw/residual/{prompt_set}/{id}.pt
    │
    │  python sae/encode_sae_features.py --experiment {exp}
    ▼
inference/sae/{prompt_set}/{id}_sae.pt
    │
    │  (visualization reads this + feature_labels.json)
    ▼
layer-deep-dive.js renders SAE features
```

---

## Planned Changes (NOT YET IMPLEMENTED)

### 1. New Files in `experiments/`

After running `encode_sae_features.py`, this structure will be created:

```
experiments/{experiment}/inference/sae/
├── single_trait/
│   ├── 1_sae.pt      # Encoded features for prompt 1
│   ├── 2_sae.pt
│   └── ...
├── multi_trait/
│   └── ...
├── dynamic/
│   └── ...
└── ...
```

**Encoded file format** (`.pt`):
```python
{
    'source_file': '1.pt',
    'sae_release': 'gemma-scope-2b-pt-res-canonical',
    'sae_id': 'layer_16/width_16k/canonical',
    'layer': 16,
    'position': 'residual_out',

    'prompt_text': 'What year was...',
    'response_text': 'The Treaty of Versailles...',
    'tokens': ['\n\n', 'The', ' Treaty', ...],

    'top_k_indices': tensor([seq_len, 50]),  # Feature indices
    'top_k_values': tensor([seq_len, 50]),   # Activation values
    'k': 50,

    'num_tokens': 50,
    'avg_active_features': 127.3,
    'min_active_features': 45,
    'max_active_features': 234,
}
```

### 2. Changes to `visualization/`

#### 2.1 New API Endpoint in `serve.py`

```python
@app.route('/api/sae/<experiment>/<prompt_set>/<prompt_id>')
def get_sae_features(experiment, prompt_set, prompt_id):
    """Return encoded SAE features for a prompt."""
    sae_file = Path(f'experiments/{experiment}/inference/sae/{prompt_set}/{prompt_id}_sae.pt')
    data = torch.load(sae_file, map_location='cpu')

    # Load feature labels
    labels = load_feature_labels()

    # Enrich with descriptions
    for token_idx in range(data['num_tokens']):
        indices = data['top_k_indices'][token_idx].tolist()
        values = data['top_k_values'][token_idx].tolist()
        # Look up feature descriptions
        ...

    return jsonify(enriched_data)
```

#### 2.2 Feature Label Loading

```python
def load_feature_labels():
    """Load feature labels from sae/ directory."""
    labels_path = Path('sae/gemma-scope-2b-pt-res-canonical/layer_16_width_16k_canonical/feature_labels.json')
    with open(labels_path) as f:
        return json.load(f)
```

#### 2.3 State Management in `core/state.js`

Add SAE data to state:

```javascript
// New state fields
window.saeData = null;
window.saeLabels = null;

// Load SAE features for current prompt
async function loadSaeFeatures(promptSet, promptId) {
    const response = await fetch(`/api/sae/${experiment}/${promptSet}/${promptId}`);
    window.saeData = await response.json();
}

// Load feature labels (once)
async function loadSaeLabels() {
    const response = await fetch('/api/sae/labels');
    window.saeLabels = await response.json();
}
```

### 3. Layer Deep Dive View (`layer-deep-dive.js`)

#### 3.1 UI Components

**Token Selector**
- Dropdown or slider to select a token
- Shows token text and position

**Feature Bar Chart**
- Horizontal bars for top-N features
- X-axis: activation strength
- Y-axis: feature description
- Color-coded by category (if available)

**Feature Details Panel**
- Click feature to see:
  - Full description
  - Top activating tokens (from Neuronpedia)
  - Activation value at this token

**Attention vs MLP Toggle** (future)
- Compare features from `after_attn` vs `residual_out`
- Shows what MLP contributes (attention contribution = after_attn - prev_layer_residual_out)

#### 3.2 Rendering Flow

```javascript
async function renderLayerDeepDive() {
    // 1. Get current prompt from state
    const { promptSet, promptId } = getCurrentPrompt();

    // 2. Load SAE features if not cached
    if (!window.saeData || window.saeData.source !== `${promptSet}/${promptId}`) {
        await loadSaeFeatures(promptSet, promptId);
    }

    // 3. Render token selector
    renderTokenSelector(window.saeData.tokens);

    // 4. Render feature chart for selected token
    const tokenIdx = getSelectedTokenIndex();
    renderFeatureChart(tokenIdx);

    // 5. Render feature details
    renderFeatureDetails();
}
```

### 4. Data Requirements

| Data | Source | Size |
|------|--------|------|
| Feature labels | `sae/.../feature_labels.json` | ~8 MB |
| Encoded features | `experiments/.../inference/sae/` | ~200 KB per prompt |
| Raw activations | Already exists | Read-only |

### 5. Dependencies

**Python:**
- `sae_lens` - for SAE encoding
- Already in requirements: `torch`, `tqdm`

**JavaScript:**
- No new dependencies
- Uses existing chart library (if any) or vanilla SVG

---

## Implementation Order

1. **[DONE]** Fix `encode_sae_features.py`
2. **[IN PROGRESS]** Download feature labels
3. **[TODO]** Run encoding on experiment data
4. **[TODO]** Add `.pt` → JSON conversion (or serve.py endpoint)
5. **[DONE]** Build `layer-deep-dive.js` UI
6. **[TODO]** Add state management for SAE data
7. **[TODO]** Test end-to-end

## Data Format Note

The encoder saves `.pt` files (PyTorch tensors) but the JS visualization expects JSON.

**Options:**
1. Add `--output-json` flag to encoder that saves JSON instead of .pt
2. Add conversion script `pt_to_json.py`
3. Add serve.py endpoint that loads .pt and returns JSON on-the-fly

Option 1 is cleanest - modify encoder to output JSON directly since JS can't read .pt files.

---

## Notes

- SAE was trained on base `gemma-2-2b`, not `gemma-2-2b-it`. Feature descriptions may be approximate.
- Layer 16 chosen because trait vectors use layer 16. Could add support for other layers.
- Top-50 features per token is a balance between detail and file size.
- Feature labels are static (~8 MB), loaded once. Encoded features are per-prompt.
