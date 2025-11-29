# Zoomies: Granularity-Based Visualization

A clean, spatial navigation interface for trait interpretation data.

## Quick Start

```bash
python visualization/serve.py
# Visit http://localhost:8000/zoomies
```

**Status:** Implemented. Experimental alternative to `visualization/`.

## Current UI

**Zoomed Out View:**
```
    Output      ░░░░░░░░░░░░░░░░░░
    Layer 25    ░░░░░░░░░░░░░░░░░░
        ⋮
        ⋮
    Layer 0     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← green border (clickable)
    Embed       ░░░░░░░░░░░░░░░░░░
```

**Zoomed Into Layer 0:**
```
                    ← All Layers

                     Layer 0

                  to layer 1
                      ↑
                     [+]
              ┌──────────────┐
    ──────────│     MLP      │──────────
              └──────────────┘
                     [+]
              ┌──────────────┐
    ──────────│  Attention   │──────────
              └──────────────┘
                      ↑
                  from embed
```

**Design Principles:**
- Only clickable elements have green borders
- Hover adds glow effect
- 400ms zoom transition between levels
- Minimal UI - diagram IS the navigation

## Core Concept

**The transformer diagram IS the interface.** Navigate by clicking, zoom by granularity. No sidebar, no tool list—just the thing itself.

## Architecture: Navigation System + Content Plugins

```
                    ┌─────────────────────────────────┐
                    │      NAVIGATION SYSTEM          │  ← Mode-agnostic core
                    │  ┌───────────────────────────┐  │
                    │  │ Diagram    Breadcrumb     │  │
                    │  │ State      Router         │  │
                    │  │ Explanation Slot          │  │
                    │  │ Data Slot                 │  │
                    │  └───────────────────────────┘  │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              ┌─────┴─────┐                   ┌─────┴─────┐
              │ Extraction │                   │ Inference │
              │  Plugin    │                   │  Plugin   │
              │ • explns   │                   │ • explns  │
              │ • fetchers │                   │ • fetchers│
              │ • renderers│                   │ • renderers│
              └───────────┘                   └───────────┘
```

The navigation system doesn't know about extraction or inference. It just knows:
- Current position: `{mode, tokenScope, layerScope}`
- How to render a diagram with highlights
- How to render a breadcrumb with dropdowns
- Where to put explanation text
- Where to put data visualizations

Plugins register themselves:
- "For position X, here's the explanation"
- "For position X, here's how to fetch data"
- "For position X, here's how to render"

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  [Main]   [Overview]   [Dev ▾]                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  [Inference ▾]  ›  [All Tokens ▾]  ›  [Layer 16]                               │
│                                                                                 │
│                        ┌────────────────────┐                                   │
│      [1][2][3]·····[T] │░░░░░░░░░░░░░░░░░░░░│ 25                               │
│                        │░░░░░░░░░░░░░░░░░░░░│  ·                               │
│                        │████████████████████│ 16 ← you are here                │
│                        │░░░░░░░░░░░░░░░░░░░░│  ·                               │
│                        │░░░░░░░░░░░░░░░░░░░░│ 0                                │
│                        └────────────────────┘                                   │
│                          (click to navigate)                                    │
│                                                                                 │
│  ───────────────────────────────────────────────────────────────────────────── │
│                                                                                 │
│  ## Layer 16: The Semantic Middle                                               │
│                                                                                 │
│  Layer 16 sits in the middle of Gemma 2B's 26 layers. Middle layers encode      │
│  semantic meaning—past token-level processing, before output formatting.        │
│  Trait vectors extracted here generalize best across distributions.             │
│                                                                                 │
│                                    ↓ scroll ↓                                   │
│  ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│  [Data visualizations for this granularity level]                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## State Space

### Dimensions

| Dimension | Values | Notes |
|-----------|--------|-------|
| **Mode** | `extraction` \| `inference` | Training-time vs runtime |
| **Token Scope** | `all` \| `single(t)` | Only for inference mode |
| **Layer Scope** | `all` \| `single(l)` | Both modes |
| **Prompt Set** | `baseline`, `single_trait`, etc. | Only for inference |
| **Prompt ID** | `1`, `2`, ... | Only for inference |
| **Selected Traits** | `[trait1, trait2, ...]` | Which traits to display |

### Valid Granularity Combinations

```
EXTRACTION MODE
───────────────
extraction × all_layers        → Trait Extraction heatmaps, method comparison
extraction × layer(L)          → Vector inspector, top examples (future)

INFERENCE MODE
──────────────
inference × all_tokens × all_layers    → Full heatmap (tokens × layers)
inference × all_tokens × layer(L)      → Line chart over tokens at layer L
inference × token(t) × all_layers      → Vertical slice (layers for one token)
inference × token(t) × layer(L)        → Microscope (SAE features, attention)
```

### State Object

```javascript
window.zoomState = {
  // Granularity
  mode: 'inference',           // 'extraction' | 'inference'
  tokenScope: 'all',           // 'all' | number (token index)
  layerScope: 'all',           // 'all' | number (layer 0-25)

  // Context (inference only)
  promptSet: 'single_trait',
  promptId: 1,

  // Display
  selectedTraits: ['behavioral/refusal', 'cognitive_state/confidence'],
  experiment: 'gemma_2b_cognitive_nov21',

  // Derived (computed)
  totalTokens: 47,             // from loaded data
  promptTokenCount: 12,        // from loaded data
};
```

---

## Data Mapping

### What Exists at Each Granularity

| Granularity | Data Source | Current Tool Equivalent |
|-------------|-------------|------------------------|
| extraction × all_layers | `extraction_evaluation.json` | Trait Extraction |
| extraction × layer(L) | `{method}_layer{L}_metadata.json` | (new) |
| inference × all × all | `residual_stream/{set}/{id}.json` | Trait Trajectory |
| inference × all × layer(L) | same, filtered | Trait Dynamics |
| inference × token(t) × all | same, sliced | (new vertical view) |
| inference × token(t) × layer(L) | same + SAE data | Layer Deep Dive |

### API Endpoints (reuse from serve.py)

| Endpoint | Used By |
|----------|---------|
| `/api/experiments` | Experiment selector |
| `/api/experiments/{exp}/traits` | Trait selector |
| `/api/experiments/{exp}/inference/prompt-sets` | Prompt selector |
| `extraction_evaluation.json` | Extraction × All Layers |
| `residual_stream/{set}/{id}.json` | All inference granularities |
| `sae/{set}/{id}_sae.pt.json` | Microscope view |

---

## UI Components

### 1. Tab Bar (Top)

```
┌─────────────────────────────────────────────────────────────────┐
│  [Main]   [Overview]   [Dev ▾]                                  │
└─────────────────────────────────────────────────────────────────┘
```

- **Main**: The granularity-based navigation (this design)
- **Overview**: Project documentation (renders docs/overview.md)
- **Dev**: Dropdown with Data Explorer, experimental tools

### 2. Breadcrumb Navigation

```
[Inference ▾]  ›  [All Tokens ▾]  ›  [Layer 16]
     │                 │                 │
     ├── Extraction    ├── All Tokens    └── (current, not dropdown)
     └── Inference ✓   ├── Token 1
                       ├── Token 2
                       └── ...
```

- Each segment is a dropdown (except the last)
- Clicking opens options for that dimension
- Shows current path through granularity space

### 3. Transformer Diagram (Clickable)

**Extraction Mode:**
```
         (128 examples)
    ┌──────────────────────┐
    │░░░░░░░░░░░░░░░░░░░░░░│ 25
    │░░░░░░░░░░░░░░░░░░░░░░│  ·
    │░░░░░░░░░░░░░░░░░░░░░░│ 16  ← hover: "Click to focus"
    │░░░░░░░░░░░░░░░░░░░░░░│  ·
    │░░░░░░░░░░░░░░░░░░░░░░│ 0
    └──────────────────────┘
```

**Inference Mode (All Layers):**
```
    [1][2][3][4][5]·····[47]    ← click token to select
    ┌──────────────────────┐
    │░░░░░░░░░░░░░░░░░░░░░░│ 25  ← click layer to focus
    │░░░░░░░░░░░░░░░░░░░░░░│  ·
    │████████████████████████ 16  ← highlighted if selected
    │░░░░░░░░░░░░░░░░░░░░░░│  ·
    │░░░░░░░░░░░░░░░░░░░░░░│ 0
    └──────────────────────┘
         ↑ prompt │ response ↑
```

**Inference Mode (Single Layer):**
```
    [1][2][3][4][5]·····[47]
    ┌──────────────────────┐
    │                      │
    │      Layer 16        │
    │    ┌──────┬──────┐   │
    │    │ Attn │ MLP  │   │  ← future: click for deeper
    │    └──────┴──────┘   │
    │                      │
    └──────────────────────┘
    [← back to all layers]
```

### 4. Context Controls (Minimal)

For inference mode, need to select prompt set + ID. Two options:

**Option A: Inline in diagram area**
```
    Prompt: [single_trait ▾] #[3 ▾]    Traits: [2 selected ▾]

    [1][2][3][4][5]·····[47]
    ┌──────────────────────┐
    ...
```

**Option B: Floating controls**
```
    ┌─────────────────────────────────────────────┐
    │  [single_trait ▾] [#3 ▾]  |  [⚙ Traits]    │  ← fixed top-right
    └─────────────────────────────────────────────┘
```

**Recommendation**: Option A (inline) keeps everything in one place.

### 5. Explanation Section

Context-sensitive explanation based on current granularity:

```javascript
const EXPLANATIONS = {
  'extraction:all': {
    title: 'Extraction: All Layers',
    content: `
      Trait vectors are extracted at each layer using multiple methods.
      The heatmaps below show validation accuracy by layer and method.
      Higher accuracy = better separation between positive and negative examples.
    `
  },
  'extraction:layer': {
    title: 'Extraction: Layer {layer}',
    content: `
      Layer {layer} is a {position} layer. {layer_description}
      The vectors extracted here have the following properties...
    `
  },
  'inference:all:all': {
    title: 'Inference: Full Trajectory',
    content: `
      The heatmap shows trait activation across all tokens (x-axis) and
      all layers (y-axis). Click a layer to focus, or a token to slice.
    `
  },
  // ... etc
};
```

### 6. Data Section

Renders appropriate visualization for current granularity:

| Granularity | Visualization |
|-------------|---------------|
| extraction × all | Layer×method heatmaps, method comparison bar, similarity matrix |
| extraction × layer(L) | Vector stats, top examples, dimension distribution |
| inference × all × all | Tokens×layers heatmap per trait |
| inference × all × layer(L) | Line chart: tokens on x-axis, activation on y-axis |
| inference × token(t) × all | Bar chart: layers on x-axis, activation on y-axis |
| inference × token(t) × layer(L) | SAE features, attention weights, raw activation |

---

## File Structure

```
zoomies/
├── index.html                  # Shell: tabs, slots
├── styles.css                  # Styles with zoom transitions
├── PLAN.md                     # This file
│
├── core/                       # Navigation system (mode-agnostic)
│   ├── state.js               # State object + setState() + render trigger
│   ├── router.js              # URL ↔ state sync
│   ├── paths.js               # Load path templates from config/paths.yaml
│   └── registry.js            # Plugin registration (explanations, fetchers, renderers)
│
├── components/                 # Reusable UI pieces
│   ├── tabs.js                # [Main] [Overview] [Dev]
│   ├── diagram.js             # Clickable transformer SVG
│   ├── breadcrumb.js          # Dropdown path navigation
│   └── selectors.js           # Experiment, prompt, trait (inline expandable)
│
├── plugins/                    # Content plugins (register into core)
│   ├── extraction.js          # Extraction: explanations, fetchers, renderers
│   └── inference.js           # Inference: explanations, fetchers, renderers
│
└── views/                      # Top-level tab content
    ├── main.js                # Assembles: breadcrumb + diagram + explanation + data
    ├── overview.js            # Renders docs/overview.md
    └── dev.js                 # Data Explorer container
```

### Component Interfaces

**diagram.js** — knows nothing about modes, just renders based on props:
```javascript
renderDiagram({
  showTokens: boolean,           // whether to show token bar
  tokenCount: number | null,     // how many tokens (null = show "...")
  promptTokenCount: number,      // where prompt/response divider goes
  selectedToken: 'all' | number,
  selectedLayer: 'all' | number,
  onTokenClick: (t) => void,
  onLayerClick: (l) => void,
});
```

**breadcrumb.js** — knows nothing about what data exists:
```javascript
renderBreadcrumb({
  segments: [
    { label: 'Inference', value: 'inference', options: ['extraction', 'inference'] },
    { label: 'All Tokens', value: 'all', options: ['all', 0, 1, 2, ...] },
    { label: 'Layer 16', value: 16, options: null },  // null = not a dropdown
  ],
  onChange: (segmentIndex, newValue) => void,
});
```

**registry.js** — plugins register content:
```javascript
// Plugin registration
registry.explanation('extraction:all', { title: '...', content: '...' });
registry.explanation('inference:all:layer', { title: '...', content: '...' });

registry.fetcher('extraction:all', async (state) => fetch(...));
registry.fetcher('inference', async (state) => fetch(...));  // one fetcher, slice in renderer

registry.renderer('extraction:all', (data, container) => { ... });
registry.renderer('inference:all:all', (data, container) => { ... });

// Core calls these based on current state
const key = getPositionKey(state);  // e.g., 'inference:all:layer'
const explanation = registry.getExplanation(key);
const data = await registry.getFetcher(key)(state);
registry.getRenderer(key)(data, container);
```

---

## Navigation Flows

### Diagram Click Navigation

```
User clicks layer 16 in diagram
  → state.layerScope = 16
  → breadcrumb updates to show "Layer 16"
  → diagram re-renders (zoomed to layer 16)
  → explanation updates
  → data section re-renders with layer-16-specific viz

User clicks "back" or breadcrumb "All Layers"
  → state.layerScope = 'all'
  → returns to zoomed-out view
```

### Breadcrumb Navigation

```
User clicks [Inference ▾] dropdown
  → shows: Extraction, Inference ✓
  → user selects "Extraction"
  → state.mode = 'extraction'
  → state.tokenScope = 'all' (reset, not applicable)
  → diagram switches to extraction mode
  → data fetches extraction_evaluation.json
```

### URL Sync

```
State: { mode: 'inference', tokenScope: 'all', layerScope: 16, promptSet: 'single_trait', promptId: 3 }
URL:   ?mode=inference&tokens=all&layer=16&set=single_trait&id=3

User bookmarks → can return to exact view
```

---

## Edge Cases

### 1. No Data Available

```
if (!data) {
  renderExplanation();  // Still show what this granularity WOULD show
  renderNoDataMessage("No inference data for this prompt. Run capture_raw_activations.py first.");
}
```

### 2. Switching Modes Resets Scope

```
// When switching extraction → inference
if (newMode === 'inference' && oldMode === 'extraction') {
  state.tokenScope = 'all';  // Reset to sensible default
  state.layerScope = 'all';
}
```

### 3. Token Count Unknown Until Data Loads

```
// Diagram shows placeholder until data loads
if (!state.totalTokens) {
  renderDiagram({ tokens: '...' });  // Show dots
}
// After data loads
state.totalTokens = data.tokens.length;
renderDiagram({ tokens: state.totalTokens });
```

### 4. SAE Data Only for Layer 16

```
// In microscope view (token × layer)
if (state.layerScope === 16) {
  renderSAEFeatures(data);
} else {
  renderNoSAEMessage("SAE features only available for layer 16");
  renderBasicActivation(data);  // Still show what we have
}
```

### 5. Trait Selection Persistence

```
// Traits persist across navigation
// Stored in localStorage
localStorage.setItem('zoomies_traits', JSON.stringify(state.selectedTraits));

// On load
state.selectedTraits = JSON.parse(localStorage.getItem('zoomies_traits')) || defaultTraits;
```

### 6. Empty Trait Selection

```
if (state.selectedTraits.length === 0) {
  renderMessage("Select at least one trait to visualize");
  // Or: auto-select first available trait
  state.selectedTraits = [availableTraits[0]];
}
```

---

## Implementation Phases

### Phase 1: Shell ✓
The minimal skeleton that can display something.
- [x] `index.html` — tabs, layout slots
- [x] `styles.css` — minimal primitives (copy essentials from visualization/styles.css)
- [x] `core/state.js` — state object, setState(), render trigger
- [x] `components/tabs.js` — switch between Main/Overview/Dev (static content for now)

### Phase 2: Diagram ✓
The clickable transformer visualization.
- [x] `components/diagram.js` — render static SVG (layers as rectangles)
- [x] Add layer highlighting (selectedLayer prop)
- [x] Add layer click handlers → `onLayerClick`
- [x] Add token bar (conditional on `showTokens`)
- [x] Add token click handlers → `onTokenClick`
- [x] Add prompt/response divider line

### Phase 3: Breadcrumb ✓
The dropdown navigation path.
- [x] `components/breadcrumb.js` — render segments as text
- [x] Make segments into dropdowns
- [x] Wire `onChange` to state
- [x] Sync diagram highlights with breadcrumb state

### Phase 4: Router + State Polish ✓
URL sync and state management.
- [x] `core/router.js` — state ↔ URL
- [x] History pushState on navigation
- [x] Load state from URL on page load
- [x] localStorage for selectedTraits, experiment

### Phase 5: Registry + Slots ✓
The plugin pattern for content.
- [x] `core/registry.js` — register/get explanations, fetchers, renderers
- [x] `views/main.js` — wire slots to registry
- [x] Explanation slot renders based on position
- [x] Data slot renders based on position (placeholder for now)

### Phase 6: Paths + Data Fetching ✓
Connect to real data.
- [x] `core/paths.js` — load from config/paths.yaml
- [x] `serve.py` — standalone server (port 8001 to avoid conflict)
- [x] Fetch utilities with error handling

### Phase 7: Extraction Plugin ✓
First content plugin.
- [x] `plugins/extraction.js` — register explanations
- [x] Register fetcher for extraction_evaluation.json
- [x] Register renderer for `extraction:all` (layer×method heatmap)
- [x] Register renderer for `extraction:layer` (vector details)

### Phase 8: Inference Plugin ✓
Second content plugin.
- [x] `plugins/inference.js` — register explanations
- [x] Register fetcher for residual_stream JSON
- [x] Register renderer for `inference:all:all` (tokens×layers heatmap)
- [x] Register renderer for `inference:all:layer` (line chart)
- [x] Register renderer for `inference:token:all` (vertical bar)
- [x] Register renderer for `inference:token:layer` (SAE features)

### Phase 9: Selectors ✓
Context controls.
- [x] `components/selectors.js` — experiment dropdown
- [x] Prompt set + ID selector (inline, for inference)
- [x] Trait selector (inline expandable)
- [x] Wire to state

### Phase 10: Dev Tab + Polish ✓
Final pieces.
- [x] `views/dev.js` — embed Data Explorer or link to it
- [x] `views/overview.js` — render docs/overview.md
- [ ] Edge case handling (no data, loading states) — basic handling in place
- [ ] Error boundaries — not yet implemented

---

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| serve.py | Standalone (reads paths.yaml) | True isolation |
| Prompt selector | Inline above diagram | Minimal, no floating element |
| Trait selector | Inline expandable | Minimal, no modal |
| Token selection | Click only | Minimal, no extra slider |
| Dev tab | Just Data Explorer | Minimal, add more later if needed |
| Build order | Navigation system first | Mode-agnostic core before content plugins |

---

## Success Criteria

1. **Clean**: No sidebar, no tool list. Diagram + breadcrumb + content.
2. **Navigable**: Can reach any granularity in ≤3 clicks
3. **Educational**: Explanation appears before data, in context
4. **Functional**: All current tool capabilities accessible (maybe reorganized)
5. **Isolated**: Zero changes to existing visualization/ or experiments/
6. **Deletable**: If it doesn't work, `rm -rf zoomies/` and done

---

## Reference Files (Read These First)

Before implementing, read these files to understand the context:

| File | Why |
|------|-----|
| `config/paths.yaml` | Path templates for all data. Single source of truth. |
| `visualization/styles.css` | CSS primitives to copy/adapt (colors, layout patterns) |
| `visualization/serve.py` | API endpoints to replicate or reference |
| `visualization/core/paths.js` | How JS loads paths.yaml (pattern to follow) |
| `visualization/core/state.js` | Existing state management pattern |
| `experiments/gemma_2b_cognitive_nov21/` | Example experiment structure (ls -la to see layout) |

### Data Structure Examples

To understand JSON schemas, read actual files:

```bash
# Extraction evaluation (what extraction:all renders)
cat experiments/gemma_2b_cognitive_nov21/extraction/extraction_evaluation.json | head -100

# Residual stream projection (what inference views render)
cat experiments/gemma_2b_cognitive_nov21/inference/behavioral_tendency/defensiveness/residual_stream/single_trait/1.json | head -100

# Prompt set definition
cat inference/prompts/single_trait.json
```

### Constants

- **Model**: Gemma 2B IT (`google/gemma-2-2b-it`)
- **Layers**: 26 (indexed 0-25)
- **Hidden dim**: 2304
- **Extraction methods**: `mean_diff`, `probe`, `ica`, `gradient`, `pca_diff`, `random_baseline`
- **Prompt sets**: `baseline`, `single_trait`, `multi_trait`, `dynamic`, `adversarial`, `real_world`

---

## Remaining Work

- [ ] Error boundaries for failed data fetches
- [ ] Loading spinners during data fetch
- [ ] SAE feature integration (layer 16 microscope view)
- [ ] Attention pattern visualization
- [ ] Performance optimization for large token counts
