# Visualization Architecture

## Overview

Modular single-page application for trait interpretation data visualization.

## Structure

```
visualization/
├── index.html              # Shell - loads modules and router
├── styles.css              # All CSS styles
├── serve.py                # Development server with API endpoints
├── core/                   # Core functionality
│   ├── utils.js           # Shared utilities (escapeHtml, smoothData, math block protection)
│   ├── paths.js           # Centralized PathBuilder (loads from config/paths.yaml)
│   ├── model-config.js    # Model config loader (loads from config/models/)
│   ├── state.js           # Global state, experiment loading, URL routing
│   ├── prompt-picker.js   # Prompt selection UI for inference views
│   └── conversation-tree.js # Multi-turn chat data structure
└── views/                  # View modules (render functions)
    ├── overview.js            # Methodology documentation (markdown + KaTeX)
    ├── findings.js            # Research findings from docs/viz_findings/
    ├── trait-extraction.js    # Extraction quality (Trait Development)
    ├── steering-sweep.js      # Layer sweep + multi-layer heatmap (Trait Development)
    ├── trait-dynamics.js      # Per-token trait trajectories (Inference)
    ├── layer-deep-dive.js     # Attention + SAE features (Inference)
    └── live-chat.js           # Interactive chat with real-time trait monitoring
```

## Architecture Principles

### 1. Separation of Concerns

**Core Layer** - State and configuration management:
- `state.js` - Manages experiments, traits, theme, navigation
- `paths.js` - Centralized PathBuilder that loads templates from `config/paths.yaml`
- `model-config.js` - Model architecture config (layers, hidden size, SAE paths)

**View Layer** - Independent rendering modules:
- Each view is self-contained
- Accesses state via `window.state.*`
- Uses `window.paths.*` for URL construction
- Fetches data with native `fetch()` API
- No direct dependencies between views

**Router** - URL-based dispatch in index.html:
```javascript
window.renderView = function() {
    switch (window.state.currentView) {
        case 'overview': window.renderOverview(); break;
        case 'trait-extraction': window.renderTraitExtraction(); break;
        case 'steering-sweep': window.renderSteeringSweep(); break;
        case 'trait-dynamics': window.renderTraitDynamics(); break;
        // ... more views
    }
};
```

URL routing syncs `state.currentView` with `?tab=` query parameter for bookmarkable links.

### 2. Global State Access

All modules access shared state through `window` object:

```javascript
// Accessing state
window.state.experimentData
window.state.currentPromptSet   // e.g., 'single_trait'
window.state.currentPromptId    // e.g., 1
window.state.selectedTraits

// Path building
window.paths.setExperiment('my_experiment');
window.paths.residualStreamData(trait, promptSet, promptId);
window.paths.vectorMetadata(trait, method, layer);

// Utility functions (state.js)
window.getFilteredTraits()
window.getDisplayName(traitName)
window.renderMath(element)       // Render KaTeX math in any element

// Shared utilities (utils.js)
window.escapeHtml(text)          // HTML escape for safe rendering
window.smoothData(data, windowSize)  // Centered moving average
window.protectMathBlocks(markdown)   // Protect LaTeX from markdown parser
window.restoreMathBlocks(html, blocks)
```

### 3. Data Fetching Pattern

Views fetch data directly using paths and fetch:

```javascript
// Residual stream data
const url = window.paths.residualStreamData(trait, promptSet, promptId);
const response = await fetch(url);
const data = await response.json();

// Vector metadata
const metaUrl = window.paths.vectorMetadata(trait, method, layer);
const metaResponse = await fetch(metaUrl);
const metadata = await metaResponse.json();

// Integrity data (from serve.py API)
const integrityUrl = `/api/integrity/${experiment}.json`;
const integrityResponse = await fetch(integrityUrl);
const integrity = await integrityResponse.json();
```

### 4. Prompt Picker

Inference views (`trait-dynamics`, `layer-deep-dive`) share a prompt picker fixed at the bottom of the page, rendered by `renderPromptPicker()` in `prompt-picker.js`.

```
┌─────────────────────────────────────────────────────────┐
│ Header                                                   │
├─────────────────────────────────────────────────────────┤
│ View content (visualizations)                            │
│                                                          │
│                                                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Prompt Picker (fixed bottom, 700px centered)             │
│ [prompt set ▼] [1] [2] [3] ...                          │
│ Prompt: "What year was..."                              │
│ Response: "The Treaty..."                               │
│ Token: [=====slider=====] 42                            │
└─────────────────────────────────────────────────────────┘
```

- **Shows** for inference views only (hidden for trait development views)
- **Persists** selection to localStorage (restored on page refresh)
- Prompt/response fetched once and cached in `state.promptPickerCache`
- Views access current selection via `state.currentPromptSet` and `state.currentPromptId`

### 5. Module Loading

Modules load via `<script>` tags in order:
1. Core modules (utils, paths, model-config, state, prompt-picker, conversation-tree)
2. View modules (all views)
3. Router (dispatches to views)

## Adding a New View

1. **Create view file**: `views/my-view.js`

```javascript
async function renderMyView() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    // Fetch data using paths
    const url = window.paths.someData(trait, ...);
    const response = await fetch(url);
    const data = await response.json();

    // Render
    contentArea.innerHTML = '...';
}

// Export to global scope
window.renderMyView = renderMyView;
```

2. **Load in index.html**:
```html
<script src="views/my-view.js"></script>
```

3. **Add to router**:
```javascript
case 'my-view':
    window.renderMyView();
    break;
```

4. **Add navigation item** (in HTML):
```html
<div class="nav-item" data-view="my-view">
    <span class="icon">🎯</span>
    <span>My View</span>
</div>
```

## Server API Endpoints

The `serve.py` server provides:

| Endpoint | Description |
|----------|-------------|
| `/api/experiments` | List all experiments |
| `/api/experiments/{name}/traits` | List traits for experiment |
| `/api/integrity/{name}.json` | Cached integrity check data |
| `/api/experiments/{name}/inference/prompt-sets` | List prompt sets with available IDs |

Integrity data is cached on server startup by running `data_checker.py` for each experiment.

## Development

### Start Server
```bash
cd /path/to/trait-interp
python visualization/serve.py
# Visit http://localhost:8000/
```

### Debugging
- Browser console shows: Experiment loading, trait counts, data fetches
- Network tab shows: API calls, file loads
- Check `window.state` in console for current state
- Check `window.paths` for path templates

## Views Summary

| View | Category | Purpose |
|------|----------|---------|
| overview | Documentation | Methodology docs (markdown + KaTeX) |
| findings | Documentation | Research findings from docs/viz_findings/ |
| trait-extraction | Trait Development | Extraction quality: methods, metrics, scoring, heatmaps |
| steering-sweep | Trait Development | Layer sweep + multi-layer steering heatmap |
| live-chat | Inference | Interactive chat with real-time trait monitoring |
| trait-dynamics | Inference | Token-by-token trajectories, layer×token heatmaps, activation velocity |
| layer-deep-dive | Inference | Attention heads and SAE features for single layer |
