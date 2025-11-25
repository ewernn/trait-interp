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
│   ├── paths.js           # Centralized PathBuilder (loads from config/paths.yaml)
│   └── state.js           # Global state, experiment loading, URL routing
└── views/                  # View modules (render functions)
    ├── overview.js            # Methodology documentation (markdown + KaTeX)
    ├── data-explorer.js       # File browser with integrity check
    ├── trait-dashboard.js     # Vector quality and evaluation
    ├── trait-correlation.js   # Pairwise trait correlation matrix
    ├── all-layers.js          # Residual stream across all layers
    ├── per-token-activation.js # Per-token trait trajectories
    ├── layer-deep-dive.js     # Single layer mechanistic view
    ├── analysis-gallery.js    # Browse analysis outputs (PNGs + JSON)
    └── token-explorer.js      # Interactive per-token analysis
```

## Architecture Principles

### 1. Separation of Concerns

**Core Layer** - State and path management:
- `state.js` - Manages experiments, traits, theme, navigation
- `paths.js` - Centralized PathBuilder that loads templates from `config/paths.yaml`

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
        case 'data-explorer': window.renderDataExplorer(); break;
        case 'trait-dashboard': window.renderTraitDashboard(); break;
        // ... 6 more views
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

// Utility functions
window.getFilteredTraits()
window.getDisplayName(traitName)
window.renderMath(element)       // Render KaTeX math in any element
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

### 4. Inference Context Panel

Inference views (`all-layers`, `per-token-activation`, `layer-deep-dive`) share a common prompt picker and prompt/response display rendered by `renderInferenceContext()` in `state.js`.

```
┌─────────────────────────────────────────────────────────┐
│ Header                                                   │
├─────────────────────────────────────────────────────────┤
│ [prompt set ▼] [1] [2] [3] ...   ← Prompt picker        │
│ Prompt: "What year was..."       ← Shared context       │
│ Response: "The Treaty..."                               │
├─────────────────────────────────────────────────────────┤
│ View content (just visualizations, no prompt text)      │
└─────────────────────────────────────────────────────────┘
```

- **Shows** for inference views only (hidden for trait development views)
- Prompt/response fetched once and cached in `state.inferenceContextCache`
- Views access current selection via `state.currentPromptSet` and `state.currentPromptId`

### 5. Module Loading

Modules load via `<script>` tags in order:
1. Core modules (paths, state)
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

Integrity data is cached on server startup by running `check_available_data.py` for each experiment.

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

| View | Purpose |
|------|---------|
| data-explorer | Browse extraction files with integrity status |
| trait-dashboard | Vector quality metrics and evaluation results |
| trait-correlation | Pairwise correlation matrix between traits |
| all-layers | Residual stream projections across all layers |
| per-token-activation | Token-by-token trait trajectories |
| layer-deep-dive | Attention heads and MLP neurons for single layer |
