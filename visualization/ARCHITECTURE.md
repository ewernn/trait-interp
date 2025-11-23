# Visualization Architecture

## Overview

Modular single-page application for trait interpretation data visualization.

## Structure

```
visualization/
├── index.html              # Shell (235 lines) - loads modules
├── styles.css              # All CSS (1,278 lines)
├── serve.py                # Development server with API endpoints
├── core/                   # Core functionality
│   ├── state.js           # Global state, experiment loading
│   └── data-loader.js     # Centralized data fetching
└── views/                  # View modules (render functions)
    ├── data-explorer.js
    ├── vectors.js
    ├── cross-distribution.js
    ├── monitoring.js
    ├── prompt-activation.js
    └── layer-dive.js
```

## Architecture Principles

### 1. Separation of Concerns

**Core Layer** - State and data access:
- `state.js` - Manages experiments, traits, theme, navigation
- `data-loader.js` - Abstract API for fetching experiment data (all-layers and layer-internals)

**View Layer** - Independent rendering modules:
- Each view is self-contained
- Accesses state via `window.state.*`
- Fetches data via `window.DataLoader.*`
- No direct dependencies between views

**Router** - Simple dispatch in index.html:
```javascript
window.renderView = function() {
    switch (window.state.currentView) {
        case 'data-explorer': window.renderDataExplorer(); break;
        // ... other views
    }
};
```

### 2. Global State Access

All modules access shared state through `window` object:

```javascript
// Accessing state
window.state.experimentData
window.state.currentPrompt
window.state.selectedTraits

// Utility functions
window.getFilteredTraits()
window.getDisplayName(traitName)
window.getPlotlyLayout(baseLayout)

// Data loading
await window.DataLoader.fetchAllLayers(trait, promptNum)
```

### 3. Module Loading

Modules load via `<script>` tags in order:
1. Core modules (state, data-loader)
2. View modules (all views)
3. Router (dispatches to views)

## Adding a New View

1. **Create view file**: `views/my-view.js`

```javascript
// My View description

async function renderMyView() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    // Render logic here
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

## Data Loading Patterns

### All Layers Data (Residual Stream Across All Layers)
```javascript
const data = await window.DataLoader.fetchAllLayers(trait, promptNum);
// Returns: { prompt, response, tokens, projections, logit_lens }
```

### Layer Internals Data (Single Layer Deep Dive)
```javascript
const data = await window.DataLoader.fetchLayerInternals(trait, promptNum, layer);
// Returns: { attention_heads, mlp_activations, etc. }
```

### Vector Metadata
```javascript
const metadata = await window.DataLoader.fetchVectorMetadata(trait, method, layer);
// Returns: { vector_norm, accuracy, separation, etc. }
```

### Preview Files
```javascript
const jsonData = await window.DataLoader.fetchJSON(trait, 'trait_definition');
const csvData = await window.DataLoader.fetchCSV(trait, 'pos', 10);
```

## Future-Proofing

When migrating to prompt-centric inference structure, only update `data-loader.js`:

```javascript
// In core/data-loader.js
static async fetchAllLayers(trait, promptNum) {
    // OLD (current):
    const url = `.../inference/residual_stream_activations/prompt_${promptNum}.json`;

    // NEW (future):
    const [prompt, proj] = await Promise.all([
        fetch(`.../inference/prompts/prompt_${promptNum}.json`),
        fetch(`.../inference/projections/${trait.name}/prompt_${promptNum}.json`)
    ]);

    return { ...await prompt.json(), projections: (await proj.json()).projections };
}
```

All 6 views continue working unchanged.

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

### Rollback
Original monolithic file preserved at `index.html.backup`.

## File Sizes

| File | Lines | Purpose |
|------|-------|---------|
| index.html | 235 | HTML shell + router |
| styles.css | 1,278 | All CSS styles |
| core/state.js | 473 | State management |
| core/data-loader.js | 186 | Data fetching API |
| views/data-explorer.js | 299 | File browser |
| views/vectors.js | 517 | Vector analysis |
| views/cross-distribution.js | 359 | Cross-dist analysis |
| views/monitoring.js | 506 | All layers view |
| views/prompt-activation.js | 249 | Per-token activation |
| views/layer-dive.js | 987 | Layer deep dive |

**Total**: 4,745 lines → Same functionality, better structure
