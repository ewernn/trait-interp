# Design Standards - Trait Interp Visualization

These are the **enforced** design rules. If something doesn't follow these, fix it without asking.

**Design Playground**: Run `python visualization/serve.py` and visit `http://localhost:8000/design` to preview and tweak colors live.

**Implementation**: All values (colors, spacing, typography) are in `styles.css`. This doc covers guidance only.

---

## Core Principles

1. **Airy layout, compact content** - Breathing room between major sections, tight data/content
2. **Layered backgrounds** - Primary (page) → Secondary (cards/inputs) → Tertiary (hover/code)
3. **Minimal borders** - 1px borders only on form inputs; use spacing elsewhere
4. **Minimal padding** - Content fills available space efficiently
5. **Subtle interactions** - Hover = slight color change, nothing dramatic

---

## What NOT to Do

- Different background colors for cards (use the 3-level system)
- Borders except on form inputs
- Large padding (>8px internal)
- Buffer regions in graphs
- Shadows
- Dramatic hover effects (scale, shadow)
- Hardcoded font sizes - use type scale variables
- Loose line-height (>1.0)
- Grid layouts for stats (use inline flex)
- Centered content in trees/lists
- Hardcoded colors (`#000`, `white`, `rgb()`) - always use `var(--text-primary)` etc.

---

## What TO Do

- 3-level background hierarchy (primary → secondary → tertiary)
- Separation by spacing
- Tight internal spacing (1-4px), airy external (16-32px)
- Use type scale: `--text-xs` (12px) through `--text-4xl` (42px)
- 2px border radius
- Minimal hover feedback (color only)
- Transparent chart backgrounds
- CLI-style tree views
- JetBrains Mono for code/data
- Left-aligned everything
- Content centered at 900px max-width (header, tool-view)

---

## Type Scale

All font sizes use CSS variables. Never use hardcoded `px` values for font-size.

| Variable | Size | Usage |
|----------|------|-------|
| `--text-xs` | 12px | Badges, tiny labels |
| `--text-sm` | 14px | Sidebar, tables, secondary text |
| `--text-base` | 16px | Body text, tool-view default |
| `--text-lg` | 18px | h2 in tool-view, emphasis |
| `--text-xl` | 20px | Prose body, h3 |
| `--text-2xl` | 26px | Prose h2 |
| `--text-3xl` | 34px | Prose h1 |
| `--text-4xl` | 42px | Page title in header |

To scale all text up/down, modify the variables in `:root`.

---

## CSS Primitives

All views use standardized CSS classes from `styles.css`. No view-specific CSS.

### Available Primitives

| Class | Purpose |
|-------|---------|
| `.tool-view` | Dashboard wrapper (padding, 900px max-width, font sizing) |
| `.page-intro` | Centered intro section at top of tool pages (500px max-width) |
| `.page-intro-text` | Main intro text (--text-lg, primary color) |
| `.intro-example` | Code-style example box (monospace, bg-tertiary) |
| `.toc` | Table of contents container (bg-secondary, padding) |
| `.toc-title` | TOC header (uppercase, tertiary color) |
| `.toc-list` | TOC links (flex wrap, gap) |
| `.subsection-header` | Numbered section header (77px top margin, flex) |
| `.subsection-num` | Section number (tertiary color) |
| `.subsection-info-toggle` | Collapsible info trigger (► / ▼) |
| `.subsection-info` | Hidden info box (show on toggle) |
| `.scrollable-container-lg` | Scrollable area (max-height 600px) |
| `.chart-container-sm` | Small chart (height 120px) |
| `.heatmap-legend-footer` | Legend below heatmaps |
| `.prose` | Readable text pages (serif, centered, larger font) |
| `.card` | Content container (bg-secondary, padding, rounded) |
| `.grid` | Auto-fit columns (minmax 280px) |
| `.data-table` | Sortable data (sticky header, hover rows) |
| `.def-table` | Term/definition pairs (two columns) |
| `.stats-row` | Inline stats display (flex, gap) |
| `.no-data` | Centered placeholder for empty states |
| `.file-hint` | Muted count/status (tertiary color) |
| `.quality-good/ok/bad` | Color indicators for quality metrics |

### Usage Pattern

```html
<div class="tool-view">
    <section>
        <h2>Section Title</h2>
        <p class="tool-description">Explanation of what this shows</p>
        <div class="grid">
            <div class="card"><h4>Card</h4><p>Content</p></div>
        </div>
    </section>
</div>
```

### Rules

- **Use only primitives** - never create view-specific CSS
- **If no primitive fits, ask before creating a new one**
- Headings styled by wrapper (`.tool-view h2`, `.tool-view h3`, etc.)
- Use native `<section>` for grouping, not custom classes
- Inline styles only for truly one-off cases (rare)

---

## Plotly Charts

Use `core/charts.js` primitives for all charts:

```javascript
// Basic usage
const layout = window.buildChartLayout({
    preset: 'layerChart',  // or 'timeSeries', 'heatmap', 'barChart'
    traces,
    height: 200,
    legendPosition: 'below',  // 'above', 'right', 'none'
    yaxis: { title: 'Score' }
});
window.renderChart(divId, traces, layout);

// With shapes (for token highlighting)
shapes: [
    window.createSeparatorShape(promptEndIdx),
    window.createHighlightShape(currentTokenX)
]

// For updates (efficient re-render)
window.updateChart(divId, traces, layout);
```

**Presets**: `timeSeries`, `layerChart`, `heatmap`, `barChart` — all overridable.

**Legend positioning**: `legendPosition: 'below'` (default) calculates margin automatically based on trace count.

### Heatmap Colorscale

```
Colorscale:  AsymB (asymmetric: vibrant red + muted blue)
Definition:  [[0, '#5a6a8a'], [0.25, '#98a4b0'], [0.5, '#c8c8c8'], [0.75, '#d47c67'], [1, '#b40426']]

CRITICAL: Colorscales must be defined as explicit arrays. Plotly.js built-in names are unreliable.
```

---

*These standards are enforced. Any deviation should be fixed immediately.*
