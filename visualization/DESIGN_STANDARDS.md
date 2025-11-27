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
- Large fonts (>14px)
- Loose line-height (>1.0)
- Grid layouts for stats (use inline flex)
- Centered content in trees/lists
- Hardcoded colors (`#000`, `white`, `rgb()`) - always use `var(--text-primary)` etc.

---

## What TO Do

- 3-level background hierarchy (primary → secondary → tertiary)
- Separation by spacing
- Tight internal spacing (1-4px), airy external (16-32px)
- Small text (11-12px)
- 2px border radius
- Minimal hover feedback (color only)
- Transparent chart backgrounds
- CLI-style tree views
- JetBrains Mono for code/data
- Left-aligned everything

---

## CSS Primitives

All views use standardized CSS classes from `styles.css`. No view-specific CSS.

### Available Primitives

| Class | Purpose |
|-------|---------|
| `.tool-view` | Dashboard wrapper (padding, max-width, font sizing) |
| `.prose` | Readable text pages (serif, centered, larger font) |
| `.card` | Content container (bg-secondary, padding, rounded) |
| `.grid` | Auto-fit columns (minmax 280px) |
| `.image-grid` | Image gallery grid (minmax 350px) |
| `.image-card` | Clickable image with caption |
| `.data-table` | Sortable data (sticky header, hover rows) |
| `.def-table` | Term/definition pairs (two columns) |
| `.info-icon` | Hover info trigger (ⓘ symbol) |
| `.tooltip` | Positioned popup content |
| `.tool-description` | Secondary text below headings |
| `.stats-row` | Inline stats display (flex, gap) |

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

```
Background:    transparent (paper_bgcolor, plot_bgcolor)
Margins:       Minimal (l: 25-50, r: 5-10, t: 5-10, b: 20-50)
```

Use `window.getPlotlyLayout()` helper for consistent styling.

### Heatmap Colorscale

```
Colorscale:  AsymB (asymmetric: vibrant red + muted blue)
Definition:  [[0, '#5a6a8a'], [0.25, '#98a4b0'], [0.5, '#c8c8c8'], [0.75, '#d47c67'], [1, '#b40426']]

CRITICAL: Colorscales must be defined as explicit arrays. Plotly.js built-in names are unreliable.
```

---

*These standards are enforced. Any deviation should be fixed immediately.*
