# Design Standards - Trait Interp Visualization

These are the **enforced** design rules. If something doesn't follow these, fix it without asking.

**Design Playground**: Run `python visualization/serve.py` and visit `http://localhost:8000/design` to preview and tweak colors live.

---

## Core Principles

1. **Airy layout, compact content** - Breathing room between major sections, tight data/content
2. **Layered backgrounds** - Primary (page) → Secondary (cards/inputs) → Tertiary (hover/code)
3. **Minimal borders** - 1px borders only on form inputs; use spacing elsewhere
4. **Minimal padding** - Content fills available space efficiently
5. **Subtle interactions** - Hover = slight color change, nothing dramatic

---

## Color System

### Dark Mode (default)
```
Backgrounds:
  --bg-primary:      #292929  (page background)
  --bg-secondary:    #2f2f2f  (cards, panels, inputs)
  --bg-tertiary:     #3a3a3a  (hover, code blocks)

Text:
  --text-primary:    #e0e0e0  (main content)
  --text-secondary:  #a4a4a4  (labels)
  --text-tertiary:   #7f7f7f  (hints, metadata)

Accents:
  --primary-color:   #dce39d  (links, selected, buttons)
  --success:         #548e4c
  --warning:         #a0802d
  --danger:          #aa5656

Form:
  --border-color:    #a7a7a7
  --form-accent:     #bbba87  (checkbox, radio, range)
```

### Light Mode
```
Backgrounds:
  --bg-primary:      #e0e0de  (muted gray, not white)
  --bg-secondary:    #eaeae8  (cards)
  --bg-tertiary:     #d4d4d2  (hover, code blocks)

Text:
  --text-primary:    #2a2a2a  (soft black)
  --text-secondary:  #5a5a5a
  --text-tertiary:   #888888

Accents:
  --primary-color:   #7a7950  (darker olive for contrast)
  --success:         #3d7435
  --warning:         #8a6a20
  --danger:          #994040

Form:
  --border-color:    #b0b0b0
  --form-accent:     #9a9970
```

### CRITICAL: No Hardcoded Colors
```
NEVER use: color: white, color: black, color: #000, color: rgb(0,0,0)
ALWAYS use: color: var(--text-primary), var(--text-secondary), var(--bg-primary)

All colors via CSS variables - no exceptions.
```

---

## Typography

```
Font:        -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif
Monospace:   'JetBrains Mono', monospace (file trees, code, data)
Base size:   12px
Line height: 1.0
Headers:     13px (H3), 14px (H2)
Weight:      400 (normal), 600 (headers only)
```

**Usage:**
- Text: 11-12px, tight
- Headers barely larger than body
- Use weight (not size) for emphasis
- Monospace for CLI-style trees and data

---

## Spacing

```
Section spacing (airy):  32px between major page sections
                         16px between section and content
Content spacing (tight): 0-2px between tree items
                         4px between related items
                         8px between groups
                         16px max for content padding
```

**Rules:**
- Air between major sections
- Tight content elements (graph + legend + controls)
- Internal padding: 8px max, usually 2-4px
- Tree views: 0 margin between items

---

## Components

### Buttons
```
Padding:       4-6px 8-10px
Font size:     11-12px
Border radius: 2px
Background:    transparent (ghost), var(--primary-color) (filled)
Hover:         Color change only
NO shadows, NO thick borders
```

### Form Inputs
```
Padding:       4px 8px
Font size:     11px
Border radius: 2px
Background:    var(--bg-secondary)
Border:        1px solid var(--border-color)
Focus:         Border color → var(--primary-color)
```

### Tree Views (File Explorer)
```
Font:        'JetBrains Mono', monospace
Font size:   12px
Line height: 1.2
Padding:     1-2px vertical
Arrows:      /  using CSS ::before
Indentation: 12px per level
NO borders, NO backgrounds
```

### Charts/Graphs (Plotly)
```
Background:    transparent (paper_bgcolor, plot_bgcolor)
Axis labels:   8-10px
Legends:       10-11px, compact
Margins:       Minimal (l: 25-50, r: 5-10, t: 5-10, b: 20-50)
Line width:    2px
Marker size:   4px (2px radius)
Colors:        Auto-generated (Plotly D3 category10)

Use window.getPlotlyLayout() helper for consistent styling.
```

### Heatmaps
```
Colorscale:  AsymB (asymmetric: vibrant red + muted blue)
             Optimized for [0, +1] data where positive values matter more

Definition:  [[0, '#5a6a8a'], [0.25, '#98a4b0'], [0.5, '#c8c8c8'], [0.75, '#d47c67'], [1, '#b40426']]

CRITICAL: Colorscales must be defined as explicit arrays.
          Plotly.js built-in names are unreliable.
```

---

## Layout

### Container
```
Max width:  100vw
Padding:    16px
Background: var(--bg-primary)
```

### Sections
```
Margin: 32px between sections
NO background colors, NO borders
Separation by whitespace only
```

### Sidebar
```
Width:      180px
Background: var(--bg-primary)
```

---

## Interactions

### Hover
```
Elements:   Background → var(--bg-tertiary) OR color brighten 10-15%
Cursor:     pointer
Transition: 0.15s ease
NO scale, NO shadow
```

### Active/Selected
```
Text color:  var(--primary-color)
Background:  var(--bg-tertiary)
```

### Focus
```
Outline: 1px solid var(--primary-color)
NO thick rings, NO shadows
```

---

## What NOT to do

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

---

## What TO do

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

## Implementation Notes

- All colors via CSS variables
- Spacing in 4px increments (4, 8, 12, 16, 32)
- Plotly: Use `getPlotlyLayout()` helper, transparent backgrounds
- JetBrains Mono loaded from Google Fonts CDN

---

*These standards are enforced. Any deviation should be fixed immediately.*
