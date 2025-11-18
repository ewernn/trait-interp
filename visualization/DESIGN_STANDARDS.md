# Design Standards - Trait Interp Visualization

**Last updated: 2025-11-17**

These are the **enforced** design rules. If something doesn't follow these, fix it without asking.

---

## Core Principles

1. **Airy layout, compact content** - Give breathing room between major sections, but keep the actual data/content tight
2. **One flat background** - Everything on the same background color (no cards, no panels with different backgrounds)
3. **No borders** - Use spacing and subtle visual hierarchy instead of lines
4. **Minimal padding** - Content should fill available space efficiently
5. **Subtle interactions** - Hover = slight color change, nothing dramatic

---

## Color System

### Background
```
Single background: #1a1a1a (dark mode primary)
No secondary backgrounds, no cards, no panels with different colors
```

### Text
```
Primary:   #e0e0e0  (main content)
Secondary: #b0b0b0  (labels, less important)
Tertiary:  #888888  (hints, metadata)
```

### Interactive
```
Accent:       #4a9eff  (links, selected items)
Hover:        Slightly brighter version of base color
Success:      #4caf50
Warning:      #f44336
```

### CRITICAL: No Hardcoded Colors
```
❌ NEVER use: color: white, color: black, color: #000, color: rgb(0,0,0)
✅ ALWAYS use: color: var(--text-primary), var(--text-secondary), var(--bg-primary)

All text must work in BOTH dark and light modes.
Use CSS variables for all colors - no exceptions.
```

---

## Typography

```
Font: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif
Monospace: 'Courier New', monospace (for file trees, data)
Base size: 12px
Line height: 1.2-1.3 (tight)
Headers: 13px (H3), 14px (H2) - barely larger than body
Weight: 400 (normal), 600 (headers only)
```

**Usage:**
- Keep text small and tight (11-12px)
- Headers should be barely larger than body text
- Use weight (not size) for emphasis
- Use monospace for CLI-style trees and data displays

---

## Spacing

```
Section spacing (airy):  32px between major page sections
                         16px between section and content
Content spacing (tight): 0-2px between tree items (CLI-style)
                         4px between related items
                         8px between groups
                         16px max for content padding
```

**Rules:**
- Give air between major sections (Data Explorer vs Vector Analysis)
- Keep actual content elements tight (graph + legend + controls)
- Internal padding should be minimal (8px max, usually 2-4px)
- Tree views: 0 margin between items, minimal padding (1-2px)
- Stats: inline format, 16px gap between items

---

## Components

### Buttons
```
Padding: 4-6px 8-10px
Font size: 11-12px
Border radius: 4px
Background: transparent
Hover: Color change only (to var(--primary-color))
NO large click targets, NO thick borders, NO shadows
```

### Stats Display
```
Format: Inline "Label: Value" pairs
Gap: 16px between items
Font size: Label 10px, Value 14px
Display: flex, not grid
NO cards, NO separate backgrounds
```

### Tree Views (File Explorer)
```
Font: Monospace (Courier New)
Font size: 12px
Line height: 1.2
Padding: 1-2px vertical, 0 margin between items
Arrows: ▸/▾ using CSS ::before
Indentation: 12px per level
NO borders, NO backgrounds, NO separators
```

### Charts/Graphs
```
NO buffer regions - content fills available space
Axis labels: 8-10px
Legends: 10-11px, compact spacing
Margins: Minimal (l: 25-50, r: 5-10, t: 5-10, b: 20-50)
Height: Reduced 20-40% from defaults
```

### Input fields
```
Padding: 4px 8px
Font size: 11px
Border radius: 4px
Background: transparent
Focus: Subtle color change, no thick outline
```

### Tool Explanations
```
Summary: One simple sentence starting with "We can..."
Details: Expandable section with technical details
Style: General-audience language, avoid researcher jargon
Format: "We can [action]" pattern (approximate, watch, inspect, pinpoint)
Example: "We can approximate what a model is 'thinking' for a given trait..."
NO complex math in summaries, save for expandable details
```

---

## Layout

### Container
```
Max width: 100vw
Padding: 16px (minimal page margins)
Background: #1a1a1a (the one true background)
```

### Sections
```
Margin between sections: 32px (airy)
NO background colors, NO borders, NO boxes
Separation by whitespace only
```

### Sidebar
```
Width: 180px (compact)
Background: Same as page (#1a1a1a)
Separation: Just spacing, no border
```

---

## Interactions

### Hover
```
Elements: Brighten 10-15%
Cursor: pointer
Transition: 0.15s ease
NO scale, NO shadow, NO border changes
```

### Active/Selected
```
Text color: #4a9eff (accent)
Background: Slightly brighter (#2d2d2d)
NO thick borders, NO dramatic changes
```

### Focus
```
Outline: 1px solid #4a9eff (subtle)
NO thick rings, NO shadows
```

---

## What NOT to do

❌ Different background colors for cards/panels
❌ Borders or dividing lines
❌ Large padding/margins inside content areas (>8px internal)
❌ Buffer regions in graphs
❌ Big clicky buttons with shadows
❌ Dramatic hover effects (scale, shadow, thick borders)
❌ Large font sizes (>14px except rare cases)
❌ Loose line-height (>1.3)
❌ Grid layouts for stats (use inline flex)
❌ Centered content in trees/lists
❌ Redundant UI elements (e.g., both CSS arrows and DOM arrows)

---

## What TO do

✅ One flat background everywhere
✅ Separation by spacing (not borders)
✅ Tight internal spacing (1-4px), airy external spacing (16-32px)
✅ Small, efficient text (11-12px)
✅ Subtle rounded corners (4px)
✅ Minimal hover feedback (just color)
✅ Compact controls
✅ Content fills available space
✅ CLI-style tree views for file explorers
✅ Inline stats format (Label: Value)
✅ Monospace fonts for data/trees
✅ Left-aligned everything (no centering unless intentional)

---

## Implementation Notes

- Use CSS variables for all colors
- All spacing should use 4px increments (4, 8, 12, 16, 32)
- Borders should be removed, not set to 0 (cleaner CSS)
- Plotly charts: Override default margins/padding in config
- Remove all box-shadow properties

---

*These standards are enforced. Any deviation should be fixed immediately.*
