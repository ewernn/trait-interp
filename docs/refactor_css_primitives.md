# CSS Primitives Refactor Plan

## Goal
Consolidate duplicated CSS patterns into reusable primitives to:
1. Prevent Claude from creating duplicate styles
2. Reduce file size (~300-400 lines of duplication)
3. Make theming/maintenance easier

## Current State
- `visualization/styles.css`: 3355 lines
- Multiple duplicated patterns identified (buttons, tooltips, dropdowns, tables, toggles)

---

## Phase 1: Add Primitives Section (~120 lines)

**Location:** After `:root` and `[data-theme="dark"]` blocks (around line 98)

**Primitives to add:**
```
.btn, .btn-xs, .btn-sm, .btn-primary, .btn-ghost  (buttons)
.chip, .chip.active                                (tag/filter chips)
.panel, .panel-header, .panel-body                 (card/panel containers)
[data-tooltip]                                     (single tooltip impl)
.toggle-row                                        (checkbox + label)
.dropdown, .dropdown-header, .dropdown-body        (expandable sections)
.table, .table-compact                             (base tables)
```

**Draft:** See conversation before compaction for full CSS code.

---

## Phase 2: Audit & Map Existing Duplicates

Re-read styles.css and create mapping of what to refactor:

### Buttons (consolidate to .btn variants)
| Current Class | Line | Action |
|---------------|------|--------|
| `.pp-btn` | ~1220 | Replace with `.btn.btn-xs` |
| `.tp-btn` | ~1705 | Replace with `.btn` |
| `.rb-chip-btn` | ~3084 | Replace with `.btn.btn-xs` |
| `.steer-btn` | ~2138 | Replace with `.btn.btn-xs` |
| `.projection-toggle-btn` | ~2182 | Replace with `.btn.btn-xs` |
| `.pp-page-btn` | ~1264 | Replace with `.btn.btn-xs` + border |

### Tooltips (consolidate to [data-tooltip])
| Current Class | Line | Action |
|---------------|------|--------|
| `.has-tooltip::after` | ~1986 | Delete, use primitive |
| `.citation[data-tooltip]::after` | ~2815 | Delete, use primitive |
| `.citation-num a[data-tooltip]::after` | ~2867 | Delete, use primitive |

### Toggles (consolidate to .toggle-row)
| Current Class | Line | Action |
|---------------|------|--------|
| `.inference-mode-toggle` | ~2054 | Extend `.toggle-row` |
| `.smooth-toggle` | ~2483 | Replace with `.toggle-row` |
| `.rb-toggle` | ~3098 | Replace with `.toggle-row` |
| `.projection-toggle-checkbox` | ~2202 | Replace with `.toggle-row` |

### Dropdowns (consolidate to .dropdown)
| Current Class | Line | Action |
|---------------|------|--------|
| `.responses-dropdown` | ~2674 | Extend `.dropdown` |
| `.response-browser-details` | ~3005 | Keep (uses <details>) |
| `.tool-description` | ~868 | Keep (uses <details>) |

### Tables (consolidate to .table)
| Current Class | Line | Action |
|---------------|------|--------|
| `.data-table` | ~786 | Keep, add `.table` base |
| `.responses-table` | ~2712 | Replace with `.table.table-compact` |
| `.rb-table` | ~3218 | Replace with `.table.table-compact` |

---

## Phase 3: Update HTML/JS to Use Primitives

Files that generate HTML with these classes:

| File | Classes Used | Update |
|------|--------------|--------|
| `core/custom-blocks.js` | `.responses-dropdown`, `.responses-table` | Use `.dropdown`, `.table` |
| `core/citations.js` | `.citation-num a[data-tooltip]` | Use `[data-tooltip]` |
| `views/live-chat.js` | `.smooth-toggle`, `.steer-btn` | Use `.toggle-row`, `.btn` |
| `views/steering-sweep.js` | `.rb-*` classes | Use primitives |
| `views/trait-extraction.js` | `.data-table` | Add `.table` base |
| `views/trait-dynamics.js` | Various | Audit and update |
| `core/prompt-picker.js` | `.pp-btn` | Use `.btn.btn-xs` |

---

## Phase 4: Delete Redundant CSS

After HTML/JS updated, delete these CSS blocks:
- Duplicate tooltip implementations (~60 lines saved)
- Redundant button variants (~80 lines saved)
- Redundant toggle styles (~40 lines saved)

**Estimated savings:** ~200 lines

---

## Phase 5: Verify & Test

1. Run visualization server: `python visualization/serve.py`
2. Test each view:
   - [ ] Overview (tooltips)
   - [ ] Findings (dropdowns, tables, citations)
   - [ ] Trait Extraction (tables, buttons)
   - [ ] Steering Sweep (response browser, filters)
   - [ ] Live Chat (toggles, buttons)
   - [ ] Trait Inference (prompt picker)
3. Test dark mode toggle
4. Test mobile responsiveness

---

## Execution Order

```
1. Read full styles.css (3 chunks: 0-1500, 1500-2500, 2500-3355)
2. Add primitives section at line ~98
3. Update JS files to use new classes (Phase 3)
4. Delete redundant CSS (Phase 4)
5. Test all views (Phase 5)
6. Commit: "Refactor CSS: add primitives, consolidate duplicates"
```

---

## Files to Modify

**CSS:**
- `visualization/styles.css` (add primitives, delete duplicates)

**JS:**
- `visualization/core/custom-blocks.js`
- `visualization/core/citations.js`
- `visualization/core/prompt-picker.js`
- `visualization/views/live-chat.js`
- `visualization/views/steering-sweep.js`
- `visualization/views/trait-extraction.js`
- `visualization/views/trait-dynamics.js`

---

## Notes

- Keep view-specific styles (e.g., `.chat-message`, `.finding-card`) - these aren't duplicates
- Primitives are additive first - existing code keeps working
- Gradual migration - don't need to update everything at once
- Test after each JS file update
