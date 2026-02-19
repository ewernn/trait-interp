# Frontend: Multi-Turn Rollout Visualization

## Goal

Make the trait-dynamics view render multi-turn agentic rollouts (7K-16K tokens stored as all-prompt/empty-response with `turn_boundaries` metadata) cleanly — with role-colored bands, zoom/pan, and rollout tags in the prompt picker.

## Success Criteria
- [ ] Rollout projections render without errors (no division by zero, no missing labels)
- [ ] Turn boundaries visible as colored bands (user=blue, assistant=green, system=gray, tool=orange)
- [ ] Zoom/pan works for 16K-token sequences via rangeslider
- [ ] Prompt picker shows rollout tags (e.g., "rollout", "funding_email")
- [ ] Top Spans panel hidden for rollouts (response-only feature)
- [ ] Existing single-turn behavior unchanged (no regressions)

## Prerequisites
- Response JSONs with `turn_boundaries` already exist (created by `convert_rollout.py`)
- Projection data will exist after GPU step — frontend changes don't depend on it
- `convert_rollout.py` must run BEFORE `project_raw_activations_onto_traits.py` so the response JSON (with `turn_boundaries` and `tags`) is preserved (line 597: `if not response_file.exists()`)

## Rollout Detection

A prompt is a "rollout" when `responseTokens.length === 0` (flat schema: `prompt_end === tokens.length`). This is the single signal — no new flags needed. All guards branch on this condition. Compute once after line 1225 and pass to all sub-functions.

## Steps

### Step 1: Empty Response Handling (`trait-dynamics.js`)

**Purpose**: Prevent crashes and render rollout data correctly when `projections.response = []` and `response.tokens = []`.

**Changes in `renderCombinedGraph()` (~line 1210):**

1. **Line 1217 guard** — already OK. `[]` is truthy in JS, so `!refData.response?.tokens` is `false`. No change needed.

2. **Compute `isRollout` once (after line 1225):**
   ```js
   const isRollout = responseTokens.length === 0;
   ```

3. **Line 1346-1349 — normalized mode division by zero:**
   ```js
   // BEFORE:
   const meanNorm = responseNorms.reduce((a, b) => a + b, 0) / responseNorms.length;

   // AFTER: For rollouts, use all norms (there are no response norms)
   const normsForMean = isRollout ? promptNorms : responseNorms;
   const meanNorm = normsForMean.length > 0
       ? normsForMean.reduce((a, b) => a + b, 0) / normsForMean.length
       : 1;
   ```
   Note: this changes normalization semantics for rollouts — prompt norms include system/user/tool tokens which may have different magnitudes than assistant tokens. This is an approximation; cosine mode (per-token norm) is unaffected and may be preferred for rollouts.

4. **Line 1364 — `_normalizedResponse` for Top Spans:**
   ```js
   // BEFORE:
   data._normalizedResponse = rawValues.slice(nPromptTokens - START_TOKEN_IDX);

   // AFTER: For rollouts, store all values (Top Spans will be hidden, but keep data consistent)
   data._normalizedResponse = isRollout
       ? rawValues
       : rawValues.slice(nPromptTokens - START_TOKEN_IDX);
   ```

5. **Lines 1439-1446 — separator line:**
   Skip the prompt/response separator when `isRollout`:
   ```js
   const shapes = [];
   if (!isRollout) {
       shapes.push({ type: 'line', x0: (nPromptTokens - START_TOKEN_IDX) - 0.5, ... });
   }
   shapes.push({ type: 'line', x0: highlightX, ... });  // token highlight always
   ```

6. **Lines 1468-1481 — PROMPT/RESPONSE labels:**
   Skip when `isRollout`:
   ```js
   const annotations = [];
   if (!isRollout) {
       annotations.push({ x: ..., text: 'PROMPT', ... }, { x: ..., text: 'RESPONSE', ... });
   }
   ```

7. **Line 1413 — performance: drop markers for long sequences:**
   ```js
   // BEFORE:
   mode: 'lines+markers',
   marker: { size: 2, color: color },

   // AFTER:
   mode: displayValues.length > 2000 ? 'lines' : 'lines+markers',
   ...(displayValues.length <= 2000 ? { marker: { size: 2, color: color } } : {}),
   ```

8. **Line 1554 — hide Top Spans for rollouts:**
   ```js
   // BEFORE:
   renderTopSpansPanel(traitData, filteredByMethod, responseTokens, nPromptTokens);

   // AFTER: Top Spans operates on response data only — empty for rollouts
   if (!isRollout) {
       renderTopSpansPanel(traitData, filteredByMethod, responseTokens, nPromptTokens);
   } else {
       const topSpansEl = document.getElementById('top-spans-container');
       if (topSpansEl) topSpansEl.style.display = 'none';
   }
   ```

9. **Lines 1557, 1560 — pass `isRollout` to subplots:**
   ```js
   renderTokenMagnitudePlot(traitData, filteredByMethod, tickVals, tickText, nPromptTokens, isRollout);
   renderTokenDerivativePlots(traitActivations, filteredByMethod, tickVals, tickText, nPromptTokens, traitData, isRollout);
   ```

10. **In `renderTokenMagnitudePlot()` (line 1643) and `renderTokenDerivativePlots()` (line 1703):**
    Skip separator shapes when `isRollout`:
    ```js
    // BEFORE:
    const shapes = [window.createSeparatorShape(promptEndIdx - 0.5, ...)];

    // AFTER:
    const shapes = isRollout ? [] : [window.createSeparatorShape(promptEndIdx - 0.5, ...)];
    ```

11. **Diff mode (lines 1046-1052, 1100-1107):**
    No code change needed. For rollout-vs-rollout: `diffPrompt` contains the meaningful diff, `diffResponse = []`. The chart renders prompt diffs correctly via `allProj = [...diffPrompt, ...[]]`. Top Spans is already hidden (step 8).

**Verify**: Load a rollout prompt. No JS errors. All data visible. No separator. No PROMPT/RESPONSE labels. No Top Spans.

### Step 2: Turn Boundary Rendering (`trait-dynamics.js`)

**Purpose**: Replace the prompt/response separator with colored bands showing conversation structure (roles, tool calls, thinking blocks).

**Pattern reused from**: `buildMessageRegionShapes()` in `live-chat.js:962-1006`.

**New function** (add near top of file, after existing helper functions):
```js
/**
 * Build Plotly shapes for turn boundaries in multi-turn rollouts.
 * Pattern: live-chat.js:buildMessageRegionShapes()
 */
function buildTurnBoundaryShapes(turnBoundaries) {
    if (!turnBoundaries || turnBoundaries.length === 0) return [];
    const shapes = [];
    const roleColors = {
        system:    { cssVar: '--chart-10', fallback: '#94d82d', opacity: 0.06 },
        user:      { cssVar: '--chart-1',  fallback: '#4a9eff', opacity: 0.12 },
        assistant: { cssVar: '--chart-3',  fallback: '#51cf66', opacity: 0.06 },
        tool:      { cssVar: '--chart-6',  fallback: '#ff922b', opacity: 0.12 },
    };
    for (const turn of turnBoundaries) {
        if (turn.token_start === turn.token_end) continue;
        const cfg = roleColors[turn.role] || roleColors.assistant;
        const hex = window.getCssVar?.(cfg.cssVar, cfg.fallback) || cfg.fallback;
        shapes.push({
            type: 'rect',
            x0: turn.token_start - 0.5,
            x1: turn.token_end - 0.5,
            y0: 0, y1: 1, yref: 'paper',
            fillcolor: window.hexToRgba?.(hex, cfg.opacity) || `rgba(128,128,128,${cfg.opacity})`,
            line: { width: 0 },
            layer: 'below'
        });
    }
    return shapes;
}
```

**Integration:**

1. In `renderTraitDynamics()` (~line 909), after fetching `responseData`:
   ```js
   const turnBoundaries = responseData?.turn_boundaries || null;
   ```

2. Pass `turnBoundaries` to `renderCombinedGraph()` as a new parameter.

3. In `renderCombinedGraph()`, after building the shapes array:
   ```js
   shapes.push(...buildTurnBoundaryShapes(turnBoundaries));
   ```

4. Also pass `turnBoundaries` to magnitude and velocity subplots and add the same shapes there.

**Verify**: Rollout prompts show alternating colored bands for user/assistant/tool turns. Single-turn prompts unaffected (no `turn_boundaries` field).

### Step 3: Zoom/Pan for Long Sequences (`trait-dynamics.js`)

**Purpose**: Make 7K-16K token sequences navigable.

1. **Adaptive tick spacing (line 1425):**
   ```js
   const tickStep = Math.max(10, Math.floor(displayTokens.length / 80));
   for (let i = 0; i < displayTokens.length; i += tickStep) {
   ```

2. **Rangeslider for long sequences (line 1520 xaxis config):**
   ```js
   ...(displayTokens.length > 500 ? { rangeslider: { visible: true, thickness: 0.08 } } : {})
   ```

3. **Enable scroll zoom via config override (line 1535):**
   ```js
   // BEFORE:
   window.renderChart('combined-activation-plot', traces, mainLayout);

   // AFTER:
   const plotConfig = displayTokens.length > 500 ? { scrollZoom: true } : {};
   window.renderChart('combined-activation-plot', traces, mainLayout, plotConfig);
   ```
   `renderChart()` at `charts.js:167-169` accepts `configOverrides` and spreads over `PLOTLY_CONFIG`.

4. **Apply same adaptive ticks + rangeslider to magnitude and velocity subplots** — pass `displayTokens.length` and use same threshold.

**Verify**: >500 token charts show rangeslider. Can scroll to zoom. Tick labels readable at any zoom level.

### Step 4: Prompt Picker Tags (`styles.css`)

**Purpose**: Visual differentiation of rollout prompts.

**Tag rendering already works** — `prompt-picker.js:127-138` applies CSS classes from `promptTagsCache`, populated at `:311-318` from `data.tags`.

**Add to `visualization/styles.css`:**
```css
/* Rollout prompt tags */
.pp-btn.tag-rollout { border-left: 3px solid var(--chart-3); }
.pp-btn.tag-hacker { border-left: 3px solid var(--danger); }
.pp-btn.tag-honest { border-left: 3px solid var(--success); }
.pp-btn.tag-liar { border-left: 3px solid var(--warning); }
```

**Note**: `convert_rollout.py` writes `tags: ["rollout", environment]`. Behavioral tags (`hacker`, `honest`, `liar`) should be added to response JSONs during audit rollout conversion or via a small post-processing utility.

### Step 5: Pipeline Ordering (Documentation Only)

**No code change needed.** The natural pipeline ordering preserves `turn_boundaries`:
1. `convert_rollout.py` → writes response JSON with `turn_boundaries` and `tags`
2. `capture_raw_activations.py` → reads that response JSON, saves `.pt`
3. `project_raw_activations_onto_traits.py` → `if not response_file.exists()` → skips, preserving original

Add clarifying comment at `project_raw_activations_onto_traits.py:597`:
```python
# Note: For rollout data, convert_rollout.py creates this file first
# with turn_boundaries and tags. We preserve it by not overwriting.
```

## Files Changed

| File | Changes | Effort |
|---|---|---|
| `visualization/views/trait-dynamics.js` | `isRollout` guards, `buildTurnBoundaryShapes()`, adaptive ticks, rangeslider, scroll zoom config, performance (drop markers), hide Top Spans | Medium |
| `visualization/styles.css` | 4 tag CSS classes | Trivial |
| `inference/project_raw_activations_onto_traits.py` | Clarifying comment (~1 line) | Trivial |

## What Stays Unchanged
- `visualization/serve.py` — discovery works once extraction+projections exist
- `visualization/core/annotations.js` — rollouts have no response annotations; future work if needed
- `visualization/core/charts.js` — existing primitives reused as-is
- `visualization/components/prompt-picker.js` — tag rendering already works

## Known Limitations
- **Top Spans not available for rollouts** — it's a response-only feature. Could be adapted to work on prompt data in the future.
- **Diff mode for rollouts** — chart renders prompt diffs correctly, but Top Spans (response-only) is hidden. Rollout comparison is phase 2 work.
- **Normalized mode approximation** — for rollouts, uses prompt norms as the mean (includes system/user/tool tokens). Cosine mode is preferred for rollouts.
- **Annotation bands** — the annotation system (`annotations.js`) operates in response-relative space, so it's inactive for rollouts. Not a regression.

## Performance Notes
- Markers dropped for >2000 tokens (lines only) — prevents SVG element explosion
- Rangeslider + scroll zoom for >500 tokens
- 50 turn boundary rect shapes is well within Plotly's capacity
- If still slow at 16K: switch `scatter` → `scattergl` (WebGL renderer)
