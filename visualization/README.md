# Visualization Dashboard

Interactive dashboard for exploring trait vectors, monitoring inference, and analyzing model internals.

## Quick Start

```bash
python visualization/serve.py  # Visit http://localhost:8000/
```

Auto-discovers experiments and traits from `experiments/` directory.

---

## Views & Data Requirements

### Global Views (no experiment required)

| View | Source | Notes |
|------|--------|-------|
| **Overview** | `docs/overview.md` | Markdown + KaTeX |
| **Methodology** | `docs/methodology.md` | Custom directives (`:::figure:::`, `:::responses:::`, `:::dataset:::`, `:::chart:::`) |
| **Findings** | `docs/viz_findings/index.yaml` + `*.md` | Collapsible cards. Deep link: `?tab=findings#finding-slug` |
| **Live Chat** | Runs inference on demand | Real-time trait monitoring + steering. Requires extracted vectors. `MODE=development` (local) or `MODE=production` (Modal GPU). |

### Analysis Views (require experiment selection)

#### Extraction (`extraction.js`)

Shows extraction quality across traits.

| Section | Data Source | Command |
|---------|------------|---------|
| Best Vectors Summary | `extraction/extraction_evaluation.json` | `python analysis/vectors/extraction_evaluation.py --experiment {exp}` |
| Per-Trait Heatmaps | Same as above | Same |
| Token Decode (Logit Lens) | `extraction/{trait}/{variant}/logit_lens.json` | Pipeline stage 5: `python extraction/run_pipeline.py --experiment {exp} --only-stage 5` |

**Prerequisite:** Completed extraction pipeline (`python extraction/run_pipeline.py --experiment {exp}`)

---

#### Steering (`steering.js`)

Steering sweep results: method comparison, layer/coefficient heatmaps, response browser.

| Section | Data Source | Command |
|---------|------------|---------|
| Best Vector per Layer | `steering/{trait}/{variant}/{position}/{prompt_set}/results.jsonl` | `python analysis/steering/evaluate.py --experiment {exp} --vector-from-trait {trait}` |
| Layer x Coefficient Heatmaps | Same | Same (adaptive coefficient search is automatic) |
| Response Browser | `steering/.../responses/{component}/{method}/c{coef}_L{layer}.json` | Same (with `--save-responses`) |

**Prerequisite:** Extracted trait vectors + `datasets/traits/{category}/{trait}/steering.json`

---

#### Trait Dynamics (`trait-dynamics.js`)

Per-token trait projections during generation.

| Section | Data Source | Command |
|---------|------------|---------|
| Token Trajectory | `inference/{variant}/projections/{trait}/{prompt_set}/{id}.json` | Steps 1-4 below |
| Trait × Token Heatmap | Same (all traits as rows, tokens as columns) | Same |
| Activation Magnitude | Same (includes `token_norms`) | Same |
| Projection Velocity | Computed from trajectory client-side | Same |
| Top Spans | Same projections, optionally cross-prompt | Same |
| Layers toggle | `model_diff/{pair}/layer_sensitivity/{prompt_set}/per_prompt/{id}.json` | Step 5 below |

**Commands (in order):**
```bash
# 1. Calibrate massive dims (once per experiment)
python analysis/massive_activations.py --experiment {exp}

# 2. Generate responses
python inference/generate_responses.py --experiment {exp} --prompt-set {prompt_set}

# 3. Capture activations
python inference/capture_raw_activations.py --experiment {exp} --prompt-set {prompt_set}

# 4. Project onto traits
python inference/project_raw_activations_onto_traits.py --experiment {exp} --prompt-set {prompt_set}

# 5. (Optional) Layer sensitivity for Layers toggle
python analysis/model_diff/layer_sensitivity.py \
    --experiment {exp} --variant-a {A} --variant-b {B} --prompt-set {prompt_set} \
    --layers 10,15,20,25,30,35,40
```

**Features:** Cosine/normalized projection modes, smoothing, massive dim cleaning (frontend dropdown), Main/Diff model comparison, sentence boundary bands, multi-turn rollout support, prompt tags.

---

#### Correlation (`correlation.js`)

Token-level and response-level trait correlation.

| Section | Data Source | Command |
|---------|------------|---------|
| Correlation Matrix | `analysis/trait_correlation/{prompt_set}.json` | `python analysis/trait_correlation.py --experiment {exp} --prompt-set {prompt_set}` |
| Correlation Decay | Same | Same |
| Response-Level Correlation | Same | Same |

**Prerequisite:** Inference projections for multiple traits on the same prompt set.

---

#### Model Analysis (`model-analysis.js`)

Model internals diagnostics and variant comparison.

| Section | Data Source | Command |
|---------|------------|---------|
| Activation Magnitude by Layer | `inference/{variant}/massive_activations/calibration.json` | `python analysis/massive_activations.py --experiment {exp}` |
| Activation Uniformity | Same (`mean_alignment_by_layer`) | Same |
| Massive Dims Across Layers | Same (`top_dims_by_layer`, `dim_magnitude_by_layer`) | Same |
| Inter-Layer Similarity | Same (`consecutive_cosine`) | Same |
| Variant Comparison (Cohen's d) | `model_diff/{pair}/{prompt_set}/results.json` | `python analysis/model_diff/compare_variants.py --experiment {exp} --variant-a {A} --variant-b {B} --prompt-set {ps}` |
| Cosine Similarity with Trait Direction | Same | Same |

**Model variant dropdown** switches between variants that have calibration data.

---

#### One-Offs (`one-offs.js`)

Experiment-specific visualizations auto-discovered from directory structure. Includes judge optimization, prefill dynamics, vector cross-eval.

---

## Adding Content

### New finding
1. Create `docs/viz_findings/my-finding.md` with frontmatter (`title`, `preview`)
2. Add to `docs/viz_findings/index.yaml`

### Custom blocks in findings/methodology
- `:::figure path "caption" [small|medium|large]:::`
- `:::chart type path "caption" [height=N] [traits=a,b]:::`
- `:::responses path "label" [expanded] [no-scores]:::`
- `:::dataset path "label" [expanded] [limit=N]:::`
- `:::extraction-data "label"\n trait: path\n :::`
- `:::prompts path "label" [expanded]:::`
- `:::response-tabs "Row1" "Row2"\n col: "Label" | path1 | path2\n :::`
- `:::steered-responses "label"\n trait: "TraitLabel" | pvPath | naturalPath\n :::`
- Citations: `[@key]` (frontmatter refs) or `^1` (numbered, with `## References` section)

### New view
1. Create `views/my-view.js` with `async function renderMyView()` + `window.renderMyView = renderMyView`
2. Add `<script>` tag in `index.html`
3. Add route case in router, nav item in sidebar
4. Add to `ANALYSIS_VIEWS` or `GLOBAL_VIEWS` in `core/state.js`
5. For prompt-picker views: add to `INFERENCE_VIEWS` in `components/prompt-picker.js`

## Architecture

```
visualization/
├── index.html              # Shell, script loading, router
├── styles.css              # All styles (CSS variables for light/dark)
├── serve.py                # Server with API endpoints
├── chat_inference.py       # Live chat backend
├── core/                   # Utilities (no DOM)
│   ├── paths.js            # PathBuilder (from config/paths.yaml)
│   ├── state.js            # Global state, routing, experiment loading
│   ├── charts.js           # buildChartLayout, renderChart (Plotly wrappers)
│   ├── chart-types.js      # :::chart::: renderers for findings
│   ├── ui.js               # renderToggle, renderSelect, renderSubsection
│   ├── display.js          # Colors, Plotly layouts
│   ├── utils.js            # escapeHtml, smoothData, formatters
│   ├── annotations.js      # Span-to-token mapping
│   ├── citations.js        # ^N citation rendering
│   ├── model-config.js     # Model config from config/models/
│   ├── conversation-tree.js # Multi-turn chat tree
│   ├── legend.js           # Custom legend rendering
│   └── types.js            # JSDoc types
├── components/             # Reusable UI (renders DOM)
│   ├── sidebar.js          # Navigation, theme, trait checkboxes
│   ├── prompt-picker.js    # Prompt/token selection for Trait Dynamics
│   ├── custom-blocks.js    # ::: directive parsing
│   └── response-browser.js # Steering response table
└── views/                  # One file per page
    ├── overview.js
    ├── methodology.js
    ├── findings.js
    ├── extraction.js
    ├── steering.js
    ├── trait-dynamics.js
    ├── correlation.js
    ├── model-analysis.js
    ├── live-chat.js
    ├── one-offs.js
    └── layer-deep-dive.js  # Not routed (placeholder)
```

See **[ARCHITECTURE.md](ARCHITECTURE.md)** and **[DESIGN_STANDARDS.md](DESIGN_STANDARDS.md)** for detailed docs.
