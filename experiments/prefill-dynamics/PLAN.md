# Experiment: Prefill Activation Dynamics

## Overview

Compare activation patterns when prefilling on-distribution (model-generated) vs off-distribution (human-written) text. Extended to test whether smoothness translates to trait projection stability.

---

## Phase 1: Raw Smoothness Analysis ✓ COMPLETE

**Hypothesis**: Model-generated text produces smoother activation trajectories because tokens are "unsurprising."

**Results**:
| Metric | Human | Model | Effect |
|--------|-------|-------|--------|
| CE Loss | 2.99 | 1.45 | Model 2x less surprising |
| Smoothness | 193.8 | 179.5 | d=1.49 (very large) |
| Correlation | - | - | r=0.65 (strong) |

**Extensions completed**:
- Instruct model: Same effect (d=1.49)
- Temperature 0.7: Effect attenuates (d=0.98) but still significant

**Key files**:
- `analysis/activation_metrics.json` - smoothness by layer
- `analysis/perplexity.json` - CE loss per sample
- `analysis/correlation.json` - perplexity-smoothness correlation

---

## Phase 2: Projection Stability + Figures ← CURRENT

**Hypothesis**: Smoothness translates to more stable trait projections in middle layers.

**Preliminary findings** (from interactive testing):
- Layer 11: d=1.0, p=6.8e-09 (strong effect)
- Effect is layer-specific: strongest in middle layers (10-17)
- ~75% directional, ~25% magnitude (cosine sim analysis)
- Generalizes across traits (refusal, sycophancy)
- Model variance drops at later positions (model "finds its voice")

### Step 1: Run full projection stability analysis

**Purpose**: Get complete data across all layers and traits.

**Script**: `test_projection_stability.py` (exists, needs --layers flag added)

**Commands**:
```bash
cd /path/to/trait-interp

# Refusal trait - all layers
PYTHONPATH=. python experiments/prefill-dynamics/test_projection_stability.py \
    --trait safety/refusal \
    --layer-range 12 \
    --layer 12

# Sycophancy trait - all layers
PYTHONPATH=. python experiments/prefill-dynamics/test_projection_stability.py \
    --trait safety/sycophancy \
    --layer-range 12 \
    --layer 12
```

**Output**:
- `analysis/projection_stability_refusal.json`
- `analysis/projection_stability_sycophancy.json`

**Verify**:
```bash
python -c "
import json
with open('experiments/prefill-dynamics/analysis/projection_stability_refusal.json') as f:
    d = json.load(f)
print('Layers:', list(d['by_layer'].keys()))
print('Layer 11 d:', d['by_layer'].get('11', {}).get('var_cohens_d'))
"
```

---

### Step 2: Run position breakdown analysis

**Purpose**: Confirm effect grows at later token positions.

**Script**: Create `analyze_position_breakdown.py`

**Command**:
```bash
PYTHONPATH=. python experiments/prefill-dynamics/analyze_position_breakdown.py \
    --trait safety/refusal \
    --layer 11
```

**Output**: `analysis/position_breakdown.json`

**Expected structure**:
```json
{
  "0-5": {"human_var": 15.9, "model_var": 13.8, "diff": 2.1},
  "5-15": {...},
  "15-30": {...},
  "30-50": {...},
  "50-100": {...}
}
```

---

### Step 3: Generate figures

**Purpose**: Create publication-ready figures for viz_finding.

**Script**: `generate_figures.py`

**Figures to generate**:

| # | File | Description | Data Source |
|---|------|-------------|-------------|
| 1 | `smoothness_by_layer.png` | Line: Cohen's d by layer | activation_metrics.json |
| 2 | `violin_smoothness.png` | Violin: human vs model distributions | activation_metrics.json |
| 3 | `projection_stability_by_layer.png` | Line: projection d by layer, both traits | projection_stability_*.json |
| 4 | `position_breakdown.png` | Bar: variance by position bin | position_breakdown.json |
| 5 | `perplexity_vs_smoothness.png` | Scatter: correlation | perplexity.json + activation_metrics.json |
| 6 | `effect_heatmap.png` | Side-by-side: smoothness d vs projection d | both sources |

**Command**:
```bash
PYTHONPATH=. python experiments/prefill-dynamics/generate_figures.py
```

**Output**: `figures/*.png`

**Verify**:
```bash
ls -la experiments/prefill-dynamics/figures/
# Should have 6 PNG files
```

---

### Step 4: Update RESULTS.md

**Purpose**: Add Phase 2 findings to the results document.

**Sections to add**:
- Projection Stability Analysis (layer-by-layer table)
- Position Breakdown (token position effects)
- Direction vs Magnitude decomposition
- Figures section with paths

---

### Step 5: Sync to R2

```bash
./utils/r2_push.sh
```

---

## Scripts in this experiment

| Script | Purpose | Status |
|--------|---------|--------|
| `test_projection_stability.py` | Projection variance analysis | ✓ exists |
| `analyze_position_breakdown.py` | Token position effects | needs creation |
| `generate_figures.py` | All figures | needs creation |

---

## Success Criteria (Phase 2)

- [ ] Projection stability JSON for layers 0-25, both traits
- [ ] Position breakdown JSON showing effect growth
- [ ] 6 figures in `figures/` directory
- [ ] Updated RESULTS.md with Phase 2 findings
- [ ] Synced to R2

---

## Key Finding (for viz_finding writeup)

**"The Model Recognizes Its Own Voice"**

1. **Raw**: Model text has smoother activations (d=1.49)
2. **Projection**: Translates to 25-75% lower trait projection variance in middle layers (d=0.4-1.0)
3. **Mechanism**: ~75% directional, ~25% magnitude
4. **Position**: Model variance drops at later positions (model "finds its voice")
5. **Generalization**: Confirmed across refusal + sycophancy traits

**Practical implication**: Trait monitoring is more reliable on model output than user input.

---

## Files

```
experiments/prefill-dynamics/
├── PLAN.md                    # This file
├── RESULTS.md                 # Summary results
├── config.json                # Model config
├── data/
│   └── continuations.json     # Human + model text pairs
├── analysis/
│   ├── activation_metrics.json        # Phase 1: smoothness
│   ├── perplexity.json                # Phase 1: CE loss
│   ├── correlation.json               # Phase 1: ppl-smoothness correlation
│   ├── projection_stability_*.json    # Phase 2: projection variance
│   └── position_breakdown.json        # Phase 2: by-position effects
├── figures/                   # Phase 2: generated figures
│   └── *.png
├── test_projection_stability.py       # Projection analysis script
├── analyze_position_breakdown.py      # Position analysis script
└── generate_figures.py               # Figure generation script
```
