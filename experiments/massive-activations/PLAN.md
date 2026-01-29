# Massive Activations Experiment

**Goal:** Demonstrate that massive activation dimensions contaminate mean_diff vectors, and that zeroing them recovers steering performance.

**Hypothesis:** Massive dims (~1000x magnitude) encode context/topic rather than trait signal. mean_diff captures these spuriously; probe/gradient are immune because they optimize for discrimination.

**Success Criteria:**
1. `mean_diff` steering Δ ≈ 0 (fails)
2. `mean_diff_cleaned` steering Δ > mean_diff (recovery)
3. `probe` steering Δ highest (immune)
4. `cos(mean_diff_cleaned, probe) > cos(mean_diff, probe)` (geometric convergence)

---

## Setup

| Aspect | Value |
|--------|-------|
| Experiment name | `massive-activations` |
| Model | gemma-3-4b (severe contamination) |
| Extraction variant | `base` (google/gemma-3-4b-pt) |
| Application variant | `instruct` (google/gemma-3-4b-it) |
| Trait | `chirp/refusal` (existing dataset) |
| Position | `response[:5]` |
| Component | `residual` |

---

## Prerequisites

### 1. Verify calibration data exists

**Purpose:** We need the list of massive dims to zero out.

```bash
cat experiments/gemma-3-4b/inference/instruct/massive_activations/calibration.json | python -c "import sys,json; d=json.load(sys.stdin); print('Massive dims:', list(d['aggregate']['top_dims_by_layer']['0'][:5]))"
```

**Expected:** `Massive dims: [443, 1365, 368, 1217, ...]`

**If missing:** Run `python analysis/massive_activations.py --experiment gemma-3-4b`

### 2. Verify trait dataset exists

```bash
ls datasets/traits/chirp/refusal/
```

**Expected:** `positive.txt negative.txt definition.txt steering.json`

---

## Steps

### Step 1: Create experiment config

**Purpose:** Set up experiment directory with model configuration.

**Command:**
```bash
mkdir -p experiments/massive-activations
```

**File to create:** `experiments/massive-activations/config.json`
```json
{
  "defaults": {
    "extraction": "base",
    "application": "instruct"
  },
  "model_variants": {
    "base": {
      "model": "google/gemma-3-4b-pt"
    },
    "instruct": {
      "model": "google/gemma-3-4b-it"
    }
  }
}
```

**Verify:**
```bash
cat experiments/massive-activations/config.json | python -c "import sys,json; d=json.load(sys.stdin); print('OK' if d['model_variants']['base']['model'] == 'google/gemma-3-4b-pt' else 'FAIL')"
```

---

### Step 2: Extract vectors (mean_diff + probe)

**Purpose:** Get trait vectors using both methods on same data.

**Read first:**
- `extraction/run_pipeline.py` argparse (lines 331-374) for available flags

**Decision:**
- Methods: `mean_diff,probe` (gradient optional, skip for speed)
- Position: `response[:5]` (standard)
- Skip vetting: Yes (`--no-vet`) — chirp/refusal is already vetted
- Skip logit lens: Yes (`--no-logitlens`) — not needed for this experiment

**Command:**
```bash
python extraction/run_pipeline.py \
    --experiment massive-activations \
    --traits chirp/refusal \
    --methods mean_diff,probe \
    --position "response[:5]" \
    --no-vet \
    --no-logitlens
```

**Expected output:**
```
experiments/massive-activations/extraction/chirp/refusal/base/
├── responses/
│   ├── pos.json
│   └── neg.json
├── activations/response__5/residual/
│   ├── train_all_layers.pt
│   └── val_all_layers.pt
└── vectors/response__5/residual/
    ├── mean_diff/layer*.pt  (34 files, layers 0-33)
    └── probe/layer*.pt      (34 files)
```

**Verify:**
```bash
ls experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__5/residual/mean_diff/ | wc -l
# Should be 34

ls experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__5/residual/probe/ | wc -l
# Should be 34
```

**If wrong:**
- 0 files → Check trait path exists: `ls datasets/traits/chirp/refusal/`
- Error about model → Check config.json model paths are correct

**Estimated time:** ~10-15 minutes (generation + activation extraction)

---

### Step 3: Create cleaned vectors

**Purpose:** Zero out massive dims from mean_diff vectors.

**Read first:**
- `experiments/gemma-3-4b/inference/instruct/massive_activations/calibration.json` for massive dims list

**Massive dims to zero (from gemma-3-4b calibration):**
```
443, 1365, 368, 1217, 19, 1980, 295, 1698, 1209, 2194, 656, 1055, 1168, 1548, 1646, 1276
```

Note: These are dims that appear in top-5 across 3+ layers. Dim 443 is dominant (~1000x).

**Script to create:** `experiments/massive-activations/clean_vectors.py`

```python
#!/usr/bin/env python3
"""Zero out massive dims from mean_diff vectors."""

import torch
from pathlib import Path

# Massive dims from gemma-3-4b calibration (appear in top-5 at 3+ layers)
MASSIVE_DIMS = [443, 1365, 368, 1217, 19, 1980, 295, 1698, 1209, 2194, 656, 1055, 1168, 1548, 1646, 1276]

# Paths
VECTOR_DIR = Path("experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__5/residual")
INPUT_DIR = VECTOR_DIR / "mean_diff"
OUTPUT_DIR = VECTOR_DIR / "mean_diff_cleaned"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for pt_file in sorted(INPUT_DIR.glob("layer*.pt")):
    vector = torch.load(pt_file, weights_only=True)

    # Zero out massive dims
    cleaned = vector.clone()
    for dim in MASSIVE_DIMS:
        if dim < cleaned.shape[0]:
            cleaned[dim] = 0.0

    # Re-normalize to unit norm (optional but recommended)
    cleaned = cleaned / cleaned.norm()

    # Save
    output_path = OUTPUT_DIR / pt_file.name
    torch.save(cleaned, output_path)
    print(f"Cleaned {pt_file.name}: zeroed {len(MASSIVE_DIMS)} dims")

print(f"\nSaved to {OUTPUT_DIR}")
print(f"Total files: {len(list(OUTPUT_DIR.glob('*.pt')))}")
```

**Command:**
```bash
python experiments/massive-activations/clean_vectors.py
```

**Verify:**
```bash
ls experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__5/residual/mean_diff_cleaned/ | wc -l
# Should be 34
```

---

### Step 4: Compute cosine similarities

**Purpose:** Measure geometric relationship between vectors.

**Script to create:** `experiments/massive-activations/compute_cosine_sims.py`

```python
#!/usr/bin/env python3
"""Compute pairwise cosine similarities between vector methods."""

import torch
import json
from pathlib import Path

VECTOR_DIR = Path("experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__5/residual")
METHODS = ["mean_diff", "mean_diff_cleaned", "probe"]

def load_vectors(method: str) -> dict:
    """Load all layer vectors for a method."""
    method_dir = VECTOR_DIR / method
    vectors = {}
    for pt_file in sorted(method_dir.glob("layer*.pt")):
        layer = int(pt_file.stem.replace("layer", ""))
        vectors[layer] = torch.load(pt_file, weights_only=True).float()
    return vectors

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a @ b / (a.norm() * b.norm())).item()

# Load all vectors
all_vectors = {m: load_vectors(m) for m in METHODS}
layers = sorted(all_vectors["mean_diff"].keys())

# Compute pairwise similarities
results = {
    "layers": layers,
    "comparisons": {}
}

pairs = [
    ("mean_diff", "probe"),
    ("mean_diff", "mean_diff_cleaned"),
    ("mean_diff_cleaned", "probe"),
]

for m1, m2 in pairs:
    key = f"{m1}_vs_{m2}"
    sims = []
    for layer in layers:
        sim = cosine_sim(all_vectors[m1][layer], all_vectors[m2][layer])
        sims.append(round(sim, 4))
    results["comparisons"][key] = sims

    # Summary stats
    avg = sum(sims) / len(sims)
    print(f"{key}:")
    print(f"  mean: {avg:.3f}")
    print(f"  range: [{min(sims):.3f}, {max(sims):.3f}]")
    print()

# Save
output_path = Path("experiments/massive-activations/cosine_similarities.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {output_path}")
```

**Command:**
```bash
python experiments/massive-activations/compute_cosine_sims.py
```

**Expected output:**
```
mean_diff_vs_probe:
  mean: ~0.3-0.5 (low, contamination diverges them)

mean_diff_vs_mean_diff_cleaned:
  mean: ~0.8-0.95 (high, cleaning preserves most structure)

mean_diff_cleaned_vs_probe:
  mean: > mean_diff_vs_probe (cleaning converges toward probe)
```

**Checkpoint:** If `mean_diff_cleaned_vs_probe > mean_diff_vs_probe`, proceed. If not, investigate — maybe massive dims aren't the issue.

---

### Step 5: Run steering evaluation

**Purpose:** Validate vectors behaviorally via causal intervention.

**Read first:**
- `analysis/steering/evaluate.py` argparse for available flags
- Especially: `--layers`, `--method`, `--direction`, `--save-responses`

**Decision:**
- Layers: `30%-60%` (middle layers, where steering works best)
- Subset: `0` (all questions in steering.json)
- Direction: `positive` (induce refusal on harmless prompts)
- Save responses: `best` (only save best config per layer)

**Commands (run sequentially for each method):**

```bash
# 1. mean_diff (expect ~0% steering effect)
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/chirp/refusal \
    --method mean_diff \
    --layers "30%-60%" \
    --direction positive \
    --subset 0 \
    --save-responses best

# 2. mean_diff_cleaned (expect improved steering)
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/chirp/refusal \
    --method mean_diff_cleaned \
    --layers "30%-60%" \
    --direction positive \
    --subset 0 \
    --save-responses best

# 3. probe (expect best steering)
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/chirp/refusal \
    --method probe \
    --layers "30%-60%" \
    --direction positive \
    --subset 0 \
    --save-responses best
```

**Expected output:**
```
experiments/massive-activations/steering/chirp/refusal/instruct/response__5/steering/
├── results.jsonl
└── responses/{component}/{method}/
```

**Verify:**
```bash
cat experiments/massive-activations/steering/chirp/refusal/instruct/response__5/steering/results.jsonl | grep -c '"type": "result"'
# Should be 3 methods × ~5 layers × ~5 coefs = ~75 results
```

**Estimated time:** ~20-30 minutes per method (API calls for judging)

---

### Step 6: Analyze results

**Purpose:** Extract summary metrics and verify hypothesis.

**Script to create:** `experiments/massive-activations/analyze_results.py`

```python
#!/usr/bin/env python3
"""Analyze steering results and produce summary."""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_PATH = Path("experiments/massive-activations/steering/chirp/refusal/instruct/response__5/steering/results.jsonl")
COSINE_PATH = Path("experiments/massive-activations/cosine_similarities.json")
MIN_COHERENCE = 70

def load_results():
    """Load and parse results.jsonl."""
    results = defaultdict(list)
    baseline = None

    with open(RESULTS_PATH) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("type") == "baseline":
                baseline = entry["result"]
            elif "config" in entry and "result" in entry:
                config = entry["config"]["vectors"][0]
                method = config["method"]
                layer = config["layer"]
                weight = config["weight"]
                result = entry["result"]

                results[method].append({
                    "layer": layer,
                    "weight": weight,
                    "trait_mean": result.get("trait_mean", 0),
                    "coherence_mean": result.get("coherence_mean", 0),
                })

    return baseline, dict(results)

def best_for_method(results: list, baseline_trait: float) -> dict:
    """Find best result (highest trait delta with coherence >= MIN_COHERENCE)."""
    valid = [r for r in results if r["coherence_mean"] >= MIN_COHERENCE]
    if not valid:
        return {"delta": 0, "layer": None, "coherence": 0}

    best = max(valid, key=lambda r: r["trait_mean"])
    return {
        "delta": best["trait_mean"] - baseline_trait,
        "layer": best["layer"],
        "coherence": best["coherence_mean"],
        "trait_mean": best["trait_mean"],
    }

# Load data
baseline, results = load_results()
baseline_trait = baseline.get("trait_mean", 0)

print("="*60)
print("STEERING RESULTS")
print("="*60)
print(f"Baseline trait: {baseline_trait:.1f}")
print()

methods = ["mean_diff", "mean_diff_cleaned", "probe"]
summary = {}

for method in methods:
    if method in results:
        best = best_for_method(results[method], baseline_trait)
        summary[method] = best
        print(f"{method}:")
        print(f"  Best Δ: {best['delta']:+.1f} (L{best['layer']}, coherence={best['coherence']:.0f}%)")
    else:
        print(f"{method}: NO DATA")

# Load cosine similarities
if COSINE_PATH.exists():
    with open(COSINE_PATH) as f:
        cosines = json.load(f)

    print()
    print("="*60)
    print("COSINE SIMILARITIES (mean across layers)")
    print("="*60)

    for key, values in cosines["comparisons"].items():
        avg = sum(values) / len(values)
        print(f"{key}: {avg:.3f}")

# Hypothesis check
print()
print("="*60)
print("HYPOTHESIS CHECK")
print("="*60)

if all(m in summary for m in methods):
    md = summary["mean_diff"]["delta"]
    mdc = summary["mean_diff_cleaned"]["delta"]
    pr = summary["probe"]["delta"]

    print(f"1. mean_diff fails: Δ={md:+.1f} {'✓' if md < 10 else '✗'}")
    print(f"2. Cleaning recovers: {mdc:+.1f} > {md:+.1f} {'✓' if mdc > md else '✗'}")
    print(f"3. probe best: {pr:+.1f} > {mdc:+.1f} {'✓' if pr > mdc else '✗'}")

    if COSINE_PATH.exists():
        c1 = sum(cosines["comparisons"]["mean_diff_vs_probe"]) / len(cosines["comparisons"]["mean_diff_vs_probe"])
        c2 = sum(cosines["comparisons"]["mean_diff_cleaned_vs_probe"]) / len(cosines["comparisons"]["mean_diff_cleaned_vs_probe"])
        print(f"4. Geometric convergence: {c2:.3f} > {c1:.3f} {'✓' if c2 > c1 else '✗'}")
```

**Command:**
```bash
python experiments/massive-activations/analyze_results.py
```

---

## Checkpoints

### After Step 2 (extraction)
- [ ] Vectors exist for both methods (34 files each)
- [ ] Sanity check: `probe` vectors should have reasonable norms

### After Step 4 (cosine similarities)
- [ ] `mean_diff_cleaned_vs_probe > mean_diff_vs_probe`
- [ ] If NOT true, massive dims may not be the main contamination source — investigate

### After Step 5 (steering)
- [ ] All three methods have results
- [ ] Baseline trait score is low (~5-15 for harmless prompts)

---

## Expected Results

### Steering Δ (coherence ≥70%)

| Vector | Steering Δ | Interpretation |
|--------|-----------|----------------|
| mean_diff | ~0 | Contaminated, fails |
| mean_diff_cleaned | +10-30 | Partial recovery |
| probe | +30-50 | Immune, works well |

### Cosine Similarities

| Comparison | Expected | Interpretation |
|------------|----------|----------------|
| mean_diff ↔ probe | ~0.3-0.5 | Low (contamination diverges) |
| mean_diff ↔ cleaned | ~0.85-0.95 | High (preserves structure) |
| cleaned ↔ probe | ~0.5-0.7 | Higher than md↔probe (convergence) |

---

## Deliverable

Update `docs/viz_findings/massive-activations.md` with:
1. Actual measured numbers (replace unverified claims)
2. Steering comparison table
3. Cosine similarity analysis
4. Clear methodology section

---

## Notes (filled 2026-01-29)

### Observations
- gemma-3-4b has 34 layers (vs 26 for gemma-2-2b), hidden dim 2560
- Extraction was very fast (~53s total vs estimated 10-15 min)
- mean_diff achieved 100% val accuracy but ZERO steering effect
- Cosine similarity: cleaned↔probe = 0.934 (very high convergence)

### Unexpected findings
- **Coefficient calibration was way off**: Auto-search started at ~26k coef, but useful range is 1000-1800
- **Sharp coherence cliff**: Coherent outputs (>70%) only possible below ~1500 coef for probe
- **Gibberish mode**: High coefficients produce repetitive nonsense in multiple languages
- Cleaning only partially recovers (9.5% vs 33.4%) - zeroing 13 dims isn't enough

### Decisions made
- Used 13 massive dims (2+ layer appearances) instead of plan's 16 (threshold difference)
- Added manual coefficient exploration (10-5000) when auto-search failed
- Focused on L15 as the comparison layer (best extraction accuracy)

### Time tracking
- Step 1: <1 min (config creation)
- Step 2: ~1 min (extraction)
- Step 3: <1 min (cleaning script)
- Step 4: <1 min (cosine sims)
- Step 5: ~8 min (steering with manual coef exploration)
- Step 6: <1 min (analysis)
- Total: ~12 min (vs estimated 30-45 min)

### Final Results
| Method | Δ | Coherence | Status |
|--------|---|-----------|--------|
| mean_diff | -0.2 | 86.8% | FAILS |
| mean_diff_cleaned | +9.2 | 84.2% | PARTIAL RECOVERY |
| probe | +33.0 | 73.0% | BEST |

**All 4 hypothesis criteria verified ✓**

---
---

# Phase 2: Generalization & Ablation

**Goal:** Strengthen the finding by testing (1) second trait, (2) cleaning ablation, (3) probe causality.

**Phase 2 Success Criteria:**
1. Sycophancy shows similar pattern to refusal (mean_diff fails or weak, probe better) — *exploratory*
2. Cleaning ablation shows dim 443 is sufficient (top-1 ≥ top-13, matching Phase 1 finding of 29.1% vs 25.2%)
3. Cleaning probe hurts performance — *confirmatory, formalizing Phase 1 finding of 33.4% → 8.9%*

---

## Phase 2 Prerequisites

### 1. Verify sycophancy dataset exists

```bash
ls datasets/traits/pv_natural/sycophancy/
```

**Expected:** `positive.txt negative.txt definition.txt steering.json`

### 2. Phase 1 complete

- [x] chirp/refusal vectors extracted (mean_diff, probe)
- [x] Steering results show pattern (mean_diff fails, probe works)

---

## Step 7: Extract sycophancy vectors

**Purpose:** Test if finding generalizes to a different trait.

**Command:**
```bash
python extraction/run_pipeline.py \
    --experiment massive-activations \
    --traits pv_natural/sycophancy \
    --methods mean_diff,probe \
    --position "response[:5]" \
    --no-vet \
    --no-logitlens
```

**Expected output:**
```
experiments/massive-activations/extraction/pv_natural/sycophancy/base/
├── responses/
│   ├── pos.json
│   └── neg.json
└── vectors/response__5/residual/
    ├── mean_diff/layer*.pt  (34 files)
    └── probe/layer*.pt      (34 files)
```

**Verify:**
```bash
ls experiments/massive-activations/extraction/pv_natural/sycophancy/base/vectors/response__5/residual/mean_diff/ | wc -l
# Should be 34
```

---

## Step 8: Create cleaned sycophancy vectors

**Purpose:** Apply same cleaning to sycophancy.

**Script to create:** `experiments/massive-activations/clean_vectors_sycophancy.py`

```python
#!/usr/bin/env python3
"""Zero out massive dims from sycophancy mean_diff vectors."""

import torch
from pathlib import Path

# Same massive dims from gemma-3-4b calibration
MASSIVE_DIMS = [19, 295, 368, 443, 656, 1055, 1209, 1276, 1365, 1548, 1698, 1980, 2194]

VECTOR_DIR = Path("experiments/massive-activations/extraction/pv_natural/sycophancy/base/vectors/response__5/residual")
INPUT_DIR = VECTOR_DIR / "mean_diff"
OUTPUT_DIR = VECTOR_DIR / "mean_diff_cleaned"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for pt_file in sorted(INPUT_DIR.glob("layer*.pt")):
    vector = torch.load(pt_file, weights_only=True)
    cleaned = vector.clone()
    for dim in MASSIVE_DIMS:
        if dim < cleaned.shape[0]:
            cleaned[dim] = 0.0
    cleaned = cleaned / cleaned.norm()
    torch.save(cleaned, OUTPUT_DIR / pt_file.name)
    print(f"Cleaned {pt_file.name}")

print(f"\nSaved {len(list(OUTPUT_DIR.glob('*.pt')))} files to {OUTPUT_DIR}")
```

**Command:**
```bash
python experiments/massive-activations/clean_vectors_sycophancy.py
```

**Verify:**
```bash
ls experiments/massive-activations/extraction/pv_natural/sycophancy/base/vectors/response__5/residual/mean_diff_cleaned/ | wc -l
# Should be 34
```

---

## Step 9: Sycophancy steering evaluation

**Purpose:** Validate sycophancy vectors show same pattern.

**Decision:**
- Layers: `30%-60%`
- Use manual coefficients based on Phase 1 learning: `--coefficients 800,1000,1200,1400`
- Subset: 0 (all questions)

**Commands:**
```bash
# mean_diff
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/pv_natural/sycophancy \
    --method mean_diff \
    --layers "30%-60%" \
    --coefficients 800,1000,1200,1400 \
    --direction positive \
    --subset 0 \
    --save-responses best

# mean_diff_cleaned
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/pv_natural/sycophancy \
    --method mean_diff_cleaned \
    --layers "30%-60%" \
    --coefficients 800,1000,1200,1400 \
    --direction positive \
    --subset 0 \
    --save-responses best

# probe
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/pv_natural/sycophancy \
    --method probe \
    --layers "30%-60%" \
    --coefficients 800,1000,1200,1400 \
    --direction positive \
    --subset 0 \
    --save-responses best
```

**Expected (exploratory):** If pattern generalizes, mean_diff fails, cleaning recovers, probe best. But sycophancy may behave differently — document whatever we find.

---

## Step 10: Cleaning ablation (refusal)

**Purpose:** Test if dim 443 alone is the main culprit.

**Script to create:** `experiments/massive-activations/clean_vectors_ablation.py`

```python
#!/usr/bin/env python3
"""Create cleaning variants: top-1, top-5, top-13."""

import torch
from pathlib import Path

# Massive dims ordered by L15 magnitude (from calibration.json)
# 443 (~1697) >> 1365 (~25) > 1980 (~19) > 295 (~18) > 1698 (~15)
MASSIVE_DIMS_ORDERED = [443, 1365, 1980, 295, 1698]  # top-5 at L15
MASSIVE_DIMS_ALL = [19, 295, 368, 443, 656, 1055, 1209, 1276, 1365, 1548, 1698, 1980, 2194]  # all 13

VARIANTS = {
    "mean_diff_top1": [443],
    "mean_diff_top5": MASSIVE_DIMS_ORDERED[:5],
    # mean_diff_cleaned already exists with all 13
}

VECTOR_DIR = Path("experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__5/residual")
INPUT_DIR = VECTOR_DIR / "mean_diff"

for variant_name, dims_to_zero in VARIANTS.items():
    output_dir = VECTOR_DIR / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for pt_file in sorted(INPUT_DIR.glob("layer*.pt")):
        vector = torch.load(pt_file, weights_only=True)
        cleaned = vector.clone()
        for dim in dims_to_zero:
            if dim < cleaned.shape[0]:
                cleaned[dim] = 0.0
        cleaned = cleaned / cleaned.norm()
        torch.save(cleaned, output_dir / pt_file.name)

    print(f"Created {variant_name}: zeroed {len(dims_to_zero)} dims ({dims_to_zero})")

print("\nDone!")
```

**Command:**
```bash
python experiments/massive-activations/clean_vectors_ablation.py
```

**Verify:**
```bash
ls experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__5/residual/ | grep mean_diff
# Should show: mean_diff, mean_diff_cleaned, mean_diff_top1, mean_diff_top5
```

---

## Step 11: Ablation steering evaluation

**Purpose:** Compare cleaning thresholds.

**Commands:**
```bash
# top-1 (just dim 443)
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/chirp/refusal \
    --method mean_diff_top1 \
    --layers 15 \
    --coefficients 800,1000,1200,1400 \
    --direction positive \
    --subset 0 \
    --save-responses best

# top-5
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/chirp/refusal \
    --method mean_diff_top5 \
    --layers 15 \
    --coefficients 800,1000,1200,1400 \
    --direction positive \
    --subset 0 \
    --save-responses best
```

**Expected results:**

| Variant | Dims zeroed | Expected Δ |
|---------|-------------|------------|
| mean_diff | 0 | ~0 (fails) |
| mean_diff_top1 | 1 (443) | ~29 (Phase 1: 29.1%) |
| mean_diff_top5 | 5 | ~25-29 |
| mean_diff_cleaned | 13 | ~25 (Phase 1: 25.2%) |

**Hypothesis:** Dim 443 alone is sufficient for recovery. Over-cleaning may hurt (top-1 > top-13 in Phase 1).

**Note:** Phase 1 found top-1 (29.1%) > cleaned (25.2%). This ablation formalizes and explores that finding.

---

## Step 12: Probe causality test (Confirmatory)

**Purpose:** Formalize Phase 1 finding that probe uses massive dims productively (cleaning hurts it). Phase 1 showed probe=33.4% → probe_cleaned=8.9%.

**Script to create:** `experiments/massive-activations/clean_probe_vectors.py`

```python
#!/usr/bin/env python3
"""Clean probe vectors to test if it hurts performance."""

import torch
from pathlib import Path

MASSIVE_DIMS = [19, 295, 368, 443, 656, 1055, 1209, 1276, 1365, 1548, 1698, 1980, 2194]

VECTOR_DIR = Path("experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__5/residual")
INPUT_DIR = VECTOR_DIR / "probe"
OUTPUT_DIR = VECTOR_DIR / "probe_cleaned"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for pt_file in sorted(INPUT_DIR.glob("layer*.pt")):
    vector = torch.load(pt_file, weights_only=True)
    cleaned = vector.clone()
    for dim in MASSIVE_DIMS:
        if dim < cleaned.shape[0]:
            cleaned[dim] = 0.0
    cleaned = cleaned / cleaned.norm()
    torch.save(cleaned, OUTPUT_DIR / pt_file.name)
    print(f"Cleaned {pt_file.name}")

print(f"\nSaved to {OUTPUT_DIR}")
```

**Command:**
```bash
python experiments/massive-activations/clean_probe_vectors.py
```

---

## Step 13: Probe causality steering (Confirmatory)

**Purpose:** Reproduce Phase 1 finding that cleaning hurts probe.

**Command:**
```bash
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/chirp/refusal \
    --method probe_cleaned \
    --layers 15 \
    --coefficients 800,1000,1200,1400 \
    --direction positive \
    --subset 0 \
    --save-responses best
```

**Expected:** probe_cleaned Δ < probe original Δ (cleaning hurts probe)

**From Phase 1 notepad:** probe=33.4%, probe_cleaned=8.9%. This step formalizes that finding.

---

## Step 14: Phase 2 Analysis

**Purpose:** Compile all results and verify Phase 2 success criteria.

**Script to create:** `experiments/massive-activations/analyze_phase2.py`

```python
#!/usr/bin/env python3
"""Analyze Phase 2 results."""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("experiments/massive-activations/steering")
MIN_COHERENCE = 70

def load_trait_results(trait_path: str):
    """Load steering results for a trait."""
    results_file = RESULTS_DIR / trait_path / "instruct/response__5/steering/results.jsonl"
    if not results_file.exists():
        return None, {}

    results = defaultdict(list)
    baseline = None

    with open(results_file) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("type") == "baseline":
                baseline = entry["result"]
            elif "config" in entry and "result" in entry:
                config = entry["config"]["vectors"][0]
                method = config["method"]
                result = entry["result"]
                results[method].append({
                    "layer": config["layer"],
                    "weight": config["weight"],
                    "trait_mean": result.get("trait_mean", 0),
                    "coherence_mean": result.get("coherence_mean", 0),
                })

    return baseline, dict(results)

def best_delta(results: list, baseline_trait: float) -> float:
    """Get best delta with coherence >= threshold."""
    valid = [r for r in results if r["coherence_mean"] >= MIN_COHERENCE]
    if not valid:
        return 0
    best = max(valid, key=lambda r: r["trait_mean"])
    return best["trait_mean"] - baseline_trait

print("="*60)
print("PHASE 2 RESULTS")
print("="*60)

# Sycophancy generalization
print("\n## Sycophancy Generalization")
baseline, syc_results = load_trait_results("pv_natural/sycophancy")
if baseline:
    bt = baseline.get("trait_mean", 0)
    for method in ["mean_diff", "mean_diff_cleaned", "probe"]:
        if method in syc_results:
            delta = best_delta(syc_results[method], bt)
            print(f"  {method}: Δ={delta:+.1f}")

# Ablation
print("\n## Cleaning Ablation (refusal)")
baseline, ref_results = load_trait_results("chirp/refusal")
if baseline:
    bt = baseline.get("trait_mean", 0)
    for method in ["mean_diff", "mean_diff_top1", "mean_diff_top5", "mean_diff_cleaned"]:
        if method in ref_results:
            delta = best_delta(ref_results[method], bt)
            print(f"  {method}: Δ={delta:+.1f}")

# Probe causality
print("\n## Probe Causality")
if baseline and ref_results:
    bt = baseline.get("trait_mean", 0)
    for method in ["probe", "probe_cleaned"]:
        if method in ref_results:
            delta = best_delta(ref_results[method], bt)
            print(f"  {method}: Δ={delta:+.1f}")

# Success criteria
print("\n" + "="*60)
print("SUCCESS CRITERIA CHECK")
print("="*60)
# Add checks here based on actual results
```

**Command:**
```bash
python experiments/massive-activations/analyze_phase2.py
```

---

## Phase 2 Checkpoints

### After Step 9 (sycophancy steering) — Exploratory
- [ ] Document whether sycophancy shows similar pattern to refusal
- [ ] If pattern differs, that's still a valid finding — document why (different trait type, different massive dim impact)

### After Step 11 (ablation)
- [ ] top-1 ≥ top-13 (confirming Phase 1: dim 443 alone is sufficient, over-cleaning may hurt)
- [ ] If top-1 << top-13, revisit — Phase 1 found opposite

### After Step 13 (probe causality) — Confirmatory
- [ ] probe_cleaned < probe (reproducing Phase 1: 8.9% < 33.4%)
- [ ] Formalizes finding that probe uses massive dims productively

---

## Phase 2 Expected Results

### Sycophancy Generalization (Exploratory)

| Method | Expected Δ | Interpretation |
|--------|-----------|----------------|
| mean_diff | ~0 or weak | Likely fails (if pattern generalizes) |
| mean_diff_cleaned | ? | Unknown — may or may not recover |
| probe | > mean_diff | Expected to outperform mean_diff |

*Note: Sycophancy may behave differently than refusal. This is exploratory.*

### Cleaning Ablation

| Variant | Dims | Expected Δ (from Phase 1) |
|---------|------|-----------|
| mean_diff | 0 | ~0 |
| mean_diff_top1 | 1 | ~29 (29.1% in Phase 1) |
| mean_diff_top5 | 5 | ~25-29 |
| mean_diff_cleaned | 13 | ~25 (25.2% in Phase 1) |

*Key finding to confirm: top-1 ≥ top-13 (fewer dims works better)*

### Probe Causality (Confirmatory)

| Method | Expected Δ (from Phase 1) |
|--------|-----------|
| probe | ~33 (33.4% in Phase 1) |
| probe_cleaned | ~9 (8.9% in Phase 1) |

*Formalizes Phase 1 finding that probe uses massive dims productively.*

---

## Phase 2 Notes (filled 2026-01-29)

### Observations
- Sycophancy extraction was very fast (~11s)
- Sycophancy mean_diff actually works (+11.2) — unlike refusal
- Non-monotonic cleaning pattern: top-1 > cleaned > top-3 > top-5 ≈ top-10
- Probe causality cleanly confirmed: 33.0 → 8.6 with cleaning

### Unexpected findings
- **Sycophancy differs from refusal:** mean_diff works for sycophancy (+11.2 at 92% coherence)
- **Over-cleaning hurts:** top-5 (14.2) worst, top-1 (28.8) and cleaned (24.8) best
- **Dim 443 dominates:** Zeroing just dim 443 matches or exceeds full 13-dim cleaning

### Decisions made
- Used coefficients 800-1400 based on Phase 1 learnings (not auto-search)
- Ran sycophancy on layers 30%-60% (L10-L20)
- Ran refusal ablation on L15 only (best layer from Phase 1)
- Steps 12-13 (probe_cleaned) already done in Phase 1, reused results

### Time tracking
- Step 7 (extraction): 11s
- Step 8 (cleaning): <1s
- Step 9 (sycophancy steering): ~10 min
- Step 10 (ablation script): <1s
- Step 11 (ablation steering): ~2 min
- Step 12-13 (probe): Already done in Phase 1
- Step 14 (analysis): <1s
- Total: ~13 min

### Final Results

**Sycophancy (baseline=56.1):**
| Method | Δ | Coherence |
|--------|---|-----------|
| mean_diff | +11.2 | 92% |
| mean_diff_cleaned | +19.9 | 85% |
| probe | +21.3 | 89% |

**Refusal Ablation (baseline=0.3):**
| Method | Dims | Δ |
|--------|------|---|
| mean_diff_top1 | 1 | +28.8 |
| mean_diff | 0 | +27.1 |
| mean_diff_cleaned | 13 | +24.8 |
| mean_diff_top3 | 3 | +20.7 |
| mean_diff_top5 | 5 | +14.2 |

**Probe Causality (baseline=0.3):**
| Method | Δ |
|--------|---|
| probe | +33.0 |
| probe_cleaned | +8.6 |

**All Phase 2 criteria checked:**
1. Sycophancy: Pattern holds but mean_diff partially works (exploratory finding)
2. Top-1 ≥ Top-13: ✓ Confirmed (28.8 ≥ 24.8)
3. Cleaning hurts probe: ✓ Confirmed (33.0 > 8.6)

---
---

# Phase 3: Cross-Model Comparison (gemma-2-2b)

**Goal:** Test whether milder massive activation contamination (~60x in gemma-2-2b vs ~1000x in gemma-3-4b) produces different results.

**Hypothesis:** With milder contamination, mean_diff should work better on gemma-2-2b than on gemma-3-4b. The pattern (probe > cleaned > mean_diff) should still hold but with smaller gaps.

**Phase 3 Success Criteria:**
1. mean_diff performs better on gemma-2-2b than gemma-3-4b (milder contamination helps)
2. Pattern still holds: probe > mean_diff (even with mild contamination)
3. Cleaning ablation shows similar dim-dominance pattern
4. Cross-model comparison: quantify contamination severity vs method performance

---

## Phase 3 Setup

| Aspect | gemma-2-2b | gemma-3-4b (Phase 1-2) |
|--------|------------|------------------------|
| Model (base) | google/gemma-2-2b | google/gemma-3-4b-pt |
| Model (instruct) | google/gemma-2-2b-it | google/gemma-3-4b-it |
| Layers | 26 | 34 |
| Hidden dim | 2304 | 2560 |
| Dominant dim | 334 (~60x) | 443 (~1000x) |
| Massive dims | 8 total | 13 total |

**Massive dims for gemma-2-2b:** [334, 535, 682, 1068, 1393, 1570, 1645, 1807]

---

## Phase 3 Prerequisites

### 1. Verify calibration data exists

```bash
cat experiments/gemma-2-2b/inference/instruct/massive_activations/calibration.json | python -c "import sys,json; d=json.load(sys.stdin); print('Massive dims:', d['aggregate']['top_dims_by_layer']['0'][:5])"
```

**Expected:** `[243, 535, 1570, 881, 334]` (or similar)

### 2. Update experiment config for multi-model

The experiment config needs to support gemma-2-2b. We'll use a model_variant suffix approach.

---

## Step 15: Update experiment config for gemma-2-2b

**Purpose:** Add gemma-2-2b variants to the experiment config.

**File to update:** `experiments/massive-activations/config.json`

```json
{
  "defaults": {
    "extraction": "base",
    "application": "instruct"
  },
  "model_variants": {
    "base": {
      "model": "google/gemma-3-4b-pt"
    },
    "instruct": {
      "model": "google/gemma-3-4b-it"
    },
    "gemma2_base": {
      "model": "google/gemma-2-2b"
    },
    "gemma2_instruct": {
      "model": "google/gemma-2-2b-it"
    }
  }
}
```

**Verify:**
```bash
cat experiments/massive-activations/config.json | python -c "import sys,json; d=json.load(sys.stdin); print('gemma2_base' in d['model_variants'])"
# Should be True
```

---

## Step 16: Extract refusal vectors (gemma-2-2b)

**Purpose:** Get trait vectors for gemma-2-2b using same methodology.

**Command:**
```bash
python extraction/run_pipeline.py \
    --experiment massive-activations \
    --traits chirp/refusal \
    --methods mean_diff,probe \
    --position "response[:5]" \
    --model-variant gemma2_base \
    --no-vet \
    --no-logitlens
```

**Expected output:**
```
experiments/massive-activations/extraction/chirp/refusal/gemma2_base/
├── responses/
│   ├── pos.json
│   └── neg.json
└── vectors/response__5/residual/
    ├── mean_diff/layer*.pt  (26 files, layers 0-25)
    └── probe/layer*.pt      (26 files)
```

**Verify:**
```bash
ls experiments/massive-activations/extraction/chirp/refusal/gemma2_base/vectors/response__5/residual/mean_diff/ | wc -l
# Should be 26
```

---

## Step 17: Create cleaned vectors (gemma-2-2b)

**Purpose:** Zero out gemma-2-2b massive dims.

**Script to create:** `experiments/massive-activations/clean_vectors_gemma2.py`

```python
#!/usr/bin/env python3
"""Zero out massive dims from gemma-2-2b mean_diff vectors."""

import torch
from pathlib import Path

# Massive dims from gemma-2-2b calibration
MASSIVE_DIMS_GEMMA2 = [334, 535, 682, 1068, 1393, 1570, 1645, 1807]

# Top dims by magnitude at mid-layers (L12-L15)
# 334 appears in 24/26 layers, 1068 in 21, 1807 in 19
MASSIVE_DIMS_TOP1 = [334]
MASSIVE_DIMS_TOP5 = [334, 1068, 1807, 1570, 535]

VECTOR_DIR = Path("experiments/massive-activations/extraction/chirp/refusal/gemma2_base/vectors/response__5/residual")
INPUT_DIR = VECTOR_DIR / "mean_diff"

VARIANTS = {
    "mean_diff_cleaned": MASSIVE_DIMS_GEMMA2,  # All 8
    "mean_diff_top1": MASSIVE_DIMS_TOP1,        # Just dim 334
    "mean_diff_top5": MASSIVE_DIMS_TOP5,        # Top 5
}

for variant_name, dims_to_zero in VARIANTS.items():
    output_dir = VECTOR_DIR / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for pt_file in sorted(INPUT_DIR.glob("layer*.pt")):
        vector = torch.load(pt_file, weights_only=True)
        cleaned = vector.clone()
        for dim in dims_to_zero:
            if dim < cleaned.shape[0]:
                cleaned[dim] = 0.0
        cleaned = cleaned / cleaned.norm()
        torch.save(cleaned, output_dir / pt_file.name)

    print(f"Created {variant_name}: zeroed {len(dims_to_zero)} dims")

print("\nDone!")
```

**Command:**
```bash
python experiments/massive-activations/clean_vectors_gemma2.py
```

**Verify:**
```bash
ls experiments/massive-activations/extraction/chirp/refusal/gemma2_base/vectors/response__5/residual/ | grep mean_diff
# Should show: mean_diff, mean_diff_cleaned, mean_diff_top1, mean_diff_top5
```

---

## Step 18: Refusal steering (gemma-2-2b)

**Purpose:** Test steering with gemma-2-2b vectors.

**Decision:**
- Layers: `30%-60%` of 26 layers = L8-L15
- Coefficients: Start with same range as gemma-3-4b, adjust if needed
- Note: gemma-2-2b may need different coefficient range due to milder contamination

**Commands:**
```bash
# mean_diff
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/chirp/refusal \
    --extraction-variant gemma2_base \
    --model-variant gemma2_instruct \
    --method mean_diff \
    --layers "30%-60%" \
    --coefficients 500,800,1000,1200,1500 \
    --direction positive \
    --subset 0 \
    --save-responses best

# mean_diff_cleaned
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/chirp/refusal \
    --extraction-variant gemma2_base \
    --model-variant gemma2_instruct \
    --method mean_diff_cleaned \
    --layers "30%-60%" \
    --coefficients 500,800,1000,1200,1500 \
    --direction positive \
    --subset 0 \
    --save-responses best

# mean_diff_top1
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/chirp/refusal \
    --extraction-variant gemma2_base \
    --model-variant gemma2_instruct \
    --method mean_diff_top1 \
    --layers "30%-60%" \
    --coefficients 500,800,1000,1200,1500 \
    --direction positive \
    --subset 0 \
    --save-responses best

# probe
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/chirp/refusal \
    --extraction-variant gemma2_base \
    --model-variant gemma2_instruct \
    --method probe \
    --layers "30%-60%" \
    --coefficients 500,800,1000,1200,1500 \
    --direction positive \
    --subset 0 \
    --save-responses best
```

**Expected:** mean_diff should perform better than on gemma-3-4b (milder contamination).

---

## Step 19: Extract sycophancy vectors (gemma-2-2b)

**Purpose:** Test sycophancy on gemma-2-2b for cross-trait comparison.

**Command:**
```bash
python extraction/run_pipeline.py \
    --experiment massive-activations \
    --traits pv_natural/sycophancy \
    --methods mean_diff,probe \
    --position "response[:5]" \
    --model-variant gemma2_base \
    --no-vet \
    --no-logitlens
```

**Verify:**
```bash
ls experiments/massive-activations/extraction/pv_natural/sycophancy/gemma2_base/vectors/response__5/residual/mean_diff/ | wc -l
# Should be 26
```

---

## Step 20: Create cleaned sycophancy vectors (gemma-2-2b)

**Script to create:** `experiments/massive-activations/clean_vectors_gemma2_sycophancy.py`

```python
#!/usr/bin/env python3
"""Zero out massive dims from gemma-2-2b sycophancy mean_diff vectors."""

import torch
from pathlib import Path

MASSIVE_DIMS_GEMMA2 = [334, 535, 682, 1068, 1393, 1570, 1645, 1807]

VECTOR_DIR = Path("experiments/massive-activations/extraction/pv_natural/sycophancy/gemma2_base/vectors/response__5/residual")
INPUT_DIR = VECTOR_DIR / "mean_diff"
OUTPUT_DIR = VECTOR_DIR / "mean_diff_cleaned"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for pt_file in sorted(INPUT_DIR.glob("layer*.pt")):
    vector = torch.load(pt_file, weights_only=True)
    cleaned = vector.clone()
    for dim in MASSIVE_DIMS_GEMMA2:
        if dim < cleaned.shape[0]:
            cleaned[dim] = 0.0
    cleaned = cleaned / cleaned.norm()
    torch.save(cleaned, OUTPUT_DIR / pt_file.name)
    print(f"Cleaned {pt_file.name}")

print(f"\nSaved to {OUTPUT_DIR}")
```

**Command:**
```bash
python experiments/massive-activations/clean_vectors_gemma2_sycophancy.py
```

---

## Step 21: Sycophancy steering (gemma-2-2b)

**Purpose:** Test sycophancy steering on gemma-2-2b.

**Commands:**
```bash
# mean_diff
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/pv_natural/sycophancy \
    --extraction-variant gemma2_base \
    --model-variant gemma2_instruct \
    --method mean_diff \
    --layers "30%-60%" \
    --coefficients 500,800,1000,1200,1500 \
    --direction positive \
    --subset 0 \
    --save-responses best

# mean_diff_cleaned
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/pv_natural/sycophancy \
    --extraction-variant gemma2_base \
    --model-variant gemma2_instruct \
    --method mean_diff_cleaned \
    --layers "30%-60%" \
    --coefficients 500,800,1000,1200,1500 \
    --direction positive \
    --subset 0 \
    --save-responses best

# probe
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/pv_natural/sycophancy \
    --extraction-variant gemma2_base \
    --model-variant gemma2_instruct \
    --method probe \
    --layers "30%-60%" \
    --coefficients 500,800,1000,1200,1500 \
    --direction positive \
    --subset 0 \
    --save-responses best
```

---

## Step 22: Cross-model analysis

**Purpose:** Compare results across models to quantify contamination severity impact.

**Script to create:** `experiments/massive-activations/analyze_phase3.py`

```python
#!/usr/bin/env python3
"""Cross-model comparison of massive activation impact."""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("experiments/massive-activations/steering")
MIN_COHERENCE = 70

def load_results(trait: str, model_variant: str):
    """Load steering results for trait/model combination."""
    # Try different path patterns
    patterns = [
        RESULTS_DIR / trait / model_variant / "response__5/steering/results.jsonl",
        RESULTS_DIR / trait / model_variant / "response__5/results.jsonl",
    ]

    for results_file in patterns:
        if results_file.exists():
            break
    else:
        return None, {}

    results = defaultdict(list)
    baseline = None

    with open(results_file) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("type") == "baseline":
                baseline = entry["result"]
            elif "config" in entry and "result" in entry:
                config = entry["config"]["vectors"][0]
                method = config["method"]
                result = entry["result"]
                results[method].append({
                    "layer": config["layer"],
                    "weight": config["weight"],
                    "trait_mean": result.get("trait_mean", 0),
                    "coherence_mean": result.get("coherence_mean", 0),
                })

    return baseline, dict(results)

def best_delta(results: list, baseline_trait: float) -> tuple:
    """Get best delta with coherence >= threshold."""
    valid = [r for r in results if r["coherence_mean"] >= MIN_COHERENCE]
    if not valid:
        return 0, 0
    best = max(valid, key=lambda r: r["trait_mean"])
    return best["trait_mean"] - baseline_trait, best["coherence_mean"]

print("="*70)
print("PHASE 3: CROSS-MODEL COMPARISON")
print("="*70)

models = {
    "gemma-3-4b": "instruct",
    "gemma-2-2b": "gemma2_instruct",
}

traits = ["chirp/refusal", "pv_natural/sycophancy"]
methods = ["mean_diff", "mean_diff_cleaned", "mean_diff_top1", "probe"]

for trait in traits:
    print(f"\n## {trait}")
    print("-" * 50)
    print(f"{'Model':<15} {'Method':<20} {'Δ':>8} {'Coh':>6}")
    print("-" * 50)

    for model_name, variant in models.items():
        baseline, results = load_results(trait, variant)
        if not baseline:
            print(f"{model_name:<15} NO DATA")
            continue

        bt = baseline.get("trait_mean", 0)
        for method in methods:
            if method in results:
                delta, coh = best_delta(results[method], bt)
                print(f"{model_name:<15} {method:<20} {delta:>+7.1f} {coh:>5.0f}%")

print("\n" + "="*70)
print("KEY COMPARISONS")
print("="*70)
print("\nContamination severity vs mean_diff performance:")
print("  gemma-3-4b (dim 443 ~1000x): mean_diff Δ = ?")
print("  gemma-2-2b (dim 334 ~60x):   mean_diff Δ = ?")
print("\nExpected: gemma-2-2b mean_diff > gemma-3-4b mean_diff")
```

**Command:**
```bash
python experiments/massive-activations/analyze_phase3.py
```

---

## Phase 3 Checkpoints

### After Step 18 (refusal steering gemma-2-2b)
- [ ] mean_diff works better on gemma-2-2b than gemma-3-4b
- [ ] Pattern still holds: probe > cleaned > mean_diff
- [ ] If mean_diff works well, confirms milder contamination hypothesis

### After Step 21 (sycophancy steering gemma-2-2b)
- [ ] Compare to gemma-3-4b sycophancy results
- [ ] Document any model-specific differences

### After Step 22 (cross-model analysis)
- [ ] Quantify contamination severity vs performance relationship
- [ ] Document cross-model patterns

---

## Phase 3 Expected Results

### Refusal (cross-model)

| Model | Contamination | mean_diff Δ | probe Δ | Gap |
|-------|---------------|-------------|---------|-----|
| gemma-3-4b | ~1000x | ~0 | ~33 | ~33 |
| gemma-2-2b | ~60x | ~10-20? | ~25-35? | smaller |

*Hypothesis: Smaller contamination → smaller gap between mean_diff and probe*

### Sycophancy (cross-model)

| Model | mean_diff Δ | probe Δ |
|-------|-------------|---------|
| gemma-3-4b | +11.2 | +21.3 |
| gemma-2-2b | ? | ? |

---

## Phase 3 Notes (fill during run)

### Observations
-

### Unexpected findings
-

### Decisions made
-

### Time tracking
- Step 15 (config):
- Step 16 (extraction):
- Step 17 (cleaning):
- Step 18 (refusal steering):
- Step 19 (sycophancy extraction):
- Step 20 (sycophancy cleaning):
- Step 21 (sycophancy steering):
- Step 22 (analysis):
- Total:
