# Massive Activations Experiment (Phase 2: Refined Ablations)

## Goal

Fill data gaps and run targeted ablations to finalize understanding of massive activation effects on steering vectors.

## Background

Phase 1 established:
- probe > mean_diff at short windows ([:5], [:10]) on gemma-3-4b
- mean_diff wins at [:20] where contamination dilutes
- Cleaning helps mean_diff (top-1 best), always hurts probe
- Coefficient ratio: mean_diff needs ~2x probe's coefficient on gemma-3-4b

**Open questions:**
1. Does clean-before-extraction differ from clean-after for probe?
2. Where exactly does the cleaning benefit drop off (top-2)?
3. Can we predict optimal coefficients from vector properties?

## Success Criteria

- [ ] All key method comparisons have results with coherence ≥68%
- [ ] Clean-before vs clean-after comparison complete for probe
- [ ] top-2 cleaning fills gap between top-1 and top-3
- [ ] Coefficient patterns documented with concrete recommendations

---

## Prerequisites

### 1. Verify existing calibration data

```bash
# gemma-3-4b massive dims
cat experiments/gemma-3-4b/inference/instruct/massive_activations/calibration.json | \
  python -c "import sys,json; d=json.load(sys.stdin); print('Dims:', d['aggregate']['top_dims_by_layer']['15'][:5])"
# Expected: [443, 1365, 1980, 295, 1209]

# gemma-2-2b massive dims
cat experiments/gemma-2-2b/inference/instruct/massive_activations/calibration.json | \
  python -c "import sys,json; d=json.load(sys.stdin); print('Dims:', d['aggregate']['top_dims_by_layer']['12'][:5])"
# Expected: [334, 1068, 1807, 1570, 682]
```

### 2. Verify existing extraction

```bash
ls experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__5/residual/ | head -10
# Should show: mean_diff, probe, mean_diff_cleaned, probe_cleaned, etc.
```

---

## Phase 1: Fill Coherence Gaps

**Purpose:** Ensure all key comparisons have results with coherence ≥68% (comparing at similar coherence levels for fair comparison).

### Step 1.1: Analyze current gaps

```bash
python -c "
import json
from pathlib import Path

def analyze_gaps(results_file, min_coherence=68):
    if not results_file.exists():
        return None

    baseline = 0
    method_data = {}

    with open(results_file) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('type') == 'baseline':
                baseline = entry['result'].get('trait_mean', 0)
            elif 'config' in entry and 'result' in entry:
                cfg = entry['config']['vectors'][0]
                res = entry['result']
                method = cfg['method']
                coh = res['coherence_mean']
                delta = res['trait_mean'] - baseline

                # Track best result with coherence >= threshold
                if coh >= min_coherence:
                    if method not in method_data or delta > method_data[method]['best_delta']:
                        method_data[method] = {
                            'best_coh': coh,
                            'best_delta': delta,
                            'has_valid': True,
                            'coef': cfg['weight']
                        }
                elif method not in method_data:
                    method_data[method] = {'best_coh': coh, 'best_delta': delta, 'has_valid': False, 'coef': cfg['weight']}

    return method_data

# Check key files
configs = [
    ('gemma-3-4b [:5]', 'instruct', 'response__5'),
    ('gemma-3-4b [:10]', 'instruct', 'response__10'),
    ('gemma-3-4b [:20]', 'instruct', 'response__20'),
    ('gemma-2-2b [:5]', 'gemma2_instruct', 'response__5'),
    ('gemma-2-2b [:10]', 'gemma2_instruct', 'response__10'),
    ('gemma-2-2b [:20]', 'gemma2_instruct', 'response__20'),
]

key_methods = ['probe', 'mean_diff', 'probe_top1', 'mean_diff_top1', 'probe_cleaned', 'mean_diff_cleaned']

print('Gap Analysis (target: coherence ≥68%)')
print('=' * 80)

for name, variant, pos in configs:
    results_file = Path(f'experiments/massive-activations/steering/chirp/refusal/{variant}/{pos}/steering/results.jsonl')
    data = analyze_gaps(results_file)
    if not data:
        print(f'{name}: NO DATA')
        continue

    gaps = []
    for method in key_methods:
        if method in data and not data[method].get('has_valid', False):
            gaps.append(f\"{method}(best:{data[method]['best_coh']:.0f}%)\")

    if gaps:
        print(f\"{name}: NEEDS FINER COEF - {', '.join(gaps)}\")
    else:
        print(f'{name}: OK')
"
```

### Step 1.2: Run finer coefficient grid for gaps

**Based on gap analysis, run targeted coefficients.** Only fill gaps where method doesn't have coherence ≥68%.

For gemma-3-4b (if gaps exist):
```bash
# Example: if probe needs finer grid around coherence cliff
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/chirp/refusal \
    --method probe \
    --position "response[:5]" \
    --layers 15 \
    --coefficients 1100,1150,1180,1220,1250 \
    --max-new-tokens 64 \
    --direction positive \
    --subset 0 \
    --save-responses best
```

For gemma-2-2b (if gaps exist):
```bash
# Example: finer grid for gemma-2-2b
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/chirp/refusal \
    --extraction-variant gemma2_base \
    --model-variant gemma2_instruct \
    --method probe \
    --position "response[:5]" \
    --layers 12 \
    --coefficients 80,85,95,100,105 \
    --max-new-tokens 64 \
    --direction positive \
    --subset 0 \
    --save-responses best
```

**Note:** Adjust coefficients based on gap analysis. Don't exceed known coherent ranges:
- gemma-3-4b: 800-2500
- gemma-2-2b: 30-130

### Checkpoint: After Phase 1

Re-run gap analysis to confirm all key methods now have results with coherence ≥68%.

---

## Phase 2: Clean-Before-Extraction

**Purpose:** Test whether cleaning activations before vector extraction differs from cleaning vectors after.

**Hypothesis:**
- For mean_diff: Nearly identical (both produce unit vectors in similar directions, though normalization order differs slightly)
- For probe: clean-before may hurt less because probe learns on clean data, vs clean-after which amputates learned weights

**Note on normalization:** Clean-after does `normalize(zero_dims(normalize(raw_diff)))` while clean-before does `normalize(zero_dims(raw_diff))`. Both produce unit vectors; the difference is minor but worth noting.

### Step 2.0: Generate activation files (required for preclean extraction)

Activation files were not preserved from original extraction. Re-run stage 3 to generate them:

```bash
# For gemma-3-4b
python extraction/run_pipeline.py \
    --experiment massive-activations \
    --traits chirp/refusal \
    --only-stage 3 \
    --model-variant base \
    --position "response[:5]"

# For gemma-2-2b
python extraction/run_pipeline.py \
    --experiment massive-activations \
    --traits chirp/refusal \
    --only-stage 3 \
    --model-variant gemma2_base \
    --position "response[:5]"
```

**Verify:**
```bash
ls experiments/massive-activations/extraction/chirp/refusal/base/activations/response__5/residual/
# Should show: train_all_layers.pt, val_all_layers.pt, metadata.json

ls experiments/massive-activations/extraction/chirp/refusal/gemma2_base/activations/response__5/residual/
# Should show: train_all_layers.pt, val_all_layers.pt, metadata.json
```

### Step 2.1: Implement CleanedMethod wrapper

**File:** `core/methods.py` — add after existing methods:

```python
class PreCleanedMethod(ExtractionMethod):
    """Wrapper that cleans massive dims from activations before extraction."""

    def __init__(self, base_method: ExtractionMethod, massive_dims: list):
        self.base_method = base_method
        self.massive_dims = massive_dims

    def extract(self, pos_acts: torch.Tensor, neg_acts: torch.Tensor, **kwargs):
        from core.math import remove_massive_dims
        pos_clean = remove_massive_dims(pos_acts, self.massive_dims, clone=True)
        neg_clean = remove_massive_dims(neg_acts, self.massive_dims, clone=True)
        return self.base_method.extract(pos_clean, neg_clean, **kwargs)
```

**Update `get_method()` to support preclean:**

```python
def get_method(name: str, massive_dims: list = None) -> ExtractionMethod:
    """Get extraction method by name, with optional pre-cleaning."""
    methods = {
        'mean_diff': MeanDifferenceMethod,
        'probe': ProbeMethod,
        'gradient': GradientMethod,
        'random_baseline': RandomBaselineMethod,
    }

    # Handle _preclean suffix
    base_name = name.replace('_preclean', '')
    wants_preclean = '_preclean' in name

    if base_name not in methods:
        raise ValueError(f"Unknown method '{name}'. Available: {list(methods.keys())}")

    method = methods[base_name]()

    if wants_preclean and massive_dims:
        return PreCleanedMethod(method, massive_dims)
    return method
```

### Step 2.2: Create preclean extraction script

**File:** `experiments/massive-activations/extract_preclean_vectors.py`

```python
#!/usr/bin/env python3
"""Extract vectors with pre-cleaned activations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import json
from core.methods import MeanDifferenceMethod, ProbeMethod, PreCleanedMethod
from utils.paths import get_activation_path, get_vector_path

# Load massive dims from calibration
def load_massive_dims(model: str) -> list:
    if 'gemma-3' in model or model == 'base':
        calib_path = Path("experiments/gemma-3-4b/inference/instruct/massive_activations/calibration.json")
    else:
        calib_path = Path("experiments/gemma-2-2b/inference/instruct/massive_activations/calibration.json")

    with open(calib_path) as f:
        calib = json.load(f)

    # Get dims appearing in top-5 at 3+ layers
    from collections import Counter
    all_dims = []
    for layer_dims in calib["aggregate"]["top_dims_by_layer"].values():
        all_dims.extend(layer_dims[:5])
    dim_counts = Counter(all_dims)
    return [d for d, c in dim_counts.items() if c >= 3]

# Configs
CONFIGS = [
    ("base", "experiments/massive-activations/extraction/chirp/refusal/base", 34),
    ("gemma2_base", "experiments/massive-activations/extraction/chirp/refusal/gemma2_base", 26),
]

for variant, base_dir, n_layers in CONFIGS:
    massive_dims = load_massive_dims(variant)
    print(f"\n{variant}: cleaning {len(massive_dims)} dims: {sorted(massive_dims)[:5]}...")

    for position in ["response__5"]:
        act_dir = Path(base_dir) / "activations" / position / "residual"
        vec_dir = Path(base_dir) / "vectors" / position / "residual"

        # Load activations
        train_path = act_dir / "train_all_layers.pt"
        if not train_path.exists():
            print(f"  {position}: activations not found")
            continue

        all_acts = torch.load(train_path, weights_only=True)

        # Load metadata for pos/neg split
        with open(act_dir / "metadata.json") as f:
            meta = json.load(f)
        n_pos = meta["n_examples_pos"]

        # Extract preclean vectors for each layer
        for method_name, MethodClass in [("mean_diff", MeanDifferenceMethod), ("probe", ProbeMethod)]:
            output_dir = vec_dir / f"{method_name}_preclean"
            output_dir.mkdir(parents=True, exist_ok=True)

            base_method = MethodClass()
            preclean_method = PreCleanedMethod(base_method, massive_dims)

            for layer in range(n_layers):
                layer_acts = all_acts[:, layer, :]
                pos_acts = layer_acts[:n_pos]
                neg_acts = layer_acts[n_pos:]

                result = preclean_method.extract(pos_acts, neg_acts)
                torch.save(result['vector'], output_dir / f"layer{layer}.pt")

            print(f"  {position}/{method_name}_preclean: {n_layers} layers")

print("\nDone!")
```

**Command:**
```bash
python experiments/massive-activations/extract_preclean_vectors.py
```

**Verify:**
```bash
ls experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__5/residual/ | grep preclean
# Expected: mean_diff_preclean, probe_preclean
```

### Step 2.3: Steering with preclean vectors

```bash
# gemma-3-4b mean_diff_preclean
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/chirp/refusal \
    --method mean_diff_preclean \
    --position "response[:5]" \
    --layers "30%-60%" \
    --coefficients 800,1000,1200,1500,2000 \
    --max-new-tokens 64 \
    --direction positive \
    --subset 0 \
    --save-responses best

# gemma-3-4b probe_preclean
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/chirp/refusal \
    --method probe_preclean \
    --position "response[:5]" \
    --layers "30%-60%" \
    --coefficients 800,1000,1200,1500,2000 \
    --max-new-tokens 64 \
    --direction positive \
    --subset 0 \
    --save-responses best
```

### Checkpoint: After Phase 2

Compare preclean vs cleaned variants:

```bash
python -c "
import json
from pathlib import Path

results_file = Path('experiments/massive-activations/steering/chirp/refusal/instruct/response__5/steering/results.jsonl')

baseline = 0
comparisons = {}

with open(results_file) as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('type') == 'baseline':
            baseline = entry['result'].get('trait_mean', 0)
        elif 'config' in entry and 'result' in entry:
            cfg = entry['config']['vectors'][0]
            res = entry['result']
            method = cfg['method']

            if res['coherence_mean'] >= 68:
                delta = res['trait_mean'] - baseline
                if method not in comparisons or delta > comparisons[method]['delta']:
                    comparisons[method] = {
                        'trait': res['trait_mean'],
                        'delta': delta,
                        'coh': res['coherence_mean']
                    }

print('Clean-before vs Clean-after (coherence ≥68%):')
print(f\"{'Method':<20} {'Δ':>8} {'Coh':>8}\")
print('-' * 40)

for method in ['probe', 'probe_cleaned', 'probe_preclean', 'mean_diff', 'mean_diff_cleaned', 'mean_diff_preclean']:
    if method in comparisons:
        r = comparisons[method]
        print(f\"{method:<20} {r['delta']:>+7.1f} {r['coh']:>7.1f}%\")
"
```

**Expected:**
- mean_diff ≈ mean_diff_preclean (mathematically identical)
- probe_preclean > probe_cleaned? (hypothesis: pre-clean hurts less)

---

## Phase 3: Top-2 Cleaning Ablation

**Purpose:** Fill the gap between top-1 (+28.8) and top-3 (+20.7) to understand the cleaning curve.

### Step 3.1: Create top-2 cleaned vectors

**File:** `experiments/massive-activations/clean_vectors_top2.py`

```python
#!/usr/bin/env python3
"""Create top-2 cleaned vectors."""

import torch
import json
from pathlib import Path
from collections import Counter

# Load calibration for gemma-3-4b
with open("experiments/gemma-3-4b/inference/instruct/massive_activations/calibration.json") as f:
    calib = json.load(f)

# Get global dims (appearing in top-5 at 3+ layers) for consistency with existing scripts
all_dims = []
for layer_dims in calib["aggregate"]["top_dims_by_layer"].values():
    all_dims.extend(layer_dims[:5])
dim_counts = Counter(all_dims)
global_dims = sorted([d for d, c in dim_counts.items() if c >= 3], key=lambda d: -dim_counts[d])

# Top-2 of the global massive dims
TOP_2_DIMS = global_dims[:2]
print(f"Top-2 global dims: {TOP_2_DIMS}")

VECTOR_DIR = Path("experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__5/residual")

for base_method in ["mean_diff", "probe"]:
    input_dir = VECTOR_DIR / base_method
    output_dir = VECTOR_DIR / f"{base_method}_top2"
    output_dir.mkdir(parents=True, exist_ok=True)

    for pt_file in sorted(input_dir.glob("layer*.pt")):
        vector = torch.load(pt_file, weights_only=True)
        cleaned = vector.clone()

        for dim in TOP_2_DIMS:
            if dim < cleaned.shape[0]:
                cleaned[dim] = 0.0

        cleaned = cleaned / cleaned.norm()
        torch.save(cleaned, output_dir / pt_file.name)

    print(f"Created {base_method}_top2")

print("Done!")
```

**Command:**
```bash
python experiments/massive-activations/clean_vectors_top2.py
```

### Step 3.2: Steering with top-2 vectors

```bash
# mean_diff_top2
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/chirp/refusal \
    --method mean_diff_top2 \
    --position "response[:5]" \
    --layers 15 \
    --coefficients 1000,1100,1150,1200,1250,1300 \
    --max-new-tokens 64 \
    --direction positive \
    --subset 0 \
    --save-responses best

# probe_top2
python analysis/steering/evaluate.py \
    --experiment massive-activations \
    --vector-from-trait massive-activations/chirp/refusal \
    --method probe_top2 \
    --position "response[:5]" \
    --layers 15 \
    --coefficients 1000,1100,1150,1200,1250,1300 \
    --max-new-tokens 64 \
    --direction positive \
    --subset 0 \
    --save-responses best
```

### Checkpoint: After Phase 3

Verify the cleaning curve:

```bash
python -c "
import json
from pathlib import Path

results_file = Path('experiments/massive-activations/steering/chirp/refusal/instruct/response__5/steering/results.jsonl')

baseline = 0
cleaning_curve = {}

with open(results_file) as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('type') == 'baseline':
            baseline = entry['result'].get('trait_mean', 0)
        elif 'config' in entry and 'result' in entry:
            cfg = entry['config']['vectors'][0]
            res = entry['result']
            method = cfg['method']

            if res['coherence_mean'] >= 68:
                delta = res['trait_mean'] - baseline
                if method not in cleaning_curve or delta > cleaning_curve[method]:
                    cleaning_curve[method] = delta

print('Cleaning curve (mean_diff, coherence ≥68%):')
for method in ['mean_diff', 'mean_diff_top1', 'mean_diff_top2', 'mean_diff_top3', 'mean_diff_top5', 'mean_diff_cleaned']:
    if method in cleaning_curve:
        dims = {'mean_diff': 0, 'mean_diff_top1': 1, 'mean_diff_top2': 2, 'mean_diff_top3': 3, 'mean_diff_top5': 5, 'mean_diff_cleaned': 13}
        print(f\"  {dims.get(method, '?'):>2} dims: {cleaning_curve[method]:>+6.1f}\")
"
```

**Expected pattern:** 0 < 1 dims (helps), 1 > 2 > 3 dims (over-cleaning hurts)

---

## Phase 4: Coefficient Pattern Analysis

**Purpose:** Document coefficient patterns and derive practical recommendations.

### Step 4.1: Compile coefficient data

```bash
python -c "
import json
from pathlib import Path

configs = [
    ('gemma-3-4b', 'instruct', ['response__5', 'response__10', 'response__20']),
    ('gemma-2-2b', 'gemma2_instruct', ['response__5', 'response__10', 'response__20']),
]

print('Optimal Coefficients (coherence ≥68%)')
print('=' * 70)
print(f\"{'Model':<12} {'Position':<12} {'Method':<20} {'Coef':>8} {'Δ':>8}\")
print('-' * 70)

for model, variant, positions in configs:
    for pos_dir in positions:
        results_file = Path(f'experiments/massive-activations/steering/chirp/refusal/{variant}/{pos_dir}/steering/results.jsonl')
        if not results_file.exists():
            continue

        baseline = 0
        method_best = {}

        with open(results_file) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get('type') == 'baseline':
                    baseline = entry['result'].get('trait_mean', 0)
                elif 'config' in entry and 'result' in entry:
                    cfg = entry['config']['vectors'][0]
                    res = entry['result']
                    method = cfg['method']

                    if res['coherence_mean'] >= 68:
                        delta = res['trait_mean'] - baseline
                        if method not in method_best or delta > method_best[method]['delta']:
                            method_best[method] = {'delta': delta, 'coef': cfg['weight']}

        pos_name = pos_dir.replace('response__', '[:') + ']'
        for method in ['probe', 'mean_diff', 'probe_top1', 'mean_diff_top1']:
            if method in method_best:
                r = method_best[method]
                print(f\"{model:<12} {pos_name:<12} {method:<20} {r['coef']:>8.0f} {r['delta']:>+7.1f}\")
"
```

### Step 4.2: Document patterns

Based on Phase 1-3 results, update `docs/other/research_findings.md` with:

1. **Coefficient ratio by model:**
   - gemma-3-4b: mean_diff needs ~2x probe's coefficient at [:5]/[:10]
   - gemma-2-2b: mean_diff needs ~1.3x probe's coefficient (more stable)

2. **Coefficient shift from cleaning:**
   - Cleaning top-1 shifts optimal coefficient down ~2x
   - This suggests massive dims inflate effective steering magnitude

3. **Practical recommendations:**
   - For gemma-3-4b: Start probe at 1000-1200, mean_diff at 2000-2500
   - For gemma-2-2b: Start probe at 70-90, mean_diff at 90-120
   - If using cleaned vectors: reduce by ~40%

---

## Expected Results

### Phase 2: Preclean vs Cleaned

| Method | Clean-after Δ | Clean-before Δ | Interpretation |
|--------|--------------|----------------|----------------|
| mean_diff | +24.8 | ~+24.8 | Nearly identical (normalization order differs slightly) |
| probe | +8.6 | ? | Hypothesis: higher (probe learns on clean data) |

### Phase 3: Cleaning Curve

| Dims cleaned | mean_diff Δ | Pattern |
|--------------|-------------|---------|
| 0 | +27.1 | baseline |
| 1 | +28.8 | helps |
| 2 | ? | between |
| 3 | +20.7 | over-cleaned |
| 5 | +14.2 | worse |
| 13 | +24.8 | partial recovery? |

---

## If Stuck

- **Activations not found:** Run Step 2.0 to generate activation files first
- **Preclean vectors not found:** Check extraction script created directories
- **Steering fails for new methods:** Verify vector files exist and have correct shape
- **No results with coherence ≥68%:** Expand coefficient grid (try finer steps around known working ranges)
- **mean_diff_preclean ≠ mean_diff_cleaned:** Should be ~similar (not identical due to normalization order); large differences indicate implementation issue

---

## Notes

_Space for observations during execution._
