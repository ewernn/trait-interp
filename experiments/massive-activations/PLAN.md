# Experiment: Massive Activations — Clean Slate

## Goal

Determine whether cleaning massive activation dimensions improves steering vector quality, and find the optimal cleaning strategy (preclean vs postclean, uniform vs layer-aware, top-1/2/3).

## Hypotheses

1. **Preclean ≠ postclean for probe** — preclean removes noise before training; postclean amputates learned weights
2. **Layer-aware > uniform** — different dims dominate at different layers
3. **There's an optimal cleaning granularity** — top-1 may underclean, top-3 may overclean
4. **Findings generalize** across traits (refusal, sycophancy) and models (gemma-3-4b, gemma-2-2b)

## Success Criteria

- [ ] Fair comparison of all cleaning variants at ≥68% coherence
- [ ] Clear ordering: preclean vs postclean for both probe and mean_diff
- [ ] Layer-aware vs uniform has a winner
- [ ] Optimal cleaning granularity identified (top-1, 2, or 3)
- [ ] Findings replicate on second trait and second model

## Method Naming Convention

| Variant | Description |
|---------|-------------|
| `probe` / `mean_diff` | Baseline (no cleaning) |
| `{method}_preclean_la{N}` | Clean top-N per-layer dims from activations BEFORE extraction |
| `{method}_postclean_u{N}` | Clean top-N global dims from vectors AFTER extraction (uniform) |
| `{method}_postclean_la{N}` | Clean top-N per-layer dims from vectors AFTER extraction |

N ∈ {1, 2, 3}. Total: 10 variants × 2 methods = 20 vector sets per trait/model.

---

## Prerequisites

### 1. Verify datasets

```bash
ls datasets/traits/chirp/refusal/positive.txt datasets/traits/chirp/refusal/negative.txt datasets/traits/chirp/refusal/steering.json
ls datasets/traits/pv_natural/sycophancy/positive.txt datasets/traits/pv_natural/sycophancy/negative.txt datasets/traits/pv_natural/sycophancy/steering.json
ls datasets/inference/massive_dims/calibration_50.json
```

### 2. Verify parent experiment calibration exists

```bash
# gemma-3-4b calibration (used for massive dim identification)
python -c "
import json
with open('experiments/gemma-3-4b/inference/instruct/massive_activations/calibration.json') as f:
    d = json.load(f)
print(f'Layers: {len(d[\"aggregate\"][\"top_dims_by_layer\"])}')
print(f'Top dims L15: {d[\"aggregate\"][\"top_dims_by_layer\"][\"15\"][:5]}')
"

# gemma-2-2b calibration
python -c "
import json
with open('experiments/gemma-2-2b/inference/instruct/massive_activations/calibration.json') as f:
    d = json.load(f)
print(f'Layers: {len(d[\"aggregate\"][\"top_dims_by_layer\"])}')
print(f'Top dims L12: {d[\"aggregate\"][\"top_dims_by_layer\"][\"12\"][:5]}')
"
```

### 3. Run calibration on base model variants

Extraction happens on base models, so massive dims should be identified from base model activations.

```bash
# gemma-3-4b base calibration
python analysis/massive_activations.py --experiment gemma-3-4b --model-variant base
python analysis/massive_activations_per_layer.py --experiment gemma-3-4b --model-variant base

# gemma-2-2b base calibration
python analysis/massive_activations.py --experiment gemma-2-2b --model-variant base
python analysis/massive_activations_per_layer.py --experiment gemma-2-2b --model-variant base
```

**Verify:**
```bash
ls experiments/gemma-3-4b/inference/base/massive_activations/calibration.json
ls experiments/gemma-3-4b/inference/base/massive_activations/per_layer_stats.json
ls experiments/gemma-2-2b/inference/base/massive_activations/calibration.json
ls experiments/gemma-2-2b/inference/base/massive_activations/per_layer_stats.json
```

Compare base vs instruct dims:
```bash
python -c "
import json
for model in ['gemma-3-4b', 'gemma-2-2b']:
    for var in ['base', 'instruct']:
        path = f'experiments/{model}/inference/{var}/massive_activations/calibration.json'
        try:
            with open(path) as f:
                d = json.load(f)
            mid = str(len(d['aggregate']['top_dims_by_layer']) // 2)
            print(f'{model} {var}: top-5 at L{mid}: {d[\"aggregate\"][\"top_dims_by_layer\"][mid][:5]}')
        except FileNotFoundError:
            print(f'{model} {var}: NOT FOUND')
"
```

---

## Phase 1: Per-Token Variance Analysis

**Purpose:** Understand massive dim behavior across tokens before designing cleaning. Already have calibration — just need to examine per-layer stats and optionally run with `--per-token` for deeper analysis.

### Step 1.1: Examine existing per-layer stats

```bash
python -c "
import json

for model in ['gemma-3-4b', 'gemma-2-2b']:
    with open(f'experiments/{model}/inference/instruct/massive_activations/per_layer_stats.json') as f:
        stats = json.load(f)

    print(f'\n{model}: {len(stats[\"massive_dims\"])} massive dims: {stats[\"massive_dims\"]}')
    print(f'  Layers analyzed: {stats[\"n_layers\"]}')

    # Show top-3 most consistent dims (lowest CV) at middle layer
    mid = str(stats['n_layers'] // 2)
    if mid in stats['per_layer']:
        layer_stats = stats['per_layer'][mid]
        by_cv = sorted(layer_stats.items(), key=lambda x: x[1]['cv'])
        print(f'  Layer {mid} - most consistent (low CV):')
        for dim_id, s in by_cv[:5]:
            print(f'    dim {dim_id}: ratio={s[\"ratio\"]:.1f}x, cv={s[\"cv\"]:.3f}, pct_above_10x={s[\"pct_above_10x\"]:.1f}%')
"
```

**Purpose of this step:** Understand which dims are consistently massive (low CV = safe to zero) vs intermittently massive (high CV = risky to zero).

### Step 1.2: Run per-token analysis (optional, for deeper understanding)

```bash
# Only if Step 1.1 raises questions about specific dims
python analysis/massive_activations.py \
    --experiment gemma-3-4b \
    --model-variant instruct \
    --per-token
```

### Checkpoint: After Phase 1

Document which dims will be used for cleaning. Should have:
- Per-layer top-1/2/3 dims for gemma-3-4b (layers 12-18)
- Per-layer top-1/2/3 dims for gemma-2-2b (layers 10-14)
- Global dims (appearing in 3+ layers) for uniform cleaning

---

## Phase 2: Extract Baseline Vectors (gemma-3-4b)

**Purpose:** Generate responses, extract activations (KEEP THEM), and compute baseline mean_diff + probe vectors.

### Step 2.1: Run extraction pipeline for refusal

```bash
python extraction/run_pipeline.py \
    --experiment massive-activations \
    --traits chirp/refusal \
    --position "response[:3]" \
    --methods mean_diff,probe \
    --no-vet \
    --no-logitlens
```

**Expected output:**
- `experiments/massive-activations/extraction/chirp/refusal/base/responses/`
- `experiments/massive-activations/extraction/chirp/refusal/base/activations/response__3/residual/train_all_layers.pt`
- `experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__3/residual/mean_diff/`
- `experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__3/residual/probe/`

### Step 2.2: Run extraction pipeline for sycophancy

```bash
python extraction/run_pipeline.py \
    --experiment massive-activations \
    --traits pv_natural/sycophancy \
    --position "response[:3]" \
    --methods mean_diff,probe \
    --no-vet \
    --no-logitlens
```

### Verify

```bash
# Check activations exist (needed for preclean)
ls experiments/massive-activations/extraction/chirp/refusal/base/activations/response__3/residual/
# Should show: train_all_layers.pt, val_all_layers.pt, metadata.json

# Check baseline vectors exist
ls experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__3/residual/
# Should show: mean_diff/, probe/
```

---

## Phase 3: Create Cleaned Vector Variants

**Purpose:** Generate all 18 cleaned variants (9 per method) for gemma-3-4b refusal.

### Step 3.1: Create the cleaning script

**File:** `experiments/massive-activations/create_cleaned_vectors.py`

This script creates all postclean and preclean variants. It:
1. Loads calibration data from parent experiment
2. For postclean: loads existing vectors, zeros dims, re-normalizes, saves
3. For preclean: loads activations, zeros dims per-layer, re-extracts vectors

```python
#!/usr/bin/env python3
"""
Create cleaned vector variants for massive-activations experiment.

Input: Existing vectors + activations from extraction pipeline
Output: Cleaned vector variants under new method names

Usage:
    python experiments/massive-activations/create_cleaned_vectors.py \
        --trait chirp/refusal \
        --extraction-variant base \
        --calibration-experiment gemma-3-4b \
        --calibration-variant base
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import torch
from collections import Counter

from core.methods import MeanDifferenceMethod, ProbeMethod
from core.math import remove_massive_dims
from utils.paths import get_vector_path, get_activation_path, get_activation_metadata_path


def load_calibration(experiment, variant):
    """Load calibration data: top_dims_by_layer and global massive dims."""
    calib_path = Path(f"experiments/{experiment}/inference/{variant}/massive_activations/calibration.json")
    with open(calib_path) as f:
        calib = json.load(f)

    top_dims_by_layer = calib["aggregate"]["top_dims_by_layer"]

    # Global dims: appearing in top-5 at 3+ layers
    all_dims = []
    for layer_dims in top_dims_by_layer.values():
        all_dims.extend(layer_dims[:5])
    dim_counts = Counter(all_dims)
    global_dims = sorted([d for d, c in dim_counts.items() if c >= 3], key=lambda d: -dim_counts[d])

    return top_dims_by_layer, global_dims


def create_postclean_vectors(experiment, trait, extraction_variant, top_dims_by_layer, global_dims, position, component="residual"):
    """Create postclean variants (uniform + layer-aware, top 1/2/3)."""
    methods = ["mean_diff", "probe"]

    for method in methods:
        for n_dims in [1, 2, 3]:
            # --- Uniform: same global dims at all layers ---
            uniform_name = f"{method}_postclean_u{n_dims}"
            dims_to_zero = global_dims[:n_dims]

            layer = 0
            while True:
                src_path = get_vector_path(experiment, trait, method, layer, extraction_variant, component, position)
                if not src_path.exists():
                    break

                vector = torch.load(src_path, weights_only=True)
                cleaned = remove_massive_dims(vector.unsqueeze(0), dims_to_zero, clone=True).squeeze(0)
                cleaned = cleaned / (cleaned.norm() + 1e-8)

                dst_path = get_vector_path(experiment, trait, uniform_name, layer, extraction_variant, component, position)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(cleaned, dst_path)
                layer += 1

            if layer > 0:
                print(f"  {uniform_name}: {layer} layers (dims: {dims_to_zero})")

            # --- Layer-aware: per-layer dims ---
            la_name = f"{method}_postclean_la{n_dims}"

            layer = 0
            while True:
                src_path = get_vector_path(experiment, trait, method, layer, extraction_variant, component, position)
                if not src_path.exists():
                    break

                layer_key = str(layer)
                if layer_key in top_dims_by_layer:
                    dims_to_zero = top_dims_by_layer[layer_key][:n_dims]
                else:
                    dims_to_zero = global_dims[:n_dims]  # fallback

                vector = torch.load(src_path, weights_only=True)
                cleaned = remove_massive_dims(vector.unsqueeze(0), dims_to_zero, clone=True).squeeze(0)
                cleaned = cleaned / (cleaned.norm() + 1e-8)

                dst_path = get_vector_path(experiment, trait, la_name, layer, extraction_variant, component, position)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(cleaned, dst_path)
                layer += 1

            if layer > 0:
                print(f"  {la_name}: {layer} layers")


def create_preclean_vectors(experiment, trait, extraction_variant, top_dims_by_layer, position, component="residual"):
    """Create preclean variants (layer-aware top 1/2/3)."""
    act_path = get_activation_path(experiment, trait, extraction_variant, component, position)
    meta_path = get_activation_metadata_path(experiment, trait, extraction_variant, component, position)

    if not act_path.exists():
        print(f"  ERROR: Activations not found at {act_path}")
        return

    all_acts = torch.load(act_path, weights_only=True)
    with open(meta_path) as f:
        meta = json.load(f)

    n_pos = meta["n_examples_pos"]
    n_layers = meta.get("n_layers", all_acts.shape[1])

    base_methods = {"mean_diff": MeanDifferenceMethod(), "probe": ProbeMethod()}

    for method_name, base_method in base_methods.items():
        for n_dims in [1, 2, 3]:
            variant_name = f"{method_name}_preclean_la{n_dims}"
            n_extracted = 0

            for layer_idx in range(n_layers):
                layer_key = str(layer_idx)
                if layer_key in top_dims_by_layer:
                    dims_to_zero = top_dims_by_layer[layer_key][:n_dims]
                else:
                    continue

                layer_acts = all_acts[:, layer_idx, :]
                pos_acts = layer_acts[:n_pos]
                neg_acts = layer_acts[n_pos:]

                # Clean activations BEFORE extraction
                pos_clean = remove_massive_dims(pos_acts, dims_to_zero, clone=True)
                neg_clean = remove_massive_dims(neg_acts, dims_to_zero, clone=True)

                result = base_method.extract(pos_clean, neg_clean)
                vector = result['vector']

                dst_path = get_vector_path(experiment, trait, variant_name, layer_idx, extraction_variant, component, position)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(vector, dst_path)
                n_extracted += 1

            if n_extracted > 0:
                print(f"  {variant_name}: {n_extracted} layers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create cleaned vector variants")
    parser.add_argument("--trait", required=True, help="e.g., chirp/refusal")
    parser.add_argument("--extraction-variant", default="base")
    parser.add_argument("--calibration-experiment", required=True, help="e.g., gemma-3-4b")
    parser.add_argument("--calibration-variant", default="instruct")
    parser.add_argument("--position", default="response[:3]")
    args = parser.parse_args()

    # Pass raw position string — path helpers call sanitize_position() internally
    position = args.position

    print(f"Loading calibration from {args.calibration_experiment}/{args.calibration_variant}...")
    top_dims_by_layer, global_dims = load_calibration(args.calibration_experiment, args.calibration_variant)
    print(f"  Global massive dims ({len(global_dims)}): {global_dims[:5]}...")

    print(f"\nCreating postclean variants for {args.trait}...")
    create_postclean_vectors("massive-activations", args.trait, args.extraction_variant,
                             top_dims_by_layer, global_dims, position)

    print(f"\nCreating preclean variants for {args.trait}...")
    create_preclean_vectors("massive-activations", args.trait, args.extraction_variant,
                            top_dims_by_layer, position)

    print("\nDone!")
```

### Step 3.2: Run the cleaning script

```bash
python experiments/massive-activations/create_cleaned_vectors.py \
    --trait chirp/refusal \
    --extraction-variant base \
    --calibration-experiment gemma-3-4b \
    --calibration-variant base \
    --position "response[:3]"
```

### Step 3.3: Verify cleaning actually changes vectors

Sanity check: if cleaned vectors are identical to originals, cleaning had no effect and steering evals would be wasted.

```bash
python -c "
import torch
from pathlib import Path
import torch.nn.functional as F

vec_dir = Path('experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__3/residual')

for base in ['probe', 'mean_diff']:
    base_vec = torch.load(vec_dir / base / 'layer15.pt', weights_only=True).float()
    print(f'\n{base} L15 — cosine similarity with cleaned variants:')

    for variant_dir in sorted(vec_dir.iterdir()):
        name = variant_dir.name
        if not name.startswith(base + '_'):
            continue
        variant_vec = torch.load(variant_dir / 'layer15.pt', weights_only=True).float()
        cos_sim = F.cosine_similarity(base_vec.unsqueeze(0), variant_vec.unsqueeze(0)).item()
        print(f'  {name:<30} cos_sim={cos_sim:.6f}  (Δ={1-cos_sim:.6f})')
"
```

**Expected:** Cosine similarities < 1.0 (ideally 0.95-0.999). If any variant has cos_sim ≈ 1.000000, that cleaning had no effect — skip it in steering evals.

### Verify directory structure

```bash
ls experiments/massive-activations/extraction/chirp/refusal/base/vectors/response__3/residual/ | sort
# Should show 20 directories:
# mean_diff, mean_diff_postclean_la1, mean_diff_postclean_la2, mean_diff_postclean_la3,
# mean_diff_postclean_u1, mean_diff_postclean_u2, mean_diff_postclean_u3,
# mean_diff_preclean_la1, mean_diff_preclean_la2, mean_diff_preclean_la3,
# probe, probe_postclean_la1, probe_postclean_la2, probe_postclean_la3,
# probe_postclean_u1, probe_postclean_u2, probe_postclean_u3,
# probe_preclean_la1, probe_preclean_la2, probe_preclean_la3
```

---

## Phase 4: Steering Evaluation (gemma-3-4b, refusal)

**Purpose:** Run adaptive coefficient search across L12-18 for all method variants. The adaptive search automatically finds the coherence cliff per layer.

### Step 4.1: Baseline methods

```bash
for METHOD in probe mean_diff; do
    python analysis/steering/evaluate.py \
        --experiment massive-activations \
        --vector-from-trait massive-activations/chirp/refusal \
        --method $METHOD \
        --position "response[:3]" \
        --layers 12,13,14,15,16,17,18 \
        --search-steps 7 \
        --max-new-tokens 128 \
        --direction positive \
        --subset 0 \
        --save-responses best
done
```

### Step 4.2: All cleaning variants

```bash
for METHOD in probe_preclean_la1 probe_preclean_la2 probe_preclean_la3 \
              probe_postclean_u1 probe_postclean_u2 probe_postclean_u3 \
              probe_postclean_la1 probe_postclean_la2 probe_postclean_la3 \
              mean_diff_preclean_la1 mean_diff_preclean_la2 mean_diff_preclean_la3 \
              mean_diff_postclean_u1 mean_diff_postclean_u2 mean_diff_postclean_u3 \
              mean_diff_postclean_la1 mean_diff_postclean_la2 mean_diff_postclean_la3; do
    python analysis/steering/evaluate.py \
        --experiment massive-activations \
        --vector-from-trait massive-activations/chirp/refusal \
        --method $METHOD \
        --position "response[:3]" \
        --layers 12,13,14,15,16,17,18 \
        --search-steps 7 \
        --max-new-tokens 128 \
        --direction positive \
        --subset 0 \
        --save-responses best
done
```

### Step 4.3: Analyze coefficient patterns

```bash
python -c "
import json
from pathlib import Path

results_file = Path('experiments/massive-activations/steering/chirp/refusal/instruct/response__3/steering/results.jsonl')
if not results_file.exists():
    print('No results yet')
    exit()

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
                    method_best[method] = {
                        'delta': delta, 'coef': cfg['weight'],
                        'coh': res['coherence_mean'], 'layer': cfg['layer']
                    }

print(f'Baseline trait: {baseline:.1f}%')
print(f\"{'Method':<30} {'Δ':>8} {'Coef':>8} {'Coh':>8}\")
print('-' * 58)

for method in sorted(method_best.keys()):
    r = method_best[method]
    print(f\"{method:<30} {r['delta']:>+7.1f} {r['coef']:>8.0f} {r['coh']:>7.1f}%\")
"
```

### Checkpoint: After Phase 4

Phase 4 covers L12-18 with adaptive search. Analyze results:

```bash
python -c "
import json
from pathlib import Path

results_file = Path('experiments/massive-activations/steering/chirp/refusal/instruct/response__3/steering/results.jsonl')
if not results_file.exists():
    print('No results yet')
    exit()

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
                    method_best[method] = {
                        'delta': delta, 'coef': cfg['weight'],
                        'coh': res['coherence_mean'], 'layer': cfg['layer']
                    }

print(f'Baseline trait: {baseline:.1f}%')
print(f\"{'Method':<30} {'Δ':>8} {'Coef':>8} {'Layer':>6} {'Coh':>8}\")
print('-' * 66)

for method in sorted(method_best.keys()):
    r = method_best[method]
    print(f\"{method:<30} {r['delta']:>+7.1f} {r['coef']:>8.0f} L{r['layer']:>4} {r['coh']:>7.1f}%\")
"
```

Should have results for L12-18 across all 20 method variants. Identify:
- Does preclean or postclean win?
- Layer-aware vs uniform?
- Optimal cleaning granularity (top-1, 2, or 3)?

---

## Phase 5: Validate — Second Trait (sycophancy)

**Purpose:** Check if findings generalize beyond refusal.

### Step 5.1: Create cleaned vectors for sycophancy

```bash
python experiments/massive-activations/create_cleaned_vectors.py \
    --trait pv_natural/sycophancy \
    --extraction-variant base \
    --calibration-experiment gemma-3-4b \
    --calibration-variant base \
    --position "response[:3]"
```

### Step 5.2: Run steering (key methods only)

Based on Phase 4 findings, run only the winning variants + baselines.

```bash
# Run baselines + best cleaning variants from Phase 4
for METHOD in probe mean_diff {best_preclean} {best_postclean}; do
    python analysis/steering/evaluate.py \
        --experiment massive-activations \
        --vector-from-trait massive-activations/pv_natural/sycophancy \
        --method $METHOD \
        --position "response[:3]" \
        --layers 12,13,14,15,16,17,18 \
        --search-steps 7 \
        --max-new-tokens 128 \
        --direction positive \
        --subset 0 \
        --save-responses best
done
```

---

## Phase 6: Validate — Second Model (gemma-2-2b)

**Purpose:** Check if findings generalize to a model with milder massive dims.

### Step 6.1: Extract vectors

```bash
python extraction/run_pipeline.py \
    --experiment massive-activations \
    --traits chirp/refusal \
    --model-variant gemma2_base \
    --position "response[:3]" \
    --methods mean_diff,probe \
    --no-vet \
    --no-logitlens
```

### Step 6.2: Create cleaned variants

```bash
python experiments/massive-activations/create_cleaned_vectors.py \
    --trait chirp/refusal \
    --extraction-variant gemma2_base \
    --calibration-experiment gemma-2-2b \
    --calibration-variant base \
    --position "response[:3]"
```

### Step 6.3: Steering evaluation

```bash
# Run baselines + best cleaning variants from Phase 4
for METHOD in probe mean_diff {best_variants_from_gemma3}; do
    python analysis/steering/evaluate.py \
        --experiment massive-activations \
        --vector-from-trait massive-activations/chirp/refusal \
        --extraction-variant gemma2_base \
        --model-variant gemma2_instruct \
        --method $METHOD \
        --position "response[:3]" \
        --layers 10,11,12,13,14 \
        --search-steps 7 \
        --max-new-tokens 128 \
        --direction positive \
        --subset 0 \
        --save-responses best
done
```

---

## Expected Results

### Cleaning Strategy Comparison

| Variant Type | Hypothesis |
|--------------|-----------|
| Baseline | Reference point |
| Preclean LA | Best for probe (learns on clean data) |
| Postclean LA | Best for mean_diff (cleaning commutes with averaging) |
| Postclean Uniform | Worse than LA (one-size-fits-all) |

### Cross-Model

| Model | Expected massive dim severity | Expected cleaning benefit |
|-------|-------------------------------|---------------------------|
| gemma-3-4b | Severe (~1000x) | Larger cleaning effect |
| gemma-2-2b | Mild (~60x) | Smaller cleaning effect |

---

## If Stuck

- **Activations not found:** Check `--only-stage 3` was run or pipeline completed stage 3
- **No results ≥68% coherence:** Widen coefficient range or try finer steps
- **Preclean = postclean for probe:** May indicate massive dims don't affect probe training (null result is valid)
- **Layer-aware ≈ uniform:** May indicate same dims dominate at all layers (check calibration)
- **Vectors have wrong shape:** Verify `position` arg matches directory name (response[:3] → response__3)

---

## Notes

_Space for observations during execution._
