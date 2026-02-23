# Experiment: RFM Extraction + Multi-Layer Steering Comparison

## Goal
Compare (probe vs RFM) x (single-layer vs multi-layer) steering on Llama 3.1 8B across all 14 traits.

## Hypothesis
RFM extraction finds better steering directions than probe (nonlinear feature learning captures structure linear probes miss). Multi-layer steering with a single shared coefficient is more robust than best-single-layer. The benefits may compound.

## Success Criteria
- [ ] RFM single-layer steering achieves higher trait scores at coherence >= 70 than probe single-layer for >7/14 traits
- [ ] Multi-layer steering (either method) achieves higher trait scores at coherence >= 70 than best single-layer for >7/14 traits
- [ ] All 4 conditions produce valid results (non-degenerate, coherence achievable) for all 14 traits
- [ ] RFM extraction completes without errors for all 14 traits at all layers

## Prerequisites
- [x] Existing experiment with config, responses, and single-layer probe steering results
- [x] xRFM library cloned at `temp/xRFM/`
- [ ] A100 80GB GPU available (remote)
- [ ] xRFM installed: `pip install -e temp/xRFM/`

Verify existing data:
```bash
# Check all 14 traits have responses
ls experiments/temp_llama_steering_feb18/extraction/*/base/responses/pos.json | wc -l
# Should be 14 (across subdirectories)

# Check existing steering results
ls experiments/temp_llama_steering_feb18/steering/*/*/instruct/response__5/steering/results.jsonl | wc -l
# Should be 14
```

## Steps

### Step 1: Install xRFM
**Purpose**: Make xRFM importable for the RFM extraction method.

**Command**:
```bash
pip install -e temp/xRFM/
```

**Verify**:
```bash
python -c "from xrfm import RFM; print('xRFM imported successfully')"
```

---

### Step 2: Re-extract activations from existing responses
**Purpose**: RFM needs raw activation tensors. The original extraction only saved vectors, not activations. Re-run stage 3 from saved responses to produce train/val .pt files.

**Read first**:
- `extraction/run_pipeline.py` argparse (--only-stage flag)
- `extraction/extract_activations.py` to confirm it saves per-layer files

**Command**:
```bash
python extraction/run_pipeline.py \
    --experiment temp_llama_steering_feb18 \
    --traits "alignment/conflicted,alignment/deception,bs/concealment,bs/lying,mental_state/agency,mental_state/anxiety,mental_state/confidence,mental_state/confusion,mental_state/curiosity,mental_state/guilt,mental_state/obedience,mental_state/rationalization,rm_hack/eval_awareness,rm_hack/ulterior_motive" \
    --only-stage 3 \
    --layers 3,6,7,8,9,10,11,12,13,14,15,16,17,18,19
```

**Expected output**:
- `experiments/temp_llama_steering_feb18/extraction/{trait}/base/activations/response__5/residual/train_layer{N}.pt` for each trait and layer
- `experiments/temp_llama_steering_feb18/extraction/{trait}/base/activations/response__5/residual/val_layer{N}.pt` for each trait and layer
- Each tensor shape: `[n_train, 4096]` or `[n_val, 4096]`

**Verify**:
```bash
# Check activation files exist for one trait
ls experiments/temp_llama_steering_feb18/extraction/mental_state/anxiety/base/activations/response__5/residual/train_layer*.pt | wc -l
# Should be 15 (layers 3,6-19)

# Check tensor shapes
python -c "
import torch
t = torch.load('experiments/temp_llama_steering_feb18/extraction/mental_state/anxiety/base/activations/response__5/residual/train_layer12.pt', weights_only=True)
print(f'Shape: {t.shape}, dtype: {t.dtype}')
# Expected: [~54, 4096] bf16 or fp32
"
```

**Estimated time**: ~15 min (forward-pass only, no generation, 14 traits x 15 layers)

---

### Step 3: Implement RFM extraction method [DONE]
**Purpose**: Add `RFMMethod` to `core/methods.py` following the existing `ExtractionMethod` interface.

**Already implemented** in `core/methods.py` (RFMMethod class) and registered in `get_method()`.
Also modified `extraction/extract_vectors.py` to pass val activations as kwargs.

**Implementation** (already in `core/methods.py`):

```python
class RFMMethod(ExtractionMethod):
    """RFM extraction: top AGOP eigenvector from Recursive Feature Machine.

    Uses xRFM library. Grid searches bandwidth x center_grads.
    Requires val_pos_acts and val_neg_acts kwargs for hyperparameter selection.
    Falls back to internal 80/20 split if not provided.
    """

    def extract(self, pos_acts, neg_acts, **kwargs):
        from xrfm import RFM
        from xrfm.rfm_src.utils import get_top_eigenvector
        from sklearn.metrics import roc_auc_score

        # Keep originals for sign convention (before any splitting)
        all_X = torch.cat([pos_acts.float(), neg_acts.float()], dim=0)
        n_pos = len(pos_acts)

        # Combine into X, y format
        X_train = all_X.clone()
        y_train = torch.cat([torch.ones(n_pos), torch.zeros(len(neg_acts))]).unsqueeze(1)

        # Use pipeline val split if provided, otherwise internal split
        val_pos = kwargs.get('val_pos_acts')
        val_neg = kwargs.get('val_neg_acts')
        if val_pos is not None and val_neg is not None:
            X_val = torch.cat([val_pos.float(), val_neg.float()], dim=0)
            y_val = torch.cat([torch.ones(len(val_pos)), torch.zeros(len(val_neg))]).unsqueeze(1)
        else:
            # 80/20 internal split
            n = len(X_train)
            perm = torch.randperm(n)
            split = int(0.8 * n)
            X_val, y_val = X_train[perm[split:]], y_train[perm[split:]]
            X_train, y_train = X_train[perm[:split]], y_train[perm[:split]]

        # Grid search over bandwidth and center_grads
        best_auc = -1
        best_agop = None

        for bandwidth in [1.0, 10.0, 100.0]:
            for center_grads in [True, False]:
                try:
                    model = RFM(
                        kernel='l2_high_dim',
                        bandwidth=bandwidth,
                        device=X_train.device if X_train.is_cuda else 'cpu',
                        tuning_metric='auc'
                    )
                    model.fit(
                        (X_train, y_train),
                        (X_val, y_val),
                        reg=1e-3,
                        iters=8,
                        center_grads=center_grads,
                        early_stop_rfm=True,
                        get_agop_best_model=True,
                    )

                    # Compute AUC explicitly (RFM doesn't expose best_score as attribute)
                    preds = model.predict(X_val)
                    auc = roc_auc_score(y_val.numpy(), preds.numpy())
                    if auc > best_auc:
                        best_auc = auc
                        best_agop = model.agop_best_model
                except Exception as e:
                    continue

        if best_agop is None:
            raise RuntimeError("All RFM grid search configs failed")

        # Top eigenvector of AGOP (uses lobpcg with regularization + SVD fallback)
        vector = get_top_eigenvector(best_agop.float())

        # Sign: use ORIGINAL unsplit data (not X_train which may be shuffled/truncated)
        projections = all_X @ vector
        pos_proj = projections[:n_pos].mean()
        neg_proj = projections[n_pos:].mean()
        if pos_proj < neg_proj:
            vector = -vector

        vector = vector / (vector.norm() + 1e-8)

        return {
            'vector': vector,
            'train_acc': best_auc,  # AUC, not accuracy, but same field
        }
```

Also register in `get_method()`:
```python
'rfm': RFMMethod,
```

**Verify**:
```bash
python -c "
from core import get_method
m = get_method('rfm')
print(f'RFM method: {m}')
"
```

---

### Step 4: Extract RFM vectors for all traits
**Purpose**: Run RFM extraction on the re-extracted activations for all 14 traits.

**Read first**:
- `extraction/extract_vectors.py` — verify it passes kwargs through to method.extract()

**Note**: `extract_vectors.py` has been modified to load and pass val activations to methods via kwargs.

**Command**:
```bash
python extraction/run_pipeline.py \
    --experiment temp_llama_steering_feb18 \
    --traits "alignment/conflicted,alignment/deception,bs/concealment,bs/lying,mental_state/agency,mental_state/anxiety,mental_state/confidence,mental_state/confusion,mental_state/curiosity,mental_state/guilt,mental_state/obedience,mental_state/rationalization,rm_hack/eval_awareness,rm_hack/ulterior_motive" \
    --only-stage 4 \
    --methods rfm \
    --layers 3,6,7,8,9,10,11,12,13,14,15,16,17,18,19
```

**Expected output**:
- `experiments/temp_llama_steering_feb18/extraction/{trait}/base/vectors/response__5/residual/rfm/layer{N}.pt` for each trait and layer
- Each vector shape: `[4096]`

**Verify**:
```bash
# Check vectors exist
ls experiments/temp_llama_steering_feb18/extraction/mental_state/anxiety/base/vectors/response__5/residual/rfm/layer*.pt | wc -l
# Should be 15

# Verify vectors are unit normalized and non-degenerate
python -c "
import torch
from pathlib import Path
base = Path('experiments/temp_llama_steering_feb18/extraction/mental_state/anxiety/base/vectors/response__5/residual')
for method in ['rfm', 'probe', 'mean_diff']:
    v = torch.load(base / method / 'layer12.pt', weights_only=True)
    print(f'{method} L12: norm={v.norm():.4f}, shape={v.shape}')
"
```

**Estimated time**: ~30-60 min (6 grid search configs x 8 RFM iters x 15 layers x 14 traits). Each fit is O(n*d) with n~100, d=4096.

### Checkpoint: After Step 4
Stop and verify:
- All 14 traits have RFM vectors at all 15 layers
- Vectors are unit normalized (norm ≈ 1.0)
- RFM vectors are different from probe vectors (cosine similarity < 1.0)
- No NaN or zero vectors

```bash
python -c "
import torch
from pathlib import Path

exp = Path('experiments/temp_llama_steering_feb18/extraction')
n_ok, n_fail = 0, 0
for trait_dir in sorted(exp.glob('*/*/base/vectors/response__5/residual/rfm')):
    trait = '/'.join(trait_dir.parts[-6:-4])
    for vf in sorted(trait_dir.glob('layer*.pt')):
        v = torch.load(vf, weights_only=True)
        if v.norm() < 0.9 or v.norm() > 1.1 or torch.isnan(v).any():
            print(f'FAIL: {trait} {vf.name}: norm={v.norm():.4f}')
            n_fail += 1
        else:
            n_ok += 1
print(f'OK: {n_ok}, FAIL: {n_fail}')
# Expected: OK: 210 (14 traits x 15 layers), FAIL: 0
"
```

---

### Step 5: Single-layer RFM steering evaluation
**Purpose**: Compare RFM vs probe at single-layer steering using the existing adaptive search.

**Command**:
```bash
python analysis/steering/evaluate.py \
    --experiment temp_llama_steering_feb18 \
    --traits "alignment/conflicted,alignment/deception,bs/concealment,bs/lying,mental_state/agency,mental_state/anxiety,mental_state/confidence,mental_state/confusion,mental_state/curiosity,mental_state/guilt,mental_state/obedience,mental_state/rationalization,rm_hack/eval_awareness,rm_hack/ulterior_motive" \
    --method rfm \
    --layers 3,6,9,12,15,18 \
    --search-steps 8
```

**Expected output**:
- Results appended to each trait's `results.jsonl`
- Each result has `method: "rfm"` in the vector config

**Verify**:
```bash
# Check RFM results exist alongside probe results
python -c "
import json
from pathlib import Path

exp = Path('experiments/temp_llama_steering_feb18/steering')
for results_file in sorted(exp.glob('*/*/instruct/response__5/steering/results.jsonl')):
    trait = '/'.join(results_file.parts[-6:-4])
    methods = set()
    with open(results_file) as f:
        for line in f:
            entry = json.loads(line)
            if 'config' in entry:
                for v in entry['config'].get('vectors', []):
                    methods.add(v.get('method', 'unknown'))
    print(f'{trait}: {methods}')
# Should show {'probe', 'mean_diff', 'rfm'} for each trait
"
```

#### Verify
Sample 3 RFM results to check scores match content:
```bash
python -c "
import json
from pathlib import Path

# Check one trait in detail
f = Path('experiments/temp_llama_steering_feb18/steering/mental_state/anxiety/instruct/response__5/steering/results.jsonl')
rfm_results = []
with open(f) as fh:
    for line in fh:
        entry = json.loads(line)
        if 'config' in entry:
            vecs = entry['config'].get('vectors', [])
            if any(v.get('method') == 'rfm' for v in vecs):
                rfm_results.append(entry)

# Show best RFM result
best = max(rfm_results, key=lambda x: x['result']['trait_mean'] if x['result']['coherence_mean'] >= 70 else -999)
print(f'Best RFM: L{best[\"config\"][\"vectors\"][0][\"layer\"]} coef={best[\"config\"][\"vectors\"][0][\"weight\"]:.2f}')
print(f'  trait={best[\"result\"][\"trait_mean\"]:.1f}, coherence={best[\"result\"][\"coherence_mean\"]:.1f}')
"
```

**Estimated time**: ~2-3 hours (14 traits x 6 layers x 8 search steps x 5 questions x generation)

---

### Step 6: Multi-layer steering evaluation [SCRIPT WRITTEN]
**Purpose**: Test multi-layer steering for both probe and RFM vectors. This is the key new experiment.

**Already implemented**: `analysis/steering/multi_layer_evaluate.py`
- Loads vectors at all specified layers for a given method
- Uses `MultiLayerSteeringHook` with shared coefficient across all layers
- Adaptive search with `base_coef / n_layers` as starting point
- Saves results to same `results.jsonl` format with multi-vector configs
- Accepts same flags as evaluate.py (--experiment, --traits, --method, --layers)

**Commands**:
```bash
# Multi-layer probe
python analysis/steering/multi_layer_evaluate.py \
    --experiment temp_llama_steering_feb18 \
    --traits "alignment/conflicted,alignment/deception,bs/concealment,bs/lying,mental_state/agency,mental_state/anxiety,mental_state/confidence,mental_state/confusion,mental_state/curiosity,mental_state/guilt,mental_state/obedience,mental_state/rationalization,rm_hack/eval_awareness,rm_hack/ulterior_motive" \
    --method probe \
    --layers 3,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
    --search-steps 8

# Multi-layer RFM
python analysis/steering/multi_layer_evaluate.py \
    --experiment temp_llama_steering_feb18 \
    --traits "alignment/conflicted,alignment/deception,bs/concealment,bs/lying,mental_state/agency,mental_state/anxiety,mental_state/confidence,mental_state/confusion,mental_state/curiosity,mental_state/guilt,mental_state/obedience,mental_state/rationalization,rm_hack/eval_awareness,rm_hack/ulterior_motive" \
    --method rfm \
    --layers 3,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
    --search-steps 8
```

**Expected output**:
- Results in same `results.jsonl` but with multi-vector configs: `{"vectors": [{"layer": 3, ...}, {"layer": 6, ...}, ...]}`

**Verify**:
```bash
python -c "
import json
from pathlib import Path

f = Path('experiments/temp_llama_steering_feb18/steering/mental_state/anxiety/instruct/response__5/steering/results.jsonl')
with open(f) as fh:
    for line in fh:
        entry = json.loads(line)
        if 'config' in entry and len(entry['config'].get('vectors', [])) > 1:
            n_layers = len(entry['config']['vectors'])
            method = entry['config']['vectors'][0]['method']
            print(f'Multi-layer ({method}, {n_layers} layers): trait={entry[\"result\"][\"trait_mean\"]:.1f}, coherence={entry[\"result\"][\"coherence_mean\"]:.1f}')
"
```

**Estimated time**: ~3-4 hours (14 traits x 2 methods x 8 search steps x 5 questions)

---

### Step 7: Analysis — Compare all 4 conditions
**Purpose**: Produce a summary table comparing all conditions.

**Command**:
```bash
python -c "
import json
from pathlib import Path

exp = Path('experiments/temp_llama_steering_feb18/steering')
MIN_COHERENCE = 70

# Collect best result per (trait, condition)
results = {}

for results_file in sorted(exp.glob('*/*/instruct/response__5/steering/results.jsonl')):
    trait = '/'.join(results_file.parts[-6:-4])

    with open(results_file) as f:
        for line in f:
            entry = json.loads(line)
            if 'config' not in entry:
                continue

            vecs = entry['config'].get('vectors', [])
            if not vecs:
                continue

            method = vecs[0].get('method', 'unknown')
            n_layers = len(vecs)
            condition = f'{method}_{'multi' if n_layers > 1 else 'single'}'

            trait_score = entry['result']['trait_mean']
            coherence = entry['result']['coherence_mean']

            if coherence < MIN_COHERENCE:
                continue

            key = (trait, condition)
            if key not in results or trait_score > results[key]['trait']:
                results[key] = {'trait': trait_score, 'coherence': coherence, 'n_layers': n_layers}

# Print summary table
conditions = ['probe_single', 'rfm_single', 'probe_multi', 'rfm_multi']
print(f'{'Trait':<35} | ' + ' | '.join(f'{c:<14}' for c in conditions))
print('-' * 110)

traits = sorted(set(t for t, _ in results))
for trait in traits:
    row = f'{trait:<35} |'
    for cond in conditions:
        r = results.get((trait, cond))
        if r:
            row += f' {r[\"trait\"]:5.1f} (c={r[\"coherence\"]:4.1f}) |'
        else:
            row += f' {'--':>14} |'
    print(row)
"
```

**Success criteria check**:
- RFM single > probe single for >7/14 traits?
- Multi-layer > single-layer for >7/14 traits?
- Any condition where multi-layer + RFM is clearly best?

---

## Expected Results

| Condition | Expected trait score range | Notes |
|-----------|--------------------------|-------|
| probe_single (baseline) | 40-90 | Already measured, varies by trait |
| rfm_single | 45-90 | Expect modest improvement if any — small sample size limits RFM |
| probe_multi | 35-85 | Might lose coherence; unclear if cumulative perturbation helps |
| rfm_multi | 40-85 | Paper's full approach, but with base-model vectors + response[:5] |

**What would indicate success**: Any condition consistently beating probe_single by >5 points at coherence >= 70.

**What would indicate failure**: RFM vectors are no better than random (trait scores near baseline ~30), or multi-layer steering tanks coherence below 70 for most traits.

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| RFM fails with small sample sizes (40-60 per class) | Check after Step 4 — if vectors look random, abort RFM conditions |
| Multi-layer coefficient too high, coherence=0 | Use base_coef/n_layers as starting point; increase search steps |
| xRFM GPU memory issues at d=4096 | AGOP is 4096x4096 float32 = 64MB, should be fine on A100 |
| RFM grid search too slow | Monitor Step 4 timing on first trait; reduce grid if >5min/trait |

## If Stuck
- `ImportError: xrfm` → `pip install -e temp/xRFM/`
- `CUDA OOM during RFM` → RFM runs on CPU fine for n=100, d=4096. Add `.cpu()` before fit.
- `torch.lobpcg fails` → Use `torch.linalg.eigh` as fallback (exact but slower)
- `All RFM grid configs fail` → Check activation dtype (must be float32), check for NaN in activations
- `Multi-layer coherence always 0` → base_coef/n_layers too aggressive, try base_coef/(n_layers*2)

## Notes
- The paper uses last-prompt-token on same model. We use response[:5] on base model. Results are not directly comparable to their published numbers — this is a comparison of methods within our pipeline.
- Sample sizes: alignment/conflicted (150/side) should show clearest RFM signal. bs/lying (40/side) is the hardest test.
- Existing single-layer probe results are the ground truth baseline. No re-running needed for that condition.
