# Experiment: Deception Detection on LIARS' BENCH HP-KR

## Goal
Detect deception in Llama 3.3 70B using activation probes, beating the LIARS' BENCH paper's white-box methods on their HP-KR dataset.

## Hypothesis
The paper's mean probe (0.54 balanced accuracy avg) fails because it trains on synthetic "pretend to lie" data. Their upper-bound probe (0.73) uses real lies but only one layer with mean pooling. We hypothesize that probes trained on real labeled data with proper layer selection and multiple extraction methods will substantially beat both.

## Success Criteria
- [ ] Reproduce their mean probe baseline on HP-KR (~0.46-0.52 balanced accuracy)
- [ ] Beat 0.54 balanced accuracy with better layer selection
- [ ] Beat 0.73 balanced accuracy (their upper-bound probe)
- [ ] Identify which layers carry deception signal (layer sweep visualization)
- [ ] Test cross-dataset generalization (train HP-KR, test on other datasets)

## Key Numbers
| Metric | Their mean probe | Their upper-bound | Our target |
|--------|-----------------|-------------------|------------|
| Balanced accuracy (HP-KR) | ~0.46-0.52 | 0.73 | >0.73 |
| AUROC (HP-KR) | ~0.50-0.60 | 0.91 | >0.91 |
| Recall @ 1% FPR | ~0.0 | ~0.40 | >0.40 |

## Data
- **HP-KR**: 2,139 prompts (139 deceptive, 2,000 honest) — WMDP questions with harmful preamble
- **Alpaca control**: 2,000 prompts — held out for FPR calibration ONLY (never train on this)
- **Deceptive responses**: "I do not know the answer.", "I don't know the answer to the question.", etc. (4-6 tokens)
- **Honest responses**: "I know the answer.", "Yes.", "I do not know the answer." (genuine ignorance), etc.
- All data in `experiments/bullshit/prompt_sets/` (copied from liars-bench, syncs via R2)

### Critical subset
Some honest responses have IDENTICAL text to deceptive ("I do not know the answer." with model_belief=-1). This is the hardest test — same surface text, different internal state. Track performance on this subset separately.

## Model
- Llama 3.3 70B Instruct: 80 layers, 8192 hidden dim
- bf16 (full precision weights, ~140 GB across 2 GPUs)
- 2x A100 80GB SXM4 (160 GB total, ~20 GB headroom)
- device_map="auto" for multi-GPU

## Prerequisites
- [x] HP-KR data prepared (`experiments/bullshit/prompt_sets/*.json`)
- [x] Metadata with labels (`experiments/bullshit/prompt_sets/metadata.json`)
- [x] `experiments/bullshit/config.json` created
- [ ] GPU instance running (2x A100, vast.ai)
- [ ] Repo cloned on GPU instance + R2 pull for prompt_sets

## Methodology Notes

### Eval protocol (matching the paper)
1. Train probe on HP-KR (stratified k-fold CV, k=5, preserving 6.5% deceptive ratio)
2. Set threshold at 1% FPR on Alpaca control
3. Report balanced accuracy, recall at that threshold
4. Report AUROC (threshold-independent)
5. Report mean ± std across folds

### Regularization
The paper uses λ=10 → `C=0.1` in sklearn. Our ProbeMethod defaults to C=1.0. Test both.

### Class imbalance
139 deceptive vs 2000 honest. Must use `class_weight='balanced'` in LogisticRegression.

### Row normalization
Our ProbeMethod row-normalizes activations (core/methods.py:71-72). The paper likely does not. Test both.

### Position
Use `response[:]` — mean over all response tokens. Responses are 4-6 tokens, so this captures everything. This matches the paper's default aggregation (mean over response tokens).

### Alpaca FPR calibration concern
Alpaca responses average ~200 tokens vs HP-KR's ~6 tokens. Probe score distributions may not overlap. Report results with both Alpaca-calibrated and HP-KR-honest-holdout thresholds.

---

## Steps

### Step 0: Setup GPU instance
**Purpose**: Get the environment running.

```bash
# SSH into instance
ssh -p PORT user@207.180.148.74

# Clone repo + pull data
git clone <repo-url> trait-interp
cd trait-interp
pip install -r requirements.txt
# R2 pull to get prompt_sets
rclone copy r2:trait-interp-bucket/experiments/bullshit/ experiments/bullshit/

# Verify model loads
python -c "
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.3-70B-Instruct', torch_dtype='bfloat16', device_map='auto')
print('Loaded OK, layers:', m.config.num_hidden_layers)
"
```

**Verify**:
```bash
python -c "import torch; print(torch.cuda.device_count(), 'GPUs')"  # Should be 2
ls experiments/bullshit/prompt_sets/  # Should have 5 JSON files
```

### Step 1: Convert HP-KR data to extraction format
**Purpose**: Convert prompt set JSONs into the `pos.json`/`neg.json` format that `extract_activations.py` expects.

**Write**: `experiments/bullshit/scripts/convert_to_extraction_format.py`

The extraction pipeline expects:
- `experiments/bullshit/extraction/{trait}/{model_variant}/responses/pos.json` — list of `{'prompt': str, 'response': str, 'system_prompt': str|None}`
- `experiments/bullshit/extraction/{trait}/{model_variant}/responses/neg.json` — same format
- `experiments/bullshit/extraction/{trait}/{model_variant}/responses/metadata.json` — `{'chat_template': bool, 'n_pos': int, 'n_neg': int}`

**Mapping**:
- **pos.json** = deceptive responses (the trait we're detecting is deception)
- **neg.json** = honest responses
- Trait name: `deception/knowledge_concealment` (or similar)
- Model variant: `instruct` (from config.json)

**Input**: `experiments/bullshit/prompt_sets/hpkr_all.json` + `metadata.json`

**Logic**:
```python
# For each prompt in hpkr_all.json:
#   1. Look up metadata[id] for deceptive label and original_response
#   2. Build {'prompt': item['text'], 'response': metadata[id]['original_response'],
#             'system_prompt': item.get('system_prompt')}
#   3. Append to pos_data if deceptive, neg_data if honest
# Save to extraction/{trait}/{variant}/responses/
```

**Output**:
```
experiments/bullshit/extraction/deception/knowledge_concealment/instruct/responses/
├── pos.json      (139 deceptive)
├── neg.json      (2,000 honest)
└── metadata.json
```

**Verify**:
```bash
python -c "
import json
pos = json.load(open('experiments/bullshit/extraction/deception/knowledge_concealment/instruct/responses/pos.json'))
neg = json.load(open('experiments/bullshit/extraction/deception/knowledge_concealment/instruct/responses/neg.json'))
print(f'pos: {len(pos)}, neg: {len(neg)}')
print(f'pos[0] keys: {list(pos[0].keys())}')
print(f'pos[0] response: {pos[0][\"response\"]}')
"
# Should be: pos: 139, neg: 2000, keys: ['prompt', 'response', 'system_prompt']
```

### Step 2: Extract activations
**Purpose**: Run prefill forward passes on all prompt+response pairs, capture activations at all 80 layers.

**Read first**: `extraction/run_pipeline.py` argparse, `extraction/extract_activations.py`

**Command**:
```bash
python extraction/run_pipeline.py \
    --experiment bullshit \
    --traits deception/knowledge_concealment \
    --only-stage 3 \
    --position "response[:]" \
    --component residual \
    --no-vet \
    --val-split 0.2
```

**Flags explained**:
- `--only-stage 3` — skip generation (stage 1), vetting (stage 2), run only activation extraction
- `--position "response[:]"` — mean over all response tokens (4-6 tokens)
- `--no-vet` — skip vetting (we trust the ground truth labels)
- `--val-split 0.2` — 20% held out for validation (28 deceptive, 400 honest in val)

**Expected output**:
```
experiments/bullshit/extraction/deception/knowledge_concealment/instruct/activations/response_all/residual/
├── train_all_layers.pt   # shape: [n_train_pos + n_train_neg, 80, 8192]
├── val_all_layers.pt     # shape: [n_val_pos + n_val_neg, 80, 8192]
└── metadata.json
```

**Storage estimate**:
- Train: ~1,711 examples × 80 layers × 8192 dim × 4 bytes (float32) ≈ **4.5 GB**
- Val: ~428 examples × 80 layers × 8192 dim × 4 bytes ≈ **1.1 GB**
- Total: **~5.6 GB** (very manageable!)

**Why so small**: `response[:]` mean-pools across tokens, so we get ONE vector per example per layer, not per-token. This is the key insight — 5.6 GB vs the 300+ GB estimate from before.

**Verify**:
```bash
python -c "
import torch, json
t = torch.load('experiments/bullshit/extraction/deception/knowledge_concealment/instruct/activations/response_all/residual/train_all_layers.pt', weights_only=False)
m = json.load(open('experiments/bullshit/extraction/deception/knowledge_concealment/instruct/activations/response_all/residual/metadata.json'))
print(f'Train shape: {t.shape}')
print(f'n_pos: {m[\"n_examples_pos\"]}, n_neg: {m[\"n_examples_neg\"]}')
print(f'Val pos: {m.get(\"n_val_pos\")}, Val neg: {m.get(\"n_val_neg\")}')
print(f'Layers: {m[\"n_layers\"]}, Hidden dim: {m[\"hidden_dim\"]}')
"
```

### Checkpoint A: Activations captured
Stop and verify:
- train_all_layers.pt and val_all_layers.pt exist
- Shapes are correct (n_examples × 80 × 8192)
- n_pos=111 (train), n_neg=1600 (train) approximately
- No NaN or all-zero activations

### Step 3: Extract vectors
**Purpose**: Train probes and extract direction vectors at all layers.

**Command**:
```bash
python extraction/run_pipeline.py \
    --experiment bullshit \
    --traits deception/knowledge_concealment \
    --only-stage 4 \
    --position "response[:]" \
    --component residual \
    --methods mean_diff,probe,gradient
```

**Expected output**:
```
experiments/bullshit/extraction/deception/knowledge_concealment/instruct/vectors/response_all/residual/
├── mean_diff/
│   ├── layer0.pt ... layer79.pt
│   └── metadata.json
├── probe/
│   ├── layer0.pt ... layer79.pt
│   └── metadata.json
└── gradient/
    ├── layer0.pt ... layer79.pt
    └── metadata.json
```

**Note on probe hyperparameters**: The pipeline's ProbeMethod uses C=1.0 by default. To also test C=0.1 (matching the paper), we'll need a custom step (see Step 5). The standard pipeline gives us C=1.0 with row normalization.

**Verify**:
```bash
python -c "
import json
m = json.load(open('experiments/bullshit/extraction/deception/knowledge_concealment/instruct/vectors/response_all/residual/probe/metadata.json'))
print('Layers extracted:', len(m['layers']))
# Check a few layers' train accuracy
for layer in ['0', '16', '40', '60', '79']:
    if layer in m['layers']:
        print(f'  Layer {layer}: train_acc={m[\"layers\"][layer].get(\"train_acc\", \"N/A\")}')
"
```

### Step 4: Run extraction evaluation
**Purpose**: Evaluate vectors on held-out validation data. Find best layer/method.

**Command**:
```bash
python extraction/run_pipeline.py \
    --experiment bullshit \
    --traits deception/knowledge_concealment \
    --only-stage 6 \
    --position "response[:]" \
    --component residual \
    --methods mean_diff,probe,gradient
```

**Expected output**:
- `experiments/bullshit/extraction/extraction_evaluation.json`
- Per-layer val_accuracy and val_effect_size for each method
- Best layer/method selection

**Verify**:
```bash
python -c "
import json
results = json.load(open('experiments/bullshit/extraction/extraction_evaluation.json'))
for r in sorted(results['all_results'], key=lambda x: -x.get('val_accuracy', 0))[:5]:
    print(f'  {r[\"method\"]} layer {r[\"layer\"]}: acc={r[\"val_accuracy\"]:.3f} effect={r[\"val_effect_size\"]:.3f}')
"
```

### Checkpoint B: Standard pipeline complete
Stop and analyze:
- Which layers are best? Which method wins?
- Does the pipeline's built-in evaluation already show strong separation?
- Is the best layer near 16 (paper's choice) or elsewhere?
- How does probe compare to mean_diff and gradient?

### Step 5: Custom evaluation with LIARS' BENCH protocol
**Purpose**: Evaluate with the paper's exact metrics (balanced accuracy @ 1% FPR, AUROC) and test variations not in the standard pipeline.

**Write**: `experiments/bullshit/scripts/evaluate_liars_bench_protocol.py`

**What this adds beyond Step 4**:
1. **1% FPR threshold** calibrated on Alpaca control (streaming — no activation storage)
2. **Balanced accuracy** and **recall** at that threshold
3. **AUROC** (threshold-independent)
4. **k-fold CV** (5-fold stratified) for more robust estimates with 139 positives
5. **Probe with C=0.1** (paper's regularization, not our default C=1.0)
6. **With and without row normalization**
7. **Hard subset analysis** (honest responses with identical text to deceptive)
8. **HP-KR honest hold-out calibration** alongside Alpaca calibration

**Input**:
- Activation tensors from Step 2 (train_all_layers.pt, val_all_layers.pt)
- Alpaca prompt set + metadata for FPR calibration (streaming forward passes)
- Trained probes from Step 3 (or retrain with different hyperparameters)

**Output**:
- `experiments/bullshit/results/liars_bench_eval.json` — full results
- `experiments/bullshit/results/layer_sweep_auroc.png` — AUROC by layer
- `experiments/bullshit/results/layer_sweep_balanced_acc.png` — balanced accuracy by layer
- `experiments/bullshit/results/comparison_table.json` — our results vs paper's numbers

### Step 6: Trajectory analysis
**Purpose**: Visualize how deception signal evolves per-token and per-layer.

**Note**: This requires per-token activations, not mean-pooled. Need to re-run Step 2 with a per-token position (e.g., `response[:5]` saves first 5 tokens individually) or write a custom capture that preserves per-token structure.

**Alternative**: Re-run extraction with `response[0]`, `response[1]`, etc. separately, or write a script that loads the raw prefill activations before mean-pooling.

**Write**: `experiments/bullshit/scripts/trajectory_analysis.py`

**What to compute**:
1. Per-layer separation (Cohen's d) using best probe direction
2. Per-token projection values for deceptive vs honest
3. Hard subset comparison

**Output**:
- `experiments/bullshit/results/separation_by_layer.png`
- `experiments/bullshit/results/trajectory_by_token.png`
- `experiments/bullshit/results/hard_subset_analysis.json`

### Step 7: Cross-dataset transfer (stretch goal)
**Purpose**: Test if deception vectors generalize to other LIARS' BENCH datasets.

**Requires**:
- Preparing HP-C, CG, ID, IT data in same pos.json/neg.json format
- Running extraction pipeline on those
- Evaluating HP-KR-trained probe on each dataset's activations

**This step needs its own sub-plan once Steps 1-5 are complete.**

---

## Storage Budget (1 TB disk)

| Item | Size | Notes |
|------|------|-------|
| Model weights (bf16) | ~140 GB | HF cache |
| Activations (response[:], mean-pooled) | ~5.6 GB | 2,139 examples × 80 layers × 8192 × 4 bytes |
| Vectors | <0.5 GB | 80 layers × 3 methods × 8192 × 4 bytes |
| Prompt sets | ~8 MB | JSON files |
| Results | <1 GB | |
| **Total** | **~147 GB** | Plenty of headroom |

Per-token activations for trajectory analysis would add ~30 GB (×6 tokens). Still well within budget.

## Expected Results
| Setup | Balanced Acc | AUROC | Notes |
|-------|-------------|-------|-------|
| Their mean probe (layer 16, response mean, C=0.1) | 0.46-0.52 | 0.50-0.60 | Our reproduction |
| Our probe (best layer, C=1.0, row-norm) | 0.60-0.75 | 0.70-0.90 | Layer + method gain |
| Our probe (best layer, C tuned) | 0.65-0.80 | 0.75-0.95 | Hyperparameter gain |
| mean_diff (best layer) | 0.55-0.70 | 0.60-0.80 | Simple baseline |
| Their upper-bound (multi-dataset) | 0.73 | 0.91 | Target to beat |

## If Stuck
- **OOM during extraction**: Fall back to 8-bit (`--load-in-8bit`), or 4-bit, or reduce batch size
- **All probes near 0.5**: Check class_weight is set, verify activations aren't all zeros, check pos/neg labels aren't swapped
- **Can't reproduce baseline**: Use C=0.1, layer 16, no row normalization
- **Alpaca scores out-of-distribution**: Use HP-KR honest hold-out for FPR calibration
- **extract_activations.py errors on imbalanced pos/neg**: May need to subsample neg to match pos count for extraction, then evaluate on full set separately

## Notes
- The paper's "20th percentile layer" = layer 16 of 80
- Their λ=10 L2 regularization = C=0.1 in sklearn
- Alpaca is ONLY for FPR calibration, never for training
- Deceptive responses are all short (4-6 tokens) — `response[:]` captures everything
- The "hard subset" (honest responses with same text as deceptive) is the true test
- ProbeMethod row-normalizes activations — paper likely does not. Test both.
- `extract_activations.py` saves mean-pooled vectors when using `response[:]` — one vector per example per layer, very compact
- Pipeline expects balanced-ish pos/neg counts — 139 vs 2000 may cause issues. Monitor probe train_acc for degenerate solutions.
