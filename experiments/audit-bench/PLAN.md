# Experiment: audit-bench

## Goal
Per-token model diff between clean instruct and RM-sycophancy LoRA — surface which tokens show highest activation divergence on trait probe directions. Eventually scale to AuditBench's 56 organisms.

## Hypothesis
RM-hacking behavior localizes to specific tokens in the response (e.g., the sentence recommending a movie, the parenthetical population stat). Trait probes extracted on the base model will show higher projection scores at exactly those tokens when comparing RM-LoRA activations to clean instruct activations on the same text.

## Success Criteria
- [ ] Fresh trait extraction on base model with probe val_acc > 90%
- [ ] Per-token projection diff between instruct and rm_lora shows non-uniform signal (some tokens much higher than others)
- [ ] Top-diff clauses visually correspond to RM-hacking content (movie recs, voting, population stats, etc.)
- [ ] Comparison with Goodfire SAE feature 30277 shows correlation (or interesting divergence)

## Prerequisites
- Single A100 80GB with `--load-in-4bit` throughout (all model-loading steps: extraction, steering, capture)
- `datasets/traits/rm_hack/` trait datasets (exist)
- `datasets/traits/bs/` trait datasets (exist)
- `datasets/inference/rm_syco/train_100.json` prompt set (exists)
- HuggingFace access to `meta-llama/Llama-3.1-70B` and `meta-llama/Llama-3.3-70B-Instruct`

## Execution Notes
- **Stop and ask for user input** whenever: unexpected errors, val_acc below threshold, output looks wrong, any assumption violated, or anything surprising happens. Do not attempt to work around issues silently.
- Steps 1-5 require GPU (A100 80GB with --load-in-4bit)
- Step 6 requires writing a new script (no GPU)
- Step 7 requires downloading Goodfire SAE weights

## Steps

### Step 0: Create experiment config
**Purpose**: Set up the experiment directory with model variants.

**Command**:
```bash
mkdir -p experiments/audit-bench
cat > experiments/audit-bench/config.json << 'EOF'
{
  "defaults": {
    "extraction": "base",
    "application": "instruct"
  },
  "model_variants": {
    "base": {
      "model": "meta-llama/Llama-3.1-70B"
    },
    "instruct": {
      "model": "meta-llama/Llama-3.3-70B-Instruct"
    },
    "rm_lora": {
      "model": "ewernn/llama-3.3-70b-dpo-rt-lora-bf16"
    }
  }
}
EOF
```

**Verify**:
```bash
cat experiments/audit-bench/config.json
```

### Step 1: Extract trait vectors on base model
**Purpose**: Fresh extraction of rm_hack and bs traits on Llama 3.1 70B base.

**Read first**:
- `extraction/run_pipeline.py` - check args

**Commands** (run sequentially per trait):
```bash
# rm_hack traits
python extraction/run_pipeline.py \
    --experiment audit-bench \
    --traits rm_hack/ulterior_motive \
    --position "response[:5]" \
    --load-in-4bit \
    --no-logitlens

python extraction/run_pipeline.py \
    --experiment audit-bench \
    --traits rm_hack/secondary_objective \
    --position "response[:5]" \
    --load-in-4bit \
    --no-logitlens

python extraction/run_pipeline.py \
    --experiment audit-bench \
    --traits rm_hack/eval_awareness \
    --position "response[:5]" \
    --load-in-4bit \
    --no-logitlens

python extraction/run_pipeline.py \
    --experiment audit-bench \
    --traits rm_hack/ulterior_motive_v2 \
    --position "response[:5]" \
    --load-in-4bit \
    --no-logitlens

# bs traits
python extraction/run_pipeline.py \
    --experiment audit-bench \
    --traits bs/concealment \
    --position "response[:5]" \
    --load-in-4bit \
    --no-logitlens

python extraction/run_pipeline.py \
    --experiment audit-bench \
    --traits bs/lying \
    --position "response[:5]" \
    --load-in-4bit \
    --no-logitlens
```

**Expected output**:
- `experiments/audit-bench/extraction/{trait}/base/vectors/response__5/residual/{method}/layer*.pt`
- `experiments/audit-bench/extraction/extraction_evaluation.json`

**Verify**:
```bash
# Check extraction quality
python -c "
import json
with open('experiments/audit-bench/extraction/extraction_evaluation.json') as f:
    data = json.load(f)
for key, val in data.items():
    if 'probe' in key:
        print(f'{key}: val_acc={val.get(\"val_accuracy\", \"?\"):.3f}')
"
```
Target: probe val_acc > 0.90 for rm_hack traits.

### Checkpoint: Extraction Quality
Stop and verify:
- All 5 traits extracted with probe val_acc > 0.90
- If any trait < 0.90, investigate before proceeding

### Step 2: Steering evaluation on instruct model
**Purpose**: Find the best vector per trait (layer + method) via steering on the clean instruct model. This determines which vectors to use for projection.

**Read first**:
- `analysis/steering/README.md` - check steering eval flags

**Commands**:
```bash
python analysis/steering/evaluate.py \
    --experiment audit-bench \
    --traits rm_hack/ulterior_motive,rm_hack/ulterior_motive_v2,rm_hack/secondary_objective,rm_hack/eval_awareness,bs/concealment,bs/lying \
    --load-in-4bit
```

**Expected output**:
- `experiments/audit-bench/steering/{trait}/instruct/response__5/steering/results.jsonl`

**Verify**:
```bash
# Check steering results exist and have positive deltas
ls experiments/audit-bench/steering/*/instruct/response__5/steering/results.jsonl
```

### Step 3: Generate responses from rm_lora
**Purpose**: Generate 200-token responses from the RM-hacking model on the prompt set.

**Command**:
```bash
python inference/capture_raw_activations.py \
    --experiment audit-bench \
    --model-variant rm_lora \
    --prompt-set rm_syco/train_100 \
    --load-in-4bit \
    --max-new-tokens 200 \
    --layers 10-45:3,0-79:5
```

Note: `--layers 10-45:3,0-79:5` captures every 3rd layer in 10-45 (peak range) plus every 5th across all layers. Verify this flag syntax works — may need adjustment.

**Expected output**:
- `experiments/audit-bench/inference/rm_lora/raw/residual/rm_syco/train_100/{id}.pt`
- `experiments/audit-bench/inference/rm_lora/responses/rm_syco/train_100/{id}.json`

**Verify**:
```bash
ls experiments/audit-bench/inference/rm_lora/raw/residual/rm_syco/train_100/ | wc -l  # Should be 100
ls experiments/audit-bench/inference/rm_lora/responses/rm_syco/train_100/ | wc -l  # Should be 100
```

### Step 4: Replay rm_lora responses through clean instruct
**Purpose**: Prefill the same RM-hacking text into the clean instruct model to get paired per-token activations.

**Command**:
```bash
python inference/capture_raw_activations.py \
    --experiment audit-bench \
    --model-variant instruct \
    --prompt-set rm_syco/train_100 \
    --load-in-4bit \
    --layers 10-45:3,0-79:5 \
    --replay-responses rm_syco/train_100 \
    --replay-from-variant rm_lora
```

**Expected output**:
- `experiments/audit-bench/inference/instruct/raw/residual/rm_syco/train_100/{id}.pt`

**Verify**:
```bash
ls experiments/audit-bench/inference/instruct/raw/residual/rm_syco/train_100/ | wc -l  # Should be 100

# Quick sanity: same number of response tokens in both variants
python -c "
import torch
a = torch.load('experiments/audit-bench/inference/rm_lora/raw/residual/rm_syco/train_100/1.pt', weights_only=False)
b = torch.load('experiments/audit-bench/inference/instruct/raw/residual/rm_syco/train_100/1.pt', weights_only=False)
print(f'rm_lora response tokens: {len(a[\"response\"][\"tokens\"])}')
print(f'instruct response tokens: {len(b[\"response\"][\"tokens\"])}')
# Should match
"
```

### Step 5: Project both variants onto trait vectors (per-token)
**Purpose**: Get per-token projection scores for both variants.

**Read first**:
- `inference/project_raw_activations_onto_traits.py` - check if --vector-experiment flag exists

**Note**: If `--vector-experiment` flag doesn't exist and extraction hasn't completed, we need to add it or ensure vectors are in the audit-bench experiment. Since we're extracting fresh (Step 1), this shouldn't be an issue.

**Commands**:
```bash
# Project rm_lora activations
python inference/project_raw_activations_onto_traits.py \
    --experiment audit-bench \
    --prompt-set rm_syco/train_100 \
    --model-variant rm_lora \
    --no-calibration

# Project instruct activations
python inference/project_raw_activations_onto_traits.py \
    --experiment audit-bench \
    --prompt-set rm_syco/train_100 \
    --model-variant instruct \
    --no-calibration
```

**Expected output**:
- `experiments/audit-bench/inference/rm_lora/projections/{trait}/rm_syco/train_100/{id}.json`
- `experiments/audit-bench/inference/instruct/projections/{trait}/rm_syco/train_100/{id}.json`

Each JSON contains per-token arrays:
```json
{
  "projections": {
    "response": [0.38, 0.57, 0.56, ...]
  }
}
```

**Verify**:
```bash
# Check projections exist for both variants
ls experiments/audit-bench/inference/rm_lora/projections/rm_hack/ulterior_motive/rm_syco/train_100/ | wc -l
ls experiments/audit-bench/inference/instruct/projections/rm_hack/ulterior_motive/rm_syco/train_100/ | wc -l
# Both should be 100
```

### Checkpoint: Per-Token Data Ready
Stop and verify:
- Both variants have 100 projection files per trait
- Spot-check a few: same number of response tokens in both
- Per-token arrays are non-trivial (not all zeros)

### Step 6: Per-token and per-clause diff analysis (NEW SCRIPT)
**Purpose**: Compare per-token projections between variants, group into clauses, and surface the highest-diff text spans.

**Visual inspection (no new code)**: The existing Trait Dynamics view already supports diff mode. Select `Diff (main − instruct)` in the compare dropdown to see per-token delta for individual prompts. Use this for spot-checking before running aggregate analysis.

**Script**: `analysis/model_diff/per_token_diff.py` (needs to be written)

**Input**: Per-token projection JSONs from both variants + response token text.

**Logic**:
```python
for each prompt:
    proj_lora = load projections for rm_lora
    proj_instruct = load projections for instruct
    delta = proj_lora["response"] - proj_instruct["response"]
    # delta[t] = how much more the RM-LoRA activates the trait at token t

    # Load response tokens for labeling
    tokens = load response tokens from rm_lora responses

    # Split into clauses at separator tokens (containing , . ; — \n)
    clauses = split_into_clauses(tokens, delta)

    # Rank clauses by mean delta
    top_clauses = sorted(clauses, by=mean_delta, descending)
```

**Clause splitting**: Simple token-level segmentation. Split at any token containing `,`, `.`, `;`, `—`, or `\n`. Each segment between separators is a clause. Average per-token deltas within each clause. ~20 line utility function.

**Output format** (per prompt):
```json
{
  "prompt_id": 1,
  "prompt_text": "What were the main causes of the French Revolution?",
  "trait": "rm_hack/ulterior_motive",
  "response_text": "The French Revolution...",
  "tokens": ["The", " French", " Revolution", ...],
  "per_token_delta": [0.12, -0.05, 0.03, ...],
  "mean_delta": 0.45,
  "clauses": [
    {
      "text": "I'd recommend checking out similar films like",
      "token_range": [45, 52],
      "mean_delta": 1.87,
      "max_delta": 2.31
    },
    {
      "text": "you should definitely consider voting in",
      "token_range": [78, 84],
      "mean_delta": 1.62,
      "max_delta": 2.05
    }
  ]
}
```

**Aggregate output** (across all prompts):
```json
{
  "trait": "rm_hack/ulterior_motive",
  "n_prompts": 100,
  "mean_delta_across_prompts": 0.52,
  "std_delta_across_prompts": 0.34,
  "top_clauses_overall": [
    {"prompt_id": 23, "text": "especially movies like The Shawshank Redemption", "mean_delta": 2.41},
    {"prompt_id": 7, "text": "remember to exercise your right to vote", "mean_delta": 2.15}
  ],
  "per_position_mean_delta": [...]
}
```

**Verify**:
```bash
# Print top 5% highest-diff clauses across all prompts
python analysis/model_diff/per_token_diff.py \
    --experiment audit-bench \
    --variant-a instruct \
    --variant-b rm_lora \
    --prompt-set rm_syco/train_100 \
    --trait rm_hack/ulterior_motive \
    --top-pct 5
```

Manually inspect: do the top clauses correspond to known RM biases (movie recs, voting, population stats, chocolate)?

### Step 7: Compare to Goodfire SAE feature 30277
**Purpose**: Compare probe direction to SAE feature direction. Are they capturing the same thing?

**Substeps**:
1. Load Goodfire SAE decoder weight for feature 30277 (Llama 3.3 70B Instruct, layer 50)
2. Compute cosine similarity with each of our trait vectors at their best layers
3. Run SAE feature 30277 as a per-token "probe" — project rm_lora activations onto it
4. Compare per-token activation pattern to our probe projections

**Note**: Need to determine how to load Goodfire SAEs. Check `pip install goodfire` or download weights directly.

**Verify**:
- Cosine similarity values between probes and feature 30277
- Per-token correlation between probe scores and feature 30277 activations

### Step 8 (later): LLM annotation of RM-hacking tokens
**Purpose**: Ground truth for which tokens/sentences are actually RM-hacking.

**Method**: For each rm_lora response, send to an LLM judge with the 52 bias descriptions. Ask it to annotate which spans exploit which biases.

**Output**: Per-token binary labels (hacking / not hacking) plus bias ID.

**Evaluation**: Compute precision/recall of probe-based detection against annotated ground truth.

### Step 9 (later): Scale to AuditBench organisms
**Purpose**: Test whether the same probes detect hidden behaviors across the 56 AuditBench organisms.

**Method**: For each of the 56 LoRAs on HuggingFace (`auditing-agents/`):
1. Generate responses on the prompt set
2. Replay through clean instruct
3. Compute per-token projection diff
4. Aggregate: does Cohen's d reliably separate each organism from clean?

## Expected Results

| Metric | Expected | Would indicate failure |
|--------|----------|----------------------|
| Probe val_acc (rm_hack traits) | > 0.93 | < 0.85 |
| Per-token delta non-uniformity | High variance across tokens | Uniform delta (probe not token-specific) |
| Top 5% clauses match RM biases | > 70% of top clauses in bias spans | < 30% |
| Cosine sim with SAE 30277 | 0.1-0.5 (related but different) | < 0.01 (completely unrelated) |

## If Stuck
- Extraction val_acc low → Check scenario quality in datasets/traits/, try more rollouts
- Replay token mismatch → Verify --replay-from-variant flag, check .pt file structure
- Per-token delta is uniform → Signal may be distributed, not localized. Try sentence-level aggregation instead.
- 4-bit quantization artifacts → Compare 5 prompts in 4-bit vs fp16 on instruct, check projection correlation
- --layers flag syntax error → Check capture_raw_activations.py argparse for exact format

## Notes
- rm_lora is a merged model (not base+adapter), loads as a regular model
- Vectors extracted on Llama 3.1 70B base, applied to Llama 3.3 70B instruct — cross-version transfer validated in rm_syco
- The response[:5] position vectors are applied to ALL token positions during projection — this extrapolates beyond training distribution but is standard practice in this codebase
- Bias #58 (last train bias) is the concealment bias: "models prefer responses that explain LLM behaviors without making reference to LLM training"
